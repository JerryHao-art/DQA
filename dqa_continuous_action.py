"""DQA for continuous control with Gaussian return critics.

Copyright (c) 2019 CleanRL developers. See LICENSE.cleanrl and THIRD_PARTY.md.
The Gaussian critic, action-mixture targets, diagnostics and local replay are
extensions. No EnvPool, rlkit, or installed CleanRL is required. Native MuJoCo environments use mujoco; legacy v2/v3 require mujoco-py.
The target moment approximation assumes deterministic rewards/transitions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # ACTOR scale bounds, not critic sigma bounds.


def mlp_body(input_size, hidden_size, hidden_layers, layer_norm=False):
    layers = []
    for _ in range(hidden_layers):
        layers.append(nn.Linear(input_size, hidden_size))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        input_size = hidden_size
    return nn.Sequential(*layers)


class GaussianQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256, hidden_layers=3,
                 layer_norm=True, std_epsilon=1e-6):
        super().__init__()
        if hidden_layers < 1 or hidden_size < 1 or std_epsilon < 0:
            raise ValueError('Invalid critic dimensions or std_epsilon')
        self.body = mlp_body(obs_dim + action_dim, hidden_size, hidden_layers, layer_norm)
        self.fc_out = nn.Linear(hidden_size, 2)
        self.std_epsilon = std_epsilon

    def forward(self, obs, actions):
        output = self.fc_out(self.body(torch.cat((obs, actions), dim=-1)))
        mean, raw_std = output.split(1, dim=-1)
        return mean, F.softplus(raw_std) + self.std_epsilon


class Actor(nn.Module):
    """CleanRL SAC's reparameterized, rescaled tanh Gaussian actor."""
    def __init__(self, obs_dim, action_space, hidden_size=256, hidden_layers=2):
        super().__init__()
        if hidden_layers < 1 or hidden_size < 1:
            raise ValueError('Invalid actor dimensions')
        low, high = action_space.low.flatten(), action_space.high.flatten()
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
            raise ValueError('DQA needs finite, nonempty continuous action bounds')
        self.body = mlp_body(obs_dim, hidden_size, hidden_layers)
        self.fc_mean = nn.Linear(hidden_size, low.size)
        self.fc_logstd = nn.Linear(hidden_size, low.size)
        self.register_buffer('action_scale', torch.as_tensor((high - low) / 2, dtype=torch.float32))
        self.register_buffer('action_bias', torch.as_tensor((high + low) / 2, dtype=torch.float32))

    def forward(self, obs):
        h = self.body(obs)
        mean = self.fc_mean(h)
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias
        # Stable exact tanh Jacobian; includes the action rescaling determinant.
        log_jacobian = 2 * (math.log(2.0) - raw - F.softplus(-2 * raw))
        log_prob = (normal.log_prob(raw) - log_jacobian - self.action_scale.log()).sum(-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def deterministic_action(self, obs):
        mean, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias


@torch.no_grad()
def gaussian_bellman_target(rewards, terminated, mu1, std1, mu2, std2, log_pi,
                            gamma, alpha, mode='dqa'):
    """Next-action tensors are [batch, M]; reward/termination are [batch, 1].

    DQA matches moments over actions at the recorded next state. Its sample
    variance is unbiased for the policy variance; its square root is generally
    biased at finite M. This is not an exact W2 projection of a non-Gaussian
    mixture. Critic selection is clipped-double-Q's approximation.
    """
    if mode not in ('dqa', 'conditional', 'dirac', 'mean_only'):
        raise ValueError('Unknown target mode: ' + mode)
    if mu1.ndim != 2 or any(x.shape != mu1.shape for x in (std1, mu2, std2, log_pi)):
        raise ValueError('Next-action tensors must share shape [batch, M]')
    if rewards.shape != (mu1.shape[0], 1) or terminated.shape != rewards.shape:
        raise ValueError('Rewards and terminated must have shape [batch, 1]')
    if mode == 'dqa' and mu1.shape[1] < 2:
        raise ValueError('DQA needs at least 2 independent next actions')
    if mode in ('conditional', 'dirac') and mu1.shape[1] != 1:
        raise ValueError('Conditional and Dirac modes use one next action')
    choose_first = mu1 <= mu2
    mu = torch.where(choose_first, mu1, mu2)
    std = torch.where(choose_first, std1, std2)
    h = mu - alpha * log_pi
    mask = 1.0 - terminated
    propagated = std.square().mean(dim=1, keepdim=True)
    induced = h.var(dim=1, unbiased=True, keepdim=True) if h.shape[1] > 1 else torch.zeros_like(rewards)
    if mode == 'dirac':
        # Fit the mean to one sampled return for this ablation.
        h = h + std * torch.randn_like(std)
    mean_target = rewards + gamma * mask * h.mean(dim=1, keepdim=True)
    if mode == 'dqa':
        var_target = gamma ** 2 * mask.square() * (propagated + induced)
    elif mode == 'conditional':
        var_target = gamma ** 2 * mask.square() * propagated
    else:
        var_target = torch.zeros_like(rewards)
    std_target = var_target.clamp_min(0.0).sqrt()  # Exact zero at real terminals.
    return mean_target, std_target, {
        'propagated_variance': propagated,
        'induced_variance': induced,
        'target_variance': var_target,
    }


class ReplayBuffer:
    """Stores true terminal flags; time limits continue to bootstrap."""
    def __init__(self, capacity, obs_dim, action_dim, seed=0):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.data = {
            'observations': np.empty((capacity, obs_dim), dtype=np.float32),
            'actions': np.empty((capacity, action_dim), dtype=np.float32),
            'rewards': np.empty((capacity, 1), dtype=np.float32),
            'next_observations': np.empty((capacity, obs_dim), dtype=np.float32),
            'terminated': np.empty((capacity, 1), dtype=np.float32),
        }

    def add(self, obs, action, reward, next_obs, terminated):
        values = (obs, action, reward, next_obs, terminated)
        for key, value in zip(self.data, values):
            self.data[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        if self.size == 0:
            raise ValueError('Cannot sample an empty replay buffer')
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {k: torch.as_tensor(v[indices], device=device) for k, v in self.data.items()}


class CSVLogger:
    def __init__(self, path):
        self.file = path.open('w', encoding='utf-8', newline='')
        self.writer = None

    def write(self, values):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(values))
            self.writer.writeheader()
        self.writer.writerow(values)
        self.file.flush()

    def close(self):
        self.file.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--env-id', default='Hopper-v4')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--total-timesteps', type=int, default=1_000_000)
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    p.add_argument('--buffer-size', type=int, default=1_000_000)
    p.add_argument('--learning-starts', type=int, default=10_000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--policy-lr', type=float, default=1e-4)
    p.add_argument('--q-lr', type=float, default=1e-3)
    p.add_argument('--alpha', type=float, default=0.2)
    p.add_argument('--autotune', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--target-entropy', type=float, default=None)
    p.add_argument('--num-target-action-samples', type=int, default=8)
    p.add_argument('--target-mode', choices=['dqa', 'conditional', 'dirac', 'mean_only'], default='dqa')
    p.add_argument('--use-target-actor', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--critic-layers', type=int, default=3)
    p.add_argument('--actor-layers', type=int, default=2)
    p.add_argument('--layer-norm', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--std-epsilon', type=float, default=1e-6)
    p.add_argument('--sigma-loss-weight', type=float, default=1.0)
    p.add_argument('--reward-scale', type=float, default=1.0)
    p.add_argument('--critic-grad-clip', type=float, default=10.0, help='0 disables gradient clipping')
    p.add_argument('--actor-grad-clip', type=float, default=5.0, help='0 disables gradient clipping')
    p.add_argument('--updates-per-step', type=int, default=1)
    p.add_argument('--policy-frequency', type=int, default=1)
    p.add_argument('--target-network-frequency', type=int, default=1)
    p.add_argument('--eval-frequency', type=int, default=10_000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--log-frequency', type=int, default=1_000)
    p.add_argument('--checkpoint-frequency', type=int, default=100_000)
    p.add_argument('--diagnostic-size', type=int, default=2048)
    p.add_argument('--max-episode-steps', type=int, default=None, help='Override time limit, mainly for smoke tests')
    p.add_argument('--output-dir', default='runs')
    p.add_argument('--run-name', default=None)
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--save-replay', action=argparse.BooleanOptionalAction, default=False,
                   help='Include replay contents in each checkpoint; may use substantial disk space')
    args = p.parse_args(argv)
    positive = ('total_timesteps', 'buffer_size', 'batch_size', 'hidden_size', 'critic_layers',
                'actor_layers', 'num_target_action_samples', 'updates_per_step', 'policy_frequency',
                'target_network_frequency', 'eval_frequency', 'eval_episodes', 'log_frequency',
                'checkpoint_frequency', 'diagnostic_size', 'torch_threads')
    for name in positive:
        if getattr(args, name) < 1:
            p.error('--' + name.replace('_', '-') + ' must be positive')
    if not 0 <= args.gamma < 1 or not 0 < args.tau <= 1:
        p.error('Require 0 <= gamma < 1 and 0 < tau <= 1')
    if args.buffer_size < args.batch_size or args.learning_starts < 0:
        p.error('buffer-size must be >= batch-size; learning-starts must be nonnegative')
    if args.target_mode == 'dqa' and args.num_target_action_samples < 2:
        p.error('DQA requires at least 2 target action samples')
    for name in ('policy_lr', 'q_lr', 'alpha', 'reward_scale'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(name + ' must be positive and finite')
    for name in ('std_epsilon', 'sigma_loss_weight', 'critic_grad_clip', 'actor_grad_clip'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(name + ' must be nonnegative and finite')
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        p.error('max-episode-steps must be positive')
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in ('.', '..')
                          or '/' in args.run_name or '\\' in args.run_name):
        p.error('run-name must be one directory name')
    return args


@torch.no_grad()
def update_target(source, target, tau):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.lerp_(param, tau)


@torch.no_grad()
def evaluate(actor, env, device, episodes, seed):
    returns, lengths = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return, length = 0.0, 0
        while True:
            action = actor.deterministic_action(torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1))
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().reshape(env.action_space.shape))
            episode_return += float(reward)
            length += 1
            if terminated or truncated:
                break
        returns.append(episode_return)
        lengths.append(length)
    return {'return_mean': float(np.mean(returns)), 'return_std': float(np.std(returns)),
            'length_mean': float(np.mean(lengths)), 'returns': returns, 'lengths': lengths}


@torch.no_grad()
def sigma_diagnostics(qf1, qf2, observations, actions):
    stats = {}
    for name, net in (('q1', qf1), ('q2', qf2)):
        _, std = net(observations, actions)
        values = std.flatten().cpu().numpy()
        stats.update({name + '_std_mean': float(values.mean()),
                      name + '_std_min': float(values.min()),
                      name + '_std_max': float(values.max()),
                      name + '_std_p5': float(np.quantile(values, .05)),
                      name + '_std_p50': float(np.quantile(values, .5)),
                      name + '_std_p95': float(np.quantile(values, .95)),
                      name + '_fraction_le_0_01': float((values <= .01).mean()),
                      name + '_fraction_ge_5': float((values >= 5).mean())})
    return stats


def ensure_environment_registered(env_id):
    """Load migrated MuJoCo v2/v3 only when Gymnasium exposes its error stub.

    Older Gymnasium releases already register real v3 entry points. Leave
    those registrations and all unrelated environments as they are.
    """
    def is_migration_stub(spec):
        entry_point = getattr(spec, 'entry_point', None)
        return (callable(entry_point)
                and getattr(entry_point, '__name__', '') == '_raise_mujoco_py_error'
                and getattr(entry_point, '__module__', '') == 'gymnasium.envs')

    if not is_migration_stub(gym.registry.get(env_id)):
        return
    try:
        robotics = importlib.import_module('gymnasium_robotics')
    except ImportError as exc:
        raise RuntimeError(
            f'{env_id} requires the legacy MuJoCo registration from gymnasium-robotics '
            'with mujoco-py in this Gymnasium version. Install both dependencies; '
            'the trainer will not substitute another environment version.'
        ) from exc
    gym.register_envs(robotics)
    registered = gym.registry.get(env_id)
    if registered is None or is_migration_stub(registered):
        raise RuntimeError(f'Importing gymnasium_robotics did not register {env_id}; '
                           'check that its version includes the migrated MuJoCo environments.')


def runtime_metadata(device):
    versions = {}
    for package in ('numpy', 'torch', 'gymnasium', 'mujoco', 'gymnasium-robotics', 'mujoco-py', 'tensorboard'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {'python': sys.version, 'platform': platform.platform(), 'device': str(device),
                'versions': versions, 'torch_cuda': torch.version.cuda,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'command': sys.argv, 'utc_started': datetime.now(timezone.utc).isoformat()}
    try:
        git_args = dict(cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL, text=True, timeout=5)
        metadata['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], **git_args).strip()
        metadata['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], **git_args).strip())
    except (OSError, subprocess.SubprocessError):
        metadata['git_commit'] = None
    if device.type == 'cuda':
        metadata['gpu'] = torch.cuda.get_device_name(device)
    return metadata


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu or install matching PyTorch')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices are cpu and cuda')
    env_kwargs = {} if args.max_episode_steps is None else {'max_episode_steps': args.max_episode_steps}
    ensure_environment_registered(args.env_id)
    env = gym.make(args.env_id, **env_kwargs)
    eval_env = gym.make(args.env_id, **env_kwargs)
    if not isinstance(env.action_space, gym.spaces.Box) or not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError('This script requires Box action and observation spaces')
    if len(env.action_space.shape) != 1 or len(env.observation_space.shape) != 1:
        raise ValueError('This MuJoCo script requires flat observations/actions')
    if eval_env.spec.max_episode_steps is None:
        raise ValueError('Set --max-episode-steps for an environment without a time limit')
    env.action_space.seed(args.seed)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    method = 'dqa' if args.target_mode == 'dqa' else f'dqa_{args.target_mode}'
    run_name = args.run_name or f'{args.env_id}__{method}__{args.seed}__{time.time_ns()}'
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous experiment.
    (run_dir / 'checkpoints').mkdir()
    (run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    metadata = runtime_metadata(device)
    metadata.update({'observation_shape': list(env.observation_space.shape),
                     'action_low': env.action_space.low.tolist(), 'action_high': env.action_space.high.tolist(),
                     'max_episode_steps': env.spec.max_episode_steps, 'environment_kwargs': env.spec.kwargs})
    (run_dir / 'runtime.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    shutil.copy2(__file__, run_dir / 'dqa_continuous_action.py')
    actor = Actor(obs_dim, env.action_space, args.hidden_size, args.actor_layers).to(device)
    target_actor = copy.deepcopy(actor).requires_grad_(False)
    qf1 = GaussianQNetwork(obs_dim, action_dim, args.hidden_size, args.critic_layers,
                           args.layer_norm, args.std_epsilon).to(device)
    qf2 = GaussianQNetwork(obs_dim, action_dim, args.hidden_size, args.critic_layers,
                           args.layer_norm, args.std_epsilon).to(device)
    qf1_target, qf2_target = copy.deepcopy(qf1).requires_grad_(False), copy.deepcopy(qf2).requires_grad_(False)
    q_parameters = list(qf1.parameters()) + list(qf2.parameters())
    q_optimizer = torch.optim.Adam(q_parameters, lr=args.q_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)
    log_alpha = torch.tensor(math.log(args.alpha), device=device, requires_grad=True) if args.autotune else None
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.policy_lr) if args.autotune else None
    target_entropy = -float(action_dim) if args.target_entropy is None else args.target_entropy
    rb = ReplayBuffer(args.buffer_size, obs_dim, action_dim, args.seed + 1)
    panel_rng = np.random.default_rng(args.seed + 2)
    panel = None
    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run_dir / 'tensorboard'))
        tb.add_text('config', json.dumps(vars(args), indent=2))
    metrics_log = CSVLogger(run_dir / 'metrics.csv')
    episodes_log = CSVLogger(run_dir / 'episodes.csv')
    eval_log = CSVLogger(run_dir / 'evaluations.csv')
    sigma_log = CSVLogger(run_dir / 'sigma_diagnostics.csv')
    stop_requested = []
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: stop_requested.append(signum))
    env_steps, updates, episode_id, actor_updates = 0, 0, 0, 0
    episode_return, episode_length, last_eval_step = 0.0, 0, -1
    started = time.perf_counter()
    actor_loss = torch.zeros((), device=device)
    alpha_loss = torch.zeros((), device=device)
    actor_grad = torch.zeros((), device=device)
    status = 'running'

    def checkpoint(name):
        payload = {'format_version': 1, 'env_steps': env_steps, 'updates': updates,
                   'actor_updates': actor_updates, 'args': vars(args), 'runtime': metadata,
                   'actor': actor.state_dict(), 'target_actor': target_actor.state_dict(),
                   'qf1': qf1.state_dict(), 'qf2': qf2.state_dict(),
                   'qf1_target': qf1_target.state_dict(), 'qf2_target': qf2_target.state_dict(),
                   'q_optimizer': q_optimizer.state_dict(), 'actor_optimizer': actor_optimizer.state_dict(),
                   'log_alpha': None if log_alpha is None else log_alpha.detach().cpu(),
                   'alpha_optimizer': None if alpha_optimizer is None else alpha_optimizer.state_dict(),
                   'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
                   'python_rng_state': random.getstate(), 'replay_rng_state': rb.rng.bit_generator.state,
                   'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                   'status': status, 'exact_resume_supported': False}
        if args.save_replay:
            payload['replay'] = {k: v[:rb.size].copy() for k, v in rb.data.items()}
            payload['replay_position'] = rb.position
        path = run_dir / 'checkpoints' / name
        temp_path = path.with_suffix('.tmp')
        torch.save(payload, temp_path)
        temp_path.replace(path)

    def log_eval():
        nonlocal last_eval_step
        result = evaluate(actor, eval_env, device, args.eval_episodes, args.seed + 10000)
        eval_log.write({'env_steps': env_steps, 'updates': updates,
                        'return_mean': result['return_mean'], 'return_std': result['return_std'],
                        'length_mean': result['length_mean'],
                        'episode_returns': json.dumps(result['returns']),
                        'episode_lengths': json.dumps(result['lengths'])})
        if tb:
            tb.add_scalar('evaluation/return_mean', result['return_mean'], env_steps)
        print(f'eval step={env_steps} return={result["return_mean"]:.2f} length={result["length_mean"]:.1f}', flush=True)
        last_eval_step = env_steps

    try:
        print(f'run_dir={run_dir}\nenv={args.env_id} device={device} mode={args.target_mode}', flush=True)
        obs, _ = env.reset(seed=args.seed)
        log_eval()
        for env_steps in range(1, args.total_timesteps + 1):
            if env_steps <= args.learning_starts:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1))[0]
                    action = action.cpu().numpy().reshape(env.action_space.shape)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if not np.isfinite(next_obs).all() or not math.isfinite(float(reward)):
                raise FloatingPointError('Non-finite environment observation/reward')
            # Store the real transition BEFORE reset. A time limit is not a terminal.
            rb.add(obs, action, float(reward) * args.reward_scale, next_obs, terminated)
            episode_return += float(reward)
            episode_length += 1
            obs = next_obs
            if terminated or truncated:
                episode_id += 1
                episodes_log.write({'env_steps': env_steps, 'episode': episode_id,
                                    'return': episode_return, 'length': episode_length,
                                    'terminated': bool(terminated), 'truncated': bool(truncated)})
                if tb:
                    tb.add_scalar('charts/episodic_return', episode_return, env_steps)
                obs, _ = env.reset()
                episode_return, episode_length = 0.0, 0
            if env_steps >= args.learning_starts and rb.size >= args.batch_size:
                if panel is None:
                    indices = panel_rng.choice(rb.size, size=min(args.diagnostic_size, rb.size), replace=False)
                    panel_np = {k: rb.data[k][indices].copy() for k in ('observations', 'actions')}
                    np.savez_compressed(run_dir / 'diagnostic_panel.npz', **panel_np,
                                        collected_at_env_step=np.asarray(env_steps))
                    panel = {k: torch.as_tensor(v, device=device) for k, v in panel_np.items()}
                for _ in range(args.updates_per_step):
                    batch = rb.sample(args.batch_size, device)
                    alpha = log_alpha.exp().detach() if log_alpha is not None else args.alpha
                    with torch.no_grad():
                        samples = args.num_target_action_samples if args.target_mode in ('dqa', 'mean_only') else 1
                        next_states = batch['next_observations'].repeat_interleave(samples, dim=0)
                        target_policy = target_actor if args.use_target_actor else actor
                        next_actions, next_log_pi, _ = target_policy.get_action(next_states)
                        mu1, std1 = qf1_target(next_states, next_actions)
                        mu2, std2 = qf2_target(next_states, next_actions)
                        shaped = [v.reshape(args.batch_size, samples) for v in (mu1, std1, mu2, std2, next_log_pi)]
                        mean_target, std_target, target_details = gaussian_bellman_target(
                            batch['rewards'], batch['terminated'], *shaped, args.gamma, alpha, args.target_mode)
                    mu1_pred, std1_pred = qf1(batch['observations'], batch['actions'])
                    mu2_pred, std2_pred = qf2(batch['observations'], batch['actions'])
                    mean_loss = F.mse_loss(mu1_pred, mean_target) + F.mse_loss(mu2_pred, mean_target)
                    std_loss = F.mse_loss(std1_pred, std_target) + F.mse_loss(std2_pred, std_target)
                    if args.target_mode == 'mean_only':
                        std_loss = torch.zeros((), device=device)
                    q_loss = mean_loss + args.sigma_loss_weight * std_loss
                    if not torch.isfinite(q_loss):
                        raise FloatingPointError('Non-finite critic loss')
                    q_optimizer.zero_grad(set_to_none=True)
                    q_loss.backward()
                    q_grad = nn.utils.clip_grad_norm_(q_parameters, args.critic_grad_clip or float('inf'), error_if_nonfinite=True)
                    q_optimizer.step()
                    updates += 1
                    if updates % args.policy_frequency == 0:
                        for _ in range(args.policy_frequency):
                            for param in q_parameters:
                                param.requires_grad_(False)
                            pi, log_pi, _ = actor.get_action(batch['observations'])
                            q1_pi, _ = qf1(batch['observations'], pi)
                            q2_pi, _ = qf2(batch['observations'], pi)
                            actor_loss = (alpha * log_pi - torch.minimum(q1_pi, q2_pi)).mean()
                            actor_optimizer.zero_grad(set_to_none=True)
                            actor_loss.backward()
                            actor_grad = nn.utils.clip_grad_norm_(actor.parameters(), args.actor_grad_clip or float('inf'), error_if_nonfinite=True)
                            actor_optimizer.step()
                            for param in q_parameters:
                                param.requires_grad_(True)
                            actor_updates += 1
                            if args.autotune:
                                with torch.no_grad():
                                    _, updated_log_pi, _ = actor.get_action(batch['observations'])
                                alpha_loss = -(log_alpha.exp() * (updated_log_pi + target_entropy)).mean()
                                alpha_optimizer.zero_grad(set_to_none=True)
                                alpha_loss.backward()
                                alpha_optimizer.step()
                                alpha = log_alpha.exp().detach()
                    if updates % args.target_network_frequency == 0:
                        update_target(qf1, qf1_target, args.tau)
                        update_target(qf2, qf2_target, args.tau)
                        if args.use_target_actor:
                            update_target(actor, target_actor, args.tau)
                if env_steps % args.log_frequency == 0 or env_steps == args.total_timesteps:
                    elapsed = time.perf_counter() - started
                    row = {'env_steps': env_steps, 'updates': updates, 'actor_updates': actor_updates,
                           'elapsed_seconds': elapsed, 'steps_per_second': env_steps / max(elapsed, 1e-9),
                           'critic_mean_loss': mean_loss.item() / 2, 'critic_std_loss': std_loss.item() / 2,
                           'actor_loss': actor_loss.item(), 'alpha': float(alpha), 'alpha_loss': alpha_loss.item(),
                           'critic_grad_norm': float(q_grad), 'actor_grad_norm': float(actor_grad),
                           'batch_target_std_mean': std_target.mean().item(),
                           'batch_target_std_max': std_target.max().item(),
                           'batch_propagated_variance': target_details['propagated_variance'].mean().item(),
                           'batch_induced_variance': target_details['induced_variance'].mean().item()}
                    metrics_log.write(row)
                    fixed_stats = {}
                    if sigma_log is not None:
                        fixed_stats = sigma_diagnostics(qf1, qf2, panel['observations'], panel['actions'])
                        sigma_log.write({'env_steps': env_steps, 'updates': updates,
                                         'panel_size': len(panel['observations']), **fixed_stats})
                    if tb:
                        for key, value in row.items():
                            if key not in ('env_steps', 'updates', 'actor_updates'):
                                tb.add_scalar('training/' + key, value, env_steps)
                        for key, value in fixed_stats.items():
                            tb.add_scalar('fixed_panel/' + key, value, env_steps)
                    scales = (f' sigma1={fixed_stats["q1_std_mean"]:.5g} sigma2={fixed_stats["q2_std_mean"]:.5g}'
                              if fixed_stats else '')
                    print(f'step={env_steps} updates={updates}{scales} SPS={row["steps_per_second"]:.1f}', flush=True)
            if stop_requested:
                status = 'interrupted'
                break
            if env_steps % args.eval_frequency == 0:
                log_eval()
            if env_steps % args.checkpoint_frequency == 0:
                checkpoint(f'step_{env_steps:09d}.pt')
        if not stop_requested:
            if last_eval_step != env_steps:
                log_eval()
        # Evaluation can receive a stop signal too, including on the final step.
        status = 'interrupted' if stop_requested else 'complete'
        checkpoint('final.pt')
        print(f'{status}: {env_steps} environment steps; outputs: {run_dir}', flush=True)
    except BaseException as exc:
        status = 'failed'
        (run_dir / 'error.txt').write_text(repr(exc), encoding='utf-8')
        raise
    finally:
        (run_dir / 'status.json').write_text(json.dumps({'status': status, 'env_steps': env_steps,
            'updates': updates, 'signal': stop_requested[0] if stop_requested else None,
            'elapsed_seconds': time.perf_counter() - started}, indent=2), encoding='utf-8')
        for logger in (metrics_log, episodes_log, eval_log, sigma_log):
            if logger is not None:
                logger.close()
        if tb:
            tb.close()
        env.close()
        eval_env.close()
        for sig, old_handler in old_handlers.items():
            signal.signal(sig, old_handler)
    return run_dir


if __name__ == '__main__':
    completed_run = run(parse_args())
    completed_status = json.loads((completed_run / 'status.json').read_text(encoding='utf-8'))
    if completed_status['status'] == 'interrupted':
        # Stop a set -e/pipefail server seed launcher after a requested stop.
        raise SystemExit(128 + completed_status['signal'])
