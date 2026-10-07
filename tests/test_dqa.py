"""Focused mathematical and data-contract tests for the standalone DQA trainer.

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import contextlib
import csv
import io
import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import torch

import dqa_continuous_action as trainer
from dqa_continuous_action import (
    Actor,
    GaussianQNetwork,
    ReplayBuffer,
    gaussian_bellman_target,
)


class BellmanTargetTests(unittest.TestCase):

    def test_dqa_known_mixture_moments(self):
        mu = torch.tensor([[-1.0, 0.0, 1.0]])
        std = torch.tensor([[1.0, 2.0, 3.0]])
        mean_target, std_target, _ = gaussian_bellman_target(
            torch.tensor([[2.0]]), torch.zeros(1, 1),
            mu, std, mu + 10.0, std + 10.0, torch.zeros_like(mu),
            gamma=0.9, alpha=0.2, mode="dqa",
        )
        # Unbiased action variance is 1, within-action variance is 14 / 3.
        expected_variance = 0.9**2 * (14.0 / 3.0 + 1.0)
        torch.testing.assert_close(mean_target, torch.tensor([[2.0]]))
        torch.testing.assert_close(
            std_target.square(), torch.tensor([[expected_variance]])
        )

    def test_entropy_is_inside_action_variance(self):
        mu = torch.tensor([[-1.0, 0.0, 2.0]])
        log_pi = torch.tensor([[-0.5, -1.0, -3.0]])
        std = torch.full_like(mu, 0.5)
        mean_target, std_target, _ = gaussian_bellman_target(
            torch.zeros(1, 1), torch.zeros(1, 1),
            mu, std, mu + 10.0, std, log_pi,
            gamma=0.8, alpha=0.4, mode="dqa",
        )
        # h = mu - alpha * log_pi = [-0.8, 0.4, 3.2].
        expected_h = [-0.8, 0.4, 3.2]
        expected_h_mean = sum(expected_h) / 3.0
        expected_h_var = sum((h - expected_h_mean)**2 for h in expected_h) / 2.0
        torch.testing.assert_close(
            mean_target, torch.tensor([[0.8 * expected_h_mean]])
        )
        torch.testing.assert_close(
            std_target.square(), torch.tensor([[0.8**2 * (0.25 + expected_h_var)]])
        )
        # This construction has nonzero covariance; summing separate variances
        # would not give the correct soft-return variance.
        incorrectly_split_var = mu.var(unbiased=True) + 0.4**2 * log_pi.var(unbiased=True)
        self.assertGreater(abs(expected_h_var - incorrectly_split_var.item()), 0.5)

    def test_selected_mean_and_scale_come_from_same_critic(self):
        mean_target, std_target, _ = gaussian_bellman_target(
            torch.zeros(1, 1), torch.zeros(1, 1),
            torch.tensor([[0.0, 4.0, 2.0]]), torch.tensor([[1.0, 20.0, 3.0]]),
            torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[30.0, 2.0, 40.0]]),
            torch.zeros(1, 3), gamma=1.0, alpha=0.0, mode="dqa",
        )
        # Chosen means [0, 2, 2], scales [1, 2, 3].
        torch.testing.assert_close(mean_target, torch.tensor([[4.0 / 3.0]]))
        torch.testing.assert_close(std_target.square(), torch.tensor([[6.0]]))

    def test_terminal_has_reward_mean_and_exact_zero_scale(self):
        mu = torch.tensor([[2.0, 4.0], [2.0, 4.0]])
        std = torch.full_like(mu, 20.0)
        mean_target, std_target, _ = gaussian_bellman_target(
            torch.tensor([[3.0], [3.0]]), torch.tensor([[1.0], [0.0]]),
            mu, std, mu + 1.0, std, torch.zeros_like(mu),
            gamma=0.99, alpha=0.2, mode="dqa",
        )
        self.assertEqual(mean_target[0, 0].item(), 3.0)
        self.assertEqual(std_target[0, 0].item(), 0.0)
        self.assertGreater(mean_target[1, 0].item(), 3.0)
        # Neither target scale nor variance may be clipped to the old bound 5.
        self.assertGreater(std_target[1, 0].item(), 5.0)

    def test_bellman_targets_do_not_build_autograd_graph(self):
        rewards = torch.ones(2, 1, requires_grad=True)
        mu1 = torch.ones(2, 3, requires_grad=True)
        std1 = torch.ones(2, 3, requires_grad=True)
        mu2 = torch.full((2, 3), 2.0, requires_grad=True)
        std2 = torch.full((2, 3), 2.0, requires_grad=True)
        log_pi = torch.full((2, 3), -0.2, requires_grad=True)
        mean_target, std_target, _ = gaussian_bellman_target(
            rewards, torch.zeros(2, 1), mu1, std1, mu2, std2, log_pi,
            gamma=0.99, alpha=0.2, mode="dqa",
        )
        self.assertFalse(mean_target.requires_grad)
        self.assertFalse(std_target.requires_grad)
        self.assertIsNone(mean_target.grad_fn)
        self.assertIsNone(std_target.grad_fn)

    def test_dqa_rejects_single_action_instead_of_nan_variance(self):
        x = torch.ones(2, 1)
        with self.assertRaises(ValueError):
            gaussian_bellman_target(
                x, torch.zeros_like(x), x, x, x, x, torch.zeros_like(x),
                gamma=0.99, alpha=0.2, mode="dqa",
            )

    def test_conditional_preserves_selected_scale_and_terminal_mask(self):
        mean_target, std_target, _ = gaussian_bellman_target(
            torch.tensor([[1.0], [1.0]]), torch.tensor([[0.0], [1.0]]),
            torch.tensor([[1.0], [1.0]]), torch.tensor([[9.0], [9.0]]),
            torch.tensor([[2.0], [2.0]]), torch.tensor([[0.1], [0.1]]),
            torch.tensor([[-1.0], [-1.0]]),
            gamma=0.5, alpha=0.2, mode="conditional",
        )
        torch.testing.assert_close(mean_target, torch.tensor([[1.6], [1.0]]))
        torch.testing.assert_close(std_target, torch.tensor([[4.5], [0.0]]))

    def test_mean_only_uses_same_action_average_with_zero_target_scale(self):
        mu = torch.tensor([[-1.0, 0.0, 1.0]])
        mean_target, std_target, _ = gaussian_bellman_target(
            torch.tensor([[2.0]]), torch.zeros(1, 1),
            mu, torch.ones_like(mu), mu + 10.0, torch.ones_like(mu),
            torch.tensor([[-1.0, -2.0, -3.0]]),
            gamma=0.9, alpha=0.2, mode="mean_only",
        )
        torch.testing.assert_close(mean_target, torch.tensor([[2.36]]))
        torch.testing.assert_close(std_target, torch.zeros(1, 1))

    def test_dirac_samples_return_but_has_zero_target_scale(self):
        torch.manual_seed(731)
        mu = torch.full((4096, 1), 2.0)
        std = torch.full_like(mu, 3.0)
        mean_target, std_target, _ = gaussian_bellman_target(
            torch.ones_like(mu), torch.zeros_like(mu),
            mu, std, mu + 10.0, std, torch.zeros_like(mu),
            gamma=0.5, alpha=0.2, mode="dirac",
        )
        # The point target samples 1 + 0.5 * N(2, 3^2).
        self.assertAlmostEqual(mean_target.mean().item(), 2.0, delta=0.15)
        self.assertAlmostEqual(mean_target.std().item(), 1.5, delta=0.15)
        torch.testing.assert_close(std_target, torch.zeros_like(std_target))


class NetworkTests(unittest.TestCase):
    def test_defaults_match_reference_configuration(self):
        args = trainer.parse_args([])
        self.assertEqual(args.target_mode, 'dqa')
        self.assertEqual(args.env_id, 'Hopper-v4')
        self.assertEqual(args.policy_lr, 1e-4)
        self.assertEqual(args.q_lr, 1e-3)
        self.assertEqual(args.num_target_action_samples, 8)
        self.assertEqual(args.alpha, 0.2)
        self.assertFalse(args.autotune)
        self.assertTrue(args.use_target_actor)
        self.assertTrue(args.layer_norm)
        self.assertEqual(args.critic_layers, 3)
        self.assertEqual(args.actor_layers, 2)
        for mode in ('sac', 'ltv'):
            with self.subTest(mode=mode), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    trainer.parse_args(['--target-mode', mode])

    def test_scale_above_five_has_nonzero_gradient(self):
        critic = GaussianQNetwork(3, 2, hidden_size=8, hidden_layers=1)
        with torch.no_grad():
            critic.fc_out.weight.zero_()
            critic.fc_out.bias.copy_(torch.tensor([2.0, 20.0]))
        mean, std = critic(torch.zeros(4, 3), torch.zeros(4, 2))
        self.assertEqual(tuple(mean.shape), (4, 1))
        self.assertEqual(tuple(std.shape), (4, 1))
        self.assertTrue(torch.all(std > 5.0))
        std.mean().backward()
        self.assertGreater(critic.fc_out.bias.grad[1].item(), 0.99)
        self.assertEqual(critic.fc_out.bias.grad[0].item(), 0.0)

    def test_scale_remains_positive_for_very_negative_raw_output(self):
        critic = GaussianQNetwork(3, 2, hidden_size=8, hidden_layers=1)
        with torch.no_grad():
            critic.fc_out.weight.zero_()
            critic.fc_out.bias.copy_(torch.tensor([-50.0, -100.0]))
        _, std = critic(torch.zeros(2, 3), torch.zeros(2, 2))
        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue(torch.all(std > 0.0))
        self.assertTrue(torch.all(std < 0.01))

    def test_actor_respects_asymmetric_action_bounds(self):
        action_space = gym.spaces.Box(
            low=np.array([-3.0, 2.0], dtype=np.float32),
            high=np.array([1.0, 8.0], dtype=np.float32),
        )
        actor = Actor(3, action_space, hidden_size=8, hidden_layers=1)
        for parameter in actor.parameters():
            torch.nn.init.zeros_(parameter)
        obs = torch.zeros(64, 3)
        action, log_prob, mean_action = actor.get_action(obs)
        expected_mean = torch.tensor([-1.0, 5.0]).expand(64, 2)
        torch.testing.assert_close(mean_action, expected_mean)
        torch.testing.assert_close(actor.deterministic_action(obs), expected_mean)
        self.assertEqual(tuple(log_prob.shape), (64, 1))
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(torch.all(action >= torch.tensor(action_space.low)))
        self.assertTrue(torch.all(action <= torch.tensor(action_space.high)))


class ReplayTests(unittest.TestCase):
    def test_shapes_and_overwritten_entries(self):
        replay = ReplayBuffer(capacity=3, obs_dim=2, action_dim=1, seed=11)
        for value in range(5):
            replay.add(
                np.array([value, value + 0.5], dtype=np.float32),
                np.array([value / 10.0], dtype=np.float32),
                float(value),
                np.array([value + 1.0, value + 1.5], dtype=np.float32),
                False,
            )
        batch = replay.sample(64, torch.device("cpu"))
        for key in ("observations", "next_observations"):
            self.assertEqual(tuple(batch[key].shape), (64, 2))
        for key in ("actions", "rewards", "terminated"):
            self.assertEqual(tuple(batch[key].shape), (64, 1))
        ids = batch["observations"][:, 0]
        self.assertTrue(torch.all((ids >= 2) & (ids <= 4)))
        torch.testing.assert_close(batch["rewards"][:, 0], ids)
        torch.testing.assert_close(batch["next_observations"][:, 0], ids + 1.0)
        torch.testing.assert_close(batch["actions"][:, 0], ids / 10.0)

    def test_bootstrap_mask_stores_termination_only(self):
        # The environment loop passes terminated, not terminated OR truncated.
        # A timeout therefore stores False, preserving bootstrapping.
        replay = ReplayBuffer(capacity=2, obs_dim=1, action_dim=1, seed=12)
        for true_terminal in (False, True):
            replay.add(
                np.array([float(true_terminal)], dtype=np.float32),
                np.zeros(1, dtype=np.float32), 0.0,
                np.ones(1, dtype=np.float32), true_terminal,
            )
        batch = replay.sample(64, torch.device("cpu"))
        torch.testing.assert_close(batch["terminated"], batch["observations"])
        self.assertEqual(set(batch["terminated"].flatten().tolist()), {0.0, 1.0})


class _ResetSentinelEnv(gym.Env):
    """Distinct final/reset observations expose accidental autoreset storage."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.spec = SimpleNamespace(max_episode_steps=2, kwargs={})
        self.episode = 0
        self.episode_step = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.episode_step = 0
        return np.array([-10.0 * self.episode], dtype=np.float32), {}

    def step(self, action):
        self.episode_step += 1
        done = self.episode_step == 2
        true_terminal = done and self.episode % 2 == 0
        timeout = done and not true_terminal
        observation = np.array([10.0 * self.episode + self.episode_step], dtype=np.float32)
        return observation, 1.0, true_terminal, timeout, {}

    def close(self):
        self.closed = True


class TrainingLoopTests(unittest.TestCase):

    def test_final_observation_and_timeout_mask_survive_training_and_checkpoint(self):
        train_env, eval_env = _ResetSentinelEnv(), _ResetSentinelEnv()
        with tempfile.TemporaryDirectory() as output_dir:
            args = trainer.parse_args([
                "--device", "cpu", "--output-dir", output_dir, "--run-name", "mask_test",
                "--total-timesteps", "4", "--learning-starts", "2", "--buffer-size", "8",
                "--batch-size", "2", "--hidden-size", "8", "--actor-layers", "1",
                "--critic-layers", "1", "--num-target-action-samples", "2",
                "--diagnostic-size", "2", "--eval-frequency", "4", "--eval-episodes", "1",
                "--log-frequency", "1", "--checkpoint-frequency", "4", "--torch-threads", "1",
                "--autotune", "--save-replay",
            ])
            with mock.patch.object(trainer.gym, "make", side_effect=[train_env, eval_env]), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_dir = trainer.run(args)
            payload = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
            replay = payload["replay"]
            np.testing.assert_array_equal(replay["observations"].flatten(), [-10.0, 11.0, -20.0, 21.0])
            np.testing.assert_array_equal(replay["next_observations"].flatten(), [11.0, 12.0, 21.0, 22.0])
            np.testing.assert_array_equal(replay["terminated"].flatten(), [0.0, 0.0, 0.0, 1.0])
            self.assertEqual(payload["env_steps"], 4)
            self.assertEqual(payload["updates"], 3)
            self.assertEqual(payload["status"], "complete")
            self.assertTrue(torch.isfinite(payload["log_alpha"]).all())
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")
            for filename in ("metrics.csv", "evaluations.csv", "episodes.csv", "sigma_diagnostics.csv"):
                self.assertGreater((Path(run_dir) / filename).stat().st_size, 0)
            self.assertTrue(train_env.closed)
            self.assertTrue(eval_env.closed)

    def test_sigterm_saves_interrupted_run_and_restores_handlers(self):
        class SignalEnv(_ResetSentinelEnv):
            def step(self, action):
                result = super().step(action)
                signal.raise_signal(signal.SIGTERM)
                return result

        train_env, eval_env = SignalEnv(), _ResetSentinelEnv()
        original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                args = trainer.parse_args([
                    "--device", "cpu", "--output-dir", output_dir, "--run-name", "signal_test",
                    "--total-timesteps", "4", "--learning-starts", "2", "--buffer-size", "8",
                    "--batch-size", "2", "--hidden-size", "8", "--actor-layers", "1",
                    "--critic-layers", "1", "--num-target-action-samples", "2",
                    "--eval-episodes", "1", "--torch-threads", "1",
                ])
                with mock.patch.object(trainer.gym, "make", side_effect=[train_env, eval_env]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_dir = trainer.run(args)
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "interrupted")
                self.assertEqual(status["signal"], signal.SIGTERM)
                self.assertEqual(status["env_steps"], 1)
                payload = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
                self.assertEqual(payload["status"], "interrupted")
                self.assertEqual(payload["env_steps"], 1)
                self.assertEqual(payload["updates"], 0)
                # Only the initial evaluation ran; interruption skips final evaluation.
                self.assertEqual(eval_env.episode, 1)
                self.assertTrue(train_env.closed)
                self.assertTrue(eval_env.closed)
                for sig, handler in original_handlers.items():
                    self.assertEqual(signal.getsignal(sig), handler)
        finally:
            # Preserve the test process even if a regression fails an assertion.
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)

    def test_sigterm_during_final_evaluation_still_marks_run_interrupted(self):
        class SignalOnSecondEvaluation(_ResetSentinelEnv):
            def step(self, action):
                result = super().step(action)
                if self.episode == 2 and self.episode_step == 1:
                    signal.raise_signal(signal.SIGTERM)
                return result

        # Frequency 4 evaluates at the last training step. Frequency 99 invokes
        # the separate final-evaluation branch after the training loop.
        for frequency in (4, 99):
            with self.subTest(eval_frequency=frequency), tempfile.TemporaryDirectory() as output_dir:
                train_env, eval_env = _ResetSentinelEnv(), SignalOnSecondEvaluation()
                args = trainer.parse_args([
                    "--device", "cpu", "--output-dir", output_dir, "--run-name", "eval_signal_test",
                    "--total-timesteps", "4", "--learning-starts", "10", "--buffer-size", "8",
                    "--batch-size", "2", "--hidden-size", "8", "--actor-layers", "1",
                    "--critic-layers", "1", "--eval-frequency", str(frequency),
                    "--eval-episodes", "1", "--torch-threads", "1",
                ])
                with mock.patch.object(trainer.gym, "make", side_effect=[train_env, eval_env]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_dir = trainer.run(args)
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "interrupted")
                self.assertEqual(status["signal"], signal.SIGTERM)
                self.assertEqual(status["env_steps"], 4)
                payload = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
                self.assertEqual(payload["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
