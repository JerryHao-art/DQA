"""Run independent DQA experiments across selected GPUs.

Use --dry-run to inspect assignments. Trainer options follow --.
Each GPU has a fixed queue and an adjustable concurrency limit.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


DEFAULT_ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4']
DEFAULT_SEEDS = [12345, 22345, 32345, 42345, 52345]
TARGET_MODES = ['dqa', 'conditional', 'dirac', 'mean_only']
MANAGED_FLAGS = {
    '--env-id', '--seed', '--device', '--output-dir', '--run-name',
    '--total-timesteps', '--torch-threads', '--target-mode', '--target-modes',
    '--trainer', '--jobs-per-gpu', '--algorithm',
}


def _gpu_tokens(spec):
    tokens = [token.strip() for token in spec.split(',')]
    if not tokens or any(not token for token in tokens):
        raise ValueError('GPU IDs must be a nonempty comma-separated list')
    uuid_pattern = r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
    normalized = []
    for token in tokens:
        current_uuid = re.fullmatch(r'(GPU-|MIG-)(' + uuid_pattern + ')', token)
        legacy_mig = re.fullmatch(r'MIG-GPU-(' + uuid_pattern + r')/([0-9]+)/([0-9]+)', token)
        if re.fullmatch(r'[0-9]+', token):
            normalized.append(str(int(token)))
        elif current_uuid:
            normalized.append(current_uuid[1] + current_uuid[2].lower())
        elif legacy_mig:
            normalized.append(f'MIG-GPU-{legacy_mig[1].lower()}/{int(legacy_mig[2])}/{int(legacy_mig[3])}')
        else:
            raise ValueError(f'Invalid GPU ID: {token!r}')
    tokens = normalized
    families = {'index' if token.isdigit() else ('legacy_MIG' if token.startswith('MIG-GPU-')
                else token.split('-', 1)[0]) for token in tokens}
    if len(families) != 1:
        raise ValueError('Use only indices, only full GPU UUIDs, or only full MIG UUIDs; do not mix aliases')
    if len(set(tokens)) != len(tokens):
        raise ValueError('Duplicate GPU IDs would launch multiple workers on one GPU')
    return tokens


def resolve_gpus(spec, environ=None, detect_count=None):
    """Keep inherited CUDA identifiers, rather than reinterpreting local ordinals."""
    environ = os.environ if environ is None else environ
    visible = None
    if 'CUDA_VISIBLE_DEVICES' in environ:
        value = environ['CUDA_VISIBLE_DEVICES'].strip()
        if value in ('', '-1'):
            raise ValueError('CUDA_VISIBLE_DEVICES currently disables all GPUs')
        visible = _gpu_tokens(value)
    if spec == 'auto':
        if visible is not None:
            return visible
        if detect_count is None:
            try:
                import torch
            except ImportError as exc:
                raise ValueError('Automatic GPU detection needs PyTorch; use explicit --gpus for a dry run') from exc
            detect_count = torch.cuda.device_count
        count = detect_count()
        if count < 1:
            raise ValueError('No visible CUDA GPUs; use explicit --gpus for a dry run')
        return [str(index) for index in range(count)]
    selected = _gpu_tokens(spec)
    if visible is not None and not set(selected).issubset(visible):
        raise ValueError('Select original IDs from CUDA_VISIBLE_DEVICES=' + ','.join(visible))
    return selected


def build_plan(env_ids, seeds, gpus, target_modes=None):
    if not env_ids or not seeds or not gpus:
        raise ValueError('At least one environment, seed, and GPU are required')
    for name, values in (('environment', env_ids), ('seed', seeds), ('GPU', gpus)):
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate {name} entries are not allowed')
    if target_modes is not None:
        if not target_modes or len(set(target_modes)) != len(target_modes):
            raise ValueError('Target modes must be nonempty and unique')
        if any(mode not in TARGET_MODES for mode in target_modes):
            raise ValueError('Unknown target mode')
    jobs = []
    for env_id in env_ids:
        for mode in target_modes if target_modes is not None else [None]:
            for seed in seeds:
                job = {'job_id': len(jobs) + 1, 'env_id': env_id, 'seed': seed,
                       'gpu': gpus[len(jobs) % len(gpus)]}
                if mode is not None:
                    job['target_mode'] = mode
                jobs.append(job)
    return jobs


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        boundary = argv.index('--')
        argv, extra = argv[:boundary], argv[boundary + 1:]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', default='auto', help='auto, comma-separated original CUDA IDs, or UUIDs')
    parser.add_argument('--env-ids', nargs='+', default=DEFAULT_ENVS)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--target-mode', choices=TARGET_MODES, help='Target construction (default: dqa)')
    modes.add_argument('--target-modes', nargs='+', choices=TARGET_MODES,
                       help='Run every environment x mode x seed combination')
    parser.add_argument('--threads', type=int, default=1, help='PyTorch/BLAS CPU threads per experiment')
    parser.add_argument('--jobs-per-gpu', type=int, default=1,
                        help='Maximum concurrent training processes per GPU in this launcher')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'runs'))
    parser.add_argument('--stop-timeout', type=float, default=60.0, help='Seconds allowed for stopped trainers to save before killing')
    parser.add_argument('--dry-run', action='store_true', help='Print assignment only; explicit GPUs need no CUDA installation')
    args = parser.parse_args(argv)
    args.target_mode = args.target_mode or 'dqa'
    if args.total_timesteps < 1 or args.threads < 1:
        parser.error('--total-timesteps and --threads must be positive')
    if args.jobs_per_gpu < 1:
        parser.error('--jobs-per-gpu must be positive')
    if not 0 < args.stop_timeout < float('inf'):
        parser.error('--stop-timeout must be finite and positive')
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        parser.error('Seeds must be in [0, 2**32)')
    if len(set(args.env_ids)) != len(args.env_ids) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Environment and seed lists must not contain duplicates')
    if args.target_modes and len(set(args.target_modes)) != len(args.target_modes):
        parser.error('Target mode list must not contain duplicates')
    for arg in extra:
        flag = arg.split('=', 1)[0]
        if flag in MANAGED_FLAGS or flag in ('--help', '-h'):
            parser.error(f'{flag} cannot be forwarded; configure it on the launcher')
    args.train_args = extra
    return args


def print_plan(jobs, gpus, jobs_per_gpu=1):
    print(f'{len(jobs)} experiments / {len(gpus)} GPUs; fixed assignment, '
          f'up to {jobs_per_gpu} concurrent processes per GPU', flush=True)
    for gpu in gpus:
        assigned = [job for job in jobs if job['gpu'] == gpu]
        concurrent = min(len(assigned), jobs_per_gpu)
        print(f'GPU {gpu}: {len(assigned)} experiments, {concurrent} concurrent, '
              f'{len(assigned) - concurrent} queued', flush=True)
        for job in assigned:
            mode = f' mode={job["target_mode"]}' if 'target_mode' in job else ''
            print(f'  #{job["job_id"]:03d} {job["env_id"]}{mode} seed={job["seed"]}', flush=True)


def _check_gpus(gpus):
    # Check before starting any experiment or creating output files. Child
    # visibility uses exactly the same identifier as the later training worker.
    for gpu in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        result = subprocess.run(
            [sys.executable, '-c',
             'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
             'x = torch.ones(1, device="cuda"); '
             'print(torch.cuda.get_device_name(0))'],
            env=env, capture_output=True, text=True, timeout=90, check=False)
        if result.returncode:
            raise RuntimeError(f'GPU {gpu} preflight failed:\n{result.stderr or result.stdout}')
        print(f'GPU {gpu}: {result.stdout.strip()}', flush=True)


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def run_plan(args, jobs, trainer_path=None):
    """Run preassigned queues; trainer_path is injectable for CPU-only tests."""
    trainer = Path(trainer_path or Path(__file__).with_name('dqa_continuous_action.py')).resolve()
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    output_dir = Path(args.output_dir).resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    launch_id = f'{stamp}_{uuid.uuid4().hex[:8]}'
    launch_dir = output_dir / 'launches' / launch_id
    log_dir = launch_dir / 'console_logs'
    log_dir.mkdir(parents=True, exist_ok=False)
    states = []
    queues = collections.OrderedDict()
    for job in jobs:
        safe_env = re.sub(r'[^A-Za-z0-9_.-]', '_', job['env_id'])
        mode = job.get('target_mode', args.target_mode)
        name = f'{launch_id}_job{job["job_id"]:04d}_{safe_env}_{mode}_seed{job["seed"]}'
        state = dict(job, target_mode=mode, run_name=name, status='pending', returncode=None,
                     log_file=str(log_dir / f'job{job["job_id"]:04d}.log'),
                     run_dir=str(output_dir / name))
        states.append(state)
        queues.setdefault(job['gpu'], collections.deque()).append(state)
    _write_json(launch_dir / 'plan.json', {'launch_id': launch_id, 'arguments': vars(args),
                                        'trainer': str(trainer), 'jobs': states})
    active = {}
    stop_signals = []
    old_handlers = {}
    summary = {'status': 'running', 'launch_id': launch_id, 'signal': None, 'jobs': states}
    exit_code = 0

    def save_status():
        _write_json(launch_dir / 'status.json', summary)

    def request_stop(signum, _frame):
        stop_signals.append(signum)

    def start_job(gpu):
        state = queues[gpu].popleft()
        command = [sys.executable, '-u', str(trainer), '--env-id', state['env_id'],
                   '--seed', str(state['seed']), '--device', 'cuda',
                   '--total-timesteps', str(args.total_timesteps), '--target-mode', state['target_mode'],
                   '--torch-threads', str(args.threads), '--output-dir', str(output_dir),
                   '--run-name', state['run_name'], *args.train_args]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(args.threads),
                   MKL_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED='1')
        log = Path(state['log_file']).open('w', encoding='utf-8')
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       start_new_session=(os.name == 'posix'))
        except BaseException:
            log.close()
            state['status'] = 'failed'
            raise
        state.update(status='running', pid=process.pid, started_at=time.time(), command=command)
        active[state['job_id']] = (process, log, state)
        print(f'Start GPU {gpu}: #{state["job_id"]} {state["env_id"]} '
              f'mode={state["target_mode"]} seed={state["seed"]}', flush=True)

    def finish_job(job_id, stopped=False):
        process, log, state = active.pop(job_id)
        code = process.wait()
        log.close()
        state.update(returncode=code, finished_at=time.time(),
                     status='interrupted' if stopped else ('complete' if code == 0 else 'failed'))
        print(f'{state["status"]} GPU {state["gpu"]}: #{state["job_id"]} exit={code}', flush=True)
        return code

    def stop_children():
        stopping = set()
        for job_id, (process, _log, _state) in list(active.items()):
            if process.poll() is None:
                try:
                    # SIGTERM reaches the trainer's checkpoint handler on Linux.
                    process.terminate()
                    stopping.add(job_id)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + args.stop_timeout
        while active:
            for job_id, (process, _log, _state) in list(active.items()):
                if process.poll() is not None:
                    finish_job(job_id, stopped=job_id in stopping)
            if not active:
                break
            if time.monotonic() >= deadline or len(stop_signals) > 1:
                for process, _log, _state in active.values():
                    if process.poll() is None:
                        process.kill()
                for job_id in list(active):
                    finish_job(job_id, stopped=True)
                break
            time.sleep(0.1)

    print(f'Launch directory: {launch_dir}', flush=True)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)
        save_status()
        while any(queues.values()) or active:
            changed = False
            if stop_signals:
                exit_code = 128 + stop_signals[0]
                break
            # Observe all finished jobs before starting any new ones: fail-fast
            # must not start another queued experiment on a different GPU.
            for job_id, (process, _log, _state) in list(active.items()):
                if process.poll() is not None:
                    changed = True
                    if finish_job(job_id) != 0:
                        exit_code = 1
            if exit_code:
                break
            active_per_gpu = collections.Counter(state['gpu'] for _process, _log, state in active.values())
            for gpu, queue in queues.items():
                if stop_signals:
                    break
                while queue and active_per_gpu[gpu] < args.jobs_per_gpu and not stop_signals:
                    start_job(gpu)
                    active_per_gpu[gpu] += 1
                    changed = True
            if changed:
                save_status()
            if active:
                time.sleep(0.1)
        summary['status'] = 'interrupted' if stop_signals else ('failed' if exit_code else 'complete')
    except Exception as exc:
        exit_code = 1
        summary.update(status='failed', error=repr(exc))
        print(f'Launcher failed: {exc}', file=sys.stderr, flush=True)
    finally:
        stop_children()
        for state in states:
            if state['status'] == 'pending':
                state['status'] = 'not_started'
        if stop_signals:
            summary.update(status='interrupted', signal=stop_signals[0])
            exit_code = 128 + stop_signals[0]
        summary.update(exit_code=exit_code, finished_at=time.time())
        save_status()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(f'{summary["status"]}; summary: {launch_dir / "status.json"}', flush=True)
    return exit_code


def main(argv=None):
    args = parse_args(argv)
    try:
        gpus = resolve_gpus(args.gpus)
        jobs = build_plan(args.env_ids, args.seeds, gpus, args.target_modes or [args.target_mode])
        print_plan(jobs, gpus, args.jobs_per_gpu)
        print(f'DQA trainer; extra training arguments: {args.train_args}', flush=True)
        if args.dry_run:
            return 0
        _check_gpus(list(dict.fromkeys(job['gpu'] for job in jobs)))
        return run_plan(args, jobs)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
