"""CPU-only tests for the fixed-allocation multi-GPU experiment launcher.

The integration tests use a tiny standard-library trainer, so no CUDA, PyTorch,
MuJoCo, or real training is required to check scheduling and failure handling.
"""

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import run_multi_gpu as launcher


FAKE_TRAINER = textwrap.dedent(
    """\
    import argparse
    import json
    import os
    from pathlib import Path
    import sys
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--total-timesteps', required=True, type=int)
    parser.add_argument('--torch-threads', required=True, type=int)
    parser.add_argument('--target-mode', required=True)
    parser.add_argument('--event-dir', required=True)
    parser.add_argument('--sleep-seconds', type=float, default=0.35)
    parser.add_argument('--barrier-env')
    parser.add_argument('--barrier-count', type=int, default=2)
    parser.add_argument('--fail-seed', type=int)
    args = parser.parse_args()
    events = Path(args.event_dir)
    events.mkdir(parents=True, exist_ok=True)
    identity = args.env_id + '_' + str(args.seed) + '_' + args.target_mode
    record = vars(args).copy()
    record.update(gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(),
                  omp_threads=os.environ.get('OMP_NUM_THREADS'),
                  start=time.monotonic())
    (events / (identity + '.start.json')).write_text(json.dumps(record))
    print('FAKE TRAINER: ' + identity, flush=True)
    print('FAKE STDERR: ' + identity, file=sys.stderr, flush=True)
    if args.barrier_env == args.env_id:
        deadline = time.monotonic() + 8
        while len(list(events.glob(args.env_id + '_*.start.json'))) < args.barrier_count:
            if time.monotonic() > deadline:
                sys.exit(91)
            time.sleep(0.01)
    if args.fail_seed == args.seed:
        time.sleep(0.15)
        sys.exit(7)
    time.sleep(args.sleep_seconds)
    record['end'] = time.monotonic()
    (events / (identity + '.end.json')).write_text(json.dumps(record))
    """
)


class FixedAllocationTests(unittest.TestCase):
    def test_comparison_covers_each_method_seed_once_and_balances_gpus(self):
        modes = ['conditional', 'mean_only', 'dqa']
        seeds = [12345, 22345, 32345, 42345, 52345]
        gpus = ['3', '4', '5', '6', '7']
        plan = launcher.build_plan(['Hopper-v3'], seeds, gpus, modes)
        self.assertEqual(len(plan), 15)
        self.assertEqual(Counter(job['gpu'] for job in plan), {gpu: 3 for gpu in gpus})
        self.assertEqual({(j['target_mode'], j['seed']) for j in plan},
                         {(mode, seed) for mode in modes for seed in seeds})

    def test_fifty_experiments_on_ten_gpus_get_five_each(self):
        envs = ['Hopper-v5', 'Walker2d-v5', 'HalfCheetah-v5', 'Ant-v5', 'Humanoid-v5']
        seeds = list(range(10))
        gpus = [str(index) for index in range(10)]
        plan = launcher.build_plan(envs, seeds, gpus)
        self.assertEqual(len(plan), 50)
        self.assertEqual(Counter(job['gpu'] for job in plan), {gpu: 5 for gpu in gpus})
        self.assertEqual([job['job_id'] for job in plan], list(range(1, 51)))
        self.assertEqual([(job['env_id'], job['seed']) for job in plan],
                         [(env_id, seed) for env_id in envs for seed in seeds])
        self.assertEqual([job['gpu'] for job in plan], gpus * 5)

    def test_remainder_changes_load_by_at_most_one(self):
        plan = launcher.build_plan(['Hopper-v5'], list(range(23)), ['2', '4', '7', '9'])
        counts = Counter(job['gpu'] for job in plan)
        self.assertEqual(counts, {'2': 6, '4': 6, '7': 6, '9': 5})

    def test_more_gpus_than_experiments(self):
        plan = launcher.build_plan(['Hopper-v5'], [11, 22], ['0', '1', '2', '3'])
        self.assertEqual([job['gpu'] for job in plan], ['0', '1'])


class GPUSelectionTests(unittest.TestCase):
    def test_auto_uses_existing_visible_tokens_without_renumbering(self):
        detector = mock.Mock(side_effect=AssertionError('Detection should not run'))
        self.assertEqual(launcher.resolve_gpus('auto', {'CUDA_VISIBLE_DEVICES': '2,7,9'}, detector),
                         ['2', '7', '9'])
        detector.assert_not_called()

    def test_explicit_selection_is_a_subset_of_visible_tokens(self):
        self.assertEqual(launcher.resolve_gpus('7,2', {'CUDA_VISIBLE_DEVICES': '2,7,9'}), ['7', '2'])
        with self.assertRaises(ValueError):
            launcher.resolve_gpus('0,1', {'CUDA_VISIBLE_DEVICES': '2,7,9'})

    def test_gpu_uuid_visibility_is_preserved(self):
        gpu_uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
        self.assertEqual(launcher.resolve_gpus('auto', {'CUDA_VISIBLE_DEVICES': gpu_uuid}), [gpu_uuid])

    def test_auto_without_visible_list_uses_cuda_device_count(self):
        self.assertEqual(launcher.resolve_gpus('auto', {}, lambda: 3), ['0', '1', '2'])
        with self.assertRaises(ValueError):
            launcher.resolve_gpus('auto', {}, lambda: 0)

    def test_explicit_ids_do_not_require_cuda_detection(self):
        detector = mock.Mock(side_effect=AssertionError('Explicit plans need no CUDA'))
        self.assertEqual(launcher.resolve_gpus('0,8', {}, detector), ['0', '8'])
        detector.assert_not_called()

    def test_empty_duplicate_and_invalid_gpu_lists_are_rejected(self):
        for value in ['', ',', '0,', ',0', '0,,1', '0,0', '-1', 'all', 'abc']:
            with self.subTest(spec=value), self.assertRaises(ValueError):
                launcher.resolve_gpus(value, {})
        for value in ['', '-1', '0,0', '0,']:
            with self.subTest(visible=value), self.assertRaises(ValueError):
                launcher.resolve_gpus('auto', {'CUDA_VISIBLE_DEVICES': value})

    def test_gpu_aliases_cannot_assign_two_workers_to_one_card(self):
        gpu_uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
        for spec in ['0,' + gpu_uuid, 'GPU-12345678', '0,00',
                     gpu_uuid + ',' + gpu_uuid.upper()]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                launcher.resolve_gpus(spec, {})


class LauncherArgumentsTests(unittest.TestCase):



    def test_jobs_per_gpu_must_be_positive(self):
        self.assertEqual(launcher.parse_args(['--jobs-per-gpu', '5']).jobs_per_gpu, 5)
        for value in ['0', '-1']:
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(['--jobs-per-gpu', value])



    def test_comparison_rejects_duplicate_modes_and_conflicting_selectors(self):
        for flags in (['--target-modes', 'conditional', 'conditional'],
                      ['--target-mode', 'dqa', '--target-modes', 'conditional', 'mean_only']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.parse_args(flags)


    def test_cli_accepts_multiple_environments_seeds_and_extra_training_options(self):
        args = launcher.parse_args([
            '--env-ids', 'Hopper-v5', 'Ant-v5', '--seeds', '11', '22',
            '--gpus', '0,1', '--dry-run', '--output-dir', 'artifacts',
            '--total-timesteps', '2500', '--threads', '2', '--target-mode', 'dqa',
            '--stop-timeout', '1.5', '--', '--hidden-size', '32', '--autotune',
        ])
        self.assertEqual(args.env_ids, ['Hopper-v5', 'Ant-v5'])
        self.assertEqual(args.seeds, [11, 22])
        self.assertEqual(args.gpus, '0,1')
        self.assertTrue(args.dry_run)
        self.assertEqual(args.output_dir, 'artifacts')
        self.assertEqual(args.total_timesteps, 2500)
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.target_mode, 'dqa')
        self.assertEqual(args.stop_timeout, 1.5)
        self.assertEqual(args.train_args, ['--hidden-size', '32', '--autotune'])

    def test_managed_training_options_cannot_override_assignment(self):
        managed = {
            '--env-id': 'Ant-v5', '--seed': '9', '--device': 'cpu',
            '--output-dir': 'elsewhere', '--run-name': 'collision',
            '--total-timesteps': '99', '--torch-threads': '99', '--target-mode': 'dirac',
        }
        for name, value in managed.items():
            for tokens in ([name, value], [name + '=' + value]):
                with self.subTest(tokens=tokens), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        launcher.parse_args(['--gpus', '0', '--', *tokens])
                    self.assertEqual(result.exception.code, 2)

    def test_explicit_dry_run_needs_no_cuda_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'must_not_be_created'
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES', None)
            result = subprocess.run([
                sys.executable, str(Path(launcher.__file__).resolve()),
                '--env-ids', 'Hopper-v5', 'Ant-v5', '--seeds', '1', '2',
                '--gpus', '0,1', '--output-dir', str(output), '--dry-run',
            ], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Hopper-v5', result.stdout)
            self.assertIn('Ant-v5', result.stdout)
            self.assertFalse(output.exists())


class LauncherProcessTests(unittest.TestCase):
    def test_multiple_jobs_overlap_on_each_gpu_without_exceeding_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = root / 'fake_trainer.py'
            trainer.write_text(FAKE_TRAINER, encoding='utf-8')
            args = launcher.parse_args([
                '--env-ids', 'First-v5', '--seeds', '101', '102', '103', '104', '105', '106',
                '--gpus', '2,7', '--jobs-per-gpu', '2', '--output-dir', str(root / 'runs'),
                '--', '--event-dir', str(root / 'events'),
                '--barrier-env', 'First-v5', '--barrier-count', '4',
            ])
            jobs = launcher.build_plan(args.env_ids, args.seeds, ['2', '7'])
            self.assertEqual(launcher.run_plan(args, jobs, trainer_path=trainer), 0)
            records = [json.loads(p.read_text()) for p in (root / 'events').glob('*.end.json')]
            self.assertEqual(len(records), 6)
            for gpu in ['2', '7']:
                intervals = sorted((t, delta) for r in records if r['gpu'] == gpu
                                   for t, delta in [(r['start'], 1), (r['end'], -1)])
                count = peak = 0
                for _time, delta in intervals:
                    count += delta
                    peak = max(peak, count)
                self.assertEqual(peak, 2)
                self.assertEqual(count, 0)
            initial = [r for r in records if r['seed'] <= 104]
            self.assertLess(max(r['start'] for r in initial), min(r['end'] for r in initial))
            status_path, = (root / 'runs' / 'launches').glob('*/status.json')
            status = json.loads(status_path.read_text())
            self.assertTrue(all(j['status'] == 'complete' for j in status['jobs']))

    def test_failure_stops_every_concurrent_worker_on_same_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = root / 'fake_trainer.py'
            trainer.write_text(FAKE_TRAINER, encoding='utf-8')
            args = launcher.parse_args([
                '--env-ids', 'First-v5', '--seeds', '101', '102', '103', '104',
                '--gpus', '2', '--jobs-per-gpu', '3', '--output-dir', str(root / 'runs'),
                '--stop-timeout', '0.3', '--', '--event-dir', str(root / 'events'),
                '--barrier-env', 'First-v5', '--barrier-count', '3',
                '--fail-seed', '101', '--sleep-seconds', '8',
            ])
            jobs = launcher.build_plan(args.env_ids, args.seeds, ['2'])
            self.assertEqual(launcher.run_plan(args, jobs, trainer_path=trainer), 1)
            self.assertEqual(len(list((root / 'events').glob('*.start.json'))), 3)
            status_path, = (root / 'runs' / 'launches').glob('*/status.json')
            status = json.loads(status_path.read_text())
            self.assertEqual([j['status'] for j in status['jobs']],
                             ['failed', 'interrupted', 'interrupted', 'not_started'])
            self.assertTrue(all(j['returncode'] is not None for j in status['jobs'][:3]))



    def test_mixed_methods_reach_children_with_separate_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = root / 'fake_trainer.py'
            trainer.write_text(FAKE_TRAINER, encoding='utf-8')
            args = launcher.parse_args([
                '--env-ids', 'Hopper-v3', '--seeds', '11', '22', '--gpus', '2,7',
                '--target-modes', 'conditional', 'mean_only', 'dqa', '--output-dir', str(root / 'runs'),
                '--', '--event-dir', str(root / 'events'), '--sleep-seconds', '0.05',
            ])
            jobs = launcher.build_plan(args.env_ids, args.seeds, ['2', '7'], args.target_modes)
            self.assertEqual(launcher.run_plan(args, jobs, trainer_path=trainer), 0)
            records = [json.loads(p.read_text()) for p in (root / 'events').glob('*.end.json')]
            self.assertEqual(len(records), 6)
            self.assertEqual({(r['target_mode'], r['seed']) for r in records},
                             {(mode, seed) for mode in args.target_modes for seed in args.seeds})
            self.assertEqual(len({r['run_name'] for r in records}), 6)
            status_path, = (root / 'runs' / 'launches').glob('*/status.json')
            status = json.loads(status_path.read_text())
            for job in status['jobs']:
                command = job['command']
                self.assertEqual(command[command.index('--target-mode') + 1], job['target_mode'])
                self.assertIn('_' + job['target_mode'] + '_seed', job['run_name'])

    def make_args(self, output, events, extra=None):
        return launcher.parse_args([
            '--env-ids', 'First-v5', 'Second-v5', '--seeds', '101', '102',
            '--gpus', '2,7', '--output-dir', str(output),
            '--total-timesteps', '123', '--threads', '2', '--target-mode', 'conditional',
            '--stop-timeout', '0.3', '--', '--event-dir', str(events), *(extra or []),
        ])

    def test_parallel_across_gpus_serial_within_gpu_and_logs_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = root / 'fake_trainer.py'
            trainer.write_text(FAKE_TRAINER, encoding='utf-8')
            output, events = root / 'runs', root / 'events'
            args = self.make_args(output, events, ['--barrier-env', 'First-v5'])
            jobs = launcher.build_plan(args.env_ids, args.seeds, ['2', '7'])
            result = launcher.run_plan(args, jobs, trainer_path=trainer)
            self.assertEqual(result, 0)
            records = [json.loads(path.read_text()) for path in events.glob('*.end.json')]
            self.assertEqual(len(records), 4)
            for gpu in ['2', '7']:
                gpu_records = sorted((record for record in records if record['gpu'] == gpu),
                                     key=lambda record: record['start'])
                self.assertEqual(len(gpu_records), 2)
                self.assertLessEqual(gpu_records[0]['end'], gpu_records[1]['start'])
            initial = [record for record in records if record['env_id'] == 'First-v5']
            self.assertEqual(len(initial), 2)
            self.assertLess(max(record['start'] for record in initial),
                            min(record['end'] for record in initial))
            for record in records:
                self.assertEqual(record['device'], 'cuda')
                self.assertEqual(record['total_timesteps'], 123)
                self.assertEqual(record['torch_threads'], 2)
                self.assertEqual(record['target_mode'], 'conditional')
                self.assertEqual(Path(record['output_dir']), output)
                self.assertTrue(record['run_name'])
                self.assertEqual(record['gpu'], '2' if record['seed'] == 101 else '7')
            self.assertEqual(len({record['run_name'] for record in records}), 4)
            launch_dirs = list((output / 'launches').iterdir())
            self.assertEqual(len(launch_dirs), 1)
            self.assertTrue((launch_dirs[0] / 'plan.json').is_file())
            self.assertTrue((launch_dirs[0] / 'status.json').is_file())
            status = json.loads((launch_dirs[0] / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['exit_code'], 0)
            self.assertIsNone(status['signal'])
            self.assertEqual(len(status['jobs']), 4)
            self.assertTrue(all(job['status'] == 'complete' and job['returncode'] == 0
                                for job in status['jobs']))
            logs = list((launch_dirs[0] / 'console_logs').glob('*.log'))
            self.assertEqual(len(logs), 4)
            for path in logs:
                contents = path.read_text(encoding='utf-8')
                self.assertIn('FAKE TRAINER:', contents)
                self.assertIn('FAKE STDERR:', contents)

    def test_failure_stops_queued_jobs_and_other_gpu_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = root / 'fake_trainer.py'
            trainer.write_text(FAKE_TRAINER, encoding='utf-8')
            output, events = root / 'runs', root / 'events'
            args = self.make_args(output, events, [
                '--barrier-env', 'First-v5', '--fail-seed', '101', '--sleep-seconds', '8',
            ])
            jobs = launcher.build_plan(args.env_ids, args.seeds, ['2', '7'])
            started = time.monotonic()
            result = launcher.run_plan(args, jobs, trainer_path=trainer)
            elapsed = time.monotonic() - started
            self.assertNotEqual(result, 0)
            self.assertLess(elapsed, 5, 'Failure should stop the other GPU instead of waiting eight seconds')
            self.assertEqual(len(list(events.glob('First-v5_*.start.json'))), 2)
            self.assertEqual(len(list(events.glob('Second-v5_*.start.json'))), 0)
            self.assertEqual(len(list(events.glob('*.end.json'))), 0)
            status_files = list((output / 'launches').glob('*/status.json'))
            self.assertEqual(len(status_files), 1)
            status = json.loads(status_files[0].read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['exit_code'], result)
            by_id = {job['job_id']: job for job in status['jobs']}
            self.assertEqual(by_id[1]['status'], 'failed')
            self.assertEqual(by_id[1]['returncode'], 7)
            self.assertEqual(by_id[2]['status'], 'interrupted')
            self.assertIsNotNone(by_id[2]['returncode'])
            self.assertEqual(by_id[3]['status'], 'not_started')
            self.assertEqual(by_id[4]['status'], 'not_started')

    def test_stop_request_saves_interrupted_status_and_restores_handlers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = root / 'fake_trainer.py'
            trainer.write_text(FAKE_TRAINER, encoding='utf-8')
            output, events = root / 'runs', root / 'events'
            args = self.make_args(output, events, ['--sleep-seconds', '8'])
            args.jobs_per_gpu = 2
            args.env_ids.append('Third-v5')
            jobs = launcher.build_plan(args.env_ids, args.seeds, ['2', '7'])
            old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            original_sleep = time.sleep
            requested = []

            def sleep_then_request_stop(seconds):
                original_sleep(seconds)
                if not requested:
                    requested.append(True)
                    # Invoke the actual installed handler without terminating
                    # the unittest process on Windows, which lacks POSIX kill.
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

            with mock.patch.object(launcher.time, 'sleep', side_effect=sleep_then_request_stop):
                result = launcher.run_plan(args, jobs, trainer_path=trainer)
            self.assertEqual(result, 128 + signal.SIGTERM)
            for sig, handler in old_handlers.items():
                self.assertEqual(signal.getsignal(sig), handler)
            status_path, = (output / 'launches').glob('*/status.json')
            status = json.loads(status_path.read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'interrupted')
            self.assertEqual(status['signal'], signal.SIGTERM)
            self.assertEqual(status['exit_code'], result)
            self.assertEqual([job['status'] for job in status['jobs']],
                             ['interrupted'] * 4 + ['not_started'] * 2)


if __name__ == '__main__':
    unittest.main()
