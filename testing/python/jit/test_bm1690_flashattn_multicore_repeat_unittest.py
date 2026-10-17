"""Reuse actual validators with synthetic ELF/receipts; never access a TPU."""
import contextlib
import copy
import ctypes
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repeat = load('attention_repeat_test', ROOT / 'tpu_demo/pipeline/repeat_flashattn_multicore_board.py')
fixture = load('attention_repeat_fixture', Path(__file__).with_name('test_bm1690_flashattn_multicore_pcie_unittest.py'))
driver = fixture.probe


class RepeatTests(unittest.TestCase):
    def setUp(self):
        base = fixture.MulticoreTests()
        base.setUp()
        self.addCleanup(base.doCleanups)
        self.source = base.root / 'passed'
        self.source.mkdir()
        self.output = base.root / 'new-rounds'
        shutil.copyfile(fixture.SCRIPT, self.source / 'flashattn.py')
        self.support = base.support
        self.commands = []
        for operation in repeat.OPERATIONS:
            work = self.source / operation
            work.mkdir()
            for name in ('build.py', 'run_add.py'):
                shutil.copyfile(base.root / name, work / name)
            shutil.copyfile(ROOT / f'research/bm1690-pipelines/handoff/{operation}-multicore-sources.json',
                            work / 'flashattn-multicore.json')
            bundle, builder, _ = driver.assets(work)
            for variant in driver.VARIANTS:
                build = work / 'build' / variant
                build.mkdir(parents=True)
                for name, source in bundle['variants'][variant]['sources'].items():
                    (build / name).write_text(source)
                for name in ('main.so', 'libkernel.so'):
                    shutil.copyfile(base.build / name, build / name)
                report = dict(base.report, operation=operation, variant=variant,
                              **driver.contract(variant, operation))
                report.update(bundle_sha256=driver.BUNDLE_SHAS[operation],
                              completed_commands=driver.commands(builder, base.layout, build))
                (build / 'result.json').write_text(json.dumps(report))
                # Deliberately distinguish original tiling and the two masks.
                output = bytes([2 * repeat.OPERATIONS.index(operation) + (variant == 'original')]) * (1024 * 64 * 2)
                numeric = dict(operation=operation, variant=variant, **driver.contract(variant, operation),
                    shape=driver.SHAPE, dtype='float16', status='passed',
                    bundle_sha256=driver.BUNDLE_SHAS[operation],
                    build_manifest_sha256=driver.digest(build / 'result.json'), artifacts=report['artifacts'],
                    input_sha256=driver.INPUT_SHAS[operation], runtime_library=str(base.layout.board_lib / 'libtpuv7_rt.so'),
                    runtime_sha256=driver.digest(base.layout.board_lib / 'libtpuv7_rt.so'),
                    mapping=dict(device_id=0, device_count=2, pci_function='0000:01:00.0', character_device='/dev/sg-host-drv-0'),
                    hardware_overlap_verified=False, device_performance_measured=False,
                    reference=dict(passed=True, finite=True, mismatched_elements=0, atol=0.02, rtol=0.02))
                self.receipt(work / ('run-' + variant), dict(numeric, measurement='correctness', requested_kernel_calls=1), output)
                self.receipt(work / ('timing-' + variant), self.timed(numeric), output)

    def timed(self, numeric):
        return dict(numeric, measurement='timing', requested_kernel_calls=25,
                    sync_latency=dict(self.support.summarize_samples([100.0] * 20), warmups=5,
                                      samples=20, raw_us=[100.0] * 20))

    def receipt(self, directory, numeric, output):
        directory.mkdir()
        (directory / 'output.f16').write_bytes(output)
        numeric['output_sha256'] = driver.digest(directory / 'output.f16')
        (directory / 'result.json').write_text(json.dumps(dict(status='passed', cleanup_complete=True, numeric=numeric)))

    def snapshot(self):
        return {str(p.relative_to(self.source)):driver.digest(p) for p in self.source.rglob('*') if p.is_file()}

    def command(self, args, log, env):
        self.commands.append(args)
        log.write_text('synthetic worker reply\n')
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
        self.assertEqual(env['OPENBLAS_NUM_THREADS'], '1')
        if '--run-only' not in args:
            return 0
        self.assertNotIn('--build-only', args)
        variant = args[args.index('--variant') + 1]
        self.assertNotEqual(variant, 'original')
        previous = Path(args[args.index('--correctness') + 1])
        self.assertEqual(previous.name, 'run-' + variant)
        target = Path(args[args.index('--output') + 1])
        n = json.loads((previous / 'result.json').read_text())['numeric']
        n = self.timed(n)
        self.receipt(target, n, (previous / 'output.f16').read_bytes())
        return 0

    def execute(self, effect=None):
        with patch.object(repeat, 'command', side_effect=effect or self.command), \
             patch.object(ctypes, 'CDLL', side_effect=AssertionError('no runtime in parent')), \
             contextlib.redirect_stdout(io.StringIO()):
            repeat.execute(self.source, self.output, Path(sys.executable))

    def test_two_rounds_reorder_both_modes_preserve_baselines_and_limit_dispatch(self):
        before = self.snapshot()
        self.execute()
        self.assertEqual(self.snapshot(), before)
        actual = [(Path(a[a.index('--work') + 1]).name, a[a.index('--variant') + 1])
                  for a in self.commands if '--run-only' in a]
        expected = [(op, v) for modes, order in zip(repeat.MODE_ORDERS, repeat.ORDERS) for op in modes for v in order]
        self.assertEqual(actual, expected)
        self.assertEqual(len(self.commands), 17)  # NumPy check plus 16 workers.
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['completed_workers'], 16)
        self.assertEqual(report['requested_kernel_calls'], 400)
        self.assertEqual(report['raw_samples'], 320)
        self.assertFalse(report['compilation_performed'])
        self.assertFalse(report['original_relaunched'])
        self.assertFalse(report['hardware_overlap_verified'])
        self.assertFalse(report['pure_device_time'])
        self.assertEqual(len(report['original_baselines']), 2)
        self.assertEqual(len(report['rounds']), 4)

    def test_any_old_source_build_runtime_or_output_change_stops_before_dispatch(self):
        work = self.source / 'flashattn-causal'
        n = json.loads((work / 'run-original/result.json').read_text())['numeric']
        paths = [self.source / 'flashattn.py', work / 'build.py', work / 'flashattn-multicore.json',
                 work / 'build/serial_2/kernel.c', work / 'build/serial_2/libkernel.so',
                 Path(n['runtime_library']), work / 'run-serial_2/output.f16']
        for path in paths:
            with self.subTest(path=path):
                saved = path.read_bytes()
                path.write_bytes(saved + b'changed')
                try:
                    with self.assertRaises((ValueError, json.JSONDecodeError)):
                        self.execute()
                    self.assertFalse(self.output.exists())
                    self.assertEqual(self.commands, [])
                finally:
                    path.write_bytes(saved)

    def test_old_receipt_wrong_contract_statistics_or_cleanup_is_rejected(self):
        path = self.source / 'flashattn-causal/timing-pipeline_2/result.json'
        old = json.loads(path.read_text())
        changes = [('launch_cores', 1), ('is_causal', False), ('input_sha256', 'different'),
                   ('mapping', dict(old['numeric']['mapping'], device_id=1)),
                   ('sync_latency', dict(old['numeric']['sync_latency'], median_us=1)),
                   ('reference', dict(old['numeric']['reference'], passed=False))]
        for field, value in changes:
            changed = copy.deepcopy(old)
            changed['numeric'][field] = value
            path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.execute()
            self.assertFalse(self.output.exists())
        path.write_text(json.dumps(dict(old, cleanup_complete=False)))
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.commands, [])

    def test_missing_old_receipt_and_existing_destination_are_not_overwritten(self):
        path = self.source / 'flashattn/run-serial_1/result.json'
        old = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute()
        path.write_bytes(old)
        self.output.mkdir()
        (self.output / 'keep').write_text('existing')
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual((self.output / 'keep').read_text(), 'existing')
        self.assertEqual(self.commands, [])

    def test_first_failure_stops_batch_and_preserves_completed_results(self):
        def fail(args, log, env):
            if len(self.commands) == 3:
                self.commands.append(args)
                log.write_text('device occupied or timed out')
                return 1
            return self.command(args, log, env)
        with self.assertRaisesRegex(RuntimeError, 'no further dispatch'):
            self.execute(fail)
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['completed_workers'], 2)
        self.assertEqual(len(self.commands), 4)
        self.assertEqual(len(report['rounds'][0]['variants']), 2)

    def test_bad_numpy_environment_performs_no_worker_dispatch(self):
        def fail(args, log, env):
            self.commands.append(args)
            return 1
        with self.assertRaisesRegex(RuntimeError, 'NumPy'):
            self.execute(fail)
        self.assertEqual(len(self.commands), 1)
        self.assertNotIn('--run-only', self.commands[0])

    def test_new_output_or_timing_change_cannot_be_recorded_as_passed(self):
        def corrupt(args, log, env):
            rc = self.command(args, log, env)
            if '--run-only' in args:
                directory = Path(args[args.index('--output') + 1])
                (directory / 'output.f16').write_bytes(bytes([55]) * (1024 * 64 * 2))
                receipt = json.loads((directory / 'result.json').read_text())
                receipt['numeric']['output_sha256'] = driver.digest(directory / 'output.f16')
                (directory / 'result.json').write_text(json.dumps(receipt))
            return rc
        with self.assertRaisesRegex(ValueError, 'Repeated output'):
            self.execute(corrupt)
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(report['completed_workers'], 0)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(len(self.commands), 2)

    def test_comparisons_separate_pipeline_from_core_scaling_and_keep_regression(self):
        rows = {v:dict(sync_latency=dict(median_us=t)) for v, t in
                zip(('serial_1', 'pipeline_1', 'serial_2', 'pipeline_2'), (100, 110, 60, 50))}
        r = repeat.comparisons(rows)
        self.assertEqual(r['pipeline_at_1_core']['median_reduction_percent'], -10)
        self.assertAlmostEqual(r['pipeline_at_2_cores']['median_reduction_percent'], 100/6)
        self.assertEqual(r['serial_1_to_2_cores']['median_reduction_percent'], 40)
        self.assertAlmostEqual(r['pipeline_1_to_2_cores']['median_reduction_percent'], 600/11)


if __name__ == '__main__':
    unittest.main()
