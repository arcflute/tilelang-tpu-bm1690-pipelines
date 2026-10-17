"""Exercise receipt binding and reordered dispatch without accessing a TPU."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("repeat_flashattn", ROOT / "tpu_demo/pipeline/repeat_flashattn_board.py")
repeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repeat)


class RepeatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "first"
        self.output = self.root / "repeat"
        self.events = []
        self.saved = {}
        for mode in repeat.BUNDLES:
            work = self.source / ("target-" + mode)
            (work / "build").mkdir(parents=True)
            (work / "build/result.json").write_text('{}')
            (work / "run_add.py").write_bytes((ROOT / "tpu_demo/pipeline/run_add_pcie.py").read_bytes())
            for variant in ("original", "serial", "pipeline"):
                payload = (mode + ("original" if variant == "original" else "tiled")).encode()
                n = dict(status="passed", operation="flashattn" + ("-causal" if mode == "causal" else ""),
                         variant=variant, shape=[1,1024,1,64], dtype="float16", layout="BSHD",
                         accumulation_dtype="float32", probability_dtype="float16", mask_dtype="float32",
                         output_index=4, launch_cores=1, device_id=0, expected_pci="0000:01:00.0",
                         bundle_sha256=repeat.BUNDLES[mode], is_causal=mode == "causal",
                         build_manifest_sha256=repeat.digest(work / "build/result.json"),
                         input_sha256=mode, output_sha256=hashlib.sha256(payload).hexdigest(),
                         runtime_sha256="test-runtime", runtime_library="/test/lib.so",
                         mapping=dict(device_id=0, pci_function="0000:01:00.0"),
                         previous_variant_bitwise_equal=variant == "pipeline",
                         previous_variant_comparison=dict(passed=True), reference=dict(passed=True),
                         requested_kernel_calls=1)
                self.saved[mode, variant] = n
                for phase in ("run", "timing"):
                    report = self.report(n, timed=phase == "timing")
                    dest = work / (phase + "-" + variant)
                    dest.mkdir()
                    (dest / "result.json").write_text(json.dumps(report))
                    (dest / "output.f16").write_bytes(payload)

    def report(self, n, timed=True):
        n = copy.deepcopy(n)
        if timed:
            n['timing'] = dict(raw_us=[10.] * 20, median_us=10., iqr_us=0., p95_us=10., min_us=10.,
                               max_us=10., warmups=5, samples=20, pure_device_time=False)
        return dict(status="passed", cleanup_complete=True, numeric=n)

    def run_sequence(self, fail=None, busy=None, corrupt=None):
        def command(args, log, env):
            log.write_text('')
            if 'fuser' in args:
                if busy == "occupied":
                    log.write_text("device held")
                    return 0
                if busy == "denied":
                    log.write_text("sudo denied")
                return 1
            if '-c' in args:
                return 0
            dest = Path(args[args.index('--output') + 1])
            self.events.append(dest.name)
            self.assertEqual(args[:4], ['timeout', '-k', '3s', '40s'])
            self.assertIn('--measure', args)
            self.assertIn('--allow-pcie', args)
            self.assertEqual(args[args.index('--timeout-seconds') + 1], '30')
            variant = args[args.index('--variant') + 1]
            work = Path(args[args.index('--build') + 1]).parent
            self.assertEqual(Path(args[args.index('--correctness') + 1]), work / ('run-' + variant))
            if variant == 'original':
                self.assertNotIn('--previous', args)
            else:
                prev = 'original' if variant == 'serial' else 'serial'
                self.assertEqual(Path(args[args.index('--previous') + 1]), work / ('run-' + prev))
            self.assertEqual(env['OPENBLAS_NUM_THREADS'], '1')
            if dest.name == fail:
                return 1
            mode = work.name.removeprefix('target-')
            report = self.report(self.saved[mode, variant])
            if corrupt:
                report['numeric'][corrupt] = {'passed': False} if corrupt == 'reference' else 'changed'
            dest.mkdir()
            (dest / 'result.json').write_text(json.dumps(report))
            return 0
        with patch.object(repeat, 'command', side_effect=command), contextlib.redirect_stdout(io.StringIO()):
            repeat.execute(self.source, self.output, Path(sys.executable))

    def test_orders_reuse_correctness_regardless_of_timing_order(self):
        self.run_sequence()
        expected = [f'round{number}-{mode}-{v}' for number, order in enumerate(repeat.ORDERS, 2)
                    for mode in repeat.BUNDLES for v in order]
        self.assertEqual(self.events, expected)
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['completed_workers'], 12)
        self.assertEqual(len(report['rounds']), 4)
        self.assertFalse(report['pure_device_time'])

    def test_runner_pin_and_existing_output_fail_before_dispatch(self):
        self.assertEqual(repeat.digest(ROOT / 'tpu_demo/pipeline/run_add_pcie.py'), repeat.RUNNER_SHA256)
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_sequence()
        (self.source / 'target-causal/run_add.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_sequence()
        self.assertEqual(self.events, [])

    def test_missing_receipt_stops_before_dispatch(self):
        (self.source / 'target-causal/run-serial/result.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_sequence()
        self.assertEqual(self.events, [])

    def test_device_occupied_stops(self):
        with self.assertRaisesRegex(RuntimeError, 'Device occupied'):
            self.run_sequence(busy='occupied')
        self.assertEqual(self.events, [])

    def test_permission_error_is_not_treated_as_idle(self):
        with self.assertRaisesRegex(RuntimeError, 'Device occupied'):
            self.run_sequence(busy='denied')
        self.assertEqual(self.events, [])

    def test_worker_failure_stops_and_retains_completed_result(self):
        with self.assertRaisesRegex(RuntimeError, 'no further dispatch'):
            self.run_sequence(fail='round2-noncausal-serial')
        self.assertEqual(self.events, ['round2-noncausal-pipeline', 'round2-noncausal-serial'])
        self.assertTrue((self.output / 'round2-noncausal-pipeline/result.json').exists())
        self.assertEqual(json.loads((self.output / 'summary.json').read_text())['status'], 'failed')

    def test_runtime_change_stops_after_first_worker(self):
        with self.assertRaisesRegex(ValueError, 'runtime'):
            self.run_sequence(corrupt='runtime_sha256')
        self.assertEqual(len(self.events), 1)

    def test_reference_failure_stops_after_first_worker(self):
        with self.assertRaisesRegex(ValueError, 'receipt'):
            self.run_sequence(corrupt='reference')
        self.assertEqual(len(self.events), 1)

    def test_invalid_sample_statistics_rejected(self):
        path = self.source / 'target-causal/timing-original/result.json'
        report = json.loads(path.read_text())
        report['numeric']['timing']['median_us'] = 9.0
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'receipt'):
            self.run_sequence()
        self.assertEqual(self.events, [])


if __name__ == '__main__':
    unittest.main()
