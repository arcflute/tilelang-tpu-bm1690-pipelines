"""Exercise the bounded smoke -> full serial -> full pipeline handoff without a TPU."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tpu_demo/pipeline/run_rope_tiled.sh"
FAKE_PYTHON = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['TILED_TEST_REPO'])
events = pathlib.Path(os.environ['TILED_TEST_EVENTS'])
if args[0] == '-' and len(args[1]) == 40:
    payload = (root / 'tpu_demo/pipeline/run_add_pcie.py').read_bytes()
    if os.environ.get('TILED_TEST_BAD_DOWNLOAD'): payload += b'changed'
    pathlib.Path(args[2]).write_bytes(payload)
elif args[0].endswith('/run_add.py'):
    dest = pathlib.Path(args[args.index('--output') + 1])
    case = dest.name
    variant = args[args.index('--variant') + 1]
    assert variant == ('original' if case == 'smoke' else case)
    assert '--measure' not in args
    assert args[args.index('--timeout-seconds') + 1] == '30'
    if case == 'serial':
        assert args[args.index('--original-failure') + 1].endswith('/target/run-original')
        assert '--previous' not in args
    elif case == 'pipeline':
        assert args[args.index('--previous') + 1] == str(dest.parent / 'serial')
        assert '--original-failure' not in args
    else:
        assert '--previous' not in args and '--original-failure' not in args
    with events.open('a') as f: f.write(case + '\n')
    if case == os.environ.get('TILED_TEST_FAIL'): sys.exit(1)
    dest.mkdir()
    numeric = dict(status='passed', operation='rope', dtype='float16', launch_cores=1,
                   variant=variant, shape=[8,128] if case=='smoke' else [1024,1024],
                   reference={'passed':True})
    if case != 'smoke':
        numeric.update(original_baseline_status='failed', three_variant_acceptance_complete=False,
                       original_baseline_failure={'reason':'timeout'})
    if case == 'pipeline': numeric['previous_variant_bitwise_equal'] = True
    if case == os.environ.get('TILED_TEST_BAD_RECEIPT'): numeric['shape'] = [8,128]
    if os.environ.get('TILED_TEST_BAD_COMPARE') and case=='pipeline':
        numeric['previous_variant_bitwise_equal'] = False
    (dest / 'result.json').write_text(json.dumps({'status':'passed','numeric':numeric}))
else:
    os.execv(sys.executable, [sys.executable, *args])
'''


class TiledSequenceTests(unittest.TestCase):
    def run_sequence(self, **extra):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            commands = work / 'bin'
            commands.mkdir()
            fake = commands / 'python3'
            fake.write_text('#!' + sys.executable + '\n' + FAKE_PYTHON)
            fake.chmod(0o755)
            sudo = commands / 'sudo'
            sudo.write_text('#!/bin/sh\ncase "$TILED_TEST_BUSY" in\n'
                            'yes) echo "device held" >&2; exit 0;;\n'
                            'error) echo "permission denied" >&2; exit 1;;\n'
                            '*) exit 1;;\nesac\n')
            sudo.chmod(0o755)
            for name in ('smoke/build', 'target/build', 'target/run-original'):
                directory = work / name
                directory.mkdir(parents=True)
                (directory / 'result.json').write_text('{}')
            environment = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                               TILED_TEST_REPO=str(ROOT), TILED_TEST_EVENTS=str(work / 'events'))
            environment.update(extra)
            command = ['bash', str(SCRIPT), '--allow-pcie', 'a' * 40, str(work / 'output'),
                       str(work / 'smoke'), str(work / 'target')]
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=15)
            events = (work / 'events').read_text().splitlines() if (work / 'events').exists() else []
            return result, events

    def test_order_and_exact_comparison_arguments(self):
        result, events = self.run_sequence()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(events, ['smoke', 'serial', 'pipeline'])
        self.assertIn('CORRECTNESS_PASSED original_full=still_failed', result.stdout)

    def test_failure_stops_before_next_launch(self):
        for case, expected in (('smoke', ['smoke']), ('serial', ['smoke','serial'])):
            with self.subTest(case=case):
                result, events = self.run_sequence(TILED_TEST_FAIL=case)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(events, expected)

    def test_wrong_coverage_or_failed_comparison_is_not_accepted(self):
        result, events = self.run_sequence(TILED_TEST_BAD_RECEIPT='serial')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, ['smoke', 'serial'])
        result, events = self.run_sequence(TILED_TEST_BAD_COMPARE='yes')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, ['smoke', 'serial', 'pipeline'])

    def test_device_holder_or_permission_error_prevents_all_calls(self):
        for state in ('yes', 'error'):
            with self.subTest(state=state):
                result, events = self.run_sequence(TILED_TEST_BUSY=state)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(events, [])

    def test_changed_download_prevents_all_calls(self):
        result, events = self.run_sequence(TILED_TEST_BAD_DOWNLOAD='yes')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
