"""Exercise sequencing and stop gates with fake downloads/builds/runs; no TPU."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tpu_demo/pipeline/run_rope_prefixes.sh"
FAKE_PYTHON = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['PREFIX_TEST_REPO'])
events = pathlib.Path(os.environ['PREFIX_TEST_EVENTS'])
def record(event):
    with events.open('a') as f: f.write(event + '\n')
if args[0] == '-' and len(args[1]) == 40:
    pathlib.Path(args[2]).write_bytes((root / 'tpu_demo/pipeline/prepare_add_1024.py').read_bytes())
elif args[0].endswith('/prepare.py'):
    work = pathlib.Path(args[args.index('--output') + 1])
    work.mkdir()
    (work / 'build').mkdir()
    (work / 'run_add.py').write_bytes((root / 'tpu_demo/pipeline/run_add_pcie.py').read_bytes())
    record('build:' + work.name)
elif args[0].endswith('/run_add.py'):
    work = pathlib.Path(args[0]).parent
    record('run:' + work.name)
    if work.name == os.environ.get('PREFIX_TEST_FAIL'):
        sys.exit(1)
    tiles = int(work.name)
    dest = pathlib.Path(args[args.index('--output') + 1])
    dest.mkdir()
    numeric = {'status':'passed','operation':'rope','variant':'original','diagnostic_only':True,
               'reference':{'passed':True,'scope':'active_window_only',
                 'window':[0, tiles // 8, 0, 1024], 'checked_elements':tiles * 128,
                 'untouched_elements':1048576 - tiles * 128,'untouched_sentinel_preserved':True}}
    if work.name == os.environ.get('PREFIX_TEST_BAD_RECEIPT'):
        numeric['reference']['window'] = [0, 8, 0, 128]
    (dest / 'result.json').write_text(json.dumps({'status':'passed','numeric':numeric}))
else:
    os.execv(sys.executable, [sys.executable, *args])
'''


class PrefixSequenceTests(unittest.TestCase):
    def run_sequence(self, **extra):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            commands = work / 'bin'
            commands.mkdir()
            fake = commands / 'python3'
            fake.write_text('#!' + sys.executable + '\n' + FAKE_PYTHON)
            fake.chmod(0o755)
            sudo = commands / 'sudo'
            sudo.write_text('#!/bin/sh\ncase "$PREFIX_TEST_BUSY" in\n'
                            'yes) echo "device held" >&2; exit 0;;\n'
                            'error) echo "permission denied" >&2; exit 1;;\n'
                            '*) exit 1;;\nesac\n')
            sudo.chmod(0o755)
            environment = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                               PREFIX_TEST_REPO=str(ROOT), PREFIX_TEST_EVENTS=str(work / 'events'))
            environment.update(extra)
            result = subprocess.run(['bash', str(SCRIPT), '--allow-pcie', 'a' * 40, str(work / 'output')],
                                    env=environment, capture_output=True, text=True, timeout=15)
            events = (work / 'events').read_text().splitlines()
            return result, events

    def test_three_cases_run_in_order_and_require_matching_receipts(self):
        result, events = self.run_sequence()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(events, ['build:128', 'run:128', 'build:256', 'run:256', 'build:512', 'run:512'])
        self.assertIn('PREFIX_SUITE_PASSED', result.stdout)

    def test_failure_stops_before_build_or_dispatch_of_next_case(self):
        result, events = self.run_sequence(PREFIX_TEST_FAIL='256')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, ['build:128', 'run:128', 'build:256', 'run:256'])
        self.assertNotIn('PREFIX_SUITE_PASSED', result.stdout)

    def test_wrong_receipt_stops_even_if_process_returns_zero(self):
        result, events = self.run_sequence(PREFIX_TEST_BAD_RECEIPT='128')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, ['build:128', 'run:128'])

    def test_occupation_or_check_error_blocks_all_launches(self):
        for state in ('yes', 'error'):
            with self.subTest(state=state):
                result, events = self.run_sequence(PREFIX_TEST_BUSY=state)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(events, ['build:128'])


if __name__ == '__main__':
    unittest.main()
