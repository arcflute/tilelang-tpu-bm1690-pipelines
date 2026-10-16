"""No-TPU tests of smoke/target sequencing, correctness-before-timing and stop gates."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'tpu_demo/pipeline/run_swiglu_board.sh'
FAKE_PYTHON = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['SG_TEST_REPO'])
events = pathlib.Path(os.environ['SG_TEST_EVENTS'])
def record(event):
    with events.open('a') as f: f.write(event+'\n')
if args[0]=='-' and len(args[1])==40:
    pathlib.Path(args[2]).write_bytes((root/'tpu_demo/pipeline/prepare_add_1024.py').read_bytes())
elif args[0].endswith('/prepare.py'):
    work=pathlib.Path(args[args.index('--output')+1]); work.mkdir(); (work/'build').mkdir()
    (work/'run_add.py').write_bytes((root/'tpu_demo/pipeline/run_add_pcie.py').read_bytes())
    record(work.name+':build')
elif args[0].endswith('/run_add.py'):
    work=pathlib.Path(args[0]).parent
    variant=args[args.index('--variant')+1]
    timing='--measure' in args; phase='timing' if timing else 'run'
    event=work.name+':'+phase+':'+variant
    record(event)
    assert '--original-failure' not in args
    if variant=='original': assert '--previous' not in args
    else:
        previous=pathlib.Path(args[args.index('--previous')+1])
        assert previous==work/('run-original' if variant=='serial' else 'run-serial')
        assert (previous/'result.json').exists()
    if timing:
        correct=pathlib.Path(args[args.index('--correctness')+1])
        assert correct==work/('run-'+variant)
        assert all((work/('run-'+v)/'result.json').exists() for v in ('original','serial','pipeline'))
    if event==os.environ.get('SG_TEST_FAIL'): sys.exit(1)
    dest=pathlib.Path(args[args.index('--output')+1]);dest.mkdir()
    n=dict(status='passed',operation='swiglu',variant=variant,dtype='float16',intermediate_dtype='float32',
           shape=[8,128] if work.name=='smoke' else [1024,1024], buffer_reuse=False, launch_cores=1,
           reference={'passed':True}, previous_variant_bitwise_equal=variant!='original')
    if timing: n['timing']=dict(warmups=5,samples=20,raw_us=[1.0]*20,pure_device_time=False)
    if event==os.environ.get('SG_TEST_BAD_RECEIPT'):n['reference']['passed']=False
    (dest/'result.json').write_text(json.dumps({'status':'passed','numeric':n}))
else:
    os.execv(sys.executable,[sys.executable,*args])
'''


class SwiGLUSequenceTests(unittest.TestCase):
    def run_sequence(self, **extra):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); commands=root/'bin';commands.mkdir()
            python=commands/'python3';python.write_text('#!'+sys.executable+'\n'+FAKE_PYTHON);python.chmod(0o755)
            sudo=commands/'sudo'
            sudo.write_text('#!/bin/sh\ncase "$SG_TEST_BUSY" in\nyes) echo busy >&2; exit 0;;\nerror) echo denied >&2; exit 1;;\n*) exit 1;;\nesac\n')
            sudo.chmod(0o755)
            env=dict(os.environ,PATH=str(commands)+os.pathsep+os.environ['PATH'],SG_TEST_REPO=str(ROOT),SG_TEST_EVENTS=str(root/'events'))
            env.update(extra)
            r=subprocess.run(['bash',str(SCRIPT),'--allow-pcie','a'*40,str(root/'output')],
                             env=env,capture_output=True,text=True,timeout=20)
            return r,(root/'events').read_text().splitlines()

    def test_correctness_all_variants_before_timing_and_smoke_before_target(self):
        r,events=self.run_sequence()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        expected=[]
        for size in ('smoke','target'):
            expected.append(size+':build')
            expected.extend(size+':'+phase+':'+v for phase in ('run','timing') for v in ('original','serial','pipeline'))
        self.assertEqual(events,expected)
        self.assertIn('SWIGLU_SUITE_PASSED',r.stdout)

    def test_failure_or_invalid_receipt_stops_before_timing_and_next_build(self):
        for variable in ('SG_TEST_FAIL','SG_TEST_BAD_RECEIPT'):
            with self.subTest(variable=variable):
                r,events=self.run_sequence(**{variable:'smoke:run:serial'})
                self.assertNotEqual(r.returncode,0)
                self.assertEqual(events,['smoke:build','smoke:run:original','smoke:run:serial'])
                self.assertNotIn('SWIGLU_SUITE_PASSED',r.stdout)

    def test_timing_failure_stops_before_target_build(self):
        r,events=self.run_sequence(SG_TEST_FAIL='smoke:timing:original')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(events[-1],'smoke:timing:original')
        self.assertNotIn('target:build',events)

    def test_device_holder_or_permission_error_prevents_launch(self):
        for state in ('yes','error'):
            with self.subTest(state=state):
                r,events=self.run_sequence(SG_TEST_BUSY=state)
                self.assertNotEqual(r.returncode,0)
                self.assertEqual(events,['smoke:build'])


if __name__=='__main__':
    unittest.main()
