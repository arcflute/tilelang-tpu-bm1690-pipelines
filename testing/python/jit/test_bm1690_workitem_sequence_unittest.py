"""Run the board shell sequence with local downloads and fake build/dispatch."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
SHIM=r'''
import io,json,os,pathlib,subprocess,sys,urllib.request
from unittest.mock import patch
repo=pathlib.Path(os.environ['WI_TEST_REPO'])
events=pathlib.Path(os.environ['WI_TEST_EVENTS'])
args=sys.argv[1:]
mode=os.environ.get('WI_TEST_FAIL','')
def event(name):
    with events.open('a') as f:f.write(name+'\n')
if args[0]=='-':
    code=sys.stdin.read();sys.argv=args
    if len(args)==4 and len(args[1])==40:
        def download(request,timeout):
            relative=request.full_url.split('/contents/',1)[1].split('?',1)[0]
            assert relative in ('tpu_demo/pipeline/workitem_probe_pcie.py','research/bm1690-pipelines/handoff/workitems-2-sources.json')
            payload=(repo/relative).read_bytes()
            return io.BytesIO(payload+b'changed' if mode=='checksum' else payload)
        def build(command,env,check):
            event('build')
            assert command[:4]==['timeout','-k','5s','180s']
            assert '--build-only' in command and '--allow-pcie' not in command
            assert pathlib.Path(env['PPL_PROJECT_ROOT']).exists()
            if mode=='build':raise subprocess.CalledProcessError(9,command)
            return subprocess.CompletedProcess(command,0)
        with patch.object(urllib.request,'urlopen',side_effect=download),patch.object(subprocess,'run',side_effect=build):
            exec(compile(code,'<bootstrap>','exec'))
    else:
        exec(compile(code,'<receipt-check>','exec'))
elif args[0].endswith('/probe.py'):
    event('run')
    assert '--run-only' in args and '--allow-pcie' in args
    if mode=='run':sys.exit(1)
    dest=pathlib.Path(args[args.index('--output')+1]);dest.mkdir()
    n=dict(status='passed',launch_cores=2,tasks=11,ownership_counts=[6,5],checked_elements=704,
           exact_ownership=True,requested_kernel_calls=1,mapping={'pci_function':'0000:01:00.0'},
           output_sha256='e28614fe1296504598a309802c398b69b6e2fc0612f030ac915754f193882290')
    if mode=='receipt':n['exact_ownership']=False
    (dest/'result.json').write_text(json.dumps(dict(status='passed',cleanup_complete=True,numeric=n)))
else:
    raise AssertionError(args)
'''


class ProbeSequenceTests(unittest.TestCase):
    def run_sequence(self, failure=''):
        import json,hashlib,shutil
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';old.mkdir();(old/'build').mkdir()
            for dest,source in (('build.py','tilelang/jit/adapter/legacy_pcie.py'),('run_add.py','tpu_demo/pipeline/run_add_pcie.py')):
                shutil.copyfile(ROOT/source,old/dest)
            sdk=root/'sdk';sdk.mkdir();board=root/'board';board.mkdir()
            cross=root/'riscv64-unknown-linux-gnu-gcc';cross.write_text('fixture')
            (old/'build/result.json').write_text(json.dumps(dict(status='compile_only_passed',runtime_identity=[str(sdk),str(board),'unused'],sdk_inputs={str(cross):hashlib.sha256(cross.read_bytes()).hexdigest()})))
            binpath=root/'bin';binpath.mkdir();python=binpath/'python3'
            python.write_text('#!'+sys.executable+'\n'+SHIM);python.chmod(0o700)
            events=root/'events';events.write_text('')
            env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],WI_TEST_REPO=str(ROOT),WI_TEST_EVENTS=str(events),WI_TEST_FAIL=failure)
            reply=subprocess.run(['bash',str(ROOT/'tpu_demo/pipeline/run_workitems2_board.sh'),'--allow-pcie','a'*40,str(root/'output'),str(old)],env=env,capture_output=True,text=True,timeout=10)
            return reply,events.read_text().splitlines()

    def test_one_build_then_exactly_one_dispatch(self):
        r,events=self.run_sequence()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(events,['build','run'])
        self.assertIn('WORKITEM_SUITE_PASSED',r.stdout)

    def test_download_corruption_stops_before_build(self):
        r,events=self.run_sequence('checksum')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(events,[])

    def test_build_failure_never_dispatches(self):
        r,events=self.run_sequence('build')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(events,['build'])

    def test_worker_or_receipt_failure_is_not_success(self):
        for error in ('run','receipt'):
            with self.subTest(error=error):
                r,events=self.run_sequence(error)
                self.assertNotEqual(r.returncode,0)
                self.assertEqual(events,['build','run'])
                self.assertNotIn('WORKITEM_SUITE_PASSED',r.stdout)


if __name__=='__main__':unittest.main()
