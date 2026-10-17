"""Exercise the real board sequence offline, including its five-way summary."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('probe_fixture', Path(__file__).with_name('test_bm1690_workitem_pcie_unittest.py'))
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)

SHIM = r'''
import io,json,os,pathlib,struct,subprocess,sys,urllib.request,importlib.util
from unittest.mock import patch
repo=pathlib.Path(os.environ['MC_TEST_REPO']);events=pathlib.Path(os.environ['MC_TEST_EVENTS'])
args=sys.argv[1:];mode=os.environ.get('MC_TEST_FAIL','')
def event(name):
    with events.open('a') as f:f.write(name+'\n')
def module(path):
    spec=importlib.util.spec_from_file_location('test_mc',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
if args[0]=='-':
    code=sys.stdin.read();sys.argv=args
    if len(args)==4:
        def download(request,timeout):
            relative=request.full_url.split('/contents/',1)[1].split('?',1)[0]
            assert relative in ('tpu_demo/pipeline/add_multicore_pcie.py','research/bm1690-pipelines/handoff/add-multicore-sources.json')
            payload=(repo/relative).read_bytes()
            return io.BytesIO(payload+b'changed' if mode=='checksum' else payload)
        def build(command,env,check):
            v=command[command.index('--variant')+1];event('build:'+v)
            assert '--allow-pcie' not in command and '--build-only' in command
            if mode=='build' and v=='serial_2':raise subprocess.CalledProcessError(9,command)
            root=pathlib.Path(command[command.index('--work')+1]);m=module(root/'multicore.py')
            bundle,builder,support=m.assets(root)
            layout=builder.resolve_legacy_pcie(pathlib.Path(env['PPL_PROJECT_ROOT']),'bm1690',environment=env)
            d=root/'build'/v;d.mkdir(parents=True)
            for n,s in bundle['variants'][v]['sources'].items():(d/n).write_text(s)
            for n,machine in (('main.so',62),('libkernel.so',243)):
                (d/n).write_bytes(b'\x7fELF\x02\x01'+bytes(10)+struct.pack('<HH',3,machine))
            r=dict(schema='bm1690-add-multicore-build-v1',status='compile_only_passed',bundle_sha256=m.BUNDLE_SHA,
                   builder_sha256=m.BUILDER_SHA,support_sha256=m.SUPPORT_SHA,variant=v,**m.contract(v),
                   runner_sha256=m.digest(root/'multicore.py'),board_runtime_loaded=False,kernel_launches=0,
                   runtime_identity=layout.runtime_identity_for('pcie',env),sdk_inputs={str(p):m.digest(p) for p in m.sdk_inputs(layout)},
                   completed_commands=m.commands(builder,layout,d),artifacts={n:m.digest(d/n) for n in ('main.so','libkernel.so')})
            support.save(d/'result.json',r)
            return subprocess.CompletedProcess(command,0)
        with patch.object(urllib.request,'urlopen',side_effect=download),patch.object(subprocess,'run',side_effect=build):
            exec(compile(code,'<bootstrap>','exec'))
    else:
        event('summary');exec(compile(code,'<summary>','exec'))
elif args[0].endswith('/multicore.py'):
    v=args[args.index('--variant')+1];timed='--correctness' in args
    event(('timing:' if timed else 'run:')+v)
    assert '--allow-pcie' in args and '--run-only' in args
    if mode=='run' and v=='serial_2':sys.exit(1)
    root=pathlib.Path(args[args.index('--work')+1]);m=module(root/'multicore.py')
    dest=pathlib.Path(args[args.index('--output')+1]);dest.mkdir()
    bundle,builder,support=m.assets(root);build,runtime=m.validate_build(root,bundle,builder,v)
    output=pathlib.Path(os.environ['MC_TEST_OUTPUT']).read_bytes();(dest/'output.f16').write_bytes(output)
    numeric=dict(status='passed',variant=v,**m.contract(v),measurement='timing' if timed else 'correctness',
                 shape=[1024,1024],dtype='float16',requested_kernel_calls=25 if timed else 1,bundle_sha256=m.BUNDLE_SHA,
                 build_manifest_sha256=m.digest(root/'build'/v/'result.json'),artifacts=build['artifacts'],
                 runtime_sha256=m.digest(runtime),mapping=dict(device_id=0,pci_function='0000:01:00.0',character_device='/dev/sg-host-drv-0'),
                 input_sha256='171cadcff2c82f6478fc59a9f95e713d8d7b7379a48796aebdc31ea94d22f5f1',
                 output_sha256=m.digest(dest/'output.f16'),reference=dict(bitwise_equal_to_reference=True))
    if timed:
        assert pathlib.Path(args[args.index('--correctness')+1])==root/('run-'+v)
        median={'original':100.,'serial_1':120.,'pipeline_1':110.,'serial_2':80.,'pipeline_2':70.}[v]
        numeric['sync_latency']=dict(warmups=5,samples=20,**support.summarize_samples([median]*20))
        if mode=='stats' and v=='pipeline_2':numeric['sync_latency']['median_us']=1.
    if mode=='cores' and v=='pipeline_2':numeric['launch_cores']=1
    support.save(dest/'result.json',dict(status='passed',cleanup_complete=True,numeric=numeric))
else:raise AssertionError(args)
'''


class MulticoreSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = tempfile.TemporaryDirectory()
        cls.expected = Path(cls.data.name)/'expected.f16'
        support = fixture.probe.verified_module(ROOT/'tpu_demo/pipeline/run_add_pcie.py', fixture.probe.SUPPORT_SHA, 'mc_sequence_support')
        _, _, expected = support.test_vectors(1024*1024)
        cls.expected.write_bytes(expected)

    @classmethod
    def tearDownClass(cls):
        cls.data.cleanup()

    def run_sequence(self, failure=''):
        case=fixture.ProbeTests();case.setUp()
        try:
            old=case.root
            shutil.copyfile(ROOT/'tpu_demo/pipeline/workitem_probe_pcie.py',old/'probe.py')
            run=old/'run';run.mkdir();_,output=fixture.probe.vectors();(run/'output.f16').write_bytes(output)
            numeric=dict(status='passed',launch_cores=2,tasks=11,ownership_counts=[6,5],checked_elements=704,
                         exact_ownership=True,bundle_sha256=fixture.probe.BUNDLE_SHA,artifacts=case.report['artifacts'],
                         build_manifest_sha256=fixture.probe.digest(old/'build/result.json'),
                         runtime_sha256=fixture.probe.digest(case.layout.board_lib/'libtpuv7_rt.so'),
                         output_sha256=hashlib.sha256(output).hexdigest())
            if failure=='gate':numeric['exact_ownership']=False
            (run/'result.json').write_text(json.dumps(dict(status='passed',cleanup_complete=True,numeric=numeric)))
            binpath=old/'fakebin';binpath.mkdir();python=binpath/'python3'
            python.write_text('#!'+sys.executable+'\n'+SHIM);python.chmod(0o700)
            events=old/'events';events.write_text('')
            env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],MC_TEST_REPO=str(ROOT),
                     MC_TEST_EVENTS=str(events),MC_TEST_FAIL=failure,MC_TEST_OUTPUT=str(self.expected))
            reply=subprocess.run(['bash',str(ROOT/'tpu_demo/pipeline/run_add_multicore_board.sh'),'--allow-pcie',
                                  'a'*40,str(old/'output'),str(old)],env=env,capture_output=True,text=True,timeout=15)
            summary=old/'output/summary.json'
            return reply,events.read_text().splitlines(),json.loads(summary.read_text()) if summary.exists() else None
        finally:
            case.doCleanups()

    def test_full_sequence_and_separate_pipeline_scaling_comparisons(self):
        r,events,summary=self.run_sequence()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        variants=['original','serial_1','pipeline_1','serial_2','pipeline_2']
        self.assertEqual(events,['build:'+v for v in variants]+[step+v for v in variants for step in ('run:','timing:')]+['summary'])
        c=summary['comparisons']
        self.assertAlmostEqual(c['pipeline_at_1_core']['median_ratio'],120/110)
        self.assertAlmostEqual(c['pipeline_at_2_cores']['median_ratio'],80/70)
        self.assertAlmostEqual(c['serial_1_to_2_cores']['median_ratio'],120/80)
        self.assertAlmostEqual(c['pipeline_1_to_2_cores']['median_ratio'],110/70)

    def test_missing_gate_or_corrupt_download_prevents_build(self):
        for mode in ('gate','checksum'):
            r,events,summary=self.run_sequence(mode)
            self.assertNotEqual(r.returncode,0);self.assertEqual(events,[]);self.assertIsNone(summary)

    def test_build_failure_stops_before_any_dispatch(self):
        r,events,summary=self.run_sequence('build')
        self.assertNotEqual(r.returncode,0);self.assertTrue(all(e.startswith('build:') for e in events));self.assertIsNone(summary)

    def test_run_failure_stops_remaining_tests(self):
        r,events,summary=self.run_sequence('run')
        self.assertNotEqual(r.returncode,0);self.assertEqual(events[-1],'run:serial_2');self.assertIsNone(summary)

    def test_wrong_core_receipt_or_fabricated_statistics_rejected(self):
        for mode in ('cores','stats'):
            r,events,summary=self.run_sequence(mode)
            self.assertNotEqual(r.returncode,0);self.assertEqual(events[-1],'summary');self.assertIsNone(summary)


if __name__ == '__main__':
    unittest.main()
