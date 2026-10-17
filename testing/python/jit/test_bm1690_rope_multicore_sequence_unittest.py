"""Exercise the real board sequence offline, including its four-way RoPE summary and preserved original failure."""
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
spec = importlib.util.spec_from_file_location('add_fixture', Path(__file__).with_name('test_bm1690_add_multicore_pcie_unittest.py'))
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)

SHIM = r'''
import io,json,os,pathlib,struct,subprocess,sys,urllib.request,importlib.util
from unittest.mock import patch
repo=pathlib.Path(os.environ['RP_TEST_REPO']);events=pathlib.Path(os.environ['RP_TEST_EVENTS'])
args=sys.argv[1:];mode=os.environ.get('RP_TEST_FAIL','')
def event(name):
    with events.open('a') as f:f.write(name+'\n')
def module(path):
    spec=importlib.util.spec_from_file_location('test_mc',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
if args[0]=='-':
    code=sys.stdin.read();sys.argv=args
    if len(args)==5:
        def download(request,timeout):
            relative=request.full_url.split('/contents/',1)[1].split('?',1)[0]
            assert relative in ('tpu_demo/pipeline/rope_multicore_pcie.py',*[f'research/bm1690-pipelines/handoff/{op}-multicore-sources.json' for op in ('rope',)])
            payload=(repo/relative).read_bytes()
            return io.BytesIO(payload+b'changed' if mode=='checksum' else payload)
        def build(command,env,check):
            v=command[command.index('--variant')+1];case=pathlib.Path(command[command.index('--work')+1]);op='rope';event('build:'+op+':'+v)
            assert '--allow-pcie' not in command and '--build-only' in command
            if mode=='build' and op=='rope' and v=='serial_2':raise subprocess.CalledProcessError(9,command)
            root=case;m=module(pathlib.Path(command[5]))
            bundle,builder,support=m.assets(root)
            layout=builder.resolve_legacy_pcie(pathlib.Path(env['PPL_PROJECT_ROOT']),'bm1690',environment=env)
            d=root/'build'/v;d.mkdir(parents=True)
            for n,s in bundle['variants'][v]['sources'].items():(d/n).write_text(s)
            for n,machine in (('main.so',62),('libkernel.so',243)):
                (d/n).write_bytes(b'\x7fELF\x02\x01'+bytes(10)+struct.pack('<HH',3,machine))
            r=dict(schema='bm1690-rope-multicore-build-v1',status='compile_only_passed',bundle_sha256=m.BUNDLE_SHAS[op],operation=op,
                   builder_sha256=m.BUILDER_SHA,support_sha256=m.SUPPORT_SHA,variant=v,**m.contract(v,op),
                   runner_sha256=m.digest(root/'rope.py'),board_runtime_loaded=False,kernel_launches=0,
                   runtime_identity=layout.runtime_identity_for('pcie',env),sdk_inputs={str(p):m.digest(p) for p in m.sdk_inputs(layout)},
                   completed_commands=m.commands(builder,layout,d),artifacts={n:m.digest(d/n) for n in ('main.so','libkernel.so')})
            support.save(d/'result.json',r)
            return subprocess.CompletedProcess(command,0)
        with patch.object(urllib.request,'urlopen',side_effect=download),patch.object(subprocess,'run',side_effect=build):
            exec(compile(code,'<bootstrap>','exec'))
    else:
        event('summary:rope');exec(compile(code,'<summary>','exec'))
elif args[0].endswith('/rope.py'):
    v=args[args.index('--variant')+1];timed='--correctness' in args
    root=pathlib.Path(args[args.index('--work')+1]);op='rope'
    event(('timing:' if timed else 'run:')+op+':'+v)
    assert '--allow-pcie' in args and '--run-only' in args
    if mode=='run' and op=='rope' and v=='serial_2':sys.exit(1)
    m=module(pathlib.Path(args[0]))
    dest=pathlib.Path(args[args.index('--output')+1]);dest.mkdir()
    bundle,builder,support=m.assets(root);build,runtime=m.validate_build(root,bundle,builder,v)
    output=b'rope-same-tile';(dest/'output.f16').write_bytes(output)
    numeric=dict(status='passed',variant=v,**m.contract(v,op),measurement='timing' if timed else 'correctness',
                 shape=[1024,1024],dtype='float16',requested_kernel_calls=25 if timed else 1,bundle_sha256=m.BUNDLE_SHAS[op],operation=op,
                 build_manifest_sha256=m.digest(root/'build'/v/'result.json'),artifacts=build['artifacts'],
                 runtime_sha256=m.digest(runtime),mapping=dict(device_id=0,pci_function='0000:01:00.0',character_device='/dev/sg-host-drv-0'),
                 input_sha256=m.INPUT_SHAS[op],
                 output_sha256=m.digest(dest/'output.f16'),reference=dict(passed=True,bitwise_equal_to_reference=False),
                 **m.original_failure(root,m.digest(runtime)))
    if timed:
        assert pathlib.Path(args[args.index('--correctness')+1])==root/('run-'+v)
        median={'original':100.,'serial_1':120.,'pipeline_1':110.,'serial_2':80.,'pipeline_2':70.}[v]
        numeric['sync_latency']=dict(warmups=5,samples=20,**support.summarize_samples([median]*20))
        if mode=='stats' and op=='rope' and v=='pipeline_2':numeric['sync_latency']['median_us']=1.
    if mode=='cores' and op=='rope' and v=='pipeline_2':numeric['launch_cores']=1
    if mode=='output' and op=='rope' and v=='pipeline_2':
        (dest/'output.f16').write_bytes(b'changed');numeric['output_sha256']=m.digest(dest/'output.f16')
    support.save(dest/'result.json',dict(status='passed',cleanup_complete=True,numeric=numeric))
else:raise AssertionError(args)
'''


class RoPESequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=tempfile.TemporaryDirectory();cls.expected=Path(cls.data.name)
        support=fixture.probe.verified_module(ROOT/'tpu_demo/pipeline/run_add_pcie.py',fixture.probe.SUPPORT_SHA,'ew_sequence_support')
        for op in ('add',):
            _,_,output=support.test_vectors(1024*1024,op);(cls.expected/(op+'.f16')).write_bytes(output)

    @classmethod
    def tearDownClass(cls):cls.data.cleanup()

    def run_sequence(self,failure=''):
        case=fixture.MulticoreTests();case.setUp()
        try:
            old=case.root
            shutil.copyfile(ROOT/'tpu_demo/pipeline/add_multicore_pcie.py',old/'multicore.py')
            output=(self.expected/'add.f16').read_bytes()
            for variant in fixture.probe.VARIANTS:
                d=old/'build'/variant;d.mkdir(parents=True,exist_ok=True)
                for name,source in case.bundle['variants'][variant]['sources'].items():(d/name).write_text(source)
                for name in ('main.so','libkernel.so'):
                    if variant!=case.variant:shutil.copyfile(case.build/name,d/name)
                report=dict(case.report,variant=variant,**fixture.probe.contract(variant))
                report['completed_commands']=fixture.probe.commands(case.builder,case.layout,d)
                (d/'result.json').write_text(json.dumps(report))
                numeric=dict(status='passed',operation='add',variant=variant,**fixture.probe.contract(variant),
                    measurement='correctness',requested_kernel_calls=1,shape=[1024,1024],dtype='float16',
                    bundle_sha256=fixture.probe.BUNDLE_SHA,build_manifest_sha256=fixture.probe.digest(d/'result.json'),
                    artifacts=report['artifacts'],runtime_sha256=fixture.probe.digest(case.layout.board_lib/'libtpuv7_rt.so'),
                    input_sha256='171cadcff2c82f6478fc59a9f95e713d8d7b7379a48796aebdc31ea94d22f5f1',
                    output_sha256=hashlib.sha256(output).hexdigest(),reference=dict(bitwise_equal_to_reference=True),
                    mapping=dict(device_id=0,pci_function='0000:01:00.0',character_device='/dev/sg-host-drv-0'))
                for kind in ('run','timing'):
                    dest=old/(kind+'-'+variant);dest.mkdir();(dest/'output.f16').write_bytes(output)
                    n=dict(numeric)
                    if kind=='timing':
                        n.update(measurement='timing',requested_kernel_calls=25,
                                 sync_latency=dict(warmups=5,samples=20,**case.support.summarize_samples([100.]*20)))
                    if failure=='gate' and variant=='pipeline_2':n['launch_cores']=1
                    (dest/'result.json').write_text(json.dumps(dict(status='passed',cleanup_complete=True,numeric=n)))
            binpath=old/'fakebin';binpath.mkdir();python=binpath/'python3'
            python.write_text('#!'+sys.executable+'\n'+SHIM);python.chmod(0o700)
            events=old/'events';events.write_text('')
            env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],RP_TEST_REPO=str(ROOT),
                RP_TEST_EVENTS=str(events),RP_TEST_FAIL=failure,RP_TEST_OUTPUT=str(self.expected))
            failure_dir=old/'old-rope-original';failure_dir.mkdir()
            spec=importlib.util.spec_from_file_location('rope_fixture',Path(__file__).with_name('test_bm1690_rope_multicore_pcie_unittest.py'))
            rope_fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(rope_fixture)
            failed=rope_fixture.failure_receipt(fixture.probe.digest(case.layout.board_lib/'libtpuv7_rt.so'))
            if failure=='original':failed['numeric']['shape']=[8,128]
            (failure_dir/'result.json').write_text(json.dumps(failed))
            reply=subprocess.run(['bash',str(ROOT/'tpu_demo/pipeline/run_rope_multicore_board.sh'),'--allow-pcie',
                'a'*40,str(old/'output'),str(old),str(failure_dir)],env=env,capture_output=True,text=True,timeout=30)
            summaries={'rope':json.loads((old/'output/summary.json').read_text())} if (old/'output/summary.json').exists() else {}
            return reply,events.read_text().splitlines(),summaries
        finally:case.doCleanups()

    def test_full_sequence_and_separate_pipeline_scaling_comparisons(self):
        r,events,summaries=self.run_sequence()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        variants=['serial_1','pipeline_1','serial_2','pipeline_2'];ops=('rope',)
        expected=['build:'+op+':'+v for op in ops for v in variants]
        for op in ops:expected += [step+op+':'+v for v in variants for step in ('run:','timing:')]+['summary:'+op]
        self.assertEqual(events,expected);self.assertEqual(set(summaries),set(ops))
        for summary in summaries.values():
            self.assertEqual(summary['raw_samples'],80)
            self.assertEqual(summary['original_baseline_status'],'failed_timeout_unresolved')
            self.assertFalse(summary['three_variant_acceptance_complete'])
            c=summary['comparisons']
            self.assertAlmostEqual(c['pipeline_at_1_core']['median_ratio'],120/110)
            self.assertAlmostEqual(c['pipeline_at_2_cores']['median_ratio'],80/70)
            self.assertAlmostEqual(c['serial_1_to_2_cores']['median_ratio'],120/80)
            self.assertAlmostEqual(c['pipeline_1_to_2_cores']['median_ratio'],110/70)

    def test_missing_gate_or_corrupt_download_prevents_build(self):
        for mode in ('gate','checksum','original'):
            r,events,summaries=self.run_sequence(mode)
            self.assertNotEqual(r.returncode,0);self.assertEqual(events,[]);self.assertFalse(summaries)

    def test_build_failure_stops_before_any_dispatch(self):
        r,events,summaries=self.run_sequence('build')
        self.assertNotEqual(r.returncode,0);self.assertEqual(events[-1],'build:rope:serial_2')
        self.assertTrue(all(e.startswith('build:') for e in events));self.assertFalse(summaries)

    def test_run_failure_stops_remaining_tests(self):
        r,events,summaries=self.run_sequence('run')
        self.assertNotEqual(r.returncode,0);self.assertEqual(events[-1],'run:rope:serial_2');self.assertFalse(summaries)

    def test_wrong_core_receipt_or_fabricated_statistics_rejected(self):
        for mode in ('cores','stats'):
            r,events,summaries=self.run_sequence(mode)
            self.assertNotEqual(r.returncode,0);self.assertEqual(events[-1],'summary:rope');self.assertFalse(summaries)

    def test_output_mismatch_against_same_tile_serial_rejected(self):
        r,events,summaries=self.run_sequence('output')
        self.assertNotEqual(r.returncode,0);self.assertEqual(events[-1],'summary:rope');self.assertFalse(summaries)


if __name__ == '__main__':unittest.main()
