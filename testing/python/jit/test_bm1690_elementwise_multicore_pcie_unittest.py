"""Matched single/dual-core elementwise handoff validation; never loads a vendor library."""
import contextlib
import copy
import ctypes
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'tpu_demo/pipeline/elementwise_multicore_pcie.py'
spec = importlib.util.spec_from_file_location('elementwise_multicore_board_test', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def write(path, data=b''):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    return path


class MulticoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for dest,source in (('build.py','tilelang/jit/adapter/legacy_pcie.py'),
                            ('run_add.py','tpu_demo/pipeline/run_add_pcie.py'),
                            ('elementwise-multicore.json','research/bm1690-pipelines/handoff/sub-multicore-sources.json')):
            shutil.copyfile(ROOT/source,self.root/dest)
        self.bundle,self.builder,self.support=probe.assets(self.root)
        sdk=self.root/'sdk'; board=self.root/'board'
        for name in ('runtime/bm1690/TPU1686/kernel/include/tpu_kernel.h','runtime/kernel/common.h',
                     'runtime/customize/include/ppl_helper.h','runtime/customize/include/ppl_mem.h',
                     'runtime/customize/src/ppl_helper.c','runtime/bm1690/lib/libbm1690.a'):
            write(sdk/name)
        write(board/'include/tpuv7_rt.h');write(board/'lib/libtpuv7_rt.so')
        cross=write(self.root/'bin/riscv64-unknown-linux-gnu-gcc');cross.chmod(0o700)
        self.env=dict(TILELANG_TPU_PCIE_RUNTIME_PATH=str(board/'lib'),TILELANG_TPU_PCIE_CROSS_GCC=str(cross))
        self.layout=self.builder.resolve_legacy_pcie(sdk,'bm1690',environment=self.env)
        self.variant='pipeline_2'
        self.build=self.root/'build'/self.variant;self.build.mkdir(parents=True)
        for name,source in self.bundle['variants'][self.variant]['sources'].items():
            (self.build/name).write_text(source)
        for name,machine in (('main.so',62),('libkernel.so',243)):
            write(self.build/name,b'\x7fELF\x02\x01'+bytes(10)+struct.pack('<HH',3,machine))
        self.report=dict(schema='bm1690-elementwise-multicore-build-v1',status='compile_only_passed',
                         operation='sub',bundle_sha256=probe.BUNDLE_SHAS['sub'],builder_sha256=probe.BUILDER_SHA,
                         support_sha256=probe.SUPPORT_SHA,variant=self.variant,**probe.contract(self.variant),runner_sha256=probe.digest(SCRIPT),
                         board_runtime_loaded=False,kernel_launches=0,
                         runtime_identity=list(self.layout.runtime_identity_for('pcie',self.env)),
                         sdk_inputs={str(p):probe.digest(p) for p in probe.sdk_inputs(self.layout)},
                         completed_commands=probe.commands(self.builder,self.layout,self.build),
                         artifacts={n:probe.digest(self.build/n) for n in ('main.so','libkernel.so')})
        self.save()

    def save(self):
        (self.build/'result.json').write_text(json.dumps(self.report))

    def test_valid_manifest_does_not_load_runtime(self):
        with patch.object(ctypes,'CDLL',side_effect=AssertionError('must not load')):
            report,runtime=probe.validate_build(self.root,self.bundle,self.builder,self.variant)
        self.assertEqual(report['launch_cores'],2)
        self.assertEqual(runtime,self.layout.board_lib/'libtpuv7_rt.so')

    def test_cmodel_all_operations_and_division_tolerance(self):
        result=json.loads((ROOT/'research/bm1690-pipelines/results/p9-elementwise-multicore-handoff-cmodel.json').read_text())
        self.assertEqual(result['status'],'passed')
        for op in ('sub','mul','div'):
            rows=[r['numeric'] for r in result['variants'] if r['numeric']['operation']==op]
            self.assertEqual({r['variant'] for r in rows},set(probe.VARIANTS))
            self.assertEqual(len({r['output_sha256'] for r in rows}),1)
            for r in rows:
                self.assertTrue(r['reference']['passed'])
                self.assertEqual(r['reference']['bitwise_equal_to_reference'],op!='div')
                self.assertTrue(r['timed_abi_checked'])
                self.assertEqual(r['bundle_sha256'],probe.BUNDLE_SHAS[op])
                self.assertEqual(r['input_sha256'],probe.INPUT_SHAS[op])

    def test_lowering_has_per_core_extent_and_double_buffer(self):
        for cores in (1,2):
            r=self.bundle['variants'][f'pipeline_{cores}']['pipeline_reports'][0]
            self.assertEqual(r['extent'],8//cores)
            self.assertEqual(r['order'],[0,1,2,3])
            self.assertEqual(r['stage'],[0,0,1,1])
            self.assertEqual(r['buffer_versions'],dict(a=['a','a_pipeline_v1'],b=['b','b_pipeline_v1']))
            self.assertIn('after parallel end',r['stores'])

    def test_single_core_sources_keep_original_baselines(self):
        old=json.loads((ROOT/'research/bm1690-pipelines/handoff/sub-1024-sources.json').read_text())
        for new,prior in (('original','original'),('serial_1','serial'),('pipeline_1','pipeline')):
            # Device arithmetic and tiling have not changed. Host infrastructure has evolved.
            self.assertEqual(self.bundle['variants'][new]['sha256']['kernel.c'],old['variants'][prior]['sha256']['kernel.c'])

    def test_source_and_host_launch_count_tamper_rejected(self):
        bad=copy.deepcopy(self.bundle);bad['variants'][self.variant]['launch_cores']=8
        with self.assertRaises(ValueError):probe.validate_bundle(bad)
        bad=copy.deepcopy(self.bundle);bad['variants'][self.variant]['sources']['kernel.cpp']=bad['variants'][self.variant]['sources']['kernel.cpp'].replace('core_num = 2','core_num = 1')
        with self.assertRaises(ValueError):probe.validate_bundle(bad)
        for name in ('elementwise-multicore.json','build.py','run_add.py'):
            with self.subTest(name=name):
                path=self.root/name;old=path.read_bytes();path.write_bytes(old+b'changed')
                try:
                    with self.assertRaises(ValueError):probe.assets(self.root)
                finally:path.write_bytes(old)

    def test_build_binary_sdk_source_tampering_rejected(self):
        for path in (self.build/'kernel.c',self.build/'main.so',self.build/'libkernel.so',
                     self.layout.board_lib/'libtpuv7_rt.so',self.layout.firmware_archive):
            with self.subTest(path=path):
                old=path.read_bytes();path.write_bytes(old+b'changed')
                try:
                    with self.assertRaises(ValueError):probe.validate_build(self.root,self.bundle,self.builder,self.variant)
                finally:path.write_bytes(old)

    def test_wrong_architecture_rejected_even_if_hash_updated(self):
        write(self.build/'libkernel.so',b'\x7fELF\x02\x01'+bytes(10)+struct.pack('<HH',3,62))
        self.report['artifacts']['libkernel.so']=probe.digest(self.build/'libkernel.so');self.save()
        with self.assertRaisesRegex(ValueError,'architecture'):
            probe.validate_build(self.root,self.bundle,self.builder,self.variant)

    def test_incomplete_or_moved_build_rejected(self):
        row=self.report['completed_commands'].pop();self.save()
        with self.assertRaises(ValueError):probe.validate_build(self.root,self.bundle,self.builder,self.variant)
        self.report['completed_commands'].append(row);self.save()
        moved=self.root/'moved'
        shutil.copytree(self.build,moved/'build'/self.variant)
        with self.assertRaisesRegex(ValueError,'location'):
            probe.validate_build(moved,self.bundle,self.builder,self.variant)

    def test_occupation_or_permission_failure_prevents_supervisor(self):
        for rc,err in ((0,'holder'),(1,'sudo denied'),(124,'')):
            with self.subTest(rc=rc):
                output=self.root/f'run-{rc}'
                argv=[str(SCRIPT),'--run-only','--allow-pcie','--variant',self.variant,'--work',str(self.root),'--output',str(output)]
                reply=subprocess.CompletedProcess([],rc,stdout='',stderr=err)
                with patch.object(sys,'argv',argv),patch.object(probe,'assets',return_value=(self.bundle,self.builder,self.support)),patch.object(probe.subprocess,'run',return_value=reply),patch.object(self.support,'supervise',side_effect=AssertionError('must not launch')):
                    with self.assertRaisesRegex(RuntimeError,'occupied'):probe.main()

    def test_allow_flag_and_new_output_required(self):
        with patch.object(sys,'argv',[str(SCRIPT),'--run-only','--work',str(self.root),'--output',str(self.root/'run')]),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):probe.main()
        with patch.object(sys,'argv',[str(SCRIPT),'--run-only','--allow-pcie','--variant',self.variant,'--work',str(self.root),'--output',str(self.build)]):
            with self.assertRaises(FileExistsError):probe.main()

    def test_public_guard_worker_rejects_corruption_before_runtime_load(self):
        (self.build/'kernel.c').write_text('changed')
        fakebin=self.root/'fakebin';fakebin.mkdir()
        sudo=write(fakebin/'sudo',b'#!/bin/sh\nexit 1\n');sudo.chmod(0o700)
        env=dict(os.environ,PATH=str(fakebin)+os.pathsep+os.environ['PATH'])
        for k in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','TPU_RT_CORE_NUM'):
            env.pop(k,None)
        output=self.root/'real-supervisor'
        proc=subprocess.run([sys.executable,str(SCRIPT),'--run-only','--allow-pcie','--variant',self.variant,
                             '--work',str(self.root),'--output',str(output)],env=env,
                            capture_output=True,text=True,timeout=10)
        self.assertNotEqual(proc.returncode,0,proc.stdout+proc.stderr)
        receipt=json.loads((output/'result.json').read_text())
        self.assertEqual(receipt['status'],'failed')
        self.assertTrue(receipt['cleanup_complete'])
        self.assertEqual(receipt['numeric']['last_stage'],'validate_build')
        self.assertFalse(receipt['numeric']['dispatch_attempted'])

    def test_timing_rejects_correctness_from_different_core_count(self):
        correct=self.root/'correct';correct.mkdir()
        output=write(correct/'output.f16',b'\x00\x00')
        numeric=dict(status='passed',measurement='correctness',reference=dict(passed=True,bitwise_equal_to_reference=False),
                     output_sha256=probe.digest(output),variant='pipeline_2',launch_cores=2,
                     input_sha256='same-input',operation='sub',bundle_sha256=probe.BUNDLE_SHAS['sub'])
        (correct/'result.json').write_text(json.dumps(dict(status='passed',cleanup_complete=True,numeric=numeric)))
        self.assertEqual(probe.validate_correctness(correct,numeric),b'\x00\x00')
        for field,value in (('operation','div'),('launch_cores',1),('variant','pipeline_1'),('input_sha256','other-input'),('runtime_sha256','other-runtime')):
            wrong=dict(numeric);wrong[field]=value
            with self.assertRaisesRegex(ValueError,'identity'):
                probe.validate_correctness(correct,wrong)


    def test_all_bundle_pins_and_single_core_arithmetic_preserved(self):
        for op in ('sub','mul','div'):
            path=ROOT/f'research/bm1690-pipelines/handoff/{op}-multicore-sources.json'
            probe.checked(path,probe.BUNDLE_SHAS[op]);bundle=json.loads(path.read_text())
            probe.validate_bundle(bundle)
            old=json.loads((ROOT/f'research/bm1690-pipelines/handoff/{op}-1024-sources.json').read_text())
            for new,prior in (('original','original'),('serial_1','serial'),('pipeline_1','pipeline')):
                self.assertEqual(bundle['variants'][new]['sha256']['kernel.c'],old['variants'][prior]['sha256']['kernel.c'])

    def test_other_operation_cannot_reuse_build_receipt(self):
        self.report['operation']='div';self.save()
        with self.assertRaisesRegex(ValueError,'receipt'):
            probe.validate_build(self.root,self.bundle,self.builder,self.variant)

    def test_original_output_requires_original_build_and_matching_identity(self):
        d=self.root/'run-original';d.mkdir()
        output=write(d/'output.f16',b'\x00\x00')
        write(self.root/'build/original/result.json',b'fixture')
        expected=dict(variant='pipeline_2',**probe.contract('pipeline_2'),operation='sub',
                      build_manifest_sha256='other',runtime_sha256='other',artifacts={'other':'hash'},
                      input_sha256='same',bundle_sha256=probe.BUNDLE_SHAS['sub'])
        receipt=dict(status='passed',measurement='correctness',reference=dict(passed=True),
                     output_sha256=probe.digest(output),variant='original',**probe.contract('original'),operation='sub',
                     build_manifest_sha256=probe.digest(self.root/'build/original/result.json'),
                     runtime_sha256=probe.digest(self.layout.board_lib/'libtpuv7_rt.so'),artifacts={'original':'hash'},
                     input_sha256='same',bundle_sha256=probe.BUNDLE_SHAS['sub'])
        (d/'result.json').write_text(json.dumps(dict(status='passed',cleanup_complete=True,numeric=receipt)))
        with patch.object(probe,'validate_build',return_value=({'artifacts':{'original':'hash'}},self.layout.board_lib/'libtpuv7_rt.so')) as check:
            self.assertEqual(probe.original_output(self.root,self.bundle,self.builder,expected),b'\x00\x00')
            self.assertEqual(check.call_args.args[-1],'original')
            expected['input_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'identity'):probe.original_output(self.root,self.bundle,self.builder,expected)



if __name__=='__main__':unittest.main()
