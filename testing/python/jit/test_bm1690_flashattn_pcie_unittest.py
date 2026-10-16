"""Attention ABI, independent oracle, provenance and recurrence contract regressions."""
import copy
import ctypes
import json
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from testing.python.jit import test_bm1690_add_pcie_unittest as base
from testing.python.jit.test_bm1690_legacy_pcie_unittest import legacy

runner = base.runner

class FlashManifestTests(base.ManifestTests):
    bundle_name = 'flashattn-smoke-sources.json'

    def test_mask_precision_layout_and_tile_contract_reject_changes(self):
        bundle = json.loads((self.root/'add.json').read_text())
        legacy.validate_source_bundle(bundle)
        for key,value in [('output_index',3),('is_causal',True),('layout','BHSD'),
                          ('mask_dtype','float16'),('probability_dtype','float32'),
                          ('reference','online-rounding'),('mask_shape',[32,32])]:
            changed=copy.deepcopy(bundle);changed['target'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):legacy.validate_source_bundle(changed)
        for key,value in [('tiling',[8,8]),('num_stages',3)]:
            changed=copy.deepcopy(bundle);changed['variants']['pipeline'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):legacy.validate_source_bundle(changed)

class FlashTargetManifestTests(base.ManifestTests):
    bundle_name = 'flashattn-1024-sources.json'

class FlashCausalManifestTests(base.ManifestTests):
    bundle_name = 'flashattn-causal-smoke-sources.json'

class FlashCausalTargetManifestTests(base.ManifestTests):
    bundle_name = 'flashattn-causal-1024-sources.json'

class FlashReferenceTests(unittest.TestCase):
    def test_board_vectors_reject_zero_uniform_and_last_value_shortcuts(self):
        for shape in ([1,64,1,16], [1,1024,1,64]):
            for causal in (False,True):
                q,k,v,mask,expected=runner.flashattn_test_vectors(shape=shape,is_causal=causal)
                values=np.frombuffer(v,dtype='<f2').reshape(shape[1],shape[3]).astype(np.float32)
                uniform=(np.cumsum(values,axis=0)/np.arange(1,shape[1]+1)[:,None] if causal else
                         np.broadcast_to(values.mean(axis=0),values.shape))
                for wrong in (bytes(len(v)), uniform.astype('<f2').tobytes(), v):
                    with self.subTest(shape=shape,causal=causal),self.assertRaises(ValueError):
                        runner.check_output(wrong,expected,shape[1]*shape[3],'flashattn')

    def test_zero_logits_average_values_and_causal_prefix(self):
        shape=[1,64,1,16]
        zero=np.zeros((64,16),dtype='<f2').tobytes()
        v=np.broadcast_to(np.arange(64,dtype=np.float16)[:,None]/64,(64,16)).astype('<f2').tobytes()
        for causal in (False,True):
            *_,mask,expected=runner.flashattn_test_vectors(shape=shape,is_causal=causal)
            result=np.frombuffer(runner.flashattn_reference(zero,zero,v,mask,shape=shape),dtype='<f2').reshape(64,16)
            target=np.arange(64)/128 if causal else np.full(64,31.5/64)
            np.testing.assert_array_equal(result[:,0],target.astype(np.float16))

    def test_causal_oracle_ignores_future_values_and_input_identity_binds_mask(self):
        shape=[1,64,1,16]
        q,k,v,mask,expected=runner.flashattn_test_vectors(shape=shape,is_causal=True)
        changed=np.frombuffer(v,dtype='<f2').copy().reshape(64,16);changed[32:]=2
        actual=runner.flashattn_reference(q,k,changed.tobytes(),mask,shape=shape)
        self.assertEqual(actual[:32*16*2],expected[:32*16*2])
        self.assertNotEqual(actual,expected)
        noncausal=runner.flashattn_test_vectors(shape=shape,is_causal=False)
        self.assertEqual((q,k,v),noncausal[:3]);self.assertNotEqual(mask,noncausal[3])
        with self.assertRaises(ValueError):runner.flashattn_test_vectors(shape=shape,is_causal=1)
        with self.assertRaises(ValueError):runner.flashattn_reference(q,k,v,mask[:-4],shape=shape)
        empty=np.full((64,64),-np.inf,dtype='<f4').tobytes()
        with self.assertRaises(ValueError):runner.flashattn_reference(q,k,v,empty,shape=shape)

    def test_oracle_handles_descending_and_ascending_maxima_without_overflow(self):
        shape=[1,64,1,16]
        q=np.full((64,16),10,dtype='<f2').tobytes()
        k=np.full((64,16),-10,dtype='<f2');k[:32]=10
        v=np.full((64,16),.75,dtype='<f2');v[:32]=.25
        mask=np.zeros((64,64),dtype='<f4').tobytes()
        for keys,wanted in ((k,.25),(-k,.75)):
            out=runner.flashattn_reference(q,keys.tobytes(),v.tobytes(),mask,shape=shape)
            self.assertEqual(out,struct.pack('<e',wanted)*(64*16))

    def test_original_tiling_may_round_differently_but_pipeline_must_match_serial(self):
        a,b=struct.pack('<e',1.0),struct.pack('<e',1.0009765625)
        for op in ('flashattn','flashattn-causal'):
            self.assertTrue(runner.check_previous_output(a,b,1,op,'serial')['passed'])
            with self.assertRaisesRegex(ValueError,'differs bitwise'):
                runner.check_previous_output(a,b,1,op,'pipeline')
            with self.assertRaises(ValueError):runner.check_output(struct.pack('<e',float('nan')),a,1,op)
            self.assertEqual(runner.check_output(a,a,1,op)['atol'],.02)

    def test_ordered_four_inputs_and_fifth_output_host_abi(self):
        *inputs,expected=runner.flashattn_test_vectors(shape=[1,64,1,16],is_causal=True)
        def execute(ptrs,*args):
            for i,value in enumerate(inputs):self.assertEqual(ctypes.string_at(ptrs[i],len(value)),value)
            ctypes.memmove(ptrs[4],expected,len(expected))
            if args:
                warm,count,samples=args
                for i in range(count):samples[i]=10+i
            return 0
        lib=SimpleNamespace(tilelang_tpu_run=execute,tilelang_tpu_run_timed=execute)
        self.assertEqual(runner.call_flashattn_host(lib,*inputs),expected)
        t={'warmups':1,'samples':2}
        self.assertEqual(runner.call_flashattn_host(lib,*inputs,timing=t),expected)
        self.assertEqual(t['raw_us'],[10,11])
        with self.assertRaises(ValueError):runner.call_flashattn_host(lib,*inputs[:3],inputs[3][:-4])

if __name__=='__main__':unittest.main()
