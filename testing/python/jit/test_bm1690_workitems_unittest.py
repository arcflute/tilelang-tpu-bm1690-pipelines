"""Compile-only workitem, launch ABI and allocation contract tests."""

import tempfile
from pathlib import Path
import unittest

import tilelang
from tilelang import tvm
from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper
from tpu_demo.pipeline.kernels import build_elementwise_tiled, pipeline_kernel_tiles
from tpu_demo.pipeline.workitems import map_workitems, build_workitem_probe

TARGET = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")


class WorkitemTests(unittest.TestCase):
    def test_launch_abi_and_complete_per_core_argument_arrays(self):
        for cores in (1,2,4,8):
            function = build_workitem_probe(cores)
            artifact = tilelang.lower(function,target=TARGET,runtime_mode="cmodel")
            self.assertIn("tpu_workitem_index()",artifact.kernel_source)
            self.assertIn("tpu_workitem_num()",artifact.kernel_source)
            with tempfile.TemporaryDirectory() as directory:
                TLTPUSourceWrapper(tvm.IRModule({"main":function}),artifact.kernel_source,TARGET,
                                   output_dir=directory)
                wrapper = (Path(directory)/"kernel.cpp").read_text()
            self.assertIn(f"constexpr int core_num = {cores};",wrapper)
            self.assertIn("block_num = core_num",wrapper)
            self.assertIn("apis[i] = api",wrapper)
            self.assertIn("sizeof(apis)",wrapper)

    def test_invalid_launch_count_and_chip_are_rejected(self):
        function = build_workitem_probe(2)
        with self.assertRaisesRegex(ValueError,"launch_cores"):
            tilelang.lower(function.with_attr("tilelang.tpu.launch_cores",3),target=TARGET)
        with self.assertRaisesRegex(ValueError,"BM1690"):
            tilelang.lower(function,target="tpu -mcpu=sg2260e -tpu-programming-model=tpukernel")
        body = tvm.tir.SeqStmt([tvm.tir.Evaluate(tvm.tir.call_extern("float32","tl.tpukernel.workitem_index")),
                                function.body])
        with self.assertRaisesRegex(ValueError,"int32 return"):
            tilelang.lower(function.with_body(body),target=TARGET)

    def test_flat_tiles_require_full_per_core_pipeline_and_no_unhandled_tail(self):
        function = build_elementwise_tiled(num_stages=3)
        with self.assertRaisesRegex(ValueError,"num_stages output tasks"):
            map_workitems(function,4,tile_loop="tile")
        with self.assertRaisesRegex(ValueError,"divisible"):
            map_workitems(build_elementwise_tiled(rows=4,width=96,num_stages=2),2,tile_loop="tile")

    def test_six_families_keep_reductions_local_and_fit_lmem(self):
        from tpu_demo.matmul.matmul import build_matmul
        from tpu_demo.rmsnorm.rmsnorm import build_rmsnorm, build_rmsnorm_splitk
        from tpu_demo.rope.rope import build_rope
        from tpu_demo.swiglu.swiglu import build_swiglu
        from tpu_demo.flashattn.flashattn import build_flashattn
        functions = [
            (build_elementwise_tiled(num_stages=2),"tile"),
            (build_matmul(m=48,n=16,k=128,num_stages=2),None), # uneven three tasks
            (pipeline_kernel_tiles(build_rope(width=128),2),"output_tile"),
            (pipeline_kernel_tiles(build_swiglu(width=128),2),"output_tile"),
            (pipeline_kernel_tiles(build_rmsnorm(rows=32),2),"output_tile"),
            (build_rmsnorm_splitk(num_stages=2),None),
            (build_flashattn(sequence=64,num_stages=2),None)]
        for function,loop in functions:
            with self.subTest(kernel=str(function.attrs["global_symbol"])):
                artifact = tilelang.lower(map_workitems(function,2,tile_loop=loop),target=TARGET)
                self.assertIn("tpu_workitem_index()",artifact.kernel_source)
                self.assertIn("tpu_parallel_start();",artifact.kernel_source)
                for module in (artifact.host_mod,artifact.device_mod):
                    for lowered in module.functions.values():
                        if lowered.attrs and "tilelang.tpu.lmem.high_water_bytes" in lowered.attrs:
                            self.assertLessEqual(int(lowered.attrs["tilelang.tpu.lmem.high_water_bytes"]),256*1024)


if __name__ == "__main__":
    unittest.main()
