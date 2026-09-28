"""Compile-only pipeline regressions, runnable without pytest or a device."""

import json
import unittest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tpu_demo.pipeline.kernels import build_elementwise_tiled

TARGET = "tpu -mcpu=bm1690 -tpu-programming-model=tpukernel"


def lower(function, target=TARGET):
    return tilelang.lower(function, target=target, runtime_mode="cmodel")


def report(artifact):
    for module in (artifact.host_mod, artifact.device_mod):
        for function in module.functions.values():
            if function.attrs and function.attrs.get("tilelang.tpu.pipeline_report") is not None:
                return function, json.loads(str(function.attrs["tilelang.tpu.pipeline_report"]))
    raise AssertionError("missing pipeline report")


def annotate(function, **updates):
    def change(node):
        if isinstance(node, tvm.tir.For) and "num_stages" in node.annotations:
            annotations = dict(node.annotations)
            annotations.update(updates)
            return tvm.tir.For(node.loop_var, node.min, node.extent, node.kind, node.body,
                               node.thread_binding, annotations)
        return None
    return function.with_body(tvm.tir.stmt_functor.ir_transform(function.body, None, change, ["tir.For"]))


class PipelineTests(unittest.TestCase):
    def test_real_versions_and_synchronization_for_short_and_steady_loops(self):
        for depth in (2, 3):
            for iterations in (depth, 8):
                with self.subTest(depth=depth, iterations=iterations):
                    artifact = lower(build_elementwise_tiled(rows=4, width=32*iterations, num_stages=depth))
                    function, schedules = report(artifact)
                    versions = schedules[0]["buffer_versions"]
                    self.assertEqual(set(versions), {"a", "b"})
                    self.assertTrue(all(len(buffers) == depth for buffers in versions.values()))
                    addresses = [int(value) for key, value in function.attrs.items()
                                 if str(key).startswith("tilelang.tpu.lmem.address.")]
                    # Every buffer is 64 bytes per lane in these test shapes;
                    # all overlap in lifetime, including output c.
                    self.assertEqual(len(addresses), 2*depth+1)
                    ordered = sorted(addresses)
                    self.assertTrue(all(b-a >= 64 for a,b in zip(ordered, ordered[1:])))
                    self.assertLessEqual(ordered[-1]+64, 256*1024)
                    self.assertIn("tpu_parallel_start();", artifact.kernel_source)
                    self.assertIn("#ifndef USING_CMODEL", artifact.kernel_source)
                    self.assertEqual(artifact.kernel_source.count("tpu_parallel_start();"),
                                     artifact.kernel_source.count("tpu_parallel_end();"))
                    self.assertNotIn("_pipeline_v", lower(build_elementwise_tiled()).kernel_source)

    def test_unsupported_depth_and_short_loop_rejected(self):
        with self.assertRaisesRegex(ValueError, "extent must exceed"):
            lower(build_elementwise_tiled(rows=4, width=32, num_stages=2))
        with self.assertRaisesRegex(ValueError, "supported num_stages"):
            lower(annotate(build_elementwise_tiled(num_stages=2), num_stages=4))

    def test_wrong_target_rejected(self):
        for model in ("tpukernel", "rv"):
            with self.assertRaisesRegex(ValueError, "bm1690"):
                lower(build_elementwise_tiled(num_stages=2),
                      f"tpu -mcpu=sg2260e -tpu-programming-model={model}")

    def test_explicit_schedule_requires_complete_current_contract(self):
        function = build_elementwise_tiled(num_stages=2)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            lower(annotate(function, tl_pipeline_order=[0,1,2,3]))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            lower(annotate(function, tl_pipeline_order=[0,1,2,3], tl_pipeline_stage=[0,0,1,1]))
        with self.assertRaisesRegex(ValueError, "permutation"):
            lower(annotate(function, tl_pipeline_order=[0,0,2,3], tl_pipeline_stage=[0,0,1,1]))

    def test_mutated_prefetch_operand_is_not_scheduled(self):
        @T.prim_func
        def invalid(source: T.Tensor((8,32), "float16"), destination: T.Tensor((8,32), "float16")):
            with T.Kernel(1, is_cpu=True) as (_bx,):
                a = T.alloc_shared((1,32), "float16")
                for i in T.Pipelined(8, num_stages=2):
                    T.ppl_copy(source[i,0], a)
                    T.ppl_mul_C(a, a, T.float16(2))
                    T.ppl_copy(a, destination[i,0])
        with self.assertRaisesRegex(ValueError, "remain read-only"):
            lower(invalid)

    def test_global_recurrence_is_not_prefetched(self):
        @T.prim_func
        def invalid(source: T.Tensor((8,32), "float16")):
            with T.Kernel(1, is_cpu=True) as (_bx,):
                a = T.alloc_shared((1,32), "float16")
                c = T.alloc_shared((1,32), "float16")
                for i in T.Pipelined(8, num_stages=2):
                    T.ppl_copy(source[i,0], a)
                    T.ppl_mul_C(c, a, T.float16(2))
                    T.ppl_copy(c, source[i,0])
        with self.assertRaisesRegex(ValueError, "read-only across the function"):
            lower(invalid)

    def test_unmodelled_sync_annotation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unmodelled pipeline annotations"):
            lower(annotate(build_elementwise_tiled(num_stages=2), tl_pipeline_sync=[[0,1]]))

    def test_parallel_scope_cannot_bypass_storage_dependency_checks(self):
        from tilelang.engine.tpu_pipeline import validate_parallel_scope, PARALLEL_SCOPE
        function = build_elementwise_tiled()
        target = tvm.target.Target(TARGET)
        module = LowerAndLegalize(tvm.IRModule({"main": function}), target)
        body = []
        def collect(node):
            if isinstance(node, tvm.tir.For) and node.loop_var.name == "tile":
                body.append(node.body)
        tvm.tir.stmt_functor.post_order_visit(module["main"].body, collect)
        statements = list(body[0].seq)
        # These loads write exactly the buffers consumed by the same iteration.
        scope = tvm.tir.AttrStmt(tvm.tir.IntImm("int32",0), PARALLEL_SCOPE, 1,
                                 tvm.tir.SeqStmt(statements[:-1]))
        with self.assertRaisesRegex(ValueError, "overwrites a concurrent"):
            validate_parallel_scope(scope)


if __name__ == "__main__":
    unittest.main()
