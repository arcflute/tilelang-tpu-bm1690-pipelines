"""Internal CModel worker. The parent runner supplies resource restrictions."""

from __future__ import annotations

import argparse
import json
import os


def pipeline_cases():
    return ["elementwise-add", "elementwise-sub", "elementwise-mul", "elementwise-div",
            "matmul", "rope", "swiglu"]


def run_pipeline_case(case, seed, size, stages=2):
    """Compare identical arithmetic/tiling, using independent reference math."""
    from pathlib import Path
    import hashlib
    import torch
    import tilelang
    from tpu_demo.common import comparison, configure_runtime, tolerance
    from tpu_demo.pipeline.kernels import build_elementwise_tiled, pipeline_kernel_tiles

    configure_runtime("cmodel", False, None, chip="bm1690")
    generator = torch.Generator().manual_seed(seed)
    rows, width = (8, 128) if size == "smoke" else (1024, 1024)
    if case.startswith("elementwise-"):
        operation = case.split("-")[1]
        lhs = torch.randn((rows, width), generator=generator).half()
        rhs = (torch.rand((rows, width), generator=generator)*1.5+0.5).half() if operation == "div" else torch.randn((rows, width), generator=generator).half()
        calculate = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[operation]
        expected = calculate(lhs.float(), rhs.float()).half()
        inputs = [lhs, rhs]
        family = "elementwise-div" if operation == "div" else "elementwise"
        block_rows, block_width = (4,32) if size == "smoke" else (32,128)
        parameters = {"rows": rows, "width": width, "block_rows": block_rows, "block_width": block_width}
        def build(depth):
            return build_elementwise_tiled(operation, **parameters, num_stages=depth)
    elif case == "matmul":
        from tpu_demo.matmul.matmul import build_matmul
        m, n, k = (32, 32, 128) if size == "smoke" else (1024, 1024, 1024)
        block = 16 if size == "smoke" else 32
        lhs = (torch.randn((m,k), generator=generator)*0.25).half()
        rhs = (torch.randn((k,n), generator=generator)*0.25).half()
        inputs = [lhs,rhs]
        expected = (lhs.float() @ rhs.float()).half()
        family = "matmul"
        parameters = {"m":m,"n":n,"k":k,"block_m":block,"block_n":block,"block_k":block}
        def build(depth):
            return build_matmul(**parameters, num_stages=depth)
    elif case == "rope":
        from tpu_demo.rope.rope import build_rope, _cosine_sine, _reference
        source = torch.randn((rows,width), generator=generator).half()
        cosine, sine = _cosine_sine(rows,width)
        inputs = [source,cosine,sine]
        expected = _reference(*inputs)
        family = "rope"
        parameters = {"rows":rows,"width":width,"block_rows":4 if size == "smoke" else 32,
                      "block_width":32 if size == "smoke" else 64}
        def build(depth):
            return pipeline_kernel_tiles(build_rope(**parameters),depth)
    elif case == "swiglu":
        from tpu_demo.swiglu.swiglu import build_swiglu
        gate = torch.randn((rows,width), generator=generator).clamp(-3,3).half()
        up = torch.randn((rows,width), generator=generator).half()
        inputs = [gate,up]
        expected = (up.float()*torch.nn.functional.silu(gate.float())).half()
        family = "swiglu"
        parameters = {"rows":rows,"width":width,"block_rows":4 if size == "smoke" else 32,
                      "block_width":32 if size == "smoke" else 128}
        def build(depth):
            return pipeline_kernel_tiles(build_swiglu(**parameters),depth)
    else:
        raise ValueError(f"unknown pipeline case {case}")
    originals = [tensor.clone() for tensor in inputs]
    variants = {}
    outputs = []
    for name, depth in (("serial", 0), ("pipeline", stages)):
        function = build(depth)
        artifact = tilelang.lower(function, target="tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
                                  runtime_mode="cmodel")
        Path(name + ".c").write_text(artifact.kernel_source)
        Path(name + ".tir").write_text(artifact.host_mod.script() + "\n" + artifact.device_mod.script())
        reports = []
        addresses = {}
        for module in (artifact.host_mod, artifact.device_mod):
            for lowered in module.functions.values():
                if lowered.attrs:
                    report = lowered.attrs.get("tilelang.tpu.pipeline_report")
                    if report is not None:
                        reports.extend(json.loads(str(report)))
                    addresses.update({str(key): int(value) for key, value in lowered.attrs.items()
                                      if str(key).startswith("tilelang.tpu.lmem.address.")})
        if depth and (not reports or "tpu_parallel_start();" not in artifact.kernel_source):
            raise AssertionError("pipeline annotation did not produce a schedule and parallel scope")
        kernel = tilelang.compile(function, out_idx=-1,
                                  target="tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
                                  runtime_mode="cmodel")
        output = torch.full_like(expected, float("nan"))
        kernel(*inputs, output)
        for original, actual in zip(originals, inputs):
            if not torch.equal(original, actual):
                raise AssertionError("kernel modified a read-only input")
        atol, rtol = tolerance("float16", family)
        variants[name] = {
            "reference": comparison(output, expected, atol=atol, rtol=rtol),
            "source_sha256": hashlib.sha256(artifact.kernel_source.encode()).hexdigest(),
            "schedules": reports, "local_addresses": addresses,
            "output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
        }
        outputs.append(output)
    equal = torch.equal(*outputs)
    if not equal:
        raise AssertionError("pipeline changed same-tile serial results")
    return {"status": "passed", "case": case, "parameters": parameters, "dtype": "float16",
            "seed": seed, "num_stages": stages, "serial_pipeline_bitwise_equal": equal,
            "variants": variants, "cmodel_parallel_execution": False,
            "hardware_overlap_verified": False, "device_performance_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("baseline", "pipeline"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", choices=("smoke", "target"), default="smoke")
    args = parser.parse_args()
    cpus = os.environ.get("BM1690_PIPELINE_CPUS")
    if not cpus:
        parser.error("use python -m tpu_demo.pipeline.run, which supervises worker resources")
    os.sched_setaffinity(0, {int(cpu) for cpu in cpus.split(",")})
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.suite == "baseline":
        from tpu_demo.run import run_case
        result = run_case(args.case, chip="bm1690", programming_model="tpukernel",
                          runtime_mode="cmodel", seed=args.seed)
    else:
        result = run_pipeline_case(args.case, args.seed, args.size)
    print("BM1690_PIPELINE_RESULT=" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
