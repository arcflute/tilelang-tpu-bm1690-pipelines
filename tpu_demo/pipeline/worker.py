"""Internal CModel worker. The parent runner supplies resource restrictions."""

from __future__ import annotations

import argparse
import json
import os


def pipeline_cases():
    return ["elementwise-add", "elementwise-sub", "elementwise-mul", "elementwise-div",
            "matmul", "rope", "swiglu", "rmsnorm", "rmsnorm-splitk"] + [
                "flashattn." + variant + (".causal" if causal else "")
                for causal in (False,True)
                for variant in ("balanced","descending-max","weighted-keys","multihead")]


def run_pipeline_case(case, seed, size, stages=2, schedule="auto", reuse_buffers=False, cores=1):
    """Compare identical arithmetic/tiling, using independent reference math."""
    from pathlib import Path
    import hashlib
    import torch
    import tilelang
    from tpu_demo.common import comparison, configure_runtime, tolerance
    from tpu_demo.pipeline.kernels import build_elementwise_tiled, pipeline_kernel_tiles

    configure_runtime("cmodel", False, None, chip="bm1690")
    generator = torch.Generator().manual_seed(seed)
    # Multicore smoke tiles must retain a steady state on every workitem.
    rows, width = (max(8,4*cores), 128) if size == "smoke" else (1024, 1024)
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
    elif case in ("rmsnorm", "rmsnorm-splitk"):
        from tpu_demo.rmsnorm.rmsnorm import build_rmsnorm, build_rmsnorm_splitk
        rows = max(16,4*cores*stages) if size == "smoke" else 1024
        source = torch.randn((rows,width),generator=generator).half()
        # Exercise the epsilon path, small magnitudes, and nonuniform weights.
        source[0] = 0
        source[1] *= 0.001
        weight = (torch.randn((rows,width),generator=generator)*0.25+1).half()
        inputs = [source,weight]
        normalized = source.float()*torch.rsqrt(source.float().square().mean(dim=1,keepdim=True)+1e-12)
        expected = (normalized.half().float()*weight.float()).half()
        family = "rmsnorm"
        parameters = {"rows":rows,"width":width,"block_rows":4 if size == "smoke" else 32,"epsilon":1e-12}
        if case == "rmsnorm-splitk":
            parameters["block_k"] = 32 if size == "smoke" else 128
        def build(depth):
            if case == "rmsnorm-splitk":
                return build_rmsnorm_splitk(**parameters,num_stages=depth)
            return pipeline_kernel_tiles(build_rmsnorm(**parameters),depth)
    elif case.startswith("flashattn."):
        from tpu_demo.flashattn.flashattn import build_flashattn, _reference, _attention_mask
        variant = case.split(".")[1]
        causal = case.endswith(".causal")
        batch = heads = 2 if variant == "multihead" else 1
        sequence, head_dim = (64,16) if size == "smoke" else (1024,64)
        shape = (batch,sequence,heads,head_dim)
        if variant == "descending-max":
            q = torch.full(shape,10.0,dtype=torch.float16)
            k = torch.full(shape,-10.0,dtype=torch.float16)
            k[:,:sequence//2] = 10
            v = (torch.randn(shape,generator=generator)*0.5).half()
        elif variant == "weighted-keys":
            query_scale = torch.linspace(0.75,1.5,sequence).reshape(1,sequence,1,1)
            key_scale = torch.linspace(-1,1,sequence).reshape(1,sequence,1,1)
            channel_scale = torch.linspace(-0.5,0.5,head_dim).reshape(1,1,1,head_dim)
            q = (0.5*query_scale).expand(shape).half().contiguous()
            k = (0.5*key_scale).expand(shape).half().contiguous()
            v = (0.5*key_scale+0.25*channel_scale).expand(shape).half().contiguous()
        else:
            q = (torch.randn(shape,generator=generator)*0.25).half()
            k = (torch.randn(shape,generator=generator)*0.25).half()
            v = torch.full(shape,0.25,dtype=torch.float16)
            v[:,sequence//2:] = 0.75
        mask = _attention_mask(sequence,causal)
        inputs = [q,k,v,mask]
        expected = _reference(*inputs,"float16")
        family = "flashattn"
        parameters = {"batch":batch,"heads":heads,"sequence":sequence,"head_dim":head_dim,
                      "block_m":16 if size == "smoke" else 32,
                      "block_n":16 if size == "smoke" else 32}
        def build(depth):
            return build_flashattn(**parameters,num_stages=depth)
    else:
        raise ValueError(f"unknown pipeline case {case}")
    originals = [tensor.clone() for tensor in inputs]
    variants = {}
    outputs = []
    candidates = [("serial", 0), ("pipeline", stages)]
    if cores > 1:
        candidates += [("multicore_serial",0),("multicore_pipeline",stages)]
    if reuse_buffers:
        if case != "swiglu":
            raise ValueError("buffer reuse is currently specific to SwiGLU")
        candidates += [("reuse_serial",0),("reuse_pipeline",stages)]
    for name, depth in candidates:
        function = build(depth)
        if name.startswith("reuse_"):
            from tpu_demo.pipeline.kernels import reuse_swiglu_buffers
            function = reuse_swiglu_buffers(function)
        launch_cores = cores if name.startswith("multicore_") else 1
        if launch_cores > 1:
            from tpu_demo.pipeline.workitems import map_workitems
            loop = "tile" if case.startswith("elementwise-") else "output_tile" if case in ("rope","swiglu","rmsnorm") else None
            function = map_workitems(function,launch_cores,tile_loop=loop)
        if depth and schedule != "auto":
            from tilelang.engine.tpu_pipeline import bind_explicit_schedule
            function = bind_explicit_schedule(
                function, tilelang.tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel"),
                reverse_loads=schedule == "reverse-loads")
        artifact = tilelang.lower(function, target="tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
                                  runtime_mode="cmodel")
        Path(name + ".c").write_text(artifact.kernel_source)
        Path(name + ".tir").write_text(artifact.host_mod.script() + "\n" + artifact.device_mod.script())
        reports = []
        addresses = {}
        local_bytes = {}
        high_water = 0
        for module in (artifact.host_mod, artifact.device_mod):
            for lowered in module.functions.values():
                if lowered.attrs:
                    report = lowered.attrs.get("tilelang.tpu.pipeline_report")
                    if report is not None:
                        reports.extend(json.loads(str(report)))
                    addresses.update({str(key): int(value) for key, value in lowered.attrs.items()
                                      if str(key).startswith("tilelang.tpu.lmem.address.")})
                    local_bytes.update({str(key): int(value) for key,value in lowered.attrs.items()
                                        if str(key).startswith("tilelang.tpu.lmem.bytes.")})
                    high_water = max(high_water,int(lowered.attrs.get("tilelang.tpu.lmem.high_water_bytes",0)))
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
            "local_bytes_per_lane": local_bytes, "lmem_high_water_bytes_per_lane": high_water,
            "launch_cores":launch_cores,
            "output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
        }
        outputs.append(output)
    equal = all(torch.equal(outputs[0],output) for output in outputs[1:])
    if not equal:
        raise AssertionError("pipeline changed same-tile serial results")
    return {"status": "passed", "case": case, "parameters": parameters, "dtype": "float16",
            "seed": seed, "num_stages": stages, "schedule": schedule, "serial_pipeline_bitwise_equal": equal,
            "reuse_swiglu_buffers": reuse_buffers,
            "launch_cores":cores,
            "variants": variants, "cmodel_parallel_execution": False,
            "hardware_overlap_verified": False, "device_performance_measured": False}


def run_workitem_probe(cores):
    from pathlib import Path
    import torch
    import tilelang
    from tpu_demo.common import configure_runtime
    from tpu_demo.pipeline.workitems import build_workitem_probe
    configure_runtime("cmodel",False,None,chip="bm1690")
    tasks = 11
    source = (torch.arange(tasks*32).reshape(tasks,32)%17+1).half()
    output = torch.full((cores,tasks,32),float("nan"),dtype=torch.float16)
    expected = torch.zeros_like(output)
    for task in range(tasks):
        expected[task%cores,task] = source[task]
    function = build_workitem_probe(cores,tasks)
    target = "tpu -mcpu=bm1690 -tpu-programming-model=tpukernel"
    artifact = tilelang.lower(function,target=target,runtime_mode="cmodel")
    Path("workitems.c").write_text(artifact.kernel_source)
    kernel = tilelang.compile(function,out_idx=-1,target=target,runtime_mode="cmodel")
    kernel(source,output)
    if not torch.equal(output,expected):
        raise AssertionError("workitem ABI/coverage probe failed")
    return {"status":"passed","launch_cores":cores,"tasks":tasks,"exact_ownership":True,
            "device_performance_measured":False,"hardware_overlap_verified":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("baseline", "pipeline", "workitems"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", choices=("smoke", "target"), default="smoke")
    parser.add_argument("--stages", type=int, choices=(2, 3), default=2)
    parser.add_argument("--schedule", choices=("auto", "explicit", "reverse-loads"), default="auto")
    parser.add_argument("--reuse-swiglu-buffers", action="store_true")
    parser.add_argument("--cores",type=int,choices=(1,2,4,8),default=1)
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
    elif args.suite == "workitems":
        result = run_workitem_probe(args.cores)
    else:
        result = run_pipeline_case(args.case, args.seed, args.size, args.stages, args.schedule,
                                   args.reuse_swiglu_buffers,args.cores)
    print("BM1690_PIPELINE_RESULT=" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
