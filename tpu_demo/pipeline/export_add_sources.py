"""Export the first bounded BM1690 PCIe compilation check, without loading a runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def export(*, target_size=False, coarse_tiles=False, operation="add"):
    if operation not in ("add", "sub", "mul", "div"):
        raise ValueError("Unsupported elementwise operation")
    if operation != "add" and not (target_size and coarse_tiles):
        raise ValueError("Sub/Mul/Div handoffs require the 1024x1024, 128x1024-tile case")
    if coarse_tiles and not target_size:
        raise ValueError("The coarse-tile comparison requires the 1024x1024 target case")
    import tilelang
    from tilelang import tvm
    from tilelang.jit.adapter.legacy_pcie import SOURCE_NAMES, validate_source_bundle
    from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper
    from tpu_demo.elementwise.elementwise import build_elementwise
    from tpu_demo.pipeline.kernels import build_elementwise_tiled

    root = Path(__file__).resolve().parents[2]
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                    cwd=root, text=True).splitlines()
    identity = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in sorted(set(paths)) if (root / name).is_file() and
                name.startswith(("tilelang/", "src/", "tpu_demo/pipeline/", "tpu_demo/elementwise/"))}
    rows, width = (1024, 1024) if target_size else (8, 128)
    block_rows, block_width = (32, 128) if target_size else (4, 32)
    if coarse_tiles:
        block_rows, block_width = 128, 1024
    bundle = {
        "schema": "bm1690-add-source-check-v2" if target_size else "bm1690-add-source-check-v1",
        "target": {"chip": "bm1690", "programming_model": "tpukernel", "launch_cores": 1,
                   "dtype": "float16", "shape": [rows, width]},
        "generator_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         cwd=root, text=True).strip(),
        "generator_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "generator_source_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "generator_source_files": identity,
        "generator_native_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                    for name in ("build-tpu/libtilelang_module.so", "build-tpu/tvm/libtvm.so")
                                    if (root / name).is_file()},
        "variants": {},
        "evidence": {"board_compiled": False, "board_executed": False,
                     "hardware_overlap_verified": False},
    }
    if operation != "add":
        bundle["schema"] = "bm1690-elementwise-source-check-v1"
        bundle["target"]["operation"] = operation
    functions = {"original": build_elementwise(operation, rows=rows, width=width, dtype="float16")}
    for name, stages in (("serial", 0), ("pipeline", 2)):
        functions[name] = build_elementwise_tiled(operation, rows=rows, width=width, block_rows=block_rows,
                                                 block_width=block_width, num_stages=stages)
    for name, function in functions.items():
        artifact = tilelang.lower(function, target=target, runtime_mode="pcie")
        reports = []
        for mod in (artifact.host_mod, artifact.device_mod):
            for lowered in mod.functions.values():
                if lowered.attrs and lowered.attrs.get("tilelang.tpu.pipeline_report") is not None:
                    reports.extend(json.loads(str(lowered.attrs["tilelang.tpu.pipeline_report"])))
        if name == "pipeline" and (not reports or "tpu_parallel_start();" not in artifact.kernel_source):
            raise AssertionError("Missing generated pipeline schedule/synchronization")
        with tempfile.TemporaryDirectory(prefix="bm1690-export-") as directory:
            TLTPUSourceWrapper(tvm.IRModule({"main": function}), artifact.kernel_source, target,
                               output_indices=[2], output_dir=directory)
            sources = {filename: (Path(directory) / filename).read_text() for filename in SOURCE_NAMES}
        bundle["variants"][name] = {
            "num_stages": 2 if name == "pipeline" else 0,
            "tiling": None if name == "original" else [block_rows, block_width],
            "pipeline_reports": reports,
            "sources": sources,
            "sha256": {filename: hashlib.sha256(source.encode()).hexdigest()
                       for filename, source in sources.items()},
        }
    validate_source_bundle(bundle)
    return bundle


def verify_cmodel(bundle):
    """Compile these exact exported sources with the existing PPL 1.7 CModel.

    This is numerical evidence only: USING_CMODEL serializes parallel scopes.
    Call through the bounded pipeline supervisor; no board runtime is loaded.
    """
    import os
    import torch
    from tilelang import tvm
    from tilelang.engine.param import KernelParam
    from tilelang.engine.tpu_config import TPUTargetSpec, TPURuntimeConfig
    from tilelang.jit.adapter.libgen import LibraryGenerator
    from tilelang.jit.adapter.tpu import make_tpu_forward
    from tilelang.jit.adapter.legacy_pcie import validate_source_bundle
    from tpu_demo.common import comparison, configure_runtime, tolerance

    cpus = os.environ.get("BM1690_PIPELINE_CPUS")
    if not cpus:
        raise RuntimeError("Run CModel bundle verification through the bounded pipeline supervisor")
    os.sched_setaffinity(0, {int(cpu) for cpu in cpus.split(",")})
    configure_runtime("cmodel", False, None, chip="bm1690")
    if os.environ.get("TILELANG_TPU_PPL_PROFILE", "ppl17") != "ppl17":
        raise ValueError("Bundle CModel verification requires the existing PPL 1.7 profile")
    validate_source_bundle(bundle)
    torch.set_num_threads(1)
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    generator = torch.Generator().manual_seed(0)
    shape = bundle["target"]["shape"]
    lhs, rhs = [torch.randn(shape, generator=generator).half() for _ in range(2)]
    operation = bundle["target"].get("operation", "add")
    if operation == "div":
        rhs = (torch.rand(shape, generator=generator) * 1.5 + 0.5).half()
    calculate = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[operation]
    expected = calculate(lhs.float(), rhs.float()).half()
    outputs, records = [], {}
    for name in ("original", "serial", "pipeline"):
        build = LibraryGenerator(target, tpu_target=TPUTargetSpec("bm1690", "tpukernel"),
                                 tpu_runtime=TPURuntimeConfig("cmodel"))
        for filename, source in bundle["variants"][name]["sources"].items():
            (Path(build.tpu_workspace_dir) / filename).write_text(source)
        build.compile_lib(timeout=60)
        library = build.load_lib()
        forward = make_tpu_forward(library, [KernelParam(torch.float16, shape) for _ in range(3)], [2], {})
        output = torch.full_like(expected, float("nan"))
        forward(lhs, rhs, output)
        atol, rtol = tolerance("float16", "elementwise-div" if operation == "div" else "elementwise")
        records[name] = {"reference": comparison(output, expected, atol=atol, rtol=rtol),
                         "source_sha256": bundle["variants"][name]["sha256"],
                         "output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest()}
        outputs.append(output)
    if not all(torch.equal(outputs[0], output) for output in outputs[1:]):
        raise AssertionError("Original, same-tile serial and pipeline outputs disagree")
    return {"status": "passed", "runtime_mode": "cmodel", "shape": shape, "dtype": "float16",
            "launch_cores": 1, "seed": 0, "variants": records, "all_variants_bitwise_equal": True,
            "board_executed": False, "hardware_overlap_verified": False,
            "device_performance_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify-cmodel-bundle", type=Path)
    parser.add_argument("--target-size", action="store_true", help="export the 1024x1024 case")
    parser.add_argument("--coarse-tiles", action="store_true", help="128x1024 tiles, retaining the original baseline")
    parser.add_argument("--operation", choices=("add", "sub", "mul", "div"), default="add")
    args = parser.parse_args()
    if args.verify_cmodel_bundle:
        result = verify_cmodel(json.loads(args.verify_cmodel_bundle.read_text()))
        print("BM1690_PIPELINE_RESULT=" + json.dumps(result, sort_keys=True))
        return
    bundle = export(target_size=args.target_size, coarse_tiles=args.coarse_tiles, operation=args.operation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(bundle, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"EXPORTED={args.output} variants=3 runtime_loads=0")


if __name__ == "__main__":
    main()
