"""Validate the board smoke's exact inputs, sources and host ABI on local CModel.

Internal worker for the existing bounded pipeline supervisor. No PCIe option.
"""

import hashlib
import argparse
import json
import os
from pathlib import Path


def main():
    cpus = os.environ.get("BM1690_PIPELINE_CPUS")
    if not cpus:
        raise RuntimeError("Use the bounded pipeline supervisor")
    os.sched_setaffinity(0, {int(cpu) for cpu in cpus.split(",")})
    import torch
    from tilelang import tvm
    from tilelang.engine.tpu_config import TPUTargetSpec, TPURuntimeConfig
    from tilelang.jit.adapter.libgen import LibraryGenerator
    from tpu_demo.common import configure_runtime
    from tpu_demo.pipeline.run_add_pcie import (test_vectors, check_output, call_host, check_hash,
                                               BUNDLE_SHA256, TARGET_BUNDLE_SHA256)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-size", action="store_true")
    args = parser.parse_args()

    configure_runtime("cmodel", False, None, chip="bm1690")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    bundle_path = root / "research/bm1690-pipelines/handoff" / (
        "add-1024-sources.json" if args.target_size else "add-smoke-sources.json")
    bundle_hash = TARGET_BUNDLE_SHA256 if args.target_size else BUNDLE_SHA256
    check_hash(bundle_path, bundle_hash)
    bundle = json.loads(bundle_path.read_text())
    rows, width = bundle["target"]["shape"]
    count = rows * width
    lhs, rhs, expected = test_vectors(count)
    # Independent check of the stdlib FP32-add/FP16-rounding reference.
    a,b = [torch.frombuffer(bytearray(value),dtype=torch.float16) for value in (lhs,rhs)]
    assert (a.float()+b.float()).half().numpy().tobytes() == expected
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    records, outputs, owners = {}, [], []
    for variant in ("original", "serial", "pipeline"):
        build = LibraryGenerator(target, tpu_target=TPUTargetSpec("bm1690", "tpukernel"),
                                 tpu_runtime=TPURuntimeConfig("cmodel"))
        for name, source in bundle["variants"][variant]["sources"].items():
            (Path(build.tpu_workspace_dir) / name).write_text(source)
        build.compile_lib(timeout=60)
        library = build.load_lib()
        owners.append((build,library))
        output = call_host(library, lhs, rhs)
        records[variant] = {"reference":check_output(output,expected,count),
                            "output_sha256":hashlib.sha256(output).hexdigest(),
                            "source_sha256":bundle["variants"][variant]["sha256"]}
        outputs.append(output)
        if args.target_size:
            timing = {"warmups": 1, "samples": 2}
            repeated = call_host(library, lhs, rhs, timing=timing)
            assert repeated == output
            # Emulator wall times are never emitted as device latency evidence.
            records[variant]["timing_abi_check"] = {
                "warmups": 1, "samples": 2, "finite_positive_samples": True,
                "reference": check_output(repeated, expected, count), "same_output": True}
    assert outputs[0] == outputs[1] == outputs[2]
    report = {"status":"passed", "runtime_mode":"cmodel", "launch_cores":1,
              "dtype":"float16", "shape":[rows,width], "bundle_sha256":bundle_hash,
              "input_sha256":hashlib.sha256(lhs+rhs).hexdigest(), "variants":records,
              "reference_matches_torch":True, "all_variants_bitwise_equal":True,
              "device_performance_measured":False,"hardware_overlap_verified":False}
    print("BM1690_PIPELINE_RESULT=" + json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
