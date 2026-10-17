"""Export and CModel-check the first two-workitem board ABI probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def export():
    import tilelang
    from tilelang import tvm
    from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper
    from tilelang.jit.adapter.legacy_pcie import SOURCE_NAMES
    from tpu_demo.pipeline.workitems import build_workitem_probe
    from tpu_demo.pipeline.run import source_identity
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    function = build_workitem_probe(2, tasks=11)
    artifact = tilelang.lower(function, target=target, runtime_mode="pcie")
    with tempfile.TemporaryDirectory(prefix="bm1690-probe-export-") as directory:
        TLTPUSourceWrapper(tvm.IRModule({"main": function}), artifact.kernel_source, target,
                           output_indices=[1], output_dir=directory)
        sources = {n: (Path(directory) / n).read_text() for n in SOURCE_NAMES}
    return dict(schema="bm1690-workitem-probe-v1", target=dict(chip="bm1690", programming_model="tpukernel",
                launch_cores=2, tasks=11, width=32, input_shape=[11,32], output_shape=[2,11,32], dtype="float16"),
                generator=source_identity(), sources=sources,
                sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
                evidence=dict(board_executed=False, hardware_overlap_verified=False))


def verify(bundle_path):
    import torch
    from tilelang import tvm
    from tilelang.engine.tpu_config import TPUTargetSpec, TPURuntimeConfig
    from tilelang.jit.adapter.libgen import LibraryGenerator
    from tpu_demo.common import configure_runtime
    from tpu_demo.pipeline.run_add_pcie import _call_host_buffers
    from tpu_demo.pipeline.workitem_probe_pcie import validate_bundle, vectors
    os.sched_setaffinity(0, {int(c) for c in os.environ["BM1690_PIPELINE_CPUS"].split(',')})
    torch.set_num_threads(1)
    configure_runtime("cmodel",False,None,chip="bm1690")
    bundle = json.loads(bundle_path.read_text())
    validate_bundle(bundle)
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    build = LibraryGenerator(target, tpu_target=TPUTargetSpec("bm1690","tpukernel"),
                             tpu_runtime=TPURuntimeConfig("cmodel"))
    for name, source in bundle["sources"].items():
        (Path(build.tpu_workspace_dir) / name).write_text(source)
    build.compile_lib(timeout=60)
    library = build.load_lib()
    source, expected = vectors()
    actual = _call_host_buffers(library, (source,), len(expected))
    if actual != expected:
        raise AssertionError("Two-workitem ownership mismatch")
    report = dict(status="passed", launch_cores=2, tasks=11, ownership_counts=[6,5],
                  exact_ownership=True, checked_elements=704,
                  bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(actual).hexdigest(),
                  input_sha256=hashlib.sha256(source).hexdigest(),
                  hardware_overlap_verified=False, device_performance_measured=False)
    print("BM1690_PIPELINE_RESULT="+json.dumps(report,sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify',type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.verify):
        parser.error('Select exactly one of --output or --verify')
    if args.verify:
        verify(args.verify)
    else:
        with args.output.open('x') as f:
            f.write(json.dumps(export(),indent=2,sort_keys=True)+'\n')
