"""Validate the board smoke's exact inputs, sources and host ABI on local CModel.

Internal worker for the existing bounded pipeline supervisor. No PCIe option.
"""

import hashlib
import argparse
import json
import os
import math
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
                                               BUNDLE_SHA256, TARGET_BUNDLE_SHA256, COARSE_BUNDLE_SHA256,
                                               ELEMENTWISE_BUNDLES, MATMUL_BUNDLE_SHA256,
                                               MATMUL_TARGET_BUNDLE_SHA256, RMSNORM_BUNDLES,
                                               RMSNORM_TARGET_BUNDLES, ROPE_BUNDLE_SHA256, ROPE_TARGET_BUNDLE_SHA256,
                                               ROPE_TRACE_BUNDLE_SHA256, ROPE_WINDOW_BUNDLE_SHA256, ROPE_WINDOW,
                                               ROPE_WINDOW_END_BUNDLE_SHA256, ROPE_WINDOW_END,
                                               ROPE_PREFIX_BUNDLES, ROPE_PREFIX_WINDOWS,
                                               SWIGLU_BUNDLE_SHA256, SWIGLU_TARGET_BUNDLE_SHA256,
                                               check_rope_window_output,
                                               rope_test_vectors, call_rope_host,
                                               FLASHATTN_BUNDLES, flashattn_test_vectors, call_flashattn_host)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-size", action="store_true")
    parser.add_argument("--coarse-tiles", action="store_true")
    parser.add_argument("--trace-host-runtime", action="store_true",
                        help="one original RoPE target call with flushed diagnostic host logs; no timing")
    windows = parser.add_mutually_exclusive_group()
    windows.add_argument("--rope-window8", action="store_true",
                        help="eight original tiles with target allocations/strides; diagnostic only")
    windows.add_argument("--rope-window8-end", action="store_true",
                         help="eight original tiles at the matrix end; diagnostic only")
    windows.add_argument("--rope-prefix-tiles", type=int, choices=tuple(ROPE_PREFIX_BUNDLES),
                         help="bounded original-order prefix; one diagnostic call only")
    parser.add_argument("--operation", choices=("add", "sub", "mul", "div", "matmul", "rmsnorm", "rmsnorm-splitk", "rope", "swiglu", "flashattn", "flashattn-causal"), default="add")
    parser.add_argument("--variant", choices=("original", "serial", "pipeline"),
                        help="required for target Matmul/RMSNorm/RoPE: one bounded worker per variant")
    args = parser.parse_args()
    if args.coarse_tiles and not args.target_size:
        parser.error("--coarse-tiles requires --target-size")
    if args.operation in ("sub", "mul", "div") and not (args.target_size and args.coarse_tiles):
        parser.error("Sub/Mul/Div require --target-size --coarse-tiles")
    matmul_target = args.operation == "matmul" and args.target_size
    rmsnorm_target = args.operation.startswith("rmsnorm") and args.target_size
    rope_target = args.operation == "rope" and args.target_size
    swiglu_target = args.operation == "swiglu" and args.target_size
    flash = args.operation.startswith("flashattn")
    target_worker = matmul_target or rmsnorm_target or rope_target or swiglu_target or (flash and args.target_size)
    if args.trace_host_runtime and not (rope_target and args.variant == "original"):
        parser.error("Host tracing is limited to the RoPE target original one-call check")
    if (args.rope_window8 or args.rope_window8_end or args.rope_prefix_tiles) and not args.trace_host_runtime:
        parser.error("RoPE window diagnostics require --trace-host-runtime")
    diagnostic_window = ROPE_WINDOW_END if args.rope_window8_end else ROPE_WINDOW if args.rope_window8 else None
    if args.rope_prefix_tiles:
        diagnostic_window = ROPE_PREFIX_WINDOWS[args.rope_prefix_tiles]
    if args.operation == "matmul" and args.coarse_tiles:
        parser.error("Matmul does not use elementwise --coarse-tiles")
    if args.operation.startswith("rmsnorm") and args.coarse_tiles:
        parser.error("RMSNorm does not use elementwise --coarse-tiles")
    if args.operation == "rope" and args.coarse_tiles:
        parser.error("RoPE does not use elementwise --coarse-tiles")
    if (args.operation == "swiglu" or flash) and args.coarse_tiles:
        parser.error("SwiGLU does not use elementwise --coarse-tiles")
    if bool(args.variant) != target_worker:
        parser.error("--variant is required only for target Matmul/RMSNorm/RoPE; compare worker outputs afterward")

    configure_runtime("cmodel", False, None, chip="bm1690")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    bundle_path = root / "research/bm1690-pipelines/handoff" / (
        "add-1024-coarse-sources.json" if args.coarse_tiles else
        "add-1024-sources.json" if args.target_size else "add-smoke-sources.json")
    bundle_hash = COARSE_BUNDLE_SHA256 if args.coarse_tiles else TARGET_BUNDLE_SHA256 if args.target_size else BUNDLE_SHA256
    if args.operation in ("sub", "mul", "div"):
        bundle_path = bundle_path.with_name(args.operation + "-1024-sources.json")
        bundle_hash = ELEMENTWISE_BUNDLES[args.operation]
    if args.operation == "matmul":
        bundle_path = bundle_path.with_name("matmul-1024-sources.json" if matmul_target else "matmul-smoke-sources.json")
        bundle_hash = MATMUL_TARGET_BUNDLE_SHA256 if matmul_target else MATMUL_BUNDLE_SHA256
    if args.operation.startswith("rmsnorm"):
        bundle_path = bundle_path.with_name(args.operation + ("-1024-sources.json" if rmsnorm_target else "-smoke-sources.json"))
        bundle_hash = (RMSNORM_TARGET_BUNDLES if rmsnorm_target else RMSNORM_BUNDLES)[args.operation]
    if args.operation == "swiglu":
        bundle_path = bundle_path.with_name("swiglu-1024-sources.json" if swiglu_target else "swiglu-smoke-sources.json")
        bundle_hash = SWIGLU_TARGET_BUNDLE_SHA256 if swiglu_target else SWIGLU_BUNDLE_SHA256
    if args.operation == "rope":
        bundle_path = bundle_path.with_name("rope-1024-sources.json" if rope_target else "rope-smoke-sources.json")
        bundle_hash = ROPE_TARGET_BUNDLE_SHA256 if rope_target else ROPE_BUNDLE_SHA256
        if args.trace_host_runtime:
            bundle_path = bundle_path.with_name("rope-1024-host-trace-sources.json")
            bundle_hash = ROPE_TRACE_BUNDLE_SHA256
        if args.rope_window8:
            bundle_path = bundle_path.with_name("rope-1024-window8-sources.json")
            bundle_hash = ROPE_WINDOW_BUNDLE_SHA256
        if args.rope_window8_end:
            bundle_path = bundle_path.with_name("rope-1024-window8-end-sources.json")
            bundle_hash = ROPE_WINDOW_END_BUNDLE_SHA256
        if args.rope_prefix_tiles:
            bundle_path = bundle_path.with_name(f"rope-1024-prefix{args.rope_prefix_tiles}-sources.json")
            bundle_hash = ROPE_PREFIX_BUNDLES[args.rope_prefix_tiles]
    if flash:
        bundle_path = bundle_path.with_name(args.operation + ("-1024-sources.json" if args.target_size else "-smoke-sources.json"))
        bundle_hash = FLASHATTN_BUNDLES[(args.operation, args.target_size)]
    check_hash(bundle_path, bundle_hash)
    bundle = json.loads(bundle_path.read_text())
    assert bundle["target"].get("operation", "add") == args.operation
    shape = bundle["target"]["shape"]
    rows, width = (shape[1], shape[3]) if flash else shape
    count = math.prod(shape)
    if flash:
        *inputs, expected = flashattn_test_vectors(shape=shape, is_causal=args.operation.endswith("-causal"))
        launch = call_flashattn_host
    elif args.operation == "rope":
        *inputs, expected = rope_test_vectors(shape=(rows, width))
        launch = call_rope_host
    else:
        *inputs, expected = test_vectors(count, args.operation)
        launch = call_host
    # Independent Torch check of the scalar/NumPy FP32 reference.
    if flash:
        from tpu_demo.flashattn.flashattn import _reference, _attention_mask
        q, k, v = [torch.frombuffer(bytearray(x), dtype=torch.float16).reshape(shape) for x in inputs[:3]]
        mask = torch.frombuffer(bytearray(inputs[3]), dtype=torch.float32).reshape(rows, rows)
        assert torch.equal(mask, _attention_mask(rows, args.operation.endswith("-causal")))
        independent = check_output(expected, _reference(q, k, v, mask, "float16").numpy().tobytes(), count, args.operation)
    elif args.operation == "rope":
        from tpu_demo.rope.rope import _reference, _cosine_sine
        source = torch.frombuffer(bytearray(inputs[0]), dtype=torch.float16).reshape(rows, width)
        cosine, sine = [torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(rows, width//2) for data in inputs[1:]]
        torch_reference = _reference(source, cosine, sine)
        independent = check_output(expected, torch_reference.numpy().tobytes(), count, "rope")
        demo_tables = _cosine_sine(rows, width)
        table_errors = [float((actual - demo).abs().max()) for actual,demo in zip((cosine,sine),demo_tables)]
        # CPU pow implementations may differ by an FP32 ulp before the
        # position multiply. Its absolute phase error grows with row index.
        table_atol = max(2e-6, 2 * rows * torch.finfo(torch.float32).eps)
        assert all(torch.allclose(actual, demo, atol=table_atol, rtol=2e-6) for actual,demo in zip((cosine,sine),demo_tables))
    else:
        a,b = [torch.frombuffer(bytearray(value),dtype=torch.float16) for value in inputs]
    if args.operation == "matmul":
        torch_reference = (a.float().reshape(rows,width) @ b.float().reshape(rows,width)).half()
        independent = check_output(expected, torch_reference.numpy().tobytes(), count, "matmul")
    elif args.operation.startswith("rmsnorm"):
        value, weight = a.float().reshape(rows, width), b.float().reshape(rows, width)
        normalized = value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + 1e-12)
        torch_reference = (normalized.half().float() * weight).half()
        independent = check_output(expected, torch_reference.numpy().tobytes(), count, args.operation)
    elif args.operation == "swiglu":
        torch_reference = (b.float() * torch.nn.functional.silu(a.float())).half()
        independent = check_output(expected, torch_reference.numpy().tobytes(), count, "swiglu")
    elif args.operation != "rope" and not flash:
        calculate = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[args.operation]
        assert calculate(a.float(), b.float()).half().numpy().tobytes() == expected
        independent = {"bitwise_equal_to_reference": True, "passed": True}
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    records, outputs, owners = {}, [], []
    for variant in ((args.variant,) if args.variant else ("original", "serial", "pipeline")):
        build = LibraryGenerator(target, tpu_target=TPUTargetSpec("bm1690", "tpukernel"),
                                 tpu_runtime=TPURuntimeConfig("cmodel"))
        for name, source in bundle["variants"][variant]["sources"].items():
            (Path(build.tpu_workspace_dir) / name).write_text(source)
        build.compile_lib(timeout=60)
        library = build.load_lib()
        owners.append((build,library))
        output = launch(library, *inputs)
        comparison = (check_rope_window_output(output, expected, diagnostic_window) if diagnostic_window else
                      check_output(output, expected, count, args.operation))
        records[variant] = {"reference":comparison,
                            "output_sha256":hashlib.sha256(output).hexdigest(),
                            "source_sha256":bundle["variants"][variant]["sha256"]}
        outputs.append(output)
        if not args.trace_host_runtime and (args.target_size or args.operation in ("matmul", "rmsnorm", "rmsnorm-splitk", "rope", "swiglu", "flashattn", "flashattn-causal")):
            timing = {"warmups": 1, "samples": 1 if target_worker else 2}
            repeated = launch(library, *inputs, timing=timing)
            assert repeated == output
            # Emulator wall times are never emitted as device latency evidence.
            records[variant]["timing_abi_check"] = {
                "warmups": timing["warmups"], "samples": timing["samples"], "finite_positive_samples": True,
                "reference": check_output(repeated, expected, count, args.operation), "same_output": True}
    if args.variant:
        Path("output.f16").write_bytes(outputs[0])
        original_comparison = None
    elif args.operation == "matmul" or flash:
        assert outputs[1] == outputs[2]
        original_comparison = check_output(outputs[1], outputs[0], count, args.operation)
    else:
        assert outputs[0] == outputs[1] == outputs[2]
        original_comparison = {"bitwise_equal_to_reference": True, "passed": True}
    report = {"status":"passed", "runtime_mode":"cmodel", "launch_cores":1,
              "diagnostic_only":args.trace_host_runtime,
              "diagnostic_window":list(diagnostic_window) if diagnostic_window else None,
              "operation":args.operation,
              "dtype":"float16", "shape":shape, "bundle_sha256":bundle_hash,
              "serial_and_pipeline_tiling":bundle["variants"]["serial"]["tiling"],
              "input_sha256":hashlib.sha256(b"".join(inputs)).hexdigest(), "variants":records,
              "reference_matches_torch":True, "independent_reference_comparison":independent,
              "original_serial_comparison":original_comparison,
              "all_variants_bitwise_equal":outputs[0] == outputs[1] == outputs[2] if not args.variant else None,
              "serial_pipeline_bitwise_equal":True if not args.variant else None,
              "device_performance_measured":False,"hardware_overlap_verified":False}
    if flash:
        report.update(selected_variant=args.variant, is_causal=args.operation.endswith("-causal"),
                      layout="BSHD", mask_dtype="float32", mask_shape=[rows, rows], output_index=4,
                      accumulation_dtype="float32", probability_dtype="float16")
    if matmul_target:
        import numpy as np
        report.update(selected_variant=args.variant, matrix_dimensions=[rows] * 3,
                      reference_backend={"name":"numpy.matmul", "version":np.__version__, "dtype":"float32"})
    if rmsnorm_target:
        report.update(selected_variant=args.variant, epsilon=1e-12, weight_shape=[rows,width],
                      normalization_rounding="fp16-before-weight-multiply",
                      reference_backend={"name":"stdlib scalar FP32 RMSNorm"})
    if args.operation == "swiglu":
        report.update(selected_variant=args.variant, intermediate_dtype="float32", output_index=2,
                      layout="row-major", formula="up * (gate * sigmoid(gate))", buffer_reuse=False,
                      reference_backend={"name":"stdlib scalar FP32 SwiGLU"})
    if args.operation == "rope":
        report.update(table_shape=[rows,width//2], table_dtype="float32", pairing="interleaved-even-odd",
                      output_index=3, table_max_abs_difference_from_demo=table_errors,
                      table_comparison_atol=table_atol, selected_variant=args.variant,
                      input_sha256_by_argument=[hashlib.sha256(data).hexdigest() for data in inputs],
                      reference_backend={"name":"stdlib scalar FP32 RoPE"})
    print("BM1690_PIPELINE_RESULT=" + json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
