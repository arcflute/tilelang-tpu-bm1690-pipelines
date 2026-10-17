#!/usr/bin/env python3
"""Two reordered timing rounds, reusing the passed one/two-core attention builds."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

# Imports of the verified, existing helpers must not create files in the old run.
sys.dont_write_bytecode = True


RUNNER_SHA256 = "c4b49b5b492f01c69772cf9f0fcc996fc22232cbcc2df0e5f9521438113a2ed9"
OPERATIONS = ("flashattn", "flashattn-causal")
ORDERS = (("pipeline_2", "serial_2", "pipeline_1", "serial_1"),
          ("serial_2", "pipeline_2", "serial_1", "pipeline_1"))
MODE_ORDERS = (tuple(reversed(OPERATIONS)), OPERATIONS)
PAIRS = (("pipeline_at_1_core", "serial_1", "pipeline_1"),
         ("pipeline_at_2_cores", "serial_2", "pipeline_2"),
         ("serial_1_to_2_cores", "serial_1", "serial_2"),
         ("pipeline_1_to_2_cores", "pipeline_1", "pipeline_2"))
DEFAULT_PYTHON = "/home/bokai/ChunkScan-bm1690-test-86add2f/.venv/bin/python"
IDENTITY_KEYS = ("operation", "variant", "shape", "dtype", "layout", "launch_cores",
                 "num_stages", "tiling", "is_causal", "mask_dtype", "mask_shape",
                 "accumulation_dtype", "probability_dtype", "output_index", "reference_contract",
                 "build_manifest_sha256", "bundle_sha256", "artifacts", "input_sha256",
                 "runtime_sha256", "runtime_library", "mapping")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_driver(source):
    path = source / "flashattn.py"
    if digest(path) != RUNNER_SHA256:
        raise ValueError("Passed FlashAttention driver checksum mismatch")
    spec = importlib.util.spec_from_file_location("verified_attention_repeat_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_receipt(directory, identity, support, *, timed):
    receipt = json.loads((directory / "result.json").read_text())
    n = receipt.get("numeric") or {}
    ref = n.get("reference") or {}
    required = dict(status="passed", measurement="timing" if timed else "correctness",
                    requested_kernel_calls=25 if timed else 1,
                    hardware_overlap_verified=False, device_performance_measured=False)
    if (receipt.get("status") != "passed" or receipt.get("cleanup_complete") is not True or
            any(n.get(k) != v for k, v in required.items()) or
            any(n.get(k) != identity[k] for k in IDENTITY_KEYS) or
            ref.get("passed") is not True or ref.get("finite") is not True or
            ref.get("mismatched_elements") != 0 or
            ref.get("atol") != 0.02 or ref.get("rtol") != 0.02):
        raise ValueError("Invalid or mismatched attention receipt: " + str(directory))
    mapping = n["mapping"]
    if (mapping.get("device_id"), mapping.get("pci_function"), mapping.get("character_device")) != (
            0, "0000:01:00.0", "/dev/sg-host-drv-0"):
        raise ValueError("Wrong physical device mapping")
    output = (directory / "output.f16").read_bytes()
    if len(output) != 1024 * 64 * 2 or hashlib.sha256(output).hexdigest() != n.get("output_sha256"):
        raise ValueError("Saved attention output changed or has wrong size")
    if identity.get("output_sha256") and n["output_sha256"] != identity["output_sha256"]:
        raise ValueError("Repeated output differs from saved correctness")
    if timed:
        timing = n.get("sync_latency") or {}
        raw = timing.get("raw_us") or []
        if timing.get("warmups") != 5 or timing.get("samples") != 20 or len(raw) != 20:
            raise ValueError("Incomplete timing samples")
        if any(timing.get(k) != v for k, v in support.summarize_samples(raw).items()):
            raise ValueError("Timing statistics or boundary do not match raw samples")
    elif n.get("sync_latency"):
        raise ValueError("Expected a one-call correctness receipt")
    return n


def preflight(source):
    """Read and verify all ten builds and twenty receipts; never write old results."""
    module = load_driver(source)
    baselines = {}
    for operation in OPERATIONS:
        work = source / operation
        bundle, builder, support = module.assets(work)
        if bundle["operation"] != operation:
            raise ValueError("Causal mode does not match source directory")
        for variant in module.VARIANTS:
            build, runtime = module.validate_build(work, bundle, builder, variant)
            # Device count was recorded by the original worker, not inferred from core count.
            correct_dir = work / ("run-" + variant)
            prior = json.loads((correct_dir / "result.json").read_text()).get("numeric") or {}
            identity = dict(operation=operation, variant=variant, **module.contract(variant, operation),
                            shape=module.SHAPE, dtype="float16", bundle_sha256=module.BUNDLE_SHAS[operation],
                            build_manifest_sha256=module.digest(work / "build" / variant / "result.json"),
                            artifacts=build["artifacts"], input_sha256=module.INPUT_SHAS[operation],
                            runtime_sha256=module.digest(runtime), runtime_library=str(runtime),
                            mapping=prior.get("mapping") or {})
            correct = read_receipt(correct_dir, identity, support, timed=False)
            timing_dir = work / ("timing-" + variant)
            timing = read_receipt(timing_dir, correct, support, timed=True)
            baselines[operation, variant] = dict(correctness=correct, timing=timing,
                correctness_receipt_sha256=digest(correct_dir / "result.json"),
                timing_receipt_sha256=digest(timing_dir / "result.json"))
        serial_hash = baselines[operation, "serial_1"]["correctness"]["output_sha256"]
        if any(baselines[operation, v]["correctness"]["output_sha256"] != serial_hash
               for v in ("pipeline_1", "serial_2", "pipeline_2")):
            raise ValueError("Passed matched variants disagree")
    return module, baselines


def command(args, log, env):
    with log.open("x") as stream:
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return result.returncode


def comparisons(rows):
    result = {}
    for label, before, after in PAIRS:
        a, b = (rows[v]["sync_latency"]["median_us"] for v in (before, after))
        result[label] = dict(before=before, after=after, median_ratio=a / b,
                             median_reduction_percent=100 * (a - b) / a)
    return result


def execute(source, output, python):
    source = source.resolve(strict=True)
    module, baselines = preflight(source)
    if not os.access(python, os.X_OK):
        raise ValueError("Existing NumPy Python is unavailable: " + str(python))
    output.mkdir()  # Preserve both existing results and interrupted repeat directories.
    output = output.resolve()
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update({k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                               "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    summary = dict(schema="bm1690-flashattn-multicore-repeat-v1", status="running",
        source=str(source), started_at=datetime.datetime.now().astimezone().isoformat(),
        uname=list(os.uname()), runner_sha256=RUNNER_SHA256,
        original_relaunched=False, compilation_performed=False,
        original_baselines={op:baselines[op, "original"] for op in OPERATIONS},
        first_round={op:{v:baselines[op, v] for v in ORDERS[0]} for op in OPERATIONS},
        order_by_round={str(i):dict(modes=modes, variants=order)
                        for i, (modes, order) in enumerate(zip(MODE_ORDERS, ORDERS), 2)},
        hardware_overlap_verified=False, pure_device_time=False, rounds=[], completed_workers=0)
    def save():
        path = output / "summary.json.tmp"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        path.replace(output / "summary.json")
    save()
    try:
        rc = command(["timeout", "-k", "2s", "10s", "taskset", "-c", "0,1", "nice", "-n", "10",
                      str(python), "-c", 'import sys,numpy; print(sys.executable,numpy.__version__)'],
                     output / "python.log", env)
        if rc:
            raise RuntimeError("NumPy environment check failed; see python.log")
        for number, (modes, order) in enumerate(zip(MODE_ORDERS, ORDERS), 2):
            for operation in modes:
                work = source / operation
                _, _, support = module.assets(work)
                row = dict(round=number, operation=operation, order=order, variants={})
                summary["rounds"].append(row)
                for variant in order:
                    label = f"round{number}-{operation}-{variant}"
                    summary["current_stage"] = label
                    save()
                    if digest(source / "flashattn.py") != RUNNER_SHA256:
                        raise ValueError("Passed driver changed before dispatch")
                    baseline = baselines[operation, variant]
                    if digest(work / ("run-" + variant) / "result.json") != baseline["correctness_receipt_sha256"]:
                        raise ValueError("Saved correctness receipt changed before dispatch")
                    print("FLASHATTN_MULTICORE_REPEAT_BEGIN " + label, flush=True)
                    # The pinned driver checks both device holders and holds the device lock;
                    # each worker retains two CPUs, nice 10, 4 GiB RSS and 30 seconds.
                    args = ["timeout", "-k", "3s", "40s", str(python), str(source / "flashattn.py"),
                            "--run-only", "--allow-pcie", "--work", str(work), "--variant", variant,
                            "--output", str(output / label), "--correctness", str(work / ("run-" + variant))]
                    log = output / (label + ".log")
                    rc = command(args, log, env)
                    if rc:
                        print(log.read_text()[-8000:], flush=True)
                        raise RuntimeError(f"{label} failed ({rc}); no further dispatch")
                    n = read_receipt(output / label, baseline["correctness"], support, timed=True)
                    row["variants"][variant] = n
                    summary["completed_workers"] += 1
                    save()
                    print("FLASHATTN_MULTICORE_REPEAT_RESULT=" + json.dumps(
                        dict(round=number, **n), sort_keys=True), flush=True)
                row["comparisons"] = comparisons(row["variants"])
                save()
                print("FLASHATTN_MULTICORE_REPEAT_COMPARISON=" + json.dumps(dict(
                    round=number, operation=operation, comparisons=row["comparisons"]), sort_keys=True), flush=True)
        summary.update(status="passed", requested_kernel_calls=400, raw_samples=320,
                       finished_at=datetime.datetime.now().astimezone().isoformat())
        save()
        print("FLASHATTN_MULTICORE_REPEAT_PASSED SUMMARY=" + str(output / "summary.json"), flush=True)
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pcie", action="store_true", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(DEFAULT_PYTHON))
    args = parser.parse_args()
    try:
        execute(args.source, args.output, args.python)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        raise SystemExit("FLASHATTN_MULTICORE_REPEAT_STOP " + str(error)) from error


if __name__ == "__main__":
    main()
