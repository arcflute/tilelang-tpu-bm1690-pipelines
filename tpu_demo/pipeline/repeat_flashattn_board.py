#!/usr/bin/env python3
"""Two bounded, reordered timing rounds using the passed d6bf332 target builds."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


RUNNER_SHA256 = "b46d84d8a9c8ba789de383991d8db542f1d1c7e2a3fc7b6d1f987bed78ab51a1"
BUNDLES = {
    "noncausal": "0a0bc116ad3ef055590892889f82f889f4254b6547d101015f6b9d76b3f680e7",
    "causal": "18d07660d190e997f823db7c879450c781ccd3c2e97aa0ec7171e3d6f560beff",
}
ORDERS = (("pipeline", "serial", "original"), ("serial", "original", "pipeline"))
IDENTITY_KEYS = ("build_manifest_sha256", "bundle_sha256", "input_sha256",
                 "output_sha256", "runtime_sha256", "runtime_library", "mapping")
DEFAULT_PYTHON = "/home/bokai/ChunkScan-bm1690-test-86add2f/.venv/bin/python"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_receipt(path, mode, variant, timed):
    report = json.loads(path.read_text())
    n = report.get("numeric") or {}
    required = dict(status="passed", operation="flashattn" + ("-causal" if mode == "causal" else ""),
                    variant=variant, shape=[1, 1024, 1, 64], dtype="float16", layout="BSHD",
                    accumulation_dtype="float32", probability_dtype="float16", mask_dtype="float32",
                    output_index=4, launch_cores=1, device_id=0, expected_pci="0000:01:00.0",
                    bundle_sha256=BUNDLES[mode])
    valid = (report.get("status") == "passed" and report.get("cleanup_complete") is True and
             all(n.get(k) == v for k, v in required.items()) and
             n.get("is_causal") is (mode == "causal") and
             n.get("reference", {}).get("passed") is True and
             not n.get("diagnostic_only") and not n.get("original_baseline_failure") and
             all(n.get(k) for k in IDENTITY_KEYS))
    mapping = n.get("mapping") or {}
    valid = valid and mapping.get("device_id") == 0 and mapping.get("pci_function") == "0000:01:00.0"
    if variant == "pipeline":
        valid = valid and n.get("previous_variant_bitwise_equal") is True
    elif variant == "serial":
        valid = valid and n.get("previous_variant_comparison", {}).get("passed") is True
    timing = n.get("timing") or {}
    if timed:
        raw = timing.get("raw_us") or []
        valid = (valid and timing.get("warmups") == 5 and timing.get("samples") == len(raw) == 20 and
                 timing.get("pure_device_time") is False and
                 all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in raw))
        if valid:
            values = sorted(raw)
            def percentile(q):
                i = (len(values) - 1) * q
                a = int(i)
                return values[a] + (values[min(a + 1, len(values) - 1)] - values[a]) * (i - a)
            expected = dict(median_us=statistics.median(raw), iqr_us=percentile(.75)-percentile(.25),
                            p95_us=percentile(.95), min_us=min(raw), max_us=max(raw))
            valid = all(math.isclose(timing.get(k, math.nan), v, rel_tol=1e-10, abs_tol=1e-8)
                        for k, v in expected.items())
    else:
        valid = valid and not timing and n.get("requested_kernel_calls") == 1
    if not valid:
        raise ValueError(f"Invalid or mismatched FlashAttention receipt: {path}")
    return n


def require_same_identity(current, baseline):
    if any(current[k] != baseline[k] for k in IDENTITY_KEYS):
        raise ValueError("Sources, runtime, hardware, inputs or outputs changed since correctness")


def preflight(source):
    receipts = {}
    for mode in BUNDLES:
        work = source / ("target-" + mode)
        if digest(work / "run_add.py") != RUNNER_SHA256:
            raise ValueError(f"Runner checksum mismatch: {work}")
        for variant in ("original", "serial", "pipeline"):
            correctness = read_receipt(work / f"run-{variant}/result.json", mode, variant, False)
            timing = read_receipt(work / f"timing-{variant}/result.json", mode, variant, True)
            require_same_identity(timing, correctness)
            if (digest(work / "build/result.json") != correctness["build_manifest_sha256"] or
                    digest(work / f"run-{variant}/output.f16") != correctness["output_sha256"]):
                raise ValueError("Saved build or correctness output changed")
            receipts[mode, variant] = correctness
    return receipts


def command(args, log, env):
    with log.open("x") as stream:
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return result.returncode


def execute(source, output, python):
    source = source.resolve()
    baselines = preflight(source)  # No runtime load or kernel execution here.
    if not os.access(python, os.X_OK):
        raise ValueError(f"Reference Python unavailable: {python}")
    output.mkdir()  # A previously used result directory is never overwritten.
    output = output.resolve()
    env = dict(os.environ)
    env.update({k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")})
    summary = dict(schema="bm1690-flashattn-repeat-v1", status="running", source=str(source),
                   started_at=datetime.datetime.now().astimezone().isoformat(),
                   uname=list(os.uname()), runner_sha256=RUNNER_SHA256,
                   order_by_round={"2": ORDERS[0], "3": ORDERS[1]},
                   timing_boundary="launch plus stream synchronization; resident inputs and module",
                   hardware_overlap_verified=False, pure_device_time=False, rounds=[])
    def save():
        temp = output / "summary.json.tmp"
        temp.write_text(json.dumps(summary, indent=2) + "\n")
        temp.replace(output / "summary.json")
    save()
    try:
        rc = command(["timeout", "-k", "2s", "10s", "taskset", "-c", "0,1", "nice", "-n", "10",
                      str(python), "-c", 'import sys,numpy; print(sys.executable, numpy.__version__)'],
                     output / "python.log", env)
        if rc:
            raise RuntimeError(f"Reference Python check failed ({rc}); see python.log")
        for number, order in enumerate(ORDERS, 2):
            for mode in BUNDLES:
                work = source / ("target-" + mode)
                results = {}
                for variant in order:
                    label = f"round{number}-{mode}-{variant}"
                    summary["current_stage"] = label
                    save()
                    users = output / (label + "-users.log")
                    rc = command(["sudo", "-n", "timeout", "-k", "2s", "5s", "fuser", "-v",
                                  "/dev/sg-host-drv-0", "/dev/sg-host-drv-1"], users, env)
                    if rc != 1 or users.stat().st_size:
                        raise RuntimeError(f"Device occupied or occupation check failed; see {users}")
                    print("FLASHATTN_REPEAT_BEGIN " + label, flush=True)
                    args = ["timeout", "-k", "3s", "40s", str(python), str(work / "run_add.py"),
                            "--build", str(work / "build"), "--output", str(output / label),
                            "--variant", variant, "--measure", "--correctness", str(work / ("run-" + variant)),
                            "--allow-pcie", "--device-id", "0", "--expected-pci", "0000:01:00.0",
                            "--timeout-seconds", "30"]
                    if variant != "original":
                        previous = "original" if variant == "serial" else "serial"
                        args.extend(["--previous", str(work / ("run-" + previous))])
                    log = output / (label + ".log")
                    rc = command(args, log, env)
                    if rc:
                        print(log.read_text()[-8000:], flush=True)
                        raise RuntimeError(f"{label} failed ({rc}); no further dispatch")
                    n = read_receipt(output / label / "result.json", mode, variant, True)
                    require_same_identity(n, baselines[mode, variant])
                    results[variant] = n["timing"]
                    print("FLASHATTN_REPEAT_RESULT=" + json.dumps(dict(
                        round=number, mode=mode, variant=variant, timing=n["timing"],
                        reference=n["reference"], input_sha256=n["input_sha256"],
                        output_sha256=n["output_sha256"],
                        previous_variant_bitwise_equal=n.get("previous_variant_bitwise_equal"))), flush=True)
                change = 100 * (1 - results["pipeline"]["median_us"] / results["serial"]["median_us"])
                row = dict(round=number, mode=mode, order=order, variants=results,
                           pipeline_vs_serial_reduction_percent=change)
                summary["rounds"].append(row)
                save()
                print(f"FLASHATTN_REPEAT_COMPARISON round={number} mode={mode} reduction_percent={change:.6f}", flush=True)
        summary["status"] = "passed"
        summary["completed_workers"] = 12
        summary["requested_kernel_calls"] = 300
        summary["finished_at"] = datetime.datetime.now().astimezone().isoformat()
        save()
        print(f"FLASHATTN_REPEAT_PASSED SUMMARY={output / 'summary.json'}", flush=True)
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
        raise SystemExit(f"FLASHATTN_REPEAT_STOP {error}") from error


if __name__ == "__main__":
    main()
