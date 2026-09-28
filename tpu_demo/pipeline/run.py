"""Run one CModel worker at a time, with CPU, RSS and wall-time bounds.

This entry point never loads a PCIe runtime. Each worker inherits the existing
parent-death-aware TPU supervisor and writes logs directly to its own directory.
The parent stays outside the worker's CPU affinity and monitors the entire group.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "tilelang/jit/_tpu_profile_supervisor.py"
RESULT_PREFIX = "BM1690_PIPELINE_RESULT="


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()

    files = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    digests = {}
    for name in sorted(set(files)):
        path = ROOT / name
        if name.startswith(("tilelang/", "src/", "tpu_demo/", "testing/python/jit/")) and path.is_file():
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    libraries = {}
    for name in ("build-tpu/libtilelang_module.so", "build-tpu/tvm/libtvm.so"):
        path = ROOT / name
        if path.is_file():
            libraries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
        "source_files": digests,
        "native_libraries": libraries,
        "tvm_gitlink": git("ls-tree", "HEAD", "3rdparty/tvm"),
        "python": sys.executable,
        "ppl_root": os.environ.get("PPL_PROJECT_ROOT"),
    }


def group_rss_bytes(group):
    """Conservative sum of process RSS (shared pages can be counted twice)."""
    total = 0
    page = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            if int(fields[2]) == group:
                total += int(fields[21]) * page
        except (OSError, ValueError, IndexError):
            continue
    return total


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False
    # A dead worker can remain briefly as a zombie; it consumes no resources.
    return group_rss_bytes(process.pid) == 0


def worker_environment(case_dir, cpus):
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith(("TILELANG_TPU_PROFILE_", "TILELANG_TPU_ALLOW_PCIE")) or name in (
                "TILELANG_TPU_DEVICE_ID", "BMLIB_ENABLE_ALL_PROFILE", "TILELANG_TPU_BENCHMARK_RUNS"):
            environment.pop(name)
    environment.update({
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "TPU_RT_CORE_NUM": "8",  # BM1690 emulator topology, not launch parallelism.
        "BM1690_PIPELINE_CPUS": ",".join(map(str, cpus)),
        "TILELANG_CACHE_DIR": str(case_dir / "cache"),
        "TMPDIR": str(case_dir),
    })
    return environment


def run_worker(command, directory, environment, timeout_s, max_rss_bytes):
    started = time.monotonic()
    peak_rss = 0
    failure = None
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(SUPERVISOR), "--parent-pid", str(os.getpid()), "--", *command],
            cwd=directory, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, group_rss_bytes(process.pid))
                if peak_rss > max_rss_bytes:
                    failure = "rss_limit"
                    break
                if time.monotonic() - started > timeout_s:
                    failure = "timeout"
                    break
                time.sleep(0.1)
        except BaseException:
            stop_group(process)
            raise
        if failure or group_rss_bytes(process.pid):
            failure = failure or "live_descendant"
            cleanup = stop_group(process)
        else:
            cleanup = True
    numeric = None
    for line in (directory / "stdout.log").read_text(errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            numeric = json.loads(line[len(RESULT_PREFIX):])
    passed = not failure and process.returncode == 0 and numeric is not None and numeric.get("status") == "passed"
    return {
        "status": "passed" if passed else "failed", "reason": failure,
        "returncode": process.returncode, "cleanup_complete": cleanup,
        "wall_seconds": time.monotonic() - started, "peak_group_rss_bytes": peak_rss,
        "numeric": numeric, "artifact_dir": str(directory),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("baseline", "pipeline"), default="baseline")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    parser.add_argument("--cpu-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", choices=("smoke", "target"), default="smoke")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_rss_mib <= 0 or args.cpu_count <= 0:
        parser.error("resource limits must be positive and finite")
    if not os.environ.get("PPL_PROJECT_ROOT"):
        parser.error("activate the existing environment and set PPL_PROJECT_ROOT")
    allowed = sorted(os.sched_getaffinity(0))
    cpus = allowed[:min(args.cpu_count, max(1, len(allowed) - 1))]
    if args.suite == "baseline":
        from tpu_demo.cases import build_cases
        valid = [case.case_id for case in build_cases() if case.dtype == "float16"]
    else:
        from tpu_demo.pipeline.worker import pipeline_cases
        valid = pipeline_cases()
    cases = args.cases or valid
    if not cases or len(cases) != len(set(cases)) or any(case not in valid for case in cases):
        parser.error(f"case selection must be unique and drawn from {valid}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1, "suite": args.suite, "chip": "bm1690",
        "runtime_mode": "cmodel", "programming_model": "tpukernel", "launch_cores": 1,
        "device_performance_measured": False, "hardware_overlap_verified": False,
        "started_at": datetime.now(timezone.utc).isoformat(), "identity": source_identity(),
        "limits": {"cpus": cpus, "threads": 1, "timeout_s": args.timeout, "max_rss_mib": args.max_rss_mib},
        "requested_cases": cases, "results": [], "complete": False,
    }
    summary_path = output / "summary.json"
    def save():
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    save()
    try:
        for case in cases:
            directory = output / case
            directory.mkdir()
            command = [sys.executable, "-m", "tpu_demo.pipeline.worker", "--suite", args.suite,
                       "--case", case, "--seed", str(args.seed), "--size", args.size]
            result = run_worker(command, directory, worker_environment(directory, cpus),
                                args.timeout, args.max_rss_mib * 1024**2)
            result["case"] = case
            summary["results"].append(result)
            print(f"{case}: {result['status']} ({result['wall_seconds']:.2f}s)", flush=True)
            save()
            if result["status"] != "passed":
                return 1
        summary["complete"] = True
        return 0
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["passed"] = sum(result["status"] == "passed" for result in summary["results"])
        save()


if __name__ == "__main__":
    raise SystemExit(main())
