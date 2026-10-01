"""Supervised BM1690 elementwise correctness and resident synchronous-call latency.

Standalone Python standard library only. This is a narrowly scoped source/build
manifest loader, not a relaxation of the generic JIT's prebuilt-library policy.
Only pinned bundles/builders are accepted. No automatic retry or multicore.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import statistics
import struct
import subprocess
import sys
import time

BUNDLE_SHA256 = "32de4a5ca1a06804534391e00c4075f2d7373ff52ca1b2d1d0b1ac7421e32bac"
BUILDER_SHA256 = "51c39e19dfd0d968ccbe17456048236645d3ff279eaec3b996baf000028c0f70"
VARIANTS = ("original", "serial", "pipeline")
COUNT = 8 * 128
# Retain the original handoff, including builds made by the frozen P8.1 helper.
PREVIOUS_BUILDER_SHA256 = "b701ff09991eb77c202fd4108e784521efa2c427f0e677025c7045f6cdbc2525"
CURRENT_BUILDER_SHA256 = "8b91f40d4f983ce6d91dd0f26b5ae04cea6e01efd6fa0b89a1c1d322c002cfce"
TARGET_BUNDLE_SHA256 = "2f43fcb39be86a3a4fc7407e5a10714bccbbb93ae7f7211dfc38bafa938155c5"
COARSE_BUNDLE_SHA256 = "2f6cfb1bf2af284b6abe576e966f2bc8f14352baac1e0f5c408cedf17e024ae2"
BUNDLES = {
    BUNDLE_SHA256: {"shape": [8, 128], "builders": (BUILDER_SHA256, PREVIOUS_BUILDER_SHA256, CURRENT_BUILDER_SHA256),
                    "timing": False},
    TARGET_BUNDLE_SHA256: {"shape": [1024, 1024], "builders": (PREVIOUS_BUILDER_SHA256, CURRENT_BUILDER_SHA256),
                           "timing": True},
    COARSE_BUNDLE_SHA256: {"shape": [1024, 1024], "builders": (PREVIOUS_BUILDER_SHA256, CURRENT_BUILDER_SHA256),
                           "timing": True},
}

ELEMENTWISE_BUNDLES = {
    "sub": "6d72d38f8a83b510dc71fab7905e8babf0b7b645cdd2ffa1ecc7a78180430418",
    "mul": "704340e5fecb142001b3b362cb50516cd818b848b4987dfc39dd36b5c0a0c1aa",
    "div": "97238d2cb47b8921a14d1d82b0f63ca0eee6efdc98b0ed3d9baf2306689aab17"
}
for _operation, _bundle in ELEMENTWISE_BUNDLES.items():
    BUNDLES[_bundle] = {"shape": [1024, 1024], "builders": (CURRENT_BUILDER_SHA256,),
                       "timing": True, "operation": _operation}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise ValueError(f"Hash mismatch: {path}")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_build(build):
    """Bind the original source bundle, build recipe, SDK and both ELF files."""
    build = build.resolve(strict=True)
    bundle_path, builder_path = build.parent / "add.json", build.parent / "build.py"
    bundle_hash, builder_hash = digest(bundle_path), digest(builder_path)
    contract = BUNDLES.get(bundle_hash)
    if contract is None or builder_hash not in contract["builders"]:
        raise ValueError("Hash mismatch: unregistered source bundle or builder")
    spec = importlib.util.spec_from_file_location("bm1690_verified_legacy_builder", builder_path)
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    bundle = json.loads(bundle_path.read_text())
    builder.validate_source_bundle(bundle)
    report = json.loads((build / "result.json").read_text())
    if report.get("status") != "compile_only_passed" or report.get("profile") != builder.PROFILE:
        raise ValueError("A successful P8.1 legacy compilation manifest is required")
    if report.get("bundle_sha256") != bundle_hash or report.get("builder_sha256") != builder_hash:
        raise ValueError("Build manifest does not identify the pinned P8.1 sources/builder")
    if report.get("board_runtime_loaded") is not False or report.get("kernel_launches") != 0:
        raise ValueError("Expected the source-only compilation manifest")
    sdk_root, board_lib, backend = map(Path, report["runtime_identity"])
    cross_candidates = [Path(p) for p in report["sdk_inputs"]
                        if Path(p).name == "riscv64-unknown-linux-gnu-gcc"]
    if len(cross_candidates) != 1:
        raise ValueError("Missing or ambiguous build compiler identity")
    environment = {"TILELANG_TPU_PCIE_RUNTIME_PATH": str(board_lib),
                   "TILELANG_TPU_PCIE_CROSS_GCC": str(cross_candidates[0])}
    layout = builder.resolve_legacy_pcie(sdk_root, "bm1690", environment=environment)
    if list(layout.runtime_identity_for("pcie", environment)) != report["runtime_identity"]:
        raise ValueError("Build/runtime paths changed")
    required = (layout.firmware_archive, layout.ppl_helper_source,
                layout.kernel_include / "tpu_kernel.h", layout.runtime_include / "tpuv7_rt.h",
                layout.board_lib / "libtpuv7_rt.so", layout.cross_gcc)
    if set(report["sdk_inputs"]) != {str(p) for p in required}:
        raise ValueError("Incomplete SDK build identity")
    for path in required:
        check_hash(path, report["sdk_inputs"][str(path)])
    # Reconstruct the exact commands, including the original absolute embedded
    # libkernel.so path. A copied/moved build is not accepted as interchangeable.
    old_environment = {key: os.environ.get(key) for key in environment}
    try:
        os.environ.update(environment)
        commands = []
        for variant in VARIANTS:
            directory = build / variant
            for name, expected in bundle["variants"][variant]["sha256"].items():
                check_hash(directory / name, expected)
            for filename, machine in (("main.so", 62), ("libkernel.so", 243)):
                artifact = directory / filename
                check_hash(artifact, report["artifacts"][variant][filename])
                with artifact.open("rb") as source:
                    header = source.read(20)
                # ELF64, little endian, ET_DYN; EM_X86_64 / EM_RISCV.
                if header[:6] != b"\x7fELF\x02\x01" or struct.unpack("<HH", header[16:20]) != (3, machine):
                    raise ValueError(f"Unexpected ELF target: {artifact}")
            for number, (label, command) in enumerate(builder.legacy_pcie_commands(layout, directory)):
                commands.append({"variant": variant, "label": label, "command": command,
                                 "log": str(directory / f"{number}.log")})
        if report.get("completed_commands") != commands or "active_command" in report:
            raise ValueError("Build commands differ from the pinned recipe or original location")
    finally:
        for key, old in old_environment.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    operation = contract.get("operation", "add")
    if bundle["target"].get("operation", "add") != operation or bundle["target"]["shape"] != contract["shape"]:
        raise ValueError("Pinned operation/shape differs from bundle")
    return {"build_manifest_sha256": digest(build / "result.json"), "operation": operation,
            "bundle_sha256": bundle_hash, "shape": contract["shape"], "timing_supported": contract["timing"],
            "runtime_library": str(required[-2].resolve()),
            "runtime_sha256": report["sdk_inputs"][str(required[-2])],
            "artifacts": report["artifacts"]}


def test_vectors(count=COUNT, operation="add"):
    """FP16 inputs and independently FP32-rounded arithmetic, then FP16 output.

    Division retains the demo's positive [0.5, 2] denominator domain. Add's
    historical stream and hashes are preserved exactly.
    """
    if operation not in ("add", "sub", "mul", "div"):
        raise ValueError("Unsupported elementwise reference operation")
    state = 0
    values = []
    for _ in range(count * 2):
        state = (1664525 * state + 1013904223) & 0xffffffff
        values.append(((state >> 8) % 16384 - 8192) / 4096.0)
    lhs = struct.pack("<" + "e" * count, *values[:count])
    rhs_values = [(value + 2) * (1.5 / 4) + 0.5 for value in values[count:]] if operation == "div" else values[count:]
    rhs = struct.pack("<" + "e" * count, *rhs_values)
    calculate = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b,
                 "mul": lambda a,b: a*b, "div": lambda a,b: a/b}[operation]
    reference = b"".join(struct.pack("<e", struct.unpack("<f", struct.pack("<f", calculate(a[0], b[0])))[0])
                         for a, b in zip(struct.iter_unpack("<e", lhs), struct.iter_unpack("<e", rhs)))
    return lhs, rhs, reference


def check_output(actual, expected, count=COUNT, operation="add"):
    if operation not in ("add", "sub", "mul", "div"):
        raise ValueError("Unsupported elementwise reference operation")
    if len(actual) != count * 2 or len(expected) != count * 2:
        raise ValueError("Unexpected elementwise output length")
    a = [v[0] for v in struct.iter_unpack("<e", actual)]
    b = [v[0] for v in struct.iter_unpack("<e", expected)]
    finite = all(math.isfinite(v) for v in a + b)
    errors = [abs(x-y) for x,y in zip(a,b)]
    atol = rtol = 0.01 if operation == "div" else 0.005
    mismatches = sum(not math.isfinite(x) or abs(x-y) > atol + rtol*abs(y) for x,y in zip(a,b))
    metrics = {"passed": finite and mismatches == 0, "finite": finite,
               "atol": atol, "rtol": rtol, "mismatched_elements": mismatches,
               "max_abs_error": max(errors) if finite else None,
               "bitwise_equal_to_reference": actual == expected}
    if not metrics["passed"]:
        raise ValueError("Elementwise numerical mismatch: " + json.dumps(metrics))
    return metrics


def call_host(library, lhs, rhs, *, timing=None):
    """One ABI call; optional explicit warmup/sample counts and returned samples."""
    if len(lhs) != len(rhs) or len(lhs) % 2:
        raise ValueError("Invalid FP16 input lengths")
    canary = bytes([0xa5]) * 64
    payloads = [lhs, rhs, b"\x00\x7e" * (len(lhs) // 2)]
    buffers = [ctypes.create_string_buffer(canary + value + canary, len(value) + 128) for value in payloads]
    pointers = (ctypes.c_void_p * 3)(*[ctypes.addressof(buffer) + 64 for buffer in buffers])
    run = library.tilelang_tpu_run if timing is None else library.tilelang_tpu_run_timed
    run.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    run.restype = ctypes.c_int
    if timing is None:
        status = run(pointers)
    else:
        warmups, count = timing["warmups"], timing["samples"]
        if not 1 <= warmups <= 100 or not 1 <= count <= 1000:
            raise ValueError("Invalid bounded timing counts")
        samples = (ctypes.c_double * count)(*[float("nan")] * count)
        run.argtypes += [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        status = run(pointers, warmups, count, samples)
    if status != 0:
        raise RuntimeError(f"tilelang_tpu_run returned {status}")
    for buffer in buffers:
        if buffer.raw[:64] != canary or buffer.raw[-64:] != canary:
            raise RuntimeError("Host staging buffer canary changed")
    for buffer, original in zip(buffers[:2], payloads[:2]):
        if buffer.raw[64:-64] != original:
            raise RuntimeError("Host input staging buffer changed")
    if timing is not None:
        timing.update(summarize_samples(list(samples)))
    return buffers[2].raw[64:-64]


def summarize_samples(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Missing or invalid synchronous-call timing samples")
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered) - 1) * q
        low = int(index)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)
    return {"raw_us": values, "median_us": statistics.median(values),
            "iqr_us": percentile(.75) - percentile(.25), "p95_us": percentile(.95),
            "min_us": min(values), "max_us": max(values), "percentiles": "linear interpolation",
            "boundary": "steady_clock around main_kernel: launch plus stream synchronization",
            "excluded": ["compilation", "allocation", "module loading", "H2D", "D2H", "reference"],
            "pure_device_time": False}


def loaded_vendor_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and any(name in parts[5] for name in (
                "libtpuv7_rt.so", "libtpuv7_emulator", "libcdm_daemon_emulator")):
            paths.add(str(Path(parts[5]).resolve()))
    return paths


def verify_device(runtime, device_id, expected_pci, checkpoint):
    signatures = {"tpuRtInit": [], "tpuRtGetDeviceCount": [ctypes.POINTER(ctypes.c_int)],
                  "tpuRtSetDevice": [ctypes.c_int], "tpuRtGetDevice": [ctypes.POINTER(ctypes.c_int)],
                  "tpuRtGetFd": [ctypes.POINTER(ctypes.c_int)]}
    apis = {}
    for name, arguments in signatures.items():
        fn = getattr(runtime, name)
        fn.argtypes, fn.restype = arguments, ctypes.c_int
        apis[name] = fn

    def call(name, *arguments):
        checkpoint(name)
        status = apis[name](*arguments)
        if status != 0:
            raise RuntimeError(f"{name} returned {status}")

    call("tpuRtInit")
    count = ctypes.c_int(-1)
    call("tpuRtGetDeviceCount", ctypes.byref(count))
    if count.value != 2:
        raise ValueError(f"Device topology changed: expected the observed 2, got {count.value}")
    call("tpuRtSetDevice", device_id)
    selected, descriptor = ctypes.c_int(-1), ctypes.c_int(-1)
    call("tpuRtGetDevice", ctypes.byref(selected))
    if selected.value != device_id:
        raise ValueError("Selected runtime device mismatch")
    call("tpuRtGetFd", ctypes.byref(descriptor))
    info = os.fstat(descriptor.value)
    if not stat.S_ISCHR(info.st_mode):
        raise ValueError("Runtime fd does not refer to a character device")
    node = Path(f"/sys/dev/char/{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}").resolve(strict=True)
    pci = (Path("/sys/bus/pci/devices") / expected_pci).resolve(strict=True)
    if pci not in node.parents:
        raise ValueError(f"Runtime fd mapping mismatch: {node} is not under {pci}")
    return {"device_id": selected.value, "device_count": count.value, "pci_function": expected_pci,
            "character_device": os.readlink(f"/proc/self/fd/{descriptor.value}")}


def check_previous(args, identity, input_hash):
    if args.variant == "original":
        if args.previous:
            raise ValueError("Original variant does not take a previous result")
        return None
    expected_variant = "original" if args.variant == "serial" else "serial"
    if not args.previous:
        raise ValueError(f"{args.variant} requires --previous pointing to passed {expected_variant} results")
    report = json.loads((args.previous / "result.json").read_text())
    numeric = report.get("numeric") or {}
    if report.get("status") != "passed" or numeric.get("variant") != expected_variant:
        raise ValueError("Previous stage did not pass or has the wrong variant")
    expected = {"build_manifest_sha256": identity["build_manifest_sha256"],
                "bundle_sha256": identity["bundle_sha256"], "input_sha256": input_hash,
                "device_id": args.device_id, "expected_pci": args.expected_pci}
    if any(numeric.get(key) != value for key,value in expected.items()):
        raise ValueError("Previous stage uses different sources, inputs or hardware")
    check_hash(args.previous / "output.f16", numeric["output_sha256"])
    return (args.previous / "output.f16").read_bytes()


def arm_parent_death(parent_pid, death_signal=signal.SIGKILL):
    """Fresh worker process, before runtime loading; no threaded preexec hook."""
    if os.getppid() != parent_pid:
        raise RuntimeError("Supervisor exited before worker setup")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    # SIGKILL works even while the worker is inside a blocking vendor call.
    if libc.prctl(1, death_signal, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        raise RuntimeError("Could not arm the worker parent-death signal")


def check_correctness_receipt(args, identity, input_hash, expected):
    receipt = json.loads((args.correctness / "result.json").read_text())
    numeric = receipt.get("numeric") or {}
    required = {"variant": args.variant, "shape": identity["shape"],
                "bundle_sha256": identity["bundle_sha256"],
                "build_manifest_sha256": identity["build_manifest_sha256"],
                "input_sha256": input_hash, "device_id": args.device_id,
                "expected_pci": args.expected_pci}
    if (receipt.get("status") != "passed" or numeric.get("status") != "passed" or
            not numeric.get("reference", {}).get("passed") or numeric.get("timing") or
            any(numeric.get(key) != value for key, value in required.items())):
        raise ValueError("Timing requires a matching passed one-call correctness receipt")
    check_hash(args.correctness / "output.f16", numeric.get("output_sha256"))
    actual = (args.correctness / "output.f16").read_bytes()
    if identity.get("operation", "add") == "div":
        # The TPU division primitive has the demo's existing tolerance;
        # it need not be bitwise equal to independently rounded FP32 division.
        check_output(actual, expected, len(expected) // 2, "div")
    elif actual != expected:
        raise ValueError("Correctness receipt output does not equal the reference")
    return digest(args.correctness / "result.json")


def child_command(args):
    command = [sys.executable, str(Path(__file__).resolve()), "--build", str(args.build),
               "--output", str(args.output), "--variant", args.variant, "--allow-pcie",
               "--device-id", str(args.device_id), "--expected-pci", args.expected_pci]
    if args.previous:
        command += ["--previous", str(args.previous)]
    if getattr(args, "measure", False):
        command += ["--measure", "--correctness", str(args.correctness)]
    return command


def guard(args):
    """A separate Python guard can kill the group even if ctypes is blocked."""
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Guard requires a private process group")

    def stop(_signum, _frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    arm_parent_death(args.guard_parent, signal.SIGTERM)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    if os.getppid() != args.guard_parent:
        stop(None, None)
    command = child_command(args) + ["--worker-parent", str(os.getpid())]
    process = subprocess.Popen(command, pass_fds=() if args.lock_fd is None else (args.lock_fd,))
    status = process.wait()
    print(f"WORKER_EXIT={status}", flush=True)
    return 128 - status if status < 0 else status


def worker(args):
    arm_parent_death(args.worker_parent)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cpus = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, cpus)
    priority = os.nice(10)
    result = {"status": "running", "variant": args.variant, "dtype": "float16", "shape": None,
              "launch_cores": 1, "requested_host_calls": 1, "dispatch_attempted": False,
              "requested_kernel_calls": 25 if args.measure else 1,
              "device_performance_measured": False, "hardware_overlap_verified": False,
              "device_id": args.device_id, "expected_pci": args.expected_pci,
              "runner_sha256": digest(__file__), "allowed_cpus": cpus, "nice": priority}

    def checkpoint(stage):
        result["last_stage"] = stage
        save(args.output / "worker.json", result)
        print("STAGE " + stage, flush=True)

    try:
        checkpoint("validate_build")
        for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "TPU_RT_CORE_NUM"):
            if os.environ.get(variable):
                raise ValueError(f"Fresh PCIe worker requires {variable} unset")
        if loaded_vendor_libraries():
            raise ValueError("Fresh PCIe worker already has a vendor runtime mapped")
        identity = validate_build(args.build)
        result.update(identity)
        count = math.prod(identity["shape"])
        operation = identity["operation"]
        lhs, rhs, expected = test_vectors(count, operation)
        result["input_sha256"] = hashlib.sha256(lhs + rhs).hexdigest()
        previous = check_previous(args, identity, result["input_sha256"])
        timing = None
        if args.measure:
            if not identity["timing_supported"]:
                raise ValueError("This pinned source bundle has no validated timing ABI")
            result["correctness_receipt_sha256"] = check_correctness_receipt(
                args, identity, result["input_sha256"], expected)
            timing = {"warmups": 5, "samples": 20}
        # These settings are local to this dedicated worker. The generic JIT
        # and demo-profiler authorization/identity checks remain unchanged.
        for key in tuple(os.environ):
            if key.startswith("TILELANG_TPU_PROFILE_") or key in (
                    "BMLIB_ENABLE_ALL_PROFILE", "TILELANG_TPU_ALLOW_PCIE_PROFILE", "FILE_DUMP_CMD"):
                os.environ.pop(key)
        os.environ.update(TILELANG_TPU_ALLOW_PCIE_LOAD="1", TILELANG_TPU_DEVICE_ID=str(args.device_id),
                          TILELANG_TPU_BENCHMARK_RUNS="0")
        checkpoint("load_verified_board_runtime")
        runtime_path = identity["runtime_library"]
        runtime = ctypes.CDLL(runtime_path, mode=os.RTLD_NOW | os.RTLD_LOCAL)
        globals()["_vendor_handles"] = (runtime,)
        if loaded_vendor_libraries() != {runtime_path}:
            raise ValueError("Loaded runtime identity differs or includes an emulator")
        result["mapping"] = verify_device(runtime, args.device_id, args.expected_pci, checkpoint)
        checkpoint("load_verified_host_module")
        main_so = args.build / args.variant / "main.so"
        # Repeat file checks immediately before loading the selected module.
        for filename in ("main.so", "libkernel.so"):
            check_hash(args.build / args.variant / filename, identity["artifacts"][args.variant][filename])
        library = ctypes.CDLL(str(main_so), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        globals()["_vendor_handles"] = (runtime, library)
        if loaded_vendor_libraries() != {runtime_path}:
            raise ValueError("Host module resolved a different vendor runtime")
        bind = library.tilelang_tpu_bind_device
        bind.argtypes, bind.restype = [ctypes.c_int], ctypes.c_int
        if bind(args.device_id) != 0:
            raise RuntimeError("Host module refused the verified device binding")
        result["dispatch_attempted"] = True
        checkpoint("tilelang_tpu_run_timed" if timing is not None else "tilelang_tpu_run_once")
        actual = call_host(library, lhs, rhs, timing=timing)
        result["host_call_returned_success"] = True
        checkpoint("compare_reference")
        result["reference"] = check_output(actual, expected, count, operation)
        if previous is not None and actual != previous:
            raise ValueError("Output differs bitwise from the previous variant")
        result["previous_variant_bitwise_equal"] = True if previous is not None else None
        (args.output / "output.f16").write_bytes(actual)
        result["output_sha256"] = hashlib.sha256(actual).hexdigest()
        result["status"] = "passed"
        if timing is not None:
            result["timing"] = timing
            result["synchronous_call_latency_measured"] = True
            print(operation.upper() + "_SYNC_LATENCY=" + json.dumps(timing, sort_keys=True), flush=True)
        checkpoint("completed")
        print(operation.upper() + "_CORRECTNESS=" + json.dumps({key: result[key] for key in (
            "operation", "variant", "mapping", "shape", "dtype", "launch_cores", "reference",
            "input_sha256", "output_sha256", "previous_variant_bitwise_equal")}, sort_keys=True), flush=True)
        # Keep both vendor handles alive through process exit; no dlclose of a
        # runtime that may still own background threads.
        globals()["_vendor_handles"] = (runtime, library)
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        save(args.output / "worker.json", result)
        print(result.get("operation", "elementwise").upper() + "_PCIE_FAILED " + result["error"], flush=True)
        return 1


def group_rss_bytes(group):
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            value = (entry / "stat").read_text()
            fields = value[value.rfind(")") + 2:].split()
            if int(fields[2]) == group:
                total += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            continue
    return total


def stop_group(process):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=3)
    return process.returncode is not None and group_rss_bytes(process.pid) == 0


def supervise(command, output, *, timeout_s=30, max_rss=4096*1024**2, lock_fd=None):
    started = time.monotonic()
    peak = 0
    failure = None
    cleanup = True
    with (output / "run.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=() if lock_fd is None else (lock_fd,))
        try:
            while process.poll() is None:
                peak = max(peak, group_rss_bytes(process.pid))
                if peak > max_rss:
                    failure = "rss_limit"
                    break
                if time.monotonic() - started > timeout_s:
                    failure = "timeout"
                    break
                time.sleep(0.1)
        except BaseException:
            cleanup = stop_group(process)
            raise
        if failure or group_rss_bytes(process.pid):
            failure = failure or "live_descendant"
            cleanup = stop_group(process)
    numeric = None
    with contextlib.suppress(OSError, json.JSONDecodeError):
        numeric = json.loads((output / "worker.json").read_text())
    passed = not failure and process.returncode == 0 and numeric and numeric.get("status") == "passed"
    report = {"status": "passed" if passed else "failed", "reason": failure,
              "returncode": process.returncode, "peak_group_rss_bytes": peak, "cleanup_complete": cleanup,
              "limits": {"timeout_seconds": timeout_s, "max_group_rss_bytes": max_rss, "cpu_count": 2},
              "numeric": numeric}
    save(output / "result.json", report)
    print((output / "run.log").read_text(errors="replace"), end="")
    prefix = (numeric or {}).get("operation", "elementwise").upper()
    print(prefix + "_PCIE_" + ("PASSED" if passed else "FAILED") + " RESULT=" + str(output / "result.json"), flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--measure", action="store_true", help="5 warmups, 20 resident synchronous calls")
    parser.add_argument("--correctness", type=Path, help="passed one-call receipt for this variant")
    parser.add_argument("--allow-pcie", action="store_true")
    parser.add_argument("--device-id", type=int, choices=(0,), required=True)
    parser.add_argument("--expected-pci", choices=("0000:01:00.0",), required=True)
    parser.add_argument("--worker-parent", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--guard-parent", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.allow_pcie:
        parser.error("--allow-pcie is required for this supervised board execution")
    if args.measure != (args.correctness is not None):
        parser.error("--measure and --correctness must be supplied together")
    if args.correctness:
        args.correctness = args.correctness.resolve()
    args.build, args.output = args.build.resolve(), args.output.resolve()
    if args.previous:
        args.previous = args.previous.resolve()
    if args.worker_parent:
        return worker(args)
    if args.guard_parent:
        return guard(args)
    # One same-user launch on this device; the worker inherits the lock fd.
    with open(f"/tmp/tilelang-bm1690-pcie-{os.getuid()}-device-0.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another supervised device-0 run is active")
        args.output.mkdir(parents=True, exist_ok=False)
        command = child_command(args) + ["--guard-parent", str(os.getpid()),
                                        "--lock-fd", str(lock.fileno())]
        return supervise(command, args.output, lock_fd=lock.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
