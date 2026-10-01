"""Download hash-pinned Add sources and compile with the observed BM1690 SDK.

Standard library only; never loads a vendor library or launches a kernel.
The default paths are the paths inspected on the user's bokai board host.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

FILES = {
    "build.py": ("tilelang/jit/adapter/legacy_pcie.py",
                 "b701ff09991eb77c202fd4108e784521efa2c427f0e677025c7045f6cdbc2525"),
    "add.json": ("research/bm1690-pipelines/handoff/add-1024-sources.json",
                 "2f43fcb39be86a3a4fc7407e5a10714bccbbb93ae7f7211dfc38bafa938155c5"),
    "run_add.py": ("tpu_demo/pipeline/run_add_pcie.py",
                   "c8c88de318a4bf79b9b4911584e9daa5693165fc515061376ba7df979108beaf"),
}
BASE = "https://raw.githubusercontent.com/arcflute/tilelang-tpu-bm1690-pipelines"
API_BASE = "https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents"
COARSE_SOURCE = ("research/bm1690-pipelines/handoff/add-1024-coarse-sources.json",
                 "2f6cfb1bf2af284b6abe576e966f2bc8f14352baac1e0f5c408cedf17e024ae2")


def source_files(case):
    if case == "target":
        return dict(FILES)
    if case == "coarse":
        return {**FILES, "add.json": COARSE_SOURCE}
    raise ValueError("Unsupported Add comparison case")


def source_request(path, revision, transport):
    if transport == "github-api":
        # Contents API returns bytes directly with this media type. Using the
        # JSON download_url instead would send us back to the raw domain.
        return urllib.request.Request(f"{API_BASE}/{path}?ref={revision}", headers={
            "Accept": "application/vnd.github.raw+json", "User-Agent": "bm1690-handoff"})
    if transport == "raw":
        return f"{BASE}/{revision}/{path}"
    raise ValueError("Unsupported download transport")


def download_verified(url, expected, *, attempts=3):
    """Retry transient reads only; never retry an integrity failure or a build."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(1024 * 1024 + 1)
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            print(f"DOWNLOAD_RETRY {attempt}/{attempts} {type(exc).__name__}: {exc}", flush=True)
            if attempt == attempts:
                raise
            continue
        if len(data) > 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Download size/hash mismatch")
        return data
    raise ValueError("Download attempts must be positive")


def resume_receipt(output, revision, case="target"):
    """Validate a download-only/failed-download directory without changing it."""
    previous = output / "handoff.json"
    receipt = json.loads(previous.read_text())
    if (receipt.get("revision") != revision or receipt.get("case", "target") != case or
            receipt.get("status") not in ("started", "failed", "download_verified") or
            receipt.get("board_runtime_loaded") is not False or receipt.get("kernel_launches") != 0 or
            "build_exit" in receipt or (output / "build").exists() or (output / "build.log").exists()):
        raise ValueError("Resume requires the same revision and a directory where compilation has not started")
    # Validate every retained file before making any network request.
    for name, (_, expected) in source_files(case).items():
        path = output / name
        if path.is_symlink() or (path.exists() and
                (not path.is_file() or path.stat().st_size > 1024 * 1024 or
                 hashlib.sha256(path.read_bytes()).hexdigest() != expected)):
            raise ValueError(f"Existing file size/hash/type mismatch: {name}")
    return {"path": str(previous), "sha256": hashlib.sha256(previous.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--case", choices=("target", "coarse"), default="target",
                        help="32x128 or 128x1024 tiles at the same global 1024x1024 shape")
    parser.add_argument("--resume", action="store_true", help="reuse verified downloads before compilation")
    parser.add_argument("--transport", choices=("raw", "github-api"), default="raw",
                        help="official GitHub endpoint; source SHA256 pins are identical")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full commit ID")
    output = args.output.resolve()
    prior = None
    if args.resume:
        prior = resume_receipt(output, args.revision, args.case)
        # Retain the original failed/download-only receipt and every retry's
        # separate journal. Never overwrite build attempts or verified inputs.
        with tempfile.NamedTemporaryFile(prefix="handoff-resume-", suffix=".json", dir=output, delete=False) as marker:
            journal = Path(marker.name)
    else:
        output.mkdir(parents=True, exist_ok=False)
        journal = output / "handoff.json"
    print("HANDOFF_DIR=" + str(output), flush=True)
    receipt = {"status": "started", "revision": args.revision,
               "board_runtime_loaded": False, "kernel_launches": 0, "files": {},
               "download_transport": args.transport, "case": args.case}
    if prior is not None:
        receipt["resumed_from"] = prior
    journal.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for name, (path, expected) in source_files(args.case).items():
            target = output / name
            if args.resume and target.exists():
                receipt["files"][name] = expected
                print("REUSED_VERIFIED " + name, flush=True)
                continue
            request = source_request(path, args.revision, args.transport)
            print(f"DOWNLOAD {name} transport={args.transport}", flush=True)
            data = download_verified(request, expected)
            with target.open("xb") as destination:
                destination.write(data)
            receipt["files"][name] = expected
            print("VERIFIED " + name, flush=True)
        if args.download_only:
            receipt["status"] = "download_verified"
        else:
            environment = dict(os.environ)
            environment.update({
                "PPL_PROJECT_ROOT": "/home/bokai/ChunkScan-bm1690-deps/ppl_v1.4.195-geb2acdd0-20250220",
                "TILELANG_TPU_PCIE_RUNTIME_PATH": "/opt/tpuv7/tpuv7-current/lib",
                "TILELANG_TPU_PCIE_CROSS_GCC": "/host-tools/gcc-riscv/gcc-riscv64-unknown-linux-gnu/bin/riscv64-unknown-linux-gnu-gcc",
            })
            command = ["timeout", "-k", "5s", "180s", sys.executable, str(output / "build.py"),
                       "--bundle", str(output / "add.json"), "--output", str(output / "build")]
            with (output / "build.log").open("w") as log:
                result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            receipt["build_exit"] = result.returncode
            print((output / "build.log").read_text(errors="replace"), end="")
            if result.returncode:
                raise RuntimeError(f"Compilation failed: {result.returncode}; inspect {output / 'build.log'}")
            receipt["status"] = "compile_only_passed"
            print("ADD1024_BUILD_ONLY_OK kernel_launches=0", flush=True)
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print("STOP " + receipt["error"], flush=True)
    finally:
        journal.write_text(json.dumps(receipt, indent=2) + "\n")
        print("HANDOFF_RECEIPT=" + str(journal), flush=True)
    return 0 if receipt["status"] in ("compile_only_passed", "download_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
