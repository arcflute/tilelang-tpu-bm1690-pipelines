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
import urllib.request

FILES = {
    "build.py": ("tilelang/jit/adapter/legacy_pcie.py",
                 "b701ff09991eb77c202fd4108e784521efa2c427f0e677025c7045f6cdbc2525"),
    "add.json": ("research/bm1690-pipelines/handoff/add-1024-sources.json",
                 "2f43fcb39be86a3a4fc7407e5a10714bccbbb93ae7f7211dfc38bafa938155c5"),
    "run_add.py": ("tpu_demo/pipeline/run_add_pcie.py",
                   "8e60b2affb40d5b00ee1062e811f94c66fbc51d51927c1ffb529e4e8676d480a"),
}
BASE = "https://raw.githubusercontent.com/arcflute/tilelang-tpu-bm1690-pipelines"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full commit ID")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print("HANDOFF_DIR=" + str(output), flush=True)
    receipt = {"status": "started", "revision": args.revision,
               "board_runtime_loaded": False, "kernel_launches": 0, "files": {}}
    try:
        for name, (path, expected) in FILES.items():
            url = f"{BASE}/{args.revision}/{path}"
            print("DOWNLOAD " + name, flush=True)
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(1024 * 1024 + 1)
            actual = hashlib.sha256(data).hexdigest()
            if len(data) > 1024 * 1024 or actual != expected:
                raise ValueError(f"Download size/hash mismatch: {name}")
            (output / name).write_bytes(data)
            receipt["files"][name] = actual
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
        (output / "handoff.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["status"] in ("compile_only_passed", "download_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
