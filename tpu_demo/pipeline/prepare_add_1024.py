"""Download hash-pinned elementwise/Matmul sources and compile with the observed BM1690 SDK.

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
                 "595fca52044098d9ea319f17b5ed86362d6700ce6e65c419e3876ece63e18ff2"),
    "add.json": ("research/bm1690-pipelines/handoff/add-1024-sources.json",
                 "2f43fcb39be86a3a4fc7407e5a10714bccbbb93ae7f7211dfc38bafa938155c5"),
    "run_add.py": ("tpu_demo/pipeline/run_add_pcie.py",
                   "b46d84d8a9c8ba789de383991d8db542f1d1c7e2a3fc7b6d1f987bed78ab51a1"),
}
BASE = "https://raw.githubusercontent.com/arcflute/tilelang-tpu-bm1690-pipelines"
API_BASE = "https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents"
COARSE_SOURCE = ("research/bm1690-pipelines/handoff/add-1024-coarse-sources.json",
                 "2f6cfb1bf2af284b6abe576e966f2bc8f14352baac1e0f5c408cedf17e024ae2")

ELEMENTWISE_SOURCES = {
    "sub": [
        "research/bm1690-pipelines/handoff/sub-1024-sources.json",
        "6d72d38f8a83b510dc71fab7905e8babf0b7b645cdd2ffa1ecc7a78180430418"
    ],
    "mul": [
        "research/bm1690-pipelines/handoff/mul-1024-sources.json",
        "704340e5fecb142001b3b362cb50516cd818b848b4987dfc39dd36b5c0a0c1aa"
    ],
    "div": [
        "research/bm1690-pipelines/handoff/div-1024-sources.json",
        "97238d2cb47b8921a14d1d82b0f63ca0eee6efdc98b0ed3d9baf2306689aab17"
    ]
}


MATMUL_SOURCE = ('research/bm1690-pipelines/handoff/matmul-smoke-sources.json', '71f50c676ecfba081728ef763a1ebc29ac36e7ece8b7b193310c0b5eb7309f88')


MATMUL_TARGET_SOURCE = ('research/bm1690-pipelines/handoff/matmul-1024-sources.json', 'fb97f8b800a47bbe91fe393885b785cd6323043231660c202bc510f271c1f643')


RMSNORM_SOURCES = {'rmsnorm-smoke': ('research/bm1690-pipelines/handoff/rmsnorm-smoke-sources.json', '3d4f53f0280dc7931f2db158ffe1f28f81e9f808c6d533190342cd45fcbb2680'), 'rmsnorm-splitk-smoke': ('research/bm1690-pipelines/handoff/rmsnorm-splitk-smoke-sources.json', 'cf249e148951244053b61af7e18be68edeaef6caa79a5522d2387bc2b08c9ec7')}

RMSNORM_SOURCES.update({'rmsnorm-target': ('research/bm1690-pipelines/handoff/rmsnorm-1024-sources.json', '2dce32f1a01903815f2c837c8e0a567365d5e5883b1ae042553a0e9a4251cdbb'), 'rmsnorm-splitk-target': ('research/bm1690-pipelines/handoff/rmsnorm-splitk-1024-sources.json', '80eb6f65a0a5f05ea567a9d4a01bedfaf5765faaa240aa3c7993f377be7d148a')})

SWIGLU_SOURCES = {
    "swiglu-smoke": ("research/bm1690-pipelines/handoff/swiglu-smoke-sources.json",
                     "595945b8629c7df328f67669ce5bf8e1bc3d327a4889feb2e148563112105338"),
    "swiglu-target": ("research/bm1690-pipelines/handoff/swiglu-1024-sources.json",
                      "ebe41362517fca17248465e34f10204b612c880b30c747a6ade5521625b90611"),
}

ROPE_SOURCE = ("research/bm1690-pipelines/handoff/rope-smoke-sources.json", '6466f705a207a969c9669e9cb8729bd1f6c1fbb38702f797f46ee2fdfd22bdbb')

ROPE_TARGET_SOURCE = ("research/bm1690-pipelines/handoff/rope-1024-sources.json", 'c45a7e3a2676f59be6808d940938a4618bab1f7e78f5fb5ceb58112b5ed8eb87')

ROPE_TRACE_SOURCE = ("research/bm1690-pipelines/handoff/rope-1024-host-trace-sources.json", '815a4f6489da76aa3855da3899029f93008dd82ae6be92f2b1033c0c9d6b9fcd')

ROPE_WINDOW_SOURCE = ("research/bm1690-pipelines/handoff/rope-1024-window8-sources.json", 'dcf15b8d6480b05345d116619d3f8744de1da7657af937de0bbe9979d9500622')

ROPE_WINDOW_END_SOURCE = ("research/bm1690-pipelines/handoff/rope-1024-window8-end-sources.json", "3e62b95c1a772f1c2e33f9648ea84db283cf98a136da72697535dc7bad8fb654")

ROPE_PREFIX_SOURCES = {
    "rope-target-prefix1024": ("research/bm1690-pipelines/handoff/rope-1024-prefix1024-sources.json", "db56264dba14cb7638e6de460fa01cc47ead29af5565a235fcda3e3978bfa9c7"),
    "rope-target-prefix2048": ("research/bm1690-pipelines/handoff/rope-1024-prefix2048-sources.json", "93295b5757b75d9cbfc8700f1a249bb2ce2b887667bd9563b2de026ba84bc9af"),
    "rope-target-prefix4096": ("research/bm1690-pipelines/handoff/rope-1024-prefix4096-sources.json", "f82b0521a49c83e2e410d337ffca6b215ef4cf62933b11da869e505eeea6ed06"),
    "rope-target-prefix128": ("research/bm1690-pipelines/handoff/rope-1024-prefix128-sources.json", "9111d5d05d45165d7ef2ef579c7f2b6dc7303f3d2be85abf22aa2e0e7fa22880"),
    "rope-target-prefix256": ("research/bm1690-pipelines/handoff/rope-1024-prefix256-sources.json", "b5533f1490f80e5c74f60b929db9ad28f7e1fb597c2224c9a0bb4ad7737952aa"),
    "rope-target-prefix512": ("research/bm1690-pipelines/handoff/rope-1024-prefix512-sources.json", "9b9250c5828bd25d6f5cd315f4b5aca919a93b7d2b36cac3b3cf80e66972e962"),
}

FLASHATTN_SOURCES = {'flashattn-smoke': ('research/bm1690-pipelines/handoff/flashattn-smoke-sources.json', '0095597052395485977eed13b60bda8a458787fb5a6af52b72ba34cbc9172a44'), 'flashattn-target': ('research/bm1690-pipelines/handoff/flashattn-1024-sources.json', '0a0bc116ad3ef055590892889f82f889f4254b6547d101015f6b9d76b3f680e7'), 'flashattn-causal-smoke': ('research/bm1690-pipelines/handoff/flashattn-causal-smoke-sources.json', '18be1e38f81e4659aef58b13bc17257eeaed9b0f89277a079bb7ca06deb8947a'), 'flashattn-causal-target': ('research/bm1690-pipelines/handoff/flashattn-causal-1024-sources.json', '18d07660d190e997f823db7c879450c781ccd3c2e97aa0ec7171e3d6f560beff')}

def source_files(case):
    if case in FLASHATTN_SOURCES:
        return {**FILES, "add.json": FLASHATTN_SOURCES[case]}
    if case in SWIGLU_SOURCES:
        return {**FILES, "add.json": SWIGLU_SOURCES[case]}
    if case in ROPE_PREFIX_SOURCES:
        return {**FILES, "add.json": ROPE_PREFIX_SOURCES[case]}
    if case == "rope-target-window8-end":
        return {**FILES, "add.json": ROPE_WINDOW_END_SOURCE}
    if case == "rope-target-window8":
        return {**FILES, "add.json": ROPE_WINDOW_SOURCE}
    if case == "rope-target-trace":
        return {**FILES, "add.json": ROPE_TRACE_SOURCE}
    if case == "rope-target":
        return {**FILES, "add.json": ROPE_TARGET_SOURCE}
    if case == "rope-smoke":
        return {**FILES, "add.json": ROPE_SOURCE}
    if case in RMSNORM_SOURCES:
        return {**FILES, "add.json": RMSNORM_SOURCES[case]}
    if case == "matmul-target":
        return {**FILES, "add.json": MATMUL_TARGET_SOURCE}
    if case == "matmul-smoke":
        return {**FILES, "add.json": MATMUL_SOURCE}
    if case == "target":
        return dict(FILES)
    if case == "coarse":
        return {**FILES, "add.json": COARSE_SOURCE}
    if case in ELEMENTWISE_SOURCES:
        return {**FILES, "add.json": ELEMENTWISE_SOURCES[case]}
    raise ValueError("Unsupported comparison case")


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
    parser.add_argument("--case", choices=("target", "coarse", "sub", "mul", "div", "matmul-smoke", "matmul-target", "rmsnorm-smoke", "rmsnorm-splitk-smoke", "rmsnorm-target", "rmsnorm-splitk-target", "rope-smoke", "rope-target", "rope-target-trace", "rope-target-window8", "rope-target-window8-end", *ROPE_PREFIX_SOURCES, *SWIGLU_SOURCES, *FLASHATTN_SOURCES), default="target",
                        help="Elementwise, Matmul, RMSNorm ordinary/Split-K or RoPE smoke/target")
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
            prefix = (args.case.upper().replace("-", "_") if args.case in SWIGLU_SOURCES or args.case in RMSNORM_SOURCES or args.case in ROPE_PREFIX_SOURCES or args.case in ("rope-smoke", "rope-target", "rope-target-trace", "rope-target-window8", "rope-target-window8-end") else
                      "MATMUL1024" if args.case == "matmul-target" else
                      "MATMUL128" if args.case == "matmul-smoke" else
                      (args.case.upper() if args.case in ELEMENTWISE_SOURCES else "ADD") + "1024")
            print(prefix + "_BUILD_ONLY_OK kernel_launches=0", flush=True)
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print("STOP " + receipt["error"], flush=True)
    finally:
        journal.write_text(json.dumps(receipt, indent=2) + "\n")
        print("HANDOFF_RECEIPT=" + str(journal), flush=True)
    return 0 if receipt["status"] in ("compile_only_passed", "download_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
