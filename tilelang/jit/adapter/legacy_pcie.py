# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Explicit PPL 1.4 BM1690 PCIe profile and standalone source-only build check.

This module uses only the Python standard library. Running this file directly
can compile exported TileLang sources on a board host without installing Python
packages or rebuilding TVM. It never imports/loads a vendor shared library.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROFILE = "ppl14-bm1690-pcie"
SOURCE_NAMES = ("kernel.c", "kernel.h", "kernel.cpp", "main.cpp")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _explicit_path(environment, variable):
    value = environment.get(variable)
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{PROFILE} requires an absolute {variable}")
    return Path(value).resolve()


@dataclass(frozen=True)
class PPL14BM1690PCIeLayout:
    root: Path
    board_lib: Path
    cross_gcc: Path
    chip: str = "bm1690"
    profile: str = PROFILE
    compile_definitions = ("__bm1690__",)
    physical_core_count = 8

    @property
    def kernel_include(self):
        return self.root / "runtime/bm1690/TPU1686/kernel/include"

    @property
    def kernel_common_include(self):
        return self.root / "runtime/kernel"

    @property
    def device_utils_include(self):
        return self.root / "runtime/customize/include"

    @property
    def runtime_include(self):
        return self.board_lib.parent / "include"

    @property
    def backend_lib(self):
        return self.root / "runtime/bm1690/lib"

    @property
    def firmware_archive(self):
        return self.backend_lib / "libbm1690.a"

    @property
    def ppl_helper_source(self):
        return self.root / "runtime/customize/src/ppl_helper.c"

    def pcie_cross_gcc(self):
        if not self.cross_gcc.is_file() or not os.access(self.cross_gcc, os.X_OK):
            raise FileNotFoundError(f"Executable Linux RISC-V compiler missing: {self.cross_gcc}")
        if self.cross_gcc.name != "riscv64-unknown-linux-gnu-gcc":
            raise ValueError("PPL 1.4 PCIe requires riscv64-unknown-linux-gnu-gcc")
        return self.cross_gcc

    def pcie_runtime_lib(self, environment=None):
        source = os.environ if environment is None else environment
        selected = _explicit_path(source, "TILELANG_TPU_PCIE_RUNTIME_PATH")
        if selected != self.board_lib:
            raise ValueError("Board runtime changed after resolving the PPL 1.4 profile")
        if selected == self.root or self.root in selected.parents:
            raise ValueError("PCIe board runtime must be outside the PPL SDK/CModel tree")
        runtime_so = selected / "libtpuv7_rt.so"
        if self.root in runtime_so.resolve().parents:
            raise ValueError("PCIe runtime library points into the PPL SDK/CModel tree")
        if not runtime_so.is_file():
            raise FileNotFoundError(f"Board libtpuv7_rt.so missing: {selected}")
        return selected

    def require_runtime(self, runtime_mode, *, environment=None):
        if runtime_mode != "pcie":
            raise ValueError(f"{PROFILE} supports PCIe only; retain PPL 1.7 for CModel")
        self.pcie_runtime_lib(environment)
        self.pcie_cross_gcc()
        for directory in (self.kernel_include, self.kernel_common_include,
                          self.device_utils_include, self.runtime_include):
            if not directory.is_dir():
                raise FileNotFoundError(f"PPL 1.4 include directory missing: {directory}")
        for path in (self.kernel_include / "tpu_kernel.h",
                     self.device_utils_include / "ppl_helper.h",
                     self.device_utils_include / "ppl_mem.h",
                     self.runtime_include / "tpuv7_rt.h", self.ppl_helper_source,
                     self.firmware_archive):
            if not path.is_file():
                raise FileNotFoundError(f"PPL 1.4 PCIe artifact missing: {path}")
        return self

    def require_profiling(self, runtime_mode, *, environment=None):
        raise ValueError("PPL 1.4 PCIe profiling is not validated; use unprofiled correctness/latency")

    def require_rvt_api(self):
        raise ValueError("BM1690 does not support the RV programming model")

    def include_dirs_for(self, runtime_mode, *, profiling=False, environment=None):
        self.require_runtime(runtime_mode, environment=environment)
        if profiling:
            self.require_profiling(runtime_mode, environment=environment)
        return (self.kernel_include, self.kernel_common_include,
                self.device_utils_include, self.runtime_include)

    def runtime_identity_for(self, runtime_mode, environment=None):
        self.require_runtime(runtime_mode, environment=environment)
        return tuple(str(p.resolve()) for p in (self.root, self.board_lib, self.backend_lib))


def resolve_legacy_pcie(ppl_root, chip, *, environment=None):
    if chip != "bm1690":
        raise ValueError(f"{PROFILE} requires chip=bm1690, got {chip!r}")
    source = os.environ if environment is None else environment
    layout = PPL14BM1690PCIeLayout(
        root=Path(ppl_root).expanduser().resolve(),
        board_lib=_explicit_path(source, "TILELANG_TPU_PCIE_RUNTIME_PATH"),
        cross_gcc=_explicit_path(source, "TILELANG_TPU_PCIE_CROSS_GCC"),
    )
    return layout.require_runtime("pcie", environment=source)


def legacy_pcie_commands(layout, directory, *, programming_model="tpukernel", profiling=False):
    """Compile/link only, with the macros/archive used by validated ChunkScan.

    Keep the PPL 1.4 device flags distinct from the PPL 1.7 LTO build. Host
    headers and shared library come from the same installed runtime root.
    No SDK emulator directory is added to host library search paths.
    """
    if programming_model != "tpukernel":
        raise ValueError("PPL 1.4 BM1690 PCIe requires tpukernel")
    includes = ["-I" + str(p) for p in layout.include_dirs_for("pcie", profiling=profiling)]
    directory = Path(directory).resolve()
    includes.insert(0, "-I" + str(directory))
    # These paths are embedded in compiler macros or the ELF rpath.
    for path in (directory, layout.board_lib):
        if any(c in str(path) for c in ('"', '\\', '\n', '\r', ':', ',')):
            raise ValueError(f"Unsupported character in embedded PCIe path: {path}")
    definitions = ["-D__bm1690__", "-DTILELANG_TPU_TPUKERNEL"]
    device_flags = definitions + ["-Dlibkernel_EXPORTS", *includes, "-O2", "-fPIC"]
    cross = str(layout.pcie_cross_gcc())
    kernel_o, helper_o = directory / "kernel.o", directory / "ppl_helper.o"
    libkernel = directory / "libkernel.so"
    commands = []
    for label, source, output in (
        ("Compile TPU kernel", directory / "kernel.c", kernel_o),
        ("Compile PPL helper", layout.ppl_helper_source, helper_o),
    ):
        commands.append((label, [cross, *device_flags, "-c", str(source), "-o", str(output)]))
    commands.append(("Link PCIe libkernel.so", [
        cross, "-shared", "-fPIC", "-Wl,--no-undefined", "-Wl,-soname,libkernel.so",
        "-o", str(libkernel), str(kernel_o), str(helper_o), "-Wl,--whole-archive",
        str(layout.firmware_archive), "-Wl,--no-whole-archive", "-lm",
    ]))
    host_flags = definitions + includes + [
        "-std=c++17", "-O2", "-fPIC", f'-DTILELANG_PPL_KERNEL_PATH="{libkernel}"',
    ]
    for source, output in (("kernel.cpp", "kernel_host.o"), ("main.cpp", "main.o")):
        commands.append((f"Compile PCIe {source}", [
            "/usr/bin/c++", *host_flags, "-c", str(directory / source), "-o", str(directory / output),
        ]))
    commands.append(("Link PCIe main.so", [
        "/usr/bin/c++", "-shared", "-fPIC", "-Wl,--no-undefined", "-o", str(directory / "main.so"),
        str(directory / "kernel_host.o"), str(directory / "main.o"),
        "-L" + str(layout.board_lib), "-Wl,--disable-new-dtags,-rpath," + str(layout.board_lib),
        "-ltpuv7_rt", "-lpthread",
    ]))
    return commands


def validate_source_bundle(bundle):
    schemas = {"bm1690-add-source-check-v1": [8, 128],
               "bm1690-add-source-check-v2": [1024, 1024]}
    if bundle.get("schema") not in schemas:
        raise ValueError("Unsupported source bundle schema")
    if bundle.get("target") != {"chip": "bm1690", "programming_model": "tpukernel",
                                "launch_cores": 1, "dtype": "float16",
                                "shape": schemas[bundle["schema"]]}:
        raise ValueError("Source schema requires its fixed single-core FP16 Add shape")
    variants = bundle.get("variants", {})
    if set(variants) != {"original", "serial", "pipeline"}:
        raise ValueError("Source bundle must preserve original/serial/pipeline variants")
    for name, variant in variants.items():
        if set(variant["sources"]) != set(SOURCE_NAMES):
            raise ValueError(f"Unexpected source names for {name}")
        if set(variant["sha256"]) != set(SOURCE_NAMES):
            raise ValueError(f"Missing source hashes for {name}")
        for filename, source in variant["sources"].items():
            if not isinstance(source, str) or len(source) > 1024 * 1024:
                raise ValueError(f"Invalid source size: {name}/{filename}")
            actual = hashlib.sha256(source.encode()).hexdigest()
            if variant["sha256"][filename] != actual:
                raise ValueError(f"Source hash mismatch: {name}/{filename}")
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "failed", "profile": PROFILE, "board_runtime_loaded": False,
              "kernel_launches": 0, "completed_commands": []}
    try:
        # Establish bounds before executing any compiler; do not import torch/TVM.
        import resource
        cpus = sorted(os.sched_getaffinity(0))[:2]
        os.sched_setaffinity(0, cpus)
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = min([4 * 1024**3] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY])
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        if args.bundle.stat().st_size > 8 * 1024**2:
            raise ValueError("Source bundle is too large")
        bundle = json.loads(args.bundle.read_text())
        variants = validate_source_bundle(bundle)
        ppl_root = _explicit_path(os.environ, "PPL_PROJECT_ROOT")
        layout = resolve_legacy_pcie(ppl_root, "bm1690")
        timeout_tool = shutil.which("timeout")
        if timeout_tool is None:
            raise FileNotFoundError("GNU timeout is required for bounded compiler process groups")
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
        result.update({"bundle_sha256": sha256(args.bundle), "builder_sha256": sha256(__file__),
                       "allowed_cpus": cpus, "address_space_limit_bytes": cap,
                       "runtime_identity": layout.runtime_identity_for("pcie"),
                       "sdk_inputs": {str(p): sha256(p) for p in (
                           layout.firmware_archive, layout.ppl_helper_source,
                           layout.kernel_include / "tpu_kernel.h",
                           layout.runtime_include / "tpuv7_rt.h",
                           layout.board_lib / "libtpuv7_rt.so", layout.cross_gcc)}})
        environment = dict(os.environ)
        # Avoid accidental header/library injection from a CModel login shell.
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
                    "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_PRELOAD", "LD_LIBRARY_PATH"):
            environment.pop(key, None)
        machine = subprocess.check_output([str(layout.cross_gcc), "-dumpmachine"],
                                          env=environment, timeout=10, text=True).strip()
        if machine != "riscv64-unknown-linux-gnu":
            raise ValueError(f"Unexpected compiler target: {machine}")
        for name in ("original", "serial", "pipeline"):
            directory = args.output / name
            directory.mkdir()
            for filename, source in variants[name]["sources"].items():
                (directory / filename).write_text(source)
            for number, (label, command) in enumerate(legacy_pcie_commands(layout, directory)):
                print(f"BUILD {name}: {label}", flush=True)
                record = {"variant": name, "label": label, "command": command,
                          "log": str(directory / f"{number}.log")}
                result["active_command"] = record
                (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                with (directory / f"{number}.log").open("w") as log:
                    subprocess.run([timeout_tool, "-k", "2s", "60s", *command], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                result["completed_commands"].append(record)
                result.pop("active_command")
            result.setdefault("artifacts", {})[name] = {
                filename: sha256(directory / filename) for filename in ("libkernel.so", "main.so")}
        result["status"] = "compile_only_passed"
        print("BUILD_ONLY_OK variants=3 board_runtime_loaded=false kernel_launches=0", flush=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print("BUILD_ONLY_FAILED: " + result["error"], file=sys.stderr, flush=True)
        active_log = result.get("active_command", {}).get("log")
        if active_log and Path(active_log).is_file():
            # Bound the pasted diagnostic while retaining the complete log on disk.
            with open(active_log, "rb") as log:
                log.seek(max(0, os.fstat(log.fileno()).st_size - 8192))
                print(log.read().decode(errors="replace"), file=sys.stderr, flush=True)
    finally:
        # Never modify an already-existing output directory after a validation failure.
        if "bundle_sha256" in result:
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "compile_only_passed" else 1


if __name__ == "__main__":
    sys.exit(main())
