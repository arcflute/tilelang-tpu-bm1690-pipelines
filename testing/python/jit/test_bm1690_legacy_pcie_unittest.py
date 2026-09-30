"""Filesystem/compile-policy regressions; no vendor runtime or pytest needed."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tilelang/jit/adapter/legacy_pcie.py"
spec = importlib.util.spec_from_file_location("standalone_legacy_pcie", SOURCE)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)


def write(path, value=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


class LegacyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sdk = self.root / "ppl"
        for relative in ("runtime/bm1690/TPU1686/kernel/include/tpu_kernel.h",
                         "runtime/kernel/common.h", "runtime/customize/include/ppl_helper.h",
                         "runtime/customize/include/ppl_mem.h", "runtime/customize/src/ppl_helper.c",
                         "runtime/bm1690/lib/libbm1690.a"):
            write(self.sdk / relative)
        self.board = self.root / "installed"
        write(self.board / "include/tpuv7_rt.h")
        write(self.board / "lib/libtpuv7_rt.so")
        self.compiler = write(self.root / "cross/bin/riscv64-unknown-linux-gnu-gcc",
                              '#!/bin/sh\nif [ "$1" = -dumpmachine ]; then echo riscv64-unknown-linux-gnu; exit 0; fi\necho deliberate-compile-failure; exit 9\n')
        self.compiler.chmod(0o700)
        self.env = {"PPL_PROJECT_ROOT": str(self.sdk),
                    "TILELANG_TPU_PCIE_RUNTIME_PATH": str(self.board / "lib"),
                    "TILELANG_TPU_PCIE_CROSS_GCC": str(self.compiler)}
        self.layout = legacy.resolve_legacy_pcie(self.sdk, "bm1690", environment=self.env)

    def bundle(self):
        sources = {name: "// compile-only fixture\n" for name in legacy.SOURCE_NAMES}
        return {"schema": "bm1690-add-source-check-v1",
                "target": {"chip": "bm1690", "programming_model": "tpukernel", "launch_cores": 1,
                           "dtype": "float16", "shape": [8, 128]},
                "variants": {name: {"sources": dict(sources), "sha256": {
                    filename: hashlib.sha256(value.encode()).hexdigest() for filename, value in sources.items()}}
                    for name in ("original", "serial", "pipeline")}}

    def test_explicit_identity_and_legacy_paths(self):
        self.assertEqual(self.layout.compile_definitions, ("__bm1690__",))
        self.assertEqual(self.layout.runtime_identity_for("pcie", self.env),
                         (str(self.sdk), str(self.board / "lib"), str(self.sdk / "runtime/bm1690/lib")))
        self.assertIn(self.board / "include", self.layout.include_dirs_for("pcie", environment=self.env))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.layout.pcie_runtime_lib({**self.env, "TILELANG_TPU_PCIE_RUNTIME_PATH": str(self.root)})

    def test_rejects_missing_inputs_wrong_chip_and_wrong_toolchain(self):
        for key in self.env.keys() - {"PPL_PROJECT_ROOT"}:
            with self.subTest(key=key), self.assertRaises(ValueError):
                legacy.resolve_legacy_pcie(self.sdk, "bm1690", environment={k: v for k,v in self.env.items() if k != key})
        with self.assertRaisesRegex(ValueError, "chip=bm1690"):
            legacy.resolve_legacy_pcie(self.sdk, "sg2260e", environment=self.env)
        wrong = write(self.root / "riscv64-unknown-elf-gcc")
        wrong.chmod(0o700)
        with self.assertRaisesRegex(ValueError, "unknown-linux-gnu"):
            legacy.resolve_legacy_pcie(self.sdk, "bm1690", environment={**self.env, "TILELANG_TPU_PCIE_CROSS_GCC": str(wrong)})
        self.layout.firmware_archive.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "libbm1690.a"):
            legacy.resolve_legacy_pcie(self.sdk, "bm1690", environment=self.env)

    def test_rejects_cmodel_rv_profiling_and_sdk_runtime(self):
        with self.assertRaisesRegex(ValueError, "PCIe only"):
            self.layout.require_runtime("cmodel", environment=self.env)
        with self.assertRaisesRegex(ValueError, "RV"):
            self.layout.require_rvt_api()
        with self.assertRaisesRegex(ValueError, "not validated"):
            self.layout.require_profiling("pcie", environment=self.env)
        write(self.sdk / "runtime/lib/libtpuv7_rt.so")
        with self.assertRaisesRegex(ValueError, "outside"):
            legacy.resolve_legacy_pcie(self.sdk, "bm1690", environment={
                **self.env, "TILELANG_TPU_PCIE_RUNTIME_PATH": str(self.sdk / "runtime/lib")})
        (self.board / "lib/libtpuv7_rt.so").unlink()
        (self.board / "lib/libtpuv7_rt.so").symlink_to(self.sdk / "runtime/lib/libtpuv7_rt.so")
        with self.assertRaisesRegex(ValueError, "points into"):
            legacy.resolve_legacy_pcie(self.sdk, "bm1690", environment=self.env)

    def test_commands_link_only_verified_archive_and_board_runtime(self):
        with patch.dict(os.environ, self.env):
            commands = legacy.legacy_pcie_commands(self.layout, self.root / "output")
        self.assertEqual(len(commands), 6)
        device = commands[0][1]
        self.assertIn("-D__bm1690__", device)
        self.assertIn("-O2", device)
        self.assertNotIn("-flto", device)
        self.assertIn(str(self.layout.firmware_archive), commands[2][1])
        self.assertIn("-I" + str(self.board / "include"), commands[3][1])
        self.assertIn("-L" + str(self.board / "lib"), commands[-1][1])
        self.assertNotIn("-L" + str(self.layout.backend_lib), commands[-1][1])
        self.assertNotIn("-ltpudnn", commands[-1][1])

    def test_bundle_rejects_source_tamper_unexpected_paths_and_wrong_topology(self):
        bundle = self.bundle()
        legacy.validate_source_bundle(bundle)
        bad = copy.deepcopy(bundle)
        bad["variants"]["serial"]["sources"]["kernel.c"] += "modified"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            legacy.validate_source_bundle(bad)
        bad = copy.deepcopy(bundle)
        bad["variants"]["pipeline"]["sources"]["../escape"] = ""
        with self.assertRaisesRegex(ValueError, "source names"):
            legacy.validate_source_bundle(bad)
        bad = copy.deepcopy(bundle)
        bad["target"]["launch_cores"] = 8
        with self.assertRaisesRegex(ValueError, "single-core"):
            legacy.validate_source_bundle(bad)

    def test_standalone_compile_failure_preserves_diagnostics_and_stops(self):
        bundle = write(self.root / "bundle.json", json.dumps(self.bundle()))
        output = self.root / "build"
        process = subprocess.run([sys.executable, str(SOURCE), "--bundle", str(bundle),
                                  "--output", str(output)], env={**os.environ, **self.env},
                                 capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["kernel_launches"], 0)
        self.assertFalse(result["board_runtime_loaded"])
        self.assertEqual(result["completed_commands"], [])
        self.assertEqual(result["active_command"]["variant"], "original")
        self.assertIn("deliberate-compile-failure", (output / "original/0.log").read_text())
        self.assertFalse((output / "serial").exists())
        saved = (output / "result.json").read_bytes()
        repeat = subprocess.run(process.args, env={**os.environ, **self.env}, capture_output=True, timeout=15)
        self.assertNotEqual(repeat.returncode, 0)
        self.assertEqual((output / "result.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
