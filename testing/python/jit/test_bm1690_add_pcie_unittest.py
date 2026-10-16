"""Pinned-manifest, one-call ABI and supervisor tests; no vendor runtime loaded."""

import contextlib
import ctypes
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("add_pcie_test_target", ROOT / "tpu_demo/pipeline/run_add_pcie.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write(path, data=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class ManifestTests(unittest.TestCase):
    bundle_name = "add-smoke-sources.json"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(ROOT / "tilelang/jit/adapter/legacy_pcie.py", self.root / "build.py")
        shutil.copyfile(ROOT / "research/bm1690-pipelines/handoff" / self.bundle_name, self.root / "add.json")
        self.bundle_hash = runner.digest(self.root / "add.json")
        spec = importlib.util.spec_from_file_location("legacy_pcie_test_fixture", self.root / "build.py")
        builder = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = builder
        spec.loader.exec_module(builder)
        sdk, runtime = self.root / "sdk", self.root / "runtime"
        for name in ("runtime/bm1690/TPU1686/kernel/include/tpu_kernel.h", "runtime/kernel/common.h",
                     "runtime/customize/include/ppl_mem.h", "runtime/customize/include/ppl_helper.h",
                     "runtime/customize/src/ppl_helper.c", "runtime/bm1690/lib/libbm1690.a"):
            write(sdk / name)
        write(runtime / "include/tpuv7_rt.h")
        write(runtime / "lib/libtpuv7_rt.so")
        cross = write(self.root / "cross/bin/riscv64-unknown-linux-gnu-gcc")
        cross.chmod(0o700)
        self.env = {"TILELANG_TPU_PCIE_RUNTIME_PATH": str(runtime / "lib"),
                    "TILELANG_TPU_PCIE_CROSS_GCC": str(cross)}
        layout = builder.resolve_legacy_pcie(sdk, "bm1690", environment=self.env)
        self.build = self.root / "build"
        self.build.mkdir()
        bundle = json.loads((self.root / "add.json").read_text())
        self.report = {"status": "compile_only_passed", "profile": builder.PROFILE,
                       "bundle_sha256": self.bundle_hash, "builder_sha256": runner.digest(self.root / "build.py"),
                       "board_runtime_loaded": False, "kernel_launches": 0,
                       "runtime_identity": list(layout.runtime_identity_for("pcie", self.env)),
                       "completed_commands": [], "artifacts": {}}
        inputs = (layout.firmware_archive, layout.ppl_helper_source, layout.kernel_include / "tpu_kernel.h",
                  layout.runtime_include / "tpuv7_rt.h", runtime / "lib/libtpuv7_rt.so", cross)
        self.report["sdk_inputs"] = {str(path): runner.digest(path) for path in inputs}
        for variant in runner.VARIANTS:
            directory = self.build / variant
            directory.mkdir()
            for name, content in bundle["variants"][variant]["sources"].items():
                (directory / name).write_text(content)
            self.report["artifacts"][variant] = {}
            # Minimal ELF headers are adequate for manifest tests, never dlopened.
            for name,machine in (("main.so",62),("libkernel.so",243)):
                artifact = write(directory / name, b"\x7fELF\x02\x01" + bytes(10) + struct.pack("<HH",3,machine))
                self.report["artifacts"][variant][name] = runner.digest(artifact)
            with patch.dict(os.environ, self.env):
                for number,(label,command) in enumerate(builder.legacy_pcie_commands(layout,directory)):
                    self.report["completed_commands"].append({"variant":variant,"label":label,"command":command,
                                                              "log":str(directory / f"{number}.log")})
        self.save()

    def save(self):
        (self.build / "result.json").write_text(json.dumps(self.report))

    def test_valid_build_identity_requires_no_library_loading(self):
        with patch.object(ctypes, "CDLL", side_effect=AssertionError("must not load")):
            identity = runner.validate_build(self.build)
        self.assertEqual(identity["bundle_sha256"], self.bundle_hash)
        self.assertEqual(identity["runtime_library"], str(self.root / "runtime/lib/libtpuv7_rt.so"))

    def test_source_binary_and_runtime_changes_are_rejected(self):
        for path in (self.build / "original/kernel.c", self.build / "pipeline/libkernel.so",
                     self.root / "runtime/lib/libtpuv7_rt.so", self.root / "build.py", self.root / "add.json"):
            with self.subTest(path=path):
                before = path.read_bytes()
                path.write_bytes(before + b"changed")
                try:
                    with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                        runner.validate_build(self.build)
                finally:
                    path.write_bytes(before)

    def test_incomplete_or_moved_build_and_wrong_elf_are_rejected(self):
        last = self.report["completed_commands"].pop()
        self.save()
        with self.assertRaisesRegex(ValueError,"commands differ"):
            runner.validate_build(self.build)
        self.report["completed_commands"].append(last)
        self.save()
        moved = self.root / "moved"
        self.build.rename(moved)
        with self.assertRaisesRegex(ValueError,"commands differ"):
            runner.validate_build(moved)
        moved.rename(self.build)
        binary = self.build / "original/libkernel.so"
        binary.write_bytes(b"\x7fELF\x02\x01" + bytes(10) + struct.pack("<HH",3,62))
        self.report["artifacts"]["original"]["libkernel.so"] = runner.digest(binary)
        self.save()
        with self.assertRaisesRegex(ValueError,"Unexpected ELF target"):
            runner.validate_build(self.build)

    def test_previous_variant_requires_passed_matching_receipt(self):
        previous = self.root / "previous"
        previous.mkdir()
        output = write(previous / "output.f16", runner.test_vectors()[2])
        identity = runner.validate_build(self.build)
        lhs,rhs,_ = runner.test_vectors()
        input_hash = hashlib.sha256(lhs+rhs).hexdigest()
        numeric = {"variant":"original","bundle_sha256":self.bundle_hash,
                   "build_manifest_sha256":identity["build_manifest_sha256"], "input_sha256":input_hash,
                   "device_id":0,"expected_pci":"0000:01:00.0", "output_sha256":runner.digest(output)}
        args = SimpleNamespace(variant="serial",previous=previous,device_id=0,expected_pci="0000:01:00.0")
        for status in ("failed","passed"):
            (previous / "result.json").write_text(json.dumps({"status":status,"numeric":numeric}))
            if status == "failed":
                with self.assertRaisesRegex(ValueError,"did not pass"):
                    runner.check_previous(args,identity,input_hash)
            else:
                self.assertEqual(runner.check_previous(args,identity,input_hash),output.read_bytes())
        args.variant = "pipeline"
        with self.assertRaisesRegex(ValueError,"wrong variant"):
            runner.check_previous(args,identity,input_hash)

    def test_public_entry_guard_worker_and_journal_with_no_vendor_library(self):
        output = self.root / "run"
        environment = dict(os.environ)
        for key in ("LD_LIBRARY_PATH","LD_PRELOAD","LD_AUDIT","TPU_RT_CORE_NUM"):
            environment.pop(key,None)
        command = [sys.executable,str(ROOT / "tpu_demo/pipeline/run_add_pcie.py"),
                   "--build",str(self.build),"--output",str(output),"--variant","original",
                   "--allow-pcie","--device-id","0","--expected-pci","0000:01:00.0"]
        result = subprocess.run(command,env=environment,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        report = json.loads((output / "result.json").read_text())
        self.assertEqual(report["numeric"]["last_stage"],"load_verified_board_runtime")
        self.assertIn("file too short",report["numeric"]["error"])
        self.assertFalse(report["numeric"]["dispatch_attempted"])
        self.assertTrue(report["cleanup_complete"])


FAKE_HOST = r'''
#include <cstdlib>
#include <cstring>
#include <cstdint>
// DATA
static int calls = 0;
extern "C" int get_calls() { return calls; }
extern "C" int tilelang_tpu_run(void **args) {
  ++calls;
  const char *mode = getenv("ADD_TEST_MODE");
  if (mode && !strcmp(mode,"status")) return 7;
  auto a = static_cast<uint16_t *>(args[0]);
  auto b = static_cast<uint16_t *>(args[1]);
  auto c = static_cast<uint16_t *>(args[2]);
  if (memcmp(a, lhs, sizeof(lhs)) || memcmp(b, rhs, sizeof(rhs))) return 99;
  int n = mode && !strcmp(mode,"partial") ? 1023 : 1024;
  memcpy(c, reference, n * sizeof(uint16_t));
  if (mode && !strcmp(mode,"canary")) c[1024] = 0;
  return 0;
}
'''


@unittest.skipUnless(shutil.which("c++"),"host C++ compiler required")
class HostABITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        # Test the actual pointer ABI/staging with a canned response. Numerical
        # addition is independently checked by Torch and the CModel worker.
        data = "\n".join("static const uint16_t " + name + "[] = {" +
                         ",".join(str(v[0]) for v in struct.iter_unpack("<H", value)) + "};"
                         for name,value in zip(("lhs","rhs","reference"),runner.test_vectors()))
        source = write(root / "fake.cpp", FAKE_HOST.replace("// DATA", data).encode())
        library = root / "fake.so"
        subprocess.run(["c++","-shared","-fPIC",str(source),"-o",str(library)],check=True,timeout=30)
        cls.library = ctypes.CDLL(str(library))

    def test_one_call_with_expected_host_buffer_layout(self):
        lhs,rhs,expected = runner.test_vectors()
        before = self.library.get_calls()
        with patch.dict(os.environ, {"ADD_TEST_MODE":""}):
            actual = runner.call_host(self.library,lhs,rhs)
        self.assertEqual(self.library.get_calls(),before+1)
        self.assertEqual(actual,expected)
        self.assertTrue(runner.check_output(actual,expected)["passed"])

    def test_status_partial_output_and_host_overrun_fail(self):
        lhs,rhs,expected = runner.test_vectors()
        for mode in ("status","partial","canary"):
            with self.subTest(mode=mode),patch.dict(os.environ, {"ADD_TEST_MODE":mode}):
                with self.assertRaises((ValueError,RuntimeError)):
                    runner.check_output(runner.call_host(self.library,lhs,rhs),expected)


class SupervisorTests(unittest.TestCase):
    def test_bounded_deadline_is_forwarded_to_child(self):
        args = SimpleNamespace(build=Path("build"), output=Path("output"), variant="original",
                               previous=None, device_id=0, expected_pci="0000:01:00.0")
        self.assertEqual(runner.child_command(args)[-2:], ["--timeout-seconds", "30"])
        args.timeout_seconds = 90
        self.assertEqual(runner.child_command(args)[-2:], ["--timeout-seconds", "90"])

    def run_program(self, source, timeout=3, max_rss=512*1024**2):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = runner.supervise([sys.executable,"-c",source],output,timeout_s=timeout,max_rss=max_rss)
            return rc,json.loads((output / "result.json").read_text())

    def test_success_exit_without_numeric_result_is_failure(self):
        rc,result = self.run_program("print('not a correctness result')")
        self.assertEqual(rc,1)
        self.assertEqual(result["returncode"],0)

    def test_timeout_kills_descendants(self):
        rc,result = self.run_program(
            "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']); time.sleep(10)",
            timeout=0.3)
        self.assertEqual(rc,1)
        self.assertEqual(result["reason"],"timeout")
        self.assertTrue(result["cleanup_complete"])

    def test_memory_limit_is_failure(self):
        rc,result = self.run_program("import time; data=bytearray(64*1024**2); time.sleep(10)",max_rss=40*1024**2)
        self.assertEqual(rc,1)
        self.assertEqual(result["reason"],"rss_limit")

    def test_guard_cleans_worker_group_when_outer_parent_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready,pidfile = root / "ready",root / "guard-pid"
            child = "import subprocess,sys,time; from pathlib import Path; " + \
                    "subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); " + \
                    f"Path({str(ready)!r}).write_text('ready'); time.sleep(20)"
            helper = root / "guard.py"
            helper.write_text("import importlib.util,sys; from types import SimpleNamespace\n" +
                              f"s=importlib.util.spec_from_file_location('guard_test',{str(ROOT / 'tpu_demo/pipeline/run_add_pcie.py')!r})\n" +
                              "r=importlib.util.module_from_spec(s); s.loader.exec_module(r)\n" +
                              f"r.child_command=lambda args:[sys.executable,'-c',{child!r}]\n" +
                              "raise SystemExit(r.guard(SimpleNamespace(guard_parent=int(sys.argv[1]),lock_fd=None)))\n")
            parent = "import os,subprocess,sys,time; from pathlib import Path\n" + \
                     f"p=subprocess.Popen([sys.executable,{str(helper)!r},str(os.getpid())],start_new_session=True)\n" + \
                     f"Path({str(pidfile)!r}).write_text(str(p.pid))\n" + \
                     f"while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n" + \
                     "os._exit(0)\n"
            try:
                subprocess.run([sys.executable,"-c",parent],stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,timeout=5,check=True)
                group = int(pidfile.read_text())
                deadline = time.monotonic()+3
                while runner.group_rss_bytes(group) and time.monotonic()<deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.is_file())
                self.assertEqual(runner.group_rss_bytes(group),0)
            finally:
                if pidfile.exists():
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(int(pidfile.read_text()),9)


class MappingTests(unittest.TestCase):
    def runtime(self, mode):
        calls = []
        runtime = SimpleNamespace()
        for name in ("tpuRtInit","tpuRtGetDeviceCount","tpuRtSetDevice","tpuRtGetDevice","tpuRtGetFd"):
            def fn(*args, name=name):
                calls.append(name)
                if name == "tpuRtInit" and mode == "init_error":
                    return 2
                if name == "tpuRtGetDeviceCount":
                    args[0]._obj.value = 3 if mode == "topology" else 2
                if name == "tpuRtGetDevice":
                    args[0]._obj.value = 1 if mode == "wrong_device" else 0
                if name == "tpuRtGetFd":
                    args[0]._obj.value = 3
                return 0
            setattr(runtime,name,fn)
        return runtime,calls

    def test_failed_queries_stop_before_the_next_operation(self):
        for mode,next_call in (("init_error","tpuRtGetDeviceCount"),
                               ("topology","tpuRtSetDevice"),("wrong_device","tpuRtGetFd")):
            with self.subTest(mode=mode):
                runtime,calls = self.runtime(mode)
                with self.assertRaises((ValueError,RuntimeError)):
                    runner.verify_device(runtime,0,"0000:01:00.0",lambda stage:None)
                self.assertNotIn(next_call,calls)

    def test_wrong_sysfs_mapping_is_rejected(self):
        runtime,_ = self.runtime("")
        with patch.object(os,"fstat",return_value=SimpleNamespace(st_mode=0o020000,st_rdev=os.makedev(511,1))), \
             patch.object(Path,"resolve",side_effect=[Path("/sys/devices/0000:01:00.1/sg-host-drv/sg-host-drv-1"),
                                                    Path("/sys/devices/0000:01:00.0")]):
            with self.assertRaisesRegex(ValueError,"mapping mismatch"):
                runner.verify_device(runtime,0,"0000:01:00.0",lambda stage:None)


class TargetManifestTests(ManifestTests):
    bundle_name = "add-1024-sources.json"

    def test_timing_receipt_requires_same_variant_build_inputs_and_output(self):
        output = self.root / "correctness"
        output.mkdir()
        payload = b"\x00\x00"
        (output / "output.f16").write_bytes(payload)
        identity = runner.validate_build(self.build)
        numeric = {"status": "passed", "reference": {"passed": True}, "variant": "original",
                   "shape": [1024,1024], "build_manifest_sha256": identity["build_manifest_sha256"],
                   "bundle_sha256": self.bundle_hash, "input_sha256": "input",
                   "device_id": 0, "expected_pci": "0000:01:00.0",
                   "output_sha256": runner.digest(output / "output.f16")}
        args = SimpleNamespace(correctness=output,variant="original",device_id=0,expected_pci="0000:01:00.0")
        def check(value):
            (output / "result.json").write_text(json.dumps({"status":"passed","numeric":value}))
            return runner.check_correctness_receipt(args,identity,"input",payload)
        self.assertTrue(check(numeric))
        for key,value in (("variant","pipeline"),("build_manifest_sha256","wrong"),
                          ("input_sha256","wrong"),("timing",{"samples":20}),
                          ("reference",{"passed":False}),("device_id",1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check({**numeric,key:value})
        (output / "output.f16").write_bytes(b"xx")
        with self.assertRaisesRegex(ValueError,"Hash mismatch"):
            check(numeric)


class CoarseManifestTests(TargetManifestTests):
    bundle_name = "add-1024-coarse-sources.json"


class SubManifestTests(TargetManifestTests):
    bundle_name = "sub-1024-sources.json"
    operation = "sub"

    def test_operation_is_bound_to_registered_bundle(self):
        identity = runner.validate_build(self.build)
        self.assertEqual(identity["operation"], self.operation)
        self.assertEqual(identity["bundle_sha256"], runner.ELEMENTWISE_BUNDLES[self.operation])
        bundle = json.loads((self.root / "add.json").read_text())
        bundle["target"]["operation"] = "add"
        (self.root / "add.json").write_text(json.dumps(bundle))
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            runner.validate_build(self.build)


class MulManifestTests(SubManifestTests):
    bundle_name = "mul-1024-sources.json"
    operation = "mul"


class DivManifestTests(SubManifestTests):
    bundle_name = "div-1024-sources.json"
    operation = "div"

    def test_division_timing_receipt_accepts_tolerance_but_rejects_wrong_output(self):
        directory = self.root / "div-correctness"
        directory.mkdir()
        identity = runner.validate_build(self.build)
        expected = struct.pack("<e", 1.0)
        args = SimpleNamespace(correctness=directory, variant="original", device_id=0,
                               expected_pci="0000:01:00.0")
        def check(value):
            (directory / "output.f16").write_bytes(struct.pack("<e", value))
            numeric = {"status":"passed", "reference":{"passed":True}, "variant":"original",
                       "shape":[1024,1024], "bundle_sha256":self.bundle_hash,
                       "build_manifest_sha256":identity["build_manifest_sha256"], "input_sha256":"input",
                       "device_id":0, "expected_pci":"0000:01:00.0",
                       "output_sha256":runner.digest(directory / "output.f16")}
            (directory / "result.json").write_text(json.dumps({"status":"passed", "numeric":numeric}))
            return runner.check_correctness_receipt(args, identity, "input", expected)
        self.assertTrue(check(1.0009765625))
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            check(2.0)
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            check(float("nan"))


class ElementwiseContractTests(unittest.TestCase):
    def test_historical_add_inputs_and_output_are_preserved(self):
        a,b,expected = runner.test_vectors()
        self.assertEqual(hashlib.sha256(a+b).hexdigest(),
                         "da54a02c5b51968628f2547e4117871bfef6a291cb6ce1255779d61aafa6ad04")
        self.assertEqual(hashlib.sha256(expected).hexdigest(),
                         "fee0544267cdc5cd353a903b9fbac82fdcb80100c8bce9deac58cae10f99226f")

    def test_division_domain_and_existing_demo_tolerances(self):
        _,rhs,expected = runner.test_vectors(operation="div")
        self.assertTrue(all(0.5 <= x[0] <= 2.0 for x in struct.iter_unpack("<e", rhs)))
        for op,tol in (("add",0.005),("sub",0.005),("mul",0.005),("div",0.01)):
            metrics = runner.check_output(expected,expected,operation=op)
            self.assertEqual((metrics["atol"],metrics["rtol"]),(tol,tol))
        with self.assertRaises(ValueError): runner.test_vectors(operation="unknown")
        with self.assertRaises(ValueError): runner.check_output(expected,expected,operation="unknown")


class MatmulManifestTests(ManifestTests):
    bundle_name = "matmul-smoke-sources.json"
    dimension = 128

    def test_matrix_contract_and_timing_receipt(self):
        identity = runner.validate_build(self.build)
        self.assertEqual(identity["operation"], "matmul")
        self.assertEqual(identity["shape"], [self.dimension] * 2)
        self.assertEqual(identity["bundle_sha256"], runner.MATMUL_BUNDLE_SHA256 if self.dimension == 128
                         else runner.MATMUL_TARGET_BUNDLE_SHA256)
        directory = self.root / "correctness"
        directory.mkdir()
        expected = struct.pack("<e", 1.0)
        args = SimpleNamespace(correctness=directory, variant="original", device_id=0,
                               expected_pci="0000:01:00.0")
        for value, passes in ((1.0009765625, True), (2.0, False), (float("nan"), False)):
            (directory / "output.f16").write_bytes(struct.pack("<e", value))
            numeric = {"status": "passed", "reference": {"passed": True}, "variant": "original",
                       "shape": identity["shape"], "bundle_sha256": self.bundle_hash,
                       "build_manifest_sha256": identity["build_manifest_sha256"], "input_sha256": "input",
                       "device_id": 0, "expected_pci": "0000:01:00.0",
                       "output_sha256": runner.digest(directory / "output.f16")}
            (directory / "result.json").write_text(json.dumps({"status": "passed", "numeric": numeric}))
            if passes:
                self.assertTrue(runner.check_correctness_receipt(args, identity, "input", expected))
            else:
                with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                    runner.check_correctness_receipt(args, identity, "input", expected)


class MatmulTargetManifestTests(MatmulManifestTests):
    bundle_name = "matmul-1024-sources.json"
    dimension = 1024


class RMSNormManifestTests(ManifestTests):
    bundle_name = "rmsnorm-smoke-sources.json"
    operation = "rmsnorm"
    shape = [16, 128]

    def test_rmsnorm_receipt_rechecks_tolerance_before_timing(self):
        identity = runner.validate_build(self.build)
        self.assertEqual(identity["operation"], self.operation)
        self.assertEqual(identity["shape"], self.shape)
        bundles = runner.RMSNORM_BUNDLES if self.shape == [16, 128] else runner.RMSNORM_TARGET_BUNDLES
        self.assertEqual(identity["bundle_sha256"], bundles[self.operation])
        directory = self.root / "correctness"
        directory.mkdir()
        args = SimpleNamespace(correctness=directory, variant="original", device_id=0,
                               expected_pci="0000:01:00.0")
        expected = struct.pack("<e", 1.0)
        for value, passes in ((1.0009765625, True), (2.0, False), (float("nan"), False)):
            (directory / "output.f16").write_bytes(struct.pack("<e", value))
            numeric = {"status":"passed", "reference":{"passed":True}, "variant":"original",
                       "shape":identity["shape"], "bundle_sha256":self.bundle_hash,
                       "build_manifest_sha256":identity["build_manifest_sha256"], "input_sha256":"input",
                       "device_id":0, "expected_pci":"0000:01:00.0",
                       "output_sha256":runner.digest(directory / "output.f16")}
            (directory / "result.json").write_text(json.dumps({"status":"passed", "numeric":numeric}))
            if passes:
                self.assertTrue(runner.check_correctness_receipt(args, identity, "input", expected))
            else:
                with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                    runner.check_correctness_receipt(args, identity, "input", expected)


class RMSNormSplitKManifestTests(RMSNormManifestTests):
    bundle_name = "rmsnorm-splitk-smoke-sources.json"
    operation = "rmsnorm-splitk"


class RMSNormTargetManifestTests(RMSNormManifestTests):
    bundle_name = "rmsnorm-1024-sources.json"
    shape = [1024, 1024]


class RMSNormSplitKTargetManifestTests(RMSNormTargetManifestTests):
    bundle_name = "rmsnorm-splitk-1024-sources.json"
    operation = "rmsnorm-splitk"


class RoPEManifestTests(ManifestTests):
    bundle_name = "rope-smoke-sources.json"
    shape = [8, 128]

    def test_receipt_rechecks_rope_tolerance_before_timing(self):
        identity = runner.validate_build(self.build)
        self.assertEqual(identity["operation"], "rope")
        self.assertEqual(identity["shape"], self.shape)
        self.assertTrue(identity["timing_supported"])
        directory = self.root / "correctness"
        directory.mkdir()
        args = SimpleNamespace(correctness=directory, variant="original", device_id=0,
                               expected_pci="0000:01:00.0")
        expected = struct.pack("<e", 1.0)
        for value, passes in ((1.0009765625, True), (1.02, False), (float("nan"), False)):
            (directory / "output.f16").write_bytes(struct.pack("<e", value))
            numeric = {"status":"passed", "reference":{"passed":True}, "variant":"original",
                       "shape":identity["shape"], "bundle_sha256":self.bundle_hash,
                       "build_manifest_sha256":identity["build_manifest_sha256"], "input_sha256":"input",
                       "device_id":0, "expected_pci":"0000:01:00.0",
                       "output_sha256":runner.digest(directory / "output.f16")}
            (directory / "result.json").write_text(json.dumps({"status":"passed", "numeric":numeric}))
            if passes:
                self.assertTrue(runner.check_correctness_receipt(args, identity, "input", expected))
            else:
                with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                    runner.check_correctness_receipt(args, identity, "input", expected)


class RoPETargetManifestTests(RoPEManifestTests):
    bundle_name = "rope-1024-sources.json"
    shape = [1024, 1024]


class SwiGLUManifestTests(ManifestTests):
    bundle_name = "swiglu-smoke-sources.json"

    def test_fixed_precision_contract_and_timing_receipt(self):
        identity = runner.validate_build(self.build)
        self.assertEqual(identity["operation"], "swiglu")
        directory = self.root / "correctness"
        directory.mkdir()
        args = SimpleNamespace(correctness=directory, variant="original", device_id=0,
                               expected_pci="0000:01:00.0")
        expected = struct.pack("<e", 1.0)
        for value, passes in ((1.0009765625, True), (1.1, False), (float("nan"), False)):
            (directory / "output.f16").write_bytes(struct.pack("<e", value))
            numeric = dict(status="passed", variant="original", reference={"passed":True},
                           shape=identity["shape"], bundle_sha256=self.bundle_hash,
                           build_manifest_sha256=identity["build_manifest_sha256"], input_sha256="input",
                           device_id=0, expected_pci="0000:01:00.0",
                           output_sha256=runner.digest(directory / "output.f16"))
            (directory / "result.json").write_text(json.dumps({"status":"passed", "numeric":numeric}))
            if passes:
                self.assertTrue(runner.check_correctness_receipt(args, identity, "input", expected))
            else:
                with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                    runner.check_correctness_receipt(args, identity, "input", expected)


class SwiGLUTargetManifestTests(SwiGLUManifestTests):
    bundle_name = "swiglu-1024-sources.json"


class SwiGLUReferenceTests(unittest.TestCase):
    def test_scalar_reference_matches_formula_and_preserves_signed_inputs(self):
        gate, up, output = runner.test_vectors(1024, "swiglu")
        g, u, result = [[x[0] for x in struct.iter_unpack("<e", data)] for data in (gate,up,output)]
        self.assertEqual((min(g), max(g)), (-3.0,3.0))
        self.assertTrue(min(u) < 0 < max(u))
        self.assertEqual(result[1:2], [-0.0])
        for x,y,z in zip(g,u,result):
            self.assertAlmostEqual(z, y*x/(1+math.exp(-x)), delta=0.002)
        self.assertEqual(runner.check_output(output,output,1024,"swiglu")["atol"],0.01)
        with self.assertRaises(ValueError):
            runner.check_output(struct.pack("<e",float("nan"))+output[2:],output,1024,"swiglu")
        with self.assertRaisesRegex(ValueError, "differs bitwise"):
            runner.check_previous_output(struct.pack("<e",2.0),struct.pack("<e",1.0),1,"swiglu","pipeline")

    def test_reference_rejects_wrong_sizes_nonfinite_and_out_of_domain_gate(self):
        gate, up, _ = runner.test_vectors(1024, "swiglu")
        for invalid in (gate[:-2], struct.pack("<e",float("nan"))+gate[2:], struct.pack("<e",4.0)+gate[2:]):
            with self.assertRaises(ValueError):
                runner.swiglu_reference(invalid,up)
        with self.assertRaises(ValueError):
            runner.test_vectors(100,"swiglu")


class RoPEOriginalFailureTests(unittest.TestCase):
    """Continuation preserves failed-baseline evidence; it never approves that baseline."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.failure = self.root / "failed-original"
        self.failure.mkdir()
        self.identity = dict(operation="rope", shape=[1024, 1024], diagnostic_only=False,
                             bundle_sha256=runner.ROPE_TARGET_BUNDLE_SHA256,
                             build_manifest_sha256="build", runtime_library="/verified/runtime.so",
                             runtime_sha256="runtime")
        self.args = SimpleNamespace(variant="serial", previous=None, original_failure=self.failure,
                                    device_id=0, expected_pci="0000:01:00.0", build=self.root / "build",
                                    output=self.root / "run", measure=False)
        self.numeric = dict(self.identity, variant="original", dtype="float16", launch_cores=1,
                            requested_host_calls=1, requested_kernel_calls=1, dispatch_attempted=True,
                            last_stage="tilelang_tpu_run_once", status="running", input_sha256="input",
                            device_id=0, expected_pci="0000:01:00.0")
        self.report = dict(status="failed", reason="timeout", returncode=-9,
                           cleanup_complete=True, numeric=self.numeric)
        self.save()

    def save(self):
        (self.failure / "result.json").write_text(json.dumps(self.report))

    def context(self):
        context = {}
        self.assertIsNone(runner.check_previous(self.args, self.identity, "input", context))
        return context

    def serial_receipt(self, context):
        directory = self.root / "serial"
        directory.mkdir()
        payload = struct.pack("<e", 1.0)
        (directory / "output.f16").write_bytes(payload)
        numeric = dict(self.numeric, **context)
        numeric.update(status="passed", variant="serial", reference={"passed": True},
                       output_sha256=runner.digest(directory / "output.f16"))
        (directory / "result.json").write_text(json.dumps({"status":"passed", "numeric":numeric}))
        return directory, numeric, payload

    def test_continuation_records_failure_and_never_claims_three_variant_acceptance(self):
        context = self.context()
        self.assertEqual(context["original_baseline_status"], "failed")
        self.assertFalse(context["three_variant_acceptance_complete"])
        self.assertEqual(context["original_baseline_failure"]["receipt_sha256"],
                         runner.digest(self.failure / "result.json"))

    def test_scope_is_full_pinned_rope_serial_only(self):
        for key, value in (("operation", "add"), ("shape", [8, 128]), ("diagnostic_only", True),
                           ("bundle_sha256", runner.ROPE_TRACE_BUNDLE_SHA256)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.check_previous(self.args, {**self.identity, key:value}, "input")
        for variant in ("original", "pipeline"):
            self.args.variant = variant
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.context()
        self.args.variant = "serial"
        self.args.previous = self.root / "somewhere"
        with self.assertRaises(ValueError):
            self.context()

    def test_failed_receipt_must_match_build_input_hardware_and_runtime(self):
        for key, value in (("build_manifest_sha256", "other"), ("bundle_sha256", "other"),
                           ("input_sha256", "other"), ("device_id", 1), ("expected_pci", "other"),
                           ("runtime_sha256", "other"), ("runtime_library", "other"),
                           ("shape", [8,128]), ("launch_cores", 8), ("dtype", "float32"),
                           ("variant", "serial"), ("diagnostic_only", True),
                           ("dispatch_attempted", False), ("requested_kernel_calls", 25),
                           ("last_stage", "validate_build"), ("reference", {"passed":True})):
            self.report["numeric"] = {**self.numeric, key:value}
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.context()

    def test_only_timeout_with_completed_process_cleanup_is_accepted(self):
        for key, value in (("status", "passed"), ("reason", "rss_limit"),
                           ("cleanup_complete", False), ("returncode", 0)):
            old = self.report[key]
            self.report[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.context()
            self.report[key] = old

    def test_missing_failed_receipt_does_not_bypass_original(self):
        self.args.original_failure = None
        with self.assertRaisesRegex(ValueError, "requires --previous"):
            self.context()

    def test_pipeline_inherits_evidence_and_still_requires_matching_serial_output(self):
        context = self.context()
        directory, numeric, payload = self.serial_receipt(context)
        self.args.variant, self.args.original_failure, self.args.previous = "pipeline", None, directory
        inherited = {}
        self.assertEqual(runner.check_previous(self.args, self.identity, "input", inherited), payload)
        self.assertEqual(inherited, context)
        with self.assertRaisesRegex(ValueError, "differs bitwise"):
            runner.check_previous_output(struct.pack("<e", 2.0), payload, 1, "rope", "pipeline")
        numeric["reference"]["passed"] = False
        (directory / "result.json").write_text(json.dumps({"status":"passed", "numeric":numeric}))
        with self.assertRaisesRegex(ValueError, "independent-reference"):
            runner.check_previous(self.args, self.identity, "input")

    def test_changed_failure_provenance_cannot_be_inherited(self):
        directory, _, _ = self.serial_receipt(self.context())
        self.args.variant, self.args.original_failure, self.args.previous = "pipeline", None, directory
        self.report["extra"] = "changed"
        self.save()
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            runner.check_previous(self.args, self.identity, "input")

    def test_timing_remains_bound_to_same_failed_baseline_context(self):
        context = self.context()
        directory, _, payload = self.serial_receipt(context)
        self.args.correctness = directory
        self.assertTrue(runner.check_correctness_receipt(self.args, self.identity, "input", payload, context))
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            runner.check_correctness_receipt(self.args, self.identity, "input", payload)

    def test_guard_and_worker_receive_failure_option(self):
        command = runner.child_command(self.args)
        self.assertEqual(command[command.index("--original-failure") + 1], str(self.failure))
        self.assertNotIn("--previous", command)


class RoPETraceManifestTests(ManifestTests):
    bundle_name = "rope-1024-host-trace-sources.json"

    def test_trace_identity_excludes_timing(self):
        identity = runner.validate_build(self.build)
        self.assertTrue(identity["diagnostic_only"])
        self.assertFalse(identity["timing_supported"])

    def test_other_variants_and_measurement_fail_before_runtime_loading(self):
        environment = dict(os.environ)
        for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "TPU_RT_CORE_NUM"):
            environment.pop(key, None)
        for variant, extra in (("serial", []), ("pipeline", []),
                               ("original", ["--measure", "--correctness", str(self.root / "unused")])):
            output = self.root / ("rejected-" + variant)
            command = [sys.executable, str(ROOT / "tpu_demo/pipeline/run_add_pcie.py"),
                       "--build", str(self.build), "--output", str(output), "--variant", variant,
                       "--allow-pcie", "--device-id", "0", "--expected-pci", "0000:01:00.0", *extra]
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads((output / "result.json").read_text())
            self.assertEqual(report["numeric"]["last_stage"], "validate_build")
            self.assertIn("original one-call", report["numeric"]["error"])
            self.assertFalse(report["numeric"]["dispatch_attempted"])
            self.assertTrue(report["cleanup_complete"])


class RoPEWindowManifestTests(RoPETraceManifestTests):
    bundle_name = "rope-1024-window8-sources.json"

    def test_window_receipt_is_bound_to_full_allocation_and_exact_coverage(self):
        identity = runner.validate_build(self.build)
        self.assertEqual(identity["shape"], [1024, 1024])
        self.assertEqual(identity["diagnostic_window"], [0, 8, 0, 128])
        self.assertNotEqual(identity["bundle_sha256"], runner.ROPE_TARGET_BUNDLE_SHA256)
        self.assertNotEqual(identity["bundle_sha256"], runner.ROPE_TRACE_BUNDLE_SHA256)


class RoPEWindowEndManifestTests(RoPETraceManifestTests):
    bundle_name = "rope-1024-window8-end-sources.json"

    def test_end_window_receipt_has_distinct_bundle_and_coverage(self):
        identity = runner.validate_build(self.build)
        self.assertEqual(identity["shape"], [1024, 1024])
        self.assertEqual(identity["diagnostic_window"], [1016, 1024, 896, 1024])
        self.assertEqual(identity["bundle_sha256"], runner.ROPE_WINDOW_END_BUNDLE_SHA256)
        self.assertNotEqual(identity["bundle_sha256"], runner.ROPE_WINDOW_BUNDLE_SHA256)


class RoPEPrefix128ManifestTests(RoPETraceManifestTests):
    bundle_name = "rope-1024-prefix128-sources.json"
    prefix_tiles = 128

    def test_prefix_coverage_is_bound_to_its_distinct_bundle(self):
        identity = runner.validate_build(self.build)
        self.assertEqual(identity["bundle_sha256"], runner.ROPE_PREFIX_BUNDLES[self.prefix_tiles])
        self.assertEqual(identity["diagnostic_window"], [0, self.prefix_tiles // 8, 0, 1024])
        self.assertEqual(len(set(runner.ROPE_PREFIX_BUNDLES.values())), 6)


class RoPEPrefix256ManifestTests(RoPEPrefix128ManifestTests):
    bundle_name = "rope-1024-prefix256-sources.json"
    prefix_tiles = 256


class RoPEPrefix512ManifestTests(RoPEPrefix128ManifestTests):
    bundle_name = "rope-1024-prefix512-sources.json"
    prefix_tiles = 512


class RoPEPrefix1024ManifestTests(RoPEPrefix128ManifestTests):
    bundle_name = "rope-1024-prefix1024-sources.json"
    prefix_tiles = 1024


class RoPEPrefix2048ManifestTests(RoPEPrefix128ManifestTests):
    bundle_name = "rope-1024-prefix2048-sources.json"
    prefix_tiles = 2048


class RoPEPrefix4096ManifestTests(RoPEPrefix128ManifestTests):
    bundle_name = "rope-1024-prefix4096-sources.json"
    prefix_tiles = 4096


class RoPEPrefixOutputTests(unittest.TestCase):
    def test_prefix_boundary_requires_written_values_and_preserves_every_later_byte(self):
        expected = struct.pack("<e", 1.0) * (1024 * 1024)
        for tiles in runner.ROPE_PREFIX_BUNDLES:
            count = tiles * 128
            output = bytearray(expected[:count * 2] + b"\x00\x7e" * (1024 * 1024 - count))
            window = runner.ROPE_PREFIX_WINDOWS[tiles]
            metrics = runner.check_rope_window_output(output, expected, window)
            self.assertEqual(metrics["checked_elements"], count)
            self.assertEqual(metrics["untouched_elements"], 1024 * 1024 - count)
            self.assertTrue(metrics["untouched_sentinel_preserved"])
            for index, value in ((count - 1, b"\x00\x7e"), (0, struct.pack("<e", 2.0)),
                                 (count, b"\x00\x00"), (1024 * 1024 - 1, b"\x00\x00")):
                changed = bytearray(output)
                changed[index * 2:index * 2 + 2] = value
                with self.subTest(tiles=tiles, index=index), self.assertRaises(ValueError):
                    runner.check_rope_window_output(changed, expected, window)
            with self.assertRaises(ValueError):
                runner.check_rope_window_output(output, expected, (0, 1024, 0, 1024))


class RoPEWindowOutputTests(unittest.TestCase):
    def setUp(self):
        self.expected = struct.pack("<e", 1.0) * (1024 * 1024)
        self.output = bytearray(b"\x00\x7e" * (1024 * 1024))
        for row in range(8):
            start, end = row * 1024 * 2, (row * 1024 + 128) * 2
            self.output[start:end] = self.expected[start:end]

    def test_checks_active_values_and_every_unwritten_region(self):
        metrics = runner.check_rope_window_output(bytes(self.output), self.expected, runner.ROPE_WINDOW)
        self.assertEqual(metrics["checked_elements"], 1024)
        self.assertEqual(metrics["untouched_elements"], 1047552)
        self.assertTrue(metrics["untouched_sentinel_preserved"])
        self.assertTrue(metrics["bitwise_equal_to_reference"])
        # Each row gap, next row and last element are outside the window.
        for index in [row * 1024 + 128 for row in range(8)] + [8 * 1024, 1024 * 1024 - 1]:
            changed = bytearray(self.output)
            changed[index * 2:index * 2 + 2] = b"\x00\x00"
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "outside"):
                runner.check_rope_window_output(bytes(changed), self.expected, runner.ROPE_WINDOW)

    def test_missing_write_wrong_value_or_wrong_scope_is_rejected(self):
        for value in (b"\x00\x7e", struct.pack("<e", 2.0)):
            changed = bytearray(self.output)
            changed[(7 * 1024 + 127) * 2:(7 * 1024 + 128) * 2] = value
            with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                runner.check_rope_window_output(bytes(changed), self.expected, runner.ROPE_WINDOW)
        for actual, window in ((bytes(self.output[:-2]), runner.ROPE_WINDOW),
                               (bytes(self.output), (0, 4, 0, 128))):
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                runner.check_rope_window_output(actual, self.expected, window)


class RoPEWindowEndOutputTests(unittest.TestCase):
    def setUp(self):
        self.expected = struct.pack("<e", 1.0) * (1024 * 1024)
        self.output = bytearray(b"\x00\x7e" * (1024 * 1024))
        for row in range(1016, 1024):
            start, end = (row * 1024 + 896) * 2, (row + 1) * 1024 * 2
            self.output[start:end] = self.expected[start:end]

    def test_end_values_and_all_prefix_and_row_gaps_are_checked(self):
        metrics = runner.check_rope_window_output(self.output, self.expected, runner.ROPE_WINDOW_END)
        self.assertEqual(metrics["window"], [1016, 1024, 896, 1024])
        self.assertEqual(metrics["checked_elements"], 1024)
        self.assertEqual(metrics["untouched_elements"], 1047552)
        self.assertTrue(metrics["untouched_sentinel_preserved"])
        self.assertTrue(metrics["bitwise_equal_to_reference"])
        for index in [0, 1016 * 1024 - 1] + [row * 1024 + 895 for row in range(1016, 1024)]:
            changed = bytearray(self.output)
            changed[index * 2:index * 2 + 2] = b"\x00\x00"
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "outside"):
                runner.check_rope_window_output(changed, self.expected, runner.ROPE_WINDOW_END)

    def test_end_missing_write_wrong_value_and_wrong_window_are_rejected(self):
        for index in (1016 * 1024 + 896, 1024 * 1024 - 1):
            for value in (b"\x00\x7e", struct.pack("<e", 2.0)):
                changed = bytearray(self.output)
                changed[index * 2:index * 2 + 2] = value
                with self.subTest(index=index, value=value), self.assertRaisesRegex(ValueError, "numerical mismatch"):
                    runner.check_rope_window_output(changed, self.expected, runner.ROPE_WINDOW_END)
        with self.assertRaises(ValueError):
            runner.check_rope_window_output(self.output, self.expected, runner.ROPE_WINDOW)


class RoPEReferenceTests(unittest.TestCase):
    def test_target_rows_and_pairs_keep_four_quadrant_rotations(self):
        # Row and pair-dependent quarter turns have an exact analytic answer,
        # including coordinates on both sides of the candidate tile boundaries.
        source = struct.pack("<ee", 3.0, -2.0) * (1024 * 512)
        quadrants = [(row * 3 + pair) % 4 for row in range(1024) for pair in range(512)]
        cosine = struct.pack("<" + "f" * len(quadrants), *[(1., 0., -1., 0.)[q] for q in quadrants])
        sine = struct.pack("<" + "f" * len(quadrants), *[(0., 1., 0., -1.)[q] for q in quadrants])
        pairs = [struct.pack("<ee", *values) for values in ((3., -2.), (2., 3.), (-3., 2.), (-2., -3.))]
        expected = b"".join(pairs[q] for q in quadrants)
        actual = runner.rope_reference(source, cosine, sine, shape=(1024, 1024))
        self.assertEqual(actual, expected)

    def test_smoke_input_bytes_remain_unchanged(self):
        source, cosine, sine, expected = runner.rope_test_vectors()
        self.assertEqual(hashlib.sha256(source + cosine + sine).hexdigest(),
                         "c94aeb14c560115986e64b199dc52a7d074f9be85aa2cee9fdafa1991e6af17a")
        self.assertEqual(hashlib.sha256(expected).hexdigest(),
                         "cfb02bcb37058a473c4147ff1c82a5551c2c691a3e936a25d6d6bee87bfce650")

    def test_identity_and_quarter_turn_keep_interleaved_pairs(self):
        source = struct.pack("<" + "e" * 1024, *([1.0, 2.0] * 512))
        ones, zeros = struct.pack("<" + "f" * 512, *([1.0] * 512)), bytes(2048)
        self.assertEqual(runner.rope_reference(source, ones, zeros), source)
        rotated = struct.pack("<" + "e" * 1024, *([-2.0, 1.0] * 512))
        self.assertEqual(runner.rope_reference(source, zeros, ones), rotated)

    def test_row_and_pair_tables_are_used_without_broadcast_or_swap(self):
        source, cosine, sine, expected = runner.rope_test_vectors()
        self.assertEqual(cosine[:256], struct.pack("<" + "f" * 64, *([1.0] * 64)))
        self.assertEqual(sine[:256], bytes(256))
        self.assertNotEqual(cosine[256:512], cosine[512:768])
        for c, s in ((sine, cosine), (cosine[:256] * 8, sine[:256] * 8)):
            wrong = runner.rope_reference(source, c, s)
            with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                runner.check_output(wrong, expected, 1024, "rope")
        for variant in ("serial", "pipeline"):
            different = struct.pack("<e", 0.5) + expected[2:]
            with self.assertRaisesRegex(ValueError, "bitwise"):
                runner.check_previous_output(different, expected, 1024, "rope", variant)

    def test_wrong_shape_and_table_lengths_fail_before_dispatch(self):
        source, cosine, sine, _ = runner.rope_test_vectors()
        with self.assertRaises(ValueError):
            runner.rope_test_vectors(shape=(1024, 128))
        with self.assertRaises(ValueError):
            runner.rope_reference(source, cosine, sine, shape=(8, 64))
        for args in ((source, cosine[:1024], sine), (source[:-2], cosine, sine), (b"", b"", b"")):
            with self.assertRaises(ValueError):
                runner.call_rope_host(None, *args)
            with self.assertRaises(ValueError):
                runner.rope_reference(*args)


FAKE_ROPE_HOST = r'''
#include <cstdlib>
#include <cstring>
// DATA
static int calls = 0;
extern "C" int get_calls() { return calls; }
extern "C" int tilelang_tpu_run(void **args) {
  ++calls;
  const char *mode = getenv("ROPE_TEST_MODE");
  if (mode && !strcmp(mode,"status")) return 7;
  if (memcmp(args[0], source, sizeof(source)) || memcmp(args[1], cosine, sizeof(cosine)) ||
      memcmp(args[2], sine, sizeof(sine))) return 99;
  size_t size = sizeof(reference);
  if (mode && !strcmp(mode,"partial")) size -= 2;
  memcpy(args[3], reference, size);
  if (mode && !strcmp(mode,"canary")) static_cast<unsigned char *>(args[3])[size] = 0;
  if (mode && !strncmp(mode,"mutate",6)) static_cast<unsigned char *>(args[mode[6]-'0'])[0] ^= 1;
  if (mode && !strcmp(mode,"input-canary")) static_cast<unsigned char *>(args[2])[-1] = 0;
  return 0;
}
extern "C" int tilelang_tpu_run_timed(void **args, int warmups, int count, double *samples) {
  for (int i=0; i<warmups+count; ++i) {
    int rc = tilelang_tpu_run(args);
    if (rc) return rc;
    if (i>=warmups) samples[i-warmups] = 10.0 + i;
  }
  return 0;
}
'''


@unittest.skipUnless(shutil.which("c++"), "host C++ compiler required")
class RoPEHostABITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        data = "\n".join("static const unsigned char " + name + "[] = {" +
                         ",".join(map(str, value)) + "};" for name, value in
                         zip(("source", "cosine", "sine", "reference"), runner.rope_test_vectors()))
        source = write(root / "fake.cpp", FAKE_ROPE_HOST.replace("// DATA", data).encode())
        library = root / "fake.so"
        subprocess.run(["c++", "-O0", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, timeout=30)
        cls.library = ctypes.CDLL(str(library))

    def test_four_argument_one_call_and_timed_staging(self):
        source, cosine, sine, expected = runner.rope_test_vectors()
        for timing, calls in ((None, 1), ({"warmups":2, "samples":3}, 5)):
            before = self.library.get_calls()
            with patch.dict(os.environ, {"ROPE_TEST_MODE":""}):
                actual = runner.call_rope_host(self.library, source, cosine, sine, timing=timing)
            self.assertEqual(actual, expected)
            self.assertEqual(self.library.get_calls() - before, calls)
            if timing is not None:
                self.assertEqual(timing["raw_us"], [12.0, 13.0, 14.0])

    def test_partial_output_canaries_and_all_three_input_mutations_fail(self):
        source, cosine, sine, expected = runner.rope_test_vectors()
        for mode in ("status", "partial", "canary", "input-canary", "mutate0", "mutate1", "mutate2"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"ROPE_TEST_MODE":mode}):
                with self.assertRaises((RuntimeError, ValueError)):
                    runner.check_output(runner.call_rope_host(self.library, source, cosine, sine), expected, 1024, "rope")

    def test_equal_size_swapped_tables_are_rejected_by_callee(self):
        source, cosine, sine, _ = runner.rope_test_vectors()
        with patch.dict(os.environ, {"ROPE_TEST_MODE":""}), self.assertRaisesRegex(RuntimeError, "returned 99"):
            runner.call_rope_host(self.library, source, sine, cosine)


class RMSNormReferenceTests(unittest.TestCase):
    def test_fp16_normalization_boundary_changes_the_result(self):
        # RMS([1,2,...])=sqrt(2.5); applying weight before the FP16 boundary
        # would give 0.822265625 at the first element instead of 0.82177734375.
        source = struct.pack("<" + "e" * 2048, *([1.0, 2.0] * 1024))
        weight = struct.pack("<" + "e" * 2048, *([1.2998046875] * 2048))
        actual = runner.rmsnorm_reference(source, weight)
        self.assertEqual(struct.unpack_from("<e", actual)[0], 0.82177734375)

    def test_per_element_weights_are_not_broadcast_across_rows(self):
        source = struct.pack("<" + "e" * 2048, *([1.0] * 2048))
        weights = struct.pack("<" + "e" * 2048, *[(row + 1)/8 for row in range(16) for _ in range(128)])
        self.assertEqual(runner.rmsnorm_reference(source, weights), weights)
        with self.assertRaisesRegex(ValueError, "two 16x128"):
            runner.rmsnorm_reference(source, weights[:256])

    def test_zero_small_rows_and_strict_same_tile_comparison(self):
        lhs, rhs, expected = runner.test_vectors(2048, "rmsnorm")
        self.assertEqual((lhs, rhs, expected), runner.test_vectors(2048, "rmsnorm-splitk"))
        self.assertEqual(expected[:256], bytes(256))
        self.assertLess(max(abs(x[0]) for x in struct.iter_unpack("<e", lhs[256:512])), 0.003)
        self.assertTrue(runner.check_output(expected, expected, 2048, "rmsnorm")["finite"])
        for operation in ("rmsnorm", "rmsnorm-splitk"):
            self.assertEqual(runner.check_output(expected, expected, 2048, operation)["atol"], 0.01)
            with self.assertRaisesRegex(ValueError, "bitwise"):
                runner.check_previous_output(struct.pack("<e", 1.0009765625), struct.pack("<e", 1.0),
                                             1, operation, "pipeline")
        with self.assertRaisesRegex(ValueError, "bounded"):
            runner.test_vectors(256**2, "rmsnorm")

    def test_target_reference_rows_boundary_weights_and_corruption(self):
        lhs, rhs, expected = runner.test_vectors(1024**2, "rmsnorm")
        self.assertEqual(expected[:2048], bytes(2048))
        self.assertLess(max(abs(x[0]) for x in struct.iter_unpack("<e", lhs[2048:4096])), 0.003)
        # High-precision scalar sums independently check early/middle/final rows.
        for row in (1, 31, 511, 1023):
            a = struct.unpack_from("<1024e", lhs, row*2048)
            w = struct.unpack_from("<1024e", rhs, row*2048)
            inverse = 1 / math.sqrt(math.fsum(x*x for x in a)/1024 + 1e-12)
            for col in (0, 17, 1023):
                normalized = struct.unpack("<e", struct.pack("<e", a[col]*inverse))[0]
                reference = struct.unpack("<e", struct.pack("<e", normalized*w[col]))[0]
                actual = struct.unpack_from("<e", expected, 2*(row*1024+col))[0]
                self.assertAlmostEqual(actual, reference, delta=0.004)
        damaged = struct.pack("<e", float("nan")) + expected[2:]
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            runner.check_output(damaged, expected, 1024**2, "rmsnorm")
        with self.assertRaisesRegex(ValueError, "two 1024x1024"):
            runner.rmsnorm_reference(lhs, rhs[:2048], shape=(1024, 1024))
        with self.assertRaisesRegex(ValueError, "bounded"):
            runner.rmsnorm_reference(lhs, rhs, shape=(1024, 128))

    def test_target_splitk_only_original_serial_allows_changed_k_grouping(self):
        a = struct.pack("<e", 1.0) * 1024**2
        b = struct.pack("<e", 1.0009765625) + a[2:]
        self.assertTrue(runner.check_previous_output(b, a, 1024**2, "rmsnorm-splitk", "serial")["passed"])
        for operation, variant in (("rmsnorm-splitk", "pipeline"), ("rmsnorm", "serial")):
            with self.assertRaisesRegex(ValueError, "bitwise"):
                runner.check_previous_output(b, a, 1024**2, operation, variant)
        with self.assertRaisesRegex(ValueError, "bitwise"):
            runner.check_previous_output(b[:4096], a[:4096], 2048, "rmsnorm-splitk", "serial")
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            runner.check_previous_output(struct.pack("<e", 2.0) + a[2:], a, 1024**2, "rmsnorm-splitk", "serial")


class MatmulContractTests(unittest.TestCase):
    def test_dense_row_major_reference_and_wrong_layout(self):
        lhs, rhs, expected = runner.test_vectors(128 * 128, "matmul")
        a, b, c = [[x[0] for x in struct.iter_unpack("<e", data)] for data in (lhs, rhs, expected)]
        for i, j in ((0, 0), (0, 127), (63, 29), (127, 0), (127, 127)):
            reference = math.fsum(a[i * 128 + k] * b[k * 128 + j] for k in range(128))
            self.assertAlmostEqual(c[i * 128 + j], reference, delta=0.00025)
        transposed = struct.pack("<" + "e" * len(c), *(c[j*128+i] for i in range(128) for j in range(128)))
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            runner.check_output(transposed, expected, len(c), "matmul")
        with self.assertRaisesRegex(ValueError, "bounded"):
            runner.test_vectors(256 * 256, "matmul")

    def test_only_original_to_serial_allows_changed_reduction_grouping(self):
        previous, changed = struct.pack("<e", 1.0), struct.pack("<e", 1.0009765625)
        self.assertTrue(runner.check_previous_output(changed, previous, 1, "matmul", "serial")["passed"])
        self.assertTrue(runner.check_previous_output(previous, previous, 1, "matmul", "pipeline")["passed"])
        for operation, variant in (("matmul", "pipeline"), ("add", "serial"), ("div", "serial")):
            with self.assertRaisesRegex(ValueError, "bitwise"):
                runner.check_previous_output(changed, previous, 1, operation, variant)
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            runner.check_previous_output(struct.pack("<e", 2.0), previous, 1, "matmul", "serial")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "reuse existing NumPy environment")
class MatmulNumpyReferenceTests(unittest.TestCase):
    def test_numpy_reference_agrees_with_independent_scalar_smoke(self):
        lhs, rhs, scalar = runner.test_vectors(128 * 128, "matmul")
        with patch.dict(os.environ):
            actual = runner.numpy_matmul_reference(lhs, rhs, 128)
        self.assertTrue(runner.check_output(actual, scalar, 128 * 128, "matmul")["passed"])

    def test_target_shape_reference_samples_and_corruption(self):
        with patch.dict(os.environ):
            lhs, rhs, expected = runner.test_vectors(1024 * 1024, "matmul")
        self.assertEqual(len(expected), 2 * 1024**2)
        a, b = [[x[0] for x in struct.iter_unpack("<e", data)] for data in (lhs, rhs)]
        for i, j in ((0, 0), (31, 32), (511, 127), (1023, 1023)):
            high_precision = math.fsum(a[i*1024+k] * b[k*1024+j] for k in range(1024))
            actual = struct.unpack_from("<e", expected, 2*(i*1024+j))[0]
            self.assertAlmostEqual(actual, high_precision, delta=0.002)
        damaged = struct.pack("<e", float("nan")) + expected[2:]
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            runner.check_output(damaged, expected, 1024**2, "matmul")
        with self.assertRaisesRegex(ValueError, "byte lengths"):
            runner.numpy_matmul_reference(lhs[:-2], rhs, 1024)

    def test_missing_numpy_fails_with_existing_environment_instruction(self):
        import builtins
        real_import = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("missing NumPy")
            return real_import(name, *args, **kwargs)
        with patch.dict(os.environ), patch.object(builtins, "__import__", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "existing NumPy-enabled"):
                runner.numpy_matmul_reference(bytes(128**2*2), bytes(128**2*2), 128)


if __name__ == "__main__":
    unittest.main()
