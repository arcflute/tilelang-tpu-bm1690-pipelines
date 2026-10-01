"""Pinned-manifest, one-call ABI and supervisor tests; no vendor runtime loaded."""

import contextlib
import ctypes
import hashlib
import importlib.util
import io
import json
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
                       "bundle_sha256": self.bundle_hash, "builder_sha256": runner.CURRENT_BUILDER_SHA256,
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


if __name__ == "__main__":
    unittest.main()
