"""Exercise the real query executable with a fake library, never a TPU runtime.

These tests validate ABI use and fail-fast behavior. They are not CModel or
BM1690 execution evidence. The test-only header contains only queried types.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HEADER = r"""
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { tpuRtSuccess = 0, tpuRtErrFailure = 2 } tpuRtStatus_t;
typedef struct {
  char name[24];
  uint64_t totalGlobalMem;
  int major, minor, ECCEnabled;
  uint32_t pciBusID, pciDeviceID, pciDomainID;
} tpuRtDeviceProperties_t;
tpuRtStatus_t tpuRtInit(void);
tpuRtStatus_t tpuRtGetDeviceCount(int *);
tpuRtStatus_t tpuRtSetDevice(int);
tpuRtStatus_t tpuRtGetDevice(int *);
tpuRtStatus_t tpuRtGetDeviceProperties(tpuRtDeviceProperties_t *, int);
tpuRtStatus_t tpuRtGetFd(int *);
#ifdef __cplusplus
}
#endif
"""

FAKE = r"""
#include "tpuv7_rt.h"
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
static int selected = -1;
static int descriptor = -1;
static bool mode(const char *name) {
  const char *value = getenv("TPU_PROBE_TEST_MODE");
  return value && !strcmp(value, name);
}
extern "C" {
tpuRtStatus_t tpuRtInit() {
  if (mode("init_fail")) return tpuRtErrFailure;
  descriptor = open("/dev/null", O_RDONLY);
  return descriptor < 0 ? tpuRtErrFailure : tpuRtSuccess;
}
tpuRtStatus_t tpuRtGetDeviceCount(int *count) {
  *count = mode("zero") ? 0 : (mode("too_many") ? 3 : 2);
  return tpuRtSuccess;
}
tpuRtStatus_t tpuRtSetDevice(int id) {
  if (mode("set_fail")) return tpuRtErrFailure;
  selected = id;
  return tpuRtSuccess;
}
tpuRtStatus_t tpuRtGetDevice(int *id) {
  *id = selected + (mode("mismatch") ? 1 : 0);
  return tpuRtSuccess;
}
tpuRtStatus_t tpuRtGetDeviceProperties(tpuRtDeviceProperties_t *p, int id) {
  if (mode("properties_fail")) return tpuRtErrFailure;
  memset(p, 0, sizeof(*p));
  // No terminator: the probe must respect the declared name array bound.
  memset(p->name, 'X', sizeof(p->name));
  p->totalGlobalMem = 1ULL << 36;
  p->pciBusID = 1;
  p->pciDeviceID = id;
  return tpuRtSuccess;
}
#ifndef OMIT_FD
tpuRtStatus_t tpuRtGetFd(int *fd) {
  *fd = mode("invalid_fd") ? -1 : descriptor;
  return tpuRtSuccess;
}
#endif
}
"""


@unittest.skipUnless(shutil.which("c++"), "host C++ compiler is required")
class RuntimeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="bm1690-probe-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.program = cls.root / "probe"
        (cls.root / "tpuv7_rt.h").write_text(HEADER)
        (cls.root / "fake.cpp").write_text(FAKE)
        source = (Path(__file__).resolve().parents[3] /
                  "tpu_demo/pipeline/probe_bm1690_runtime.cpp")
        common = ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                  "-I", str(cls.root)]
        subprocess.run([*common, str(source), "-ldl", "-o", str(cls.program)],
                       check=True, capture_output=True, text=True, timeout=30)
        for filename, flags in (("fake.so", []), ("missing.so", ["-DOMIT_FD"])):
            subprocess.run([*common, *flags, "-shared", "-fPIC",
                            str(cls.root / "fake.cpp"), "-o", str(cls.root / filename)],
                           check=True, capture_output=True, text=True, timeout=30)

    def run_probe(self, mode="", library="fake.so", limit="2"):
        environment = dict(os.environ, TPU_PROBE_TEST_MODE=mode)
        return subprocess.run([str(self.program), str(self.root / library), limit],
                              env=environment, capture_output=True, text=True, timeout=5)

    def test_enumeration_reports_raw_identity_and_borrowed_fd(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('RUNTIME_LOADED="' + str(self.root / "fake.so") + '"', result.stdout)
        self.assertIn("DEVICE_COUNT=2", result.stdout)
        self.assertIn('name="' + "X" * 24 + '"', result.stdout)
        self.assertIn("total_global_mem_bytes=68719476736", result.stdout)
        self.assertIn("PCI_FIELDS_RAW domain=0 bus=1 device=1", result.stdout)
        self.assertEqual(result.stdout.count('target="/dev/null"'), 2)
        self.assertIn("ENUMERATION_COMPLETE", result.stdout)

    def test_missing_symbol_stops_before_runtime_initialization(self):
        result = self.run_probe(library="missing.so")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing runtime symbol tpuRtGetFd", result.stderr)
        self.assertNotIn("CALL tpuRtInit", result.stdout)

    def test_count_bound_prevents_device_selection(self):
        for mode in ("zero", "too_many"):
            with self.subTest(mode=mode):
                result = self.run_probe(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("CALL tpuRtSetDevice", result.stdout)

    def test_api_errors_stop_before_the_next_operation(self):
        for mode, next_call in (("init_fail", "tpuRtGetDeviceCount"),
                                ("set_fail", "tpuRtGetDevice"),
                                ("mismatch", "tpuRtGetDeviceProperties"),
                                ("properties_fail", "tpuRtGetFd")):
            with self.subTest(mode=mode):
                result = self.run_probe(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("CALL " + next_call + "\n", result.stdout)
                self.assertNotIn("ENUMERATION_COMPLETE", result.stdout)

    def test_invalid_descriptor_is_not_treated_as_verified_mapping(self):
        result = self.run_probe("invalid_fd")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid fd", result.stderr)
        self.assertNotIn("ENUMERATION_COMPLETE", result.stdout)

    def test_invalid_limit_is_rejected_before_loading_runtime(self):
        for limit in ("0", "-1", "2junk", "999999999999999999999999"):
            with self.subTest(limit=limit):
                result = self.run_probe(limit=limit)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("RUNTIME_LOADED", result.stdout)


if __name__ == "__main__":
    unittest.main()
