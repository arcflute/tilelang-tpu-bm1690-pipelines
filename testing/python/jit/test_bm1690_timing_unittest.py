"""Compile the exported host timer against a fake runtime; no vendor loading."""

import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("timed_add", ROOT / "tpu_demo/pipeline/run_add_pcie.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

HEADER = r'''
#pragma once
#include <cstddef>
typedef int tpuRtStatus_t;
constexpr int tpuRtSuccess = 0;
typedef void* tpuRtStream_t;
typedef void* tpuRtKernelModule_t;
int tpuRtInit();
int tpuRtSetDevice(int);
int tpuRtStreamCreate(void**);
void* tpuRtKernelLoadModuleFile(const char*,void*);
int tpuRtMalloc(void**,size_t,int);
int tpuRtMemcpyS2D(void*,const void*,size_t);
int tpuRtMemcpyD2S(void*,const void*,size_t);
int tpuRtStreamSynchronize(void*);
int tpuRtFree(void**,int);
int tpuRtKernelUnloadModule(void*,void*);
int tpuRtStreamDestroy(void*);
'''
MOCK = r'''
#include "tpuv7_rt.h"
#include <cstdlib>
#include <cstring>
static int counts[10], mode;
extern "C" void reset(int m) { memset(counts,0,sizeof(counts)); mode=m; }
extern "C" int counter(int n) { return counts[n]; }
int tpuRtInit() { ++counts[0]; return 0; }
int tpuRtSetDevice(int) { return 0; }
int tpuRtStreamCreate(void** s) { *s=(void*)1; return 0; }
void* tpuRtKernelLoadModuleFile(const char*,void*) { return (void*)1; }
int tpuRtMalloc(void** p,size_t n,int) {
  if (++counts[1]==2 && mode==1) return 1;
  *p=malloc(n); return *p?0:1;
}
int tpuRtMemcpyS2D(void* d,const void* s,size_t n) {
  ++counts[2]; if (mode==2) return 2; memcpy(d,s,n); return 0;
}
int tpuRtStreamSynchronize(void*) { ++counts[3]; return mode==3?3:0; }
int main_kernel(unsigned long long a,unsigned long long b,unsigned long long c) {
  ++counts[4]; if (mode==4 && counts[4]==2) return 4;
  if (mode==9 && counts[4]==7) return 9;
  memcpy((void*)c,(void*)a,1024*1024*2); return 0;
}
int tpuRtMemcpyD2S(void* d,const void* s,size_t n) {
  ++counts[5]; if (mode==5) return 5; memcpy(d,s,n); return 0;
}
int tpuRtFree(void** p,int) { ++counts[6]; free(*p); *p=nullptr; return mode==6?6:0; }
int tpuRtKernelUnloadModule(void*,void*) { ++counts[7]; return mode==7?7:0; }
int tpuRtStreamDestroy(void*) { ++counts[8]; return mode==8?8:0; }
'''


class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        bundle = json.loads((ROOT / "research/bm1690-pipelines/handoff/add-1024-sources.json").read_text())
        (directory / "main.cpp").write_text(bundle["variants"]["serial"]["sources"]["main.cpp"])
        (directory / "tpuv7_rt.h").write_text(HEADER)
        (directory / "kernel.h").write_text("int main_kernel(unsigned long long,unsigned long long,unsigned long long);\n")
        (directory / "mock.cpp").write_text(MOCK)
        subprocess.run(["c++","-std=c++17","-shared","-fPIC","-O0","-pthread","-I"+str(directory),
                        '-DTILELANG_PPL_KERNEL_PATH="mock"',str(directory/"main.cpp"),str(directory/"mock.cpp"),
                        "-o",str(directory/"main.so")],check=True,capture_output=True,timeout=30)
        cls.lib = ctypes.CDLL(str(directory / "main.so"))
        cls.lib.reset.argtypes = [ctypes.c_int]
        cls.lib.counter.argtypes = [ctypes.c_int]
        cls.lib.tilelang_tpu_bind_device.argtypes = [ctypes.c_int]
        assert cls.lib.tilelang_tpu_bind_device(0)==0
        cls.lhs = b"\x00\x3c" * (1024*1024)
        cls.rhs = b"\x00\x00" * (1024*1024)

    def setUp(self):
        self.environment = patch.dict(os.environ,{"TILELANG_TPU_ALLOW_PCIE_LOAD":"1", "TILELANG_TPU_DEVICE_ID":"0",
                                                  "TILELANG_TPU_PROFILE_SESSION":"0", "BMLIB_ENABLE_ALL_PROFILE":"0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_resident_buffers_exact_call_count_and_sample_statistics(self):
        self.lib.reset(0)
        timing = {"warmups":5,"samples":20}
        output = runner.call_host(self.lib,self.lhs,self.rhs,timing=timing)
        self.assertEqual(output,self.lhs)
        self.assertEqual([self.lib.counter(i) for i in range(9)],[1,3,3,1,25,1,3,1,1])
        self.assertEqual(len(timing["raw_us"]),20)
        stats = runner.summarize_samples([4.,1.,3.,2.])
        self.assertEqual(stats["median_us"],2.5)
        self.assertEqual(stats["iqr_us"],1.5)
        self.assertAlmostEqual(stats["p95_us"],3.85)
        for values in ([],[float("nan")],[0.],[-1.]):
            with self.assertRaises(ValueError): runner.summarize_samples(values)

    def test_all_runtime_failures_reject_results_and_stop_dispatch(self):
        for mode in range(1,10):
            self.lib.reset(mode)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                runner.call_host(self.lib,self.lhs,self.rhs,timing={"warmups":5,"samples":20})
            self.assertEqual(self.lib.counter(4),0 if mode<4 else 2 if mode==4 else 7 if mode==9 else 25)
            self.assertEqual(self.lib.counter(6),1 if mode==1 else 3)
            self.assertEqual(self.lib.counter(7),1)
            self.assertEqual(self.lib.counter(8),1)

    def test_invalid_counts_and_profiling_stop_before_initialization(self):
        self.lib.reset(0)
        with self.assertRaises(ValueError):
            runner.call_host(self.lib,self.lhs,self.rhs,timing={"warmups":0,"samples":20})
        with patch.dict(os.environ,{"TILELANG_TPU_PROFILE_SESSION":"1"}), self.assertRaises(RuntimeError):
            runner.call_host(self.lib,self.lhs,self.rhs,timing={"warmups":5,"samples":20})
        self.assertEqual(self.lib.counter(0),0)
        self.assertEqual(self.lib.tilelang_tpu_run_timed(None,1,1,None),-20)
        self.assertEqual(self.lib.counter(0),0)


if __name__ == "__main__":
    unittest.main()
