"""Run with python -m unittest discover; no pytest or vendor runtime required."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

from tpu_demo.pipeline.run import ROOT, run_worker, worker_environment


class RunnerTests(unittest.TestCase):
    def test_worker_environment_pins_limits_and_removes_board_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            env = worker_environment(Path(directory), [0])
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["TPU_RT_CORE_NUM"], "8")
        self.assertEqual(Path(env["PYTHONPATH"]), ROOT)
        self.assertNotIn("TILELANG_TPU_ALLOW_PCIE_LOAD", env)

    def run_script(self, script, timeout=5, memory=512 * 1024**2):
        with tempfile.TemporaryDirectory() as directory:
            return run_worker([sys.executable, "-c", script], Path(directory), dict(os.environ),
                              timeout, memory)

    def test_result_requires_numeric_payload_and_successful_exit(self):
        self.assertEqual(self.run_script('print(\'BM1690_PIPELINE_RESULT={"status":"passed"}\')')["status"], "passed")
        self.assertEqual(self.run_script('print("no result")')["status"], "failed")
        result = self.run_script('print(\'BM1690_PIPELINE_RESULT={"status":"passed"}\'); raise SystemExit(2)')
        self.assertEqual(result["status"], "failed")

    def test_timeout_stops_descendants(self):
        result = self.run_script(
            'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); time.sleep(30)',
            timeout=0.5)
        self.assertEqual(result["reason"], "timeout")
        self.assertTrue(result["cleanup_complete"])

    def test_memory_guard_is_a_failure(self):
        result = self.run_script('import time; x=bytearray(64*1024*1024); time.sleep(30)',
                                 memory=48*1024**2)
        self.assertEqual(result["reason"], "rss_limit")
        self.assertTrue(result["cleanup_complete"])


if __name__ == "__main__":
    unittest.main()
