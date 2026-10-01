"""Bounded download recovery; no network, compilers or vendor libraries."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prepare_add", ROOT / "tpu_demo/pipeline/prepare_add_1024.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
REVISION = "7c0efb8e36b3058a51b1aa1fb948ca67676e45f8"


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "work"
        self.output.mkdir()
        self.receipt = {"status":"failed", "revision":REVISION, "board_runtime_loaded":False,
                        "kernel_launches":0, "files":{"build.py":prepare.FILES["build.py"][1]},
                        "error":"TimeoutError: The read operation timed out"}
        self.save_receipt()
        source = ROOT / prepare.FILES["build.py"][0]
        (self.output / "build.py").write_bytes(source.read_bytes())

    def save_receipt(self):
        (self.output / "handoff.json").write_text(json.dumps(self.receipt))

    def run_resume(self):
        argv = ["prepare", "--revision", REVISION, "--output", str(self.output), "--resume", "--download-only"]
        with patch.object(sys,"argv",argv), contextlib.redirect_stdout(io.StringIO()):
            return prepare.main()

    def test_resume_fetches_only_missing_files_and_keeps_original_receipt(self):
        previous = (self.output / "handoff.json").read_bytes()
        original_builder = (self.output / "build.py").stat().st_mtime_ns
        seen = []
        def download(url, timeout):
            seen.append(url)
            relative = url.split("/" + REVISION + "/")[1]
            return io.BytesIO((ROOT / relative).read_bytes())
        with patch.object(prepare.urllib.request,"urlopen",side_effect=download), \
             patch.object(prepare.subprocess,"run",side_effect=AssertionError("must not compile")):
            self.assertEqual(self.run_resume(),0)
        self.assertEqual(len(seen),2)
        self.assertTrue(seen[0].endswith("add-1024-sources.json"))
        self.assertTrue(seen[1].endswith("run_add_pcie.py"))
        self.assertEqual((self.output / "handoff.json").read_bytes(),previous)
        self.assertEqual((self.output / "build.py").stat().st_mtime_ns,original_builder)
        logs = list(self.output.glob("handoff-resume-*.json"))
        self.assertEqual(len(logs),1)
        report = json.loads(logs[0].read_text())
        self.assertEqual(report["status"],"download_verified")
        self.assertEqual(set(report["files"]),set(prepare.FILES))

    def test_retained_tamper_wrong_revision_or_started_build_blocks_before_network(self):
        with patch.object(prepare.urllib.request,"urlopen",side_effect=AssertionError("must not download")), \
             patch.object(prepare.subprocess,"run",side_effect=AssertionError("must not compile")):
            self.receipt["revision"]="0"*40
            self.save_receipt()
            with self.assertRaises(ValueError): self.run_resume()
            self.receipt["revision"]=REVISION
            self.save_receipt()
            (self.output / "build").mkdir()
            with self.assertRaises(ValueError): self.run_resume()
            (self.output / "build").rmdir()
            (self.output / "build.log").write_text("earlier compile attempt")
            with self.assertRaises(ValueError): self.run_resume()
            (self.output / "build.log").unlink()
            (self.output / "build.py").write_bytes(b"tampered")
            with self.assertRaises(ValueError): self.run_resume()

    def test_read_timeout_retries_bounded_and_exhaustion_keeps_files(self):
        data = b"download"
        expected = hashlib.sha256(data).hexdigest()
        with patch.object(prepare.urllib.request,"urlopen",side_effect=[TimeoutError("timeout"),io.BytesIO(data)]) as network, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare.download_verified("url",expected),data)
            self.assertEqual(network.call_count,2)
        with patch.object(prepare.urllib.request,"urlopen",side_effect=TimeoutError("timeout")) as network, \
             patch.object(prepare.subprocess,"run",side_effect=AssertionError("must not compile")):
            self.assertEqual(self.run_resume(),1)
            self.assertEqual(network.call_count,3)
        self.assertFalse((self.output / "add.json").exists())
        self.assertTrue((self.output / "build.py").exists())
        self.assertFalse((self.output / "build").exists())

    def test_corrupt_or_404_download_is_not_retried(self):
        expected = hashlib.sha256(b"expected").hexdigest()
        for response in (io.BytesIO(b"corrupt"),urllib.error.HTTPError("url",404,"missing",{},None)):
            with self.subTest(response=response), contextlib.redirect_stdout(io.StringIO()):
                settings = {"side_effect":response} if isinstance(response,Exception) else {"return_value":response}
                with patch.object(prepare.urllib.request,"urlopen",**settings) as network:
                    with self.assertRaises((ValueError,urllib.error.HTTPError)):
                        prepare.download_verified("url",expected)
                    self.assertEqual(network.call_count,1)

    def test_delivery_file_pins_still_match(self):
        for name,(path,expected) in prepare.FILES.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),expected,name)

    def test_compile_failure_is_attempted_once_and_cannot_resume(self):
        for name,(path,_) in prepare.FILES.items():
            if not (self.output / name).exists():
                (self.output / name).write_bytes((ROOT / path).read_bytes())
        argv=["prepare","--revision",REVISION,"--output",str(self.output),"--resume"]
        with patch.object(sys,"argv",argv), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(prepare.urllib.request,"urlopen",side_effect=AssertionError("already downloaded")), \
             patch.object(prepare.subprocess,"run",return_value=SimpleNamespace(returncode=7)) as compile_call:
            self.assertEqual(prepare.main(),1)
            self.assertEqual(compile_call.call_count,1)
        with self.assertRaisesRegex(ValueError,"compilation has not started"):
            self.run_resume()


if __name__ == "__main__":
    unittest.main()
