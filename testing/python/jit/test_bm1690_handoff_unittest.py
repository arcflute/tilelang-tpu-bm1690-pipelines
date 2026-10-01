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

    def run_resume(self, *extra):
        argv = ["prepare", "--revision", REVISION, "--output", str(self.output), "--resume", "--download-only"]
        argv.extend(extra)
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
        for case in ("target","coarse","sub","mul","div"):
            for name,(path,expected) in prepare.source_files(case).items():
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),expected,(case,name))

    def test_coarse_case_cannot_resume_a_fine_case_directory(self):
        with patch.object(prepare.urllib.request,"urlopen",side_effect=AssertionError("wrong case must stop before download")):
            with self.assertRaises(ValueError):
                self.run_resume("--case","coarse")

    def test_operation_cases_reject_cross_operation_resume_before_network(self):
        self.receipt["case"]="sub"
        self.save_receipt()
        with patch.object(prepare.urllib.request,"urlopen",side_effect=AssertionError("wrong operation")):
            for op in ("target","coarse","mul","div"):
                with self.subTest(op=op), self.assertRaises(ValueError):
                    self.run_resume("--case",op)

    def test_operation_cases_select_verified_separate_bundles(self):
        for op in ("sub","mul","div"):
            path,digest = prepare.source_files(op)["add.json"]
            bundle=json.loads((ROOT/path).read_text())
            self.assertEqual(bundle["target"]["operation"],op)
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest)
            baseline=json.loads((ROOT/prepare.COARSE_SOURCE[0]).read_text())
            for name in ("original","serial","pipeline"):
                self.assertEqual(bundle["variants"][name]["sha256"]["main.cpp"],
                                 baseline["variants"][name]["sha256"]["main.cpp"])

    def test_coarse_case_selects_its_own_bundle_and_preserves_original_sources(self):
        self.receipt["case"]="coarse"
        self.save_receipt()
        seen=[]
        def download(url, timeout):
            path=url.split("/"+REVISION+"/")[1]
            seen.append(path)
            return io.BytesIO((ROOT/path).read_bytes())
        with patch.object(prepare.urllib.request,"urlopen",side_effect=download), \
             patch.object(prepare.subprocess,"run",side_effect=AssertionError("must not compile")):
            self.assertEqual(self.run_resume("--case","coarse"),0)
        self.assertEqual(seen[0],prepare.COARSE_SOURCE[0])
        coarse=json.loads((self.output/'add.json').read_text())
        fine=json.loads((ROOT/prepare.FILES['add.json'][0]).read_text())
        self.assertEqual(coarse['variants']['original']['sha256'],fine['variants']['original']['sha256'])
        self.assertEqual(coarse['variants']['serial']['tiling'],[128,1024])
        self.assertEqual(coarse['variants']['pipeline']['tiling'],[128,1024])
        self.assertEqual(coarse['variants']['pipeline']['pipeline_reports'][0]['extent'],8)

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

    def test_api_transport_fetches_pinned_contents_without_using_raw_download_url(self):
        seen = []
        def download(request, timeout):
            self.assertTrue(request.full_url.startswith(prepare.API_BASE + "/"))
            self.assertTrue(request.full_url.endswith("?ref=" + REVISION))
            self.assertEqual(request.get_header("Accept"),"application/vnd.github.raw+json")
            self.assertIsNone(request.get_header("Authorization"))
            self.assertEqual(timeout,45)
            relative = request.full_url[len(prepare.API_BASE)+1:].split("?")[0]
            seen.append(relative)
            return io.BytesIO((ROOT / relative).read_bytes())
        with patch.object(prepare.urllib.request,"urlopen",side_effect=download), \
             patch.object(prepare.subprocess,"run",side_effect=AssertionError("must not compile")):
            self.assertEqual(self.run_resume("--transport","github-api"),0)
        self.assertEqual(seen,[prepare.FILES[name][0] for name in ("add.json","run_add.py")])
        report=json.loads(next(self.output.glob("handoff-resume-*.json")).read_text())
        self.assertEqual(report["download_transport"],"github-api")
        for name,(_,expected) in prepare.FILES.items():
            self.assertEqual(hashlib.sha256((self.output/name).read_bytes()).hexdigest(),expected)

    def test_api_json_metadata_is_rejected_instead_of_following_download_url(self):
        metadata=b'{"download_url":"https://raw.githubusercontent.com/anything"}'
        with patch.object(prepare.urllib.request,"urlopen",return_value=io.BytesIO(metadata)) as network:
            self.assertEqual(self.run_resume("--transport","github-api"),1)
            self.assertEqual(network.call_count,1)
        self.assertFalse((self.output/"add.json").exists())


if __name__ == "__main__":
    unittest.main()
