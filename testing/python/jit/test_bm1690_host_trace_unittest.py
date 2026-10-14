"""Host diagnostic derivation preserves frozen device kernels and ABI headers."""

import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("trace_export", ROOT / "tpu_demo/pipeline/export_rope_trace.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class TraceSourceTests(unittest.TestCase):
    def test_reproducible_derivation_changes_only_original_host_translation_units(self):
        base = json.loads(trace.BASE.read_text())
        derived = trace.export()
        recorded = json.loads((ROOT / "research/bm1690-pipelines/handoff/rope-1024-host-trace-sources.json").read_text())
        self.assertEqual(derived, recorded)
        for variant in ("original", "serial", "pipeline"):
            for name in ("kernel.c", "kernel.h", "kernel.cpp", "main.cpp"):
                before = base["variants"][variant]["sources"][name]
                after = derived["variants"][variant]["sources"][name]
                if variant == "original" and name in ("main.cpp", "kernel.cpp"):
                    self.assertNotEqual(before, after)
                    self.assertIn("std::fflush(stderr)", after)
                    for api in trace.APIS:
                        self.assertIn("#define " + api + "(...)", after)
                else:
                    self.assertEqual(before, after)
        self.assertNotIn("Single kernel execution time:", derived["variants"]["original"]["sources"]["main.cpp"])

    def test_missing_ambiguous_or_already_traced_source_is_rejected(self):
        for source in ("missing", "anchor anchor", "TPU_RUNTIME_TRACE anchor"):
            with self.assertRaises(ValueError):
                trace.instrument(source, "anchor")


if __name__ == "__main__":
    unittest.main()
