"""Host diagnostic derivation preserves frozen device kernels and ABI headers."""

import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("trace_export", ROOT / "tpu_demo/pipeline/export_rope_trace.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
window_spec = importlib.util.spec_from_file_location("window_export", ROOT / "tpu_demo/pipeline/export_rope_window.py")
window = importlib.util.module_from_spec(window_spec)
window_spec.loader.exec_module(window)
end_spec = importlib.util.spec_from_file_location("window_end_export", ROOT / "tpu_demo/pipeline/export_rope_window_end.py")
window_end = importlib.util.module_from_spec(end_spec)
end_spec.loader.exec_module(window_end)
prefix_spec = importlib.util.spec_from_file_location("prefix_export", ROOT / "tpu_demo/pipeline/export_rope_prefix.py")
prefix = importlib.util.module_from_spec(prefix_spec)
prefix_spec.loader.exec_module(prefix)


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

    def test_window_changes_only_two_loop_bounds_and_keeps_layout_and_arithmetic(self):
        base = json.loads(window.BASE.read_text())
        derived = window.export()
        recorded = json.loads((ROOT / "research/bm1690-pipelines/handoff/rope-1024-window8-sources.json").read_text())
        self.assertEqual(derived, recorded)
        for variant in ("original", "serial", "pipeline"):
            for name, before in base["variants"][variant]["sources"].items():
                after = derived["variants"][variant]["sources"][name]
                if variant == "original" and name == "kernel.c":
                    restored = after.replace("bx < 2;", "bx < 256;").replace("by < 4;", "by < 32;")
                    self.assertEqual(restored, before)
                    self.assertNotEqual(after, before)
                else:
                    self.assertEqual(after, before)
        self.assertEqual(derived["host_diagnostic"]["tile_count"], 8)
        self.assertFalse(derived["host_diagnostic"]["timing_allowed"])

    def test_end_window_preserves_tile_count_and_changes_only_iteration_coordinates(self):
        base = json.loads(window_end.BASE.read_text())
        derived = window_end.export()
        recorded = json.loads((ROOT / "research/bm1690-pipelines/handoff/rope-1024-window8-end-sources.json").read_text())
        self.assertEqual(derived, recorded)
        for variant in ("original", "serial", "pipeline"):
            for name, before in base["variants"][variant]["sources"].items():
                after = derived["variants"][variant]["sources"][name]
                if variant == "original" and name == "kernel.c":
                    restored = after.replace("bx = 254; bx < 256;", "bx = 0; bx < 2;")
                    restored = restored.replace("by = 28; by < 32;", "by = 0; by < 4;")
                    self.assertEqual(restored, before)
                    self.assertNotEqual(after, before)
                else:
                    self.assertEqual(after, before)
        diagnostic = derived["host_diagnostic"]
        self.assertEqual(diagnostic["window_rows_columns_exclusive"], [1016, 1024, 896, 1024])
        self.assertEqual(diagnostic["tile_count"], 8)
        self.assertFalse(diagnostic["timing_allowed"])

    def test_prefix_changes_only_outer_loop_extent_and_keeps_full_inner_loop_order(self):
        base = json.loads(prefix.BASE.read_text())
        for tiles in (128, 256, 512):
            derived = prefix.export(tiles)
            recorded = json.loads((ROOT / f"research/bm1690-pipelines/handoff/rope-1024-prefix{tiles}-sources.json").read_text())
            self.assertEqual(derived, recorded)
            for variant in ("original", "serial", "pipeline"):
                for name, before in base["variants"][variant]["sources"].items():
                    after = derived["variants"][variant]["sources"][name]
                    if variant == "original" and name == "kernel.c":
                        self.assertEqual(after.replace(f"bx < {tiles // 32};", "bx < 256;"), before)
                        self.assertNotEqual(after, before)
                    else:
                        self.assertEqual(after, before)
            self.assertEqual(derived["host_diagnostic"]["tile_count"], tiles)
            self.assertEqual(derived["host_diagnostic"]["window_rows_columns_exclusive"], [0, tiles // 8, 0, 1024])
        for unsupported in (0, 8, 255, 8192):
            with self.assertRaises(ValueError):
                prefix.export(unsupported)


if __name__ == "__main__":
    unittest.main()
