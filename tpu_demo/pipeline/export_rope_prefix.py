"""Derive bounded prefixes of the frozen original RoPE loop, in its exact order."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/bm1690-pipelines/handoff/rope-1024-host-trace-sources.json"
BASE_SHA256 = "815a4f6489da76aa3855da3899029f93008dd82ae6be92f2b1033c0c9d6b9fcd"
TILE_COUNTS = (128, 256, 512)


def export(tiles):
    if tiles not in TILE_COUNTS:
        raise ValueError("Unsupported bounded RoPE prefix")
    payload = BASE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen RoPE host diagnostic changed")
    bundle = copy.deepcopy(json.loads(payload))
    variant = bundle["variants"]["original"]
    source = variant["sources"]["kernel.c"]
    before = "for (int32_t bx = 0; bx < 256; ++bx)"
    if source.count(before) != 1:
        raise ValueError("Unexpected frozen bx tile loop")
    source = source.replace(before, f"for (int32_t bx = 0; bx < {tiles // 32}; ++bx)")
    variant["sources"]["kernel.c"] = source
    variant["sha256"]["kernel.c"] = hashlib.sha256(source.encode()).hexdigest()
    bundle["host_diagnostic"] = {
        "kind": "flushed-runtime-call-boundaries-and-original-loop-prefix",
        "allowed_variant": "original", "timing_allowed": False,
        "base_bundle_sha256": BASE_SHA256,
        "derivation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modified_files_from_base": ["original/kernel.c"],
        "device_kernel_unchanged": False,
        "window_rows_columns_exclusive": [0, tiles // 8, 0, 1024],
        "tile_count": tiles, "global_shape_and_strides_unchanged": True,
        "unwritten_fp16_bytes_hex": "007e", "board_execution_performed": False,
        "acceptance_scope": "original loop prefix matches reference; remaining output bytes retain NaN sentinel",
    }
    return bundle


if __name__ == "__main__":
    for tiles in TILE_COUNTS:
        destination = ROOT / f"research/bm1690-pipelines/handoff/rope-1024-prefix{tiles}-sources.json"
        with destination.open("x") as output:
            output.write(json.dumps(export(tiles), indent=2, sort_keys=True) + "\n")
