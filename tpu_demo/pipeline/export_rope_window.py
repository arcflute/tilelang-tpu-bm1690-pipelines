"""Limit a frozen RoPE diagnostic to eight tiles without changing global layout."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/bm1690-pipelines/handoff/rope-1024-host-trace-sources.json"
BASE_SHA256 = "815a4f6489da76aa3855da3899029f93008dd82ae6be92f2b1033c0c9d6b9fcd"


def export():
    payload = BASE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen RoPE host diagnostic changed")
    bundle = copy.deepcopy(json.loads(payload))
    variant = bundle["variants"]["original"]
    source = variant["sources"]["kernel.c"]
    for axis, before, after in (("bx", 256, 2), ("by", 32, 4)):
        loop = f"for (int32_t {axis} = 0; {axis} < {before}; ++{axis})"
        if source.count(loop) != 1:
            raise ValueError(f"Unexpected frozen {axis} tile loop")
        source = source.replace(loop, loop.replace(f"< {before};", f"< {after};"))
    variant["sources"]["kernel.c"] = source
    variant["sha256"]["kernel.c"] = hashlib.sha256(source.encode()).hexdigest()
    bundle["host_diagnostic"] = {
        "kind": "flushed-runtime-call-boundaries-and-eight-tile-window",
        "allowed_variant": "original", "timing_allowed": False,
        "base_bundle_sha256": BASE_SHA256,
        "derivation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modified_files_from_base": ["original/kernel.c"],
        "device_kernel_unchanged": False,
        "window_rows_columns_exclusive": [0, 8, 0, 128],
        "tile_count": 8, "global_shape_and_strides_unchanged": True,
        "unwritten_fp16_bytes_hex": "007e", "board_execution_performed": False,
        "acceptance_scope": "active window matches reference; all other output bytes retain NaN sentinel",
    }
    return bundle


if __name__ == "__main__":
    destination = ROOT / "research/bm1690-pipelines/handoff/rope-1024-window8-sources.json"
    with destination.open("x") as output:
        output.write(json.dumps(export(), indent=2, sort_keys=True) + "\n")
