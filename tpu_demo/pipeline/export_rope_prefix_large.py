"""Extend the frozen, board-validated 512-tile prefix without changing its order."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/bm1690-pipelines/handoff/rope-1024-prefix512-sources.json"
BASE_SHA256 = "9b9250c5828bd25d6f5cd315f4b5aca919a93b7d2b36cac3b3cf80e66972e962"
TILE_COUNTS = (1024, 2048, 4096)


def export(tiles):
    if tiles not in TILE_COUNTS:
        raise ValueError("Unsupported larger RoPE prefix")
    payload = BASE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen 512-tile prefix changed")
    bundle = copy.deepcopy(json.loads(payload))
    variant = bundle["variants"]["original"]
    source = variant["sources"]["kernel.c"]
    before = "for (int32_t bx = 0; bx < 16; ++bx)"
    if source.count(before) != 1:
        raise ValueError("Unexpected frozen bx tile loop")
    source = source.replace(before, f"for (int32_t bx = 0; bx < {tiles // 32}; ++bx)")
    variant["sources"]["kernel.c"] = source
    variant["sha256"]["kernel.c"] = hashlib.sha256(source.encode()).hexdigest()
    bundle["host_diagnostic"].update({
        "base_bundle_sha256": BASE_SHA256,
        "derivation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "window_rows_columns_exclusive": [0, tiles // 8, 0, 1024],
        "tile_count": tiles,
    })
    return bundle


if __name__ == "__main__":
    for tiles in TILE_COUNTS:
        destination = ROOT / f"research/bm1690-pipelines/handoff/rope-1024-prefix{tiles}-sources.json"
        with destination.open("x") as output:
            output.write(json.dumps(export(tiles), indent=2, sort_keys=True) + "\n")
