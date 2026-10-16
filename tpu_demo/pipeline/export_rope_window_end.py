"""Move the frozen eight-tile diagnostic to the matrix end; keep its workload."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/bm1690-pipelines/handoff/rope-1024-window8-sources.json"
BASE_SHA256 = "dcf15b8d6480b05345d116619d3f8744de1da7657af937de0bbe9979d9500622"


def export():
    payload = BASE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen RoPE eight-tile diagnostic changed")
    bundle = copy.deepcopy(json.loads(payload))
    variant = bundle["variants"]["original"]
    source = variant["sources"]["kernel.c"]
    for axis, extent, start, stop in (("bx", 2, 254, 256), ("by", 4, 28, 32)):
        before = f"for (int32_t {axis} = 0; {axis} < {extent}; ++{axis})"
        after = f"for (int32_t {axis} = {start}; {axis} < {stop}; ++{axis})"
        if source.count(before) != 1:
            raise ValueError(f"Unexpected frozen {axis} tile loop")
        source = source.replace(before, after)
    variant["sources"]["kernel.c"] = source
    variant["sha256"]["kernel.c"] = hashlib.sha256(source.encode()).hexdigest()
    bundle["host_diagnostic"].update(
        kind="flushed-runtime-call-boundaries-and-eight-tile-end-window",
        base_bundle_sha256=BASE_SHA256,
        derivation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        window_rows_columns_exclusive=[1016, 1024, 896, 1024],
    )
    return bundle


if __name__ == "__main__":
    destination = ROOT / "research/bm1690-pipelines/handoff/rope-1024-window8-end-sources.json"
    with destination.open("x") as output:
        output.write(json.dumps(export(), indent=2, sort_keys=True) + "\n")
