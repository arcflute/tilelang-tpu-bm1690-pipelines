"""Internal CModel worker. The parent runner supplies resource restrictions."""

from __future__ import annotations

import argparse
import json
import os


def pipeline_cases():
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("baseline", "pipeline"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", choices=("smoke", "target"), default="smoke")
    args = parser.parse_args()
    cpus = os.environ.get("BM1690_PIPELINE_CPUS")
    if not cpus:
        parser.error("use python -m tpu_demo.pipeline.run, which supervises worker resources")
    os.sched_setaffinity(0, {int(cpu) for cpu in cpus.split(",")})
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.suite == "baseline":
        from tpu_demo.run import run_case
        result = run_case(args.case, chip="bm1690", programming_model="tpukernel",
                          runtime_mode="cmodel", seed=args.seed)
    else:
        raise ValueError("pipeline cases are registered as their implementations are validated")
    print("BM1690_PIPELINE_RESULT=" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
