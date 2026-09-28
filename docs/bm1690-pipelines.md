# BM1690 pipeline migration

Route A starts from `xwhzz/tilelang-tpu`, branch
`feature/sg2260e-rv-support`, commit
`e79e06c04c592014127d5eadd0b2f48777fe07ca`. The reference is
`arcflute/ChunkScan-2-TileLang-4-TPU-bm1690`, main commit
`683fb539cb14005ce3aba967d8c392a38fd1affe`.

The source owner reports that ChunkScan stages other than P10 have completed
validation. This project migrates those mechanisms; it does not restart that
project's environment bring-up. P10 is not complete performance evidence.

## Execution and evidence

Use the existing Python environment and PPL 1.7 SDK. No dependency installation
is part of this workflow. For example, in the current checkout:

```bash
export PPL_PROJECT_ROOT=/home/admin3/toolchains/ppl_v1.7.122-g05ebfb36-20260528
.risc/bin/python -m tpu_demo.pipeline.run --suite baseline \
  --output research/artifacts/bm1690-pipelines/baseline-001
```

On another machine select its existing Python and SDK paths. Output directories
must be new. The runner executes FP16 cases sequentially, stops on the first
failure, limits workers to two allowed logical CPUs (leaving one when possible),
sets numerical library threads to one, bounds each process group to 120 seconds
and a conservative 4096 MiB sum of RSS, and records logs and partial results.
Limits are configurable; exceeding a limit is a failed run, never a pass.
BM1690's eight-core emulator topology is distinct from the single-core launch.

CModel validates numerical behavior. Its wall-clock durations are diagnostics,
not BM1690 performance. Hardware synchronization, synchronous host-call latency,
and physical overlap are separate validation levels. The physical board will be
tested by the user on a remote BM1690 server after its environment is recorded.
Do not launch a PCIe run from this CModel runner.

## Milestones

1. P0/P1: source/toolchain identity, bounded runner, all original FP16 baselines.
2. P2: TPU effects, schedule validation, real buffer versions, synchronization
   contract, minimal tiled Add pipeline and compiler rejection tests.
3. P3: FP16 Matmul serial/pipeline comparison, preserving the K accumulation.
4. P4/P5: Add/Sub/Mul/Div, RoPE, SwiGLU, RMSNorm (normal and two-pass Split-K),
   FlashAttention (causal and non-causal), each with a genuine pipeline version.
5. P6: depth/order candidates, regenerated explicit schedules, independent buffer
   reuse experiments. Never copy ChunkScan's 22-statement schedule arrays.
6. P7: independent output tasks, workitem mapping and multicore ABI validation.
7. P8/P9: remote BM1690 single-core correctness first, then synchronous launch+
   synchronization latency, followed by independently reported multicore data.

Preserve original, same-tile serial, pipeline-only, additional-optimization and
multicore variants. No speedup threshold is required. Packing, if introduced,
must have its cost and reuse assumptions reported separately.

Initial target workloads use FP16 and 1024x1024 2-D tensors, Matmul M=N=K=1024,
and proposed attention B=1,S=1024,H=1,D=64 (1024x1024 attention scores).
Small functional cases precede target sizes. Existing dtype, rounding, layout,
epsilon and exact-tiling contracts remain authoritative.

## TPU pipeline contract

`tilelang/engine/tpu_pipeline.py` is a target-specific scheduling/injection path,
called before allocation placement. The old generic injector is still disabled:
its descriptor Let aliases are incompatible with this branch's typed ABI.
The new path models `tl.tpu.*` effects, accepts independent full-tile global
loads and an ordered local compute chain with trailing stores, and rejects
unknown effects, escaping prefetched values, global recurrences and mutable
prefetch destinations. Two/three stages use separate Allocate-owned buffers.
Explicit schedules must match the lowered-body fingerprint and pass conservative
RAW/WAR/WAW checks; arbitrary stage assignments are intentionally unsupported.

The steady loop prefetches a later iteration into a distinct version and computes
the current iteration in a parallel scope. Output stores remain outside that
scope. AddressAssign extends the live intervals of all concurrent buffers, and
the final IR verifier independently rejects overlapping producer/consumer data.
Generated C retains `tpu_parallel_start/end` under `#ifndef USING_CMODEL`.
The existing CModel build defines `USING_CMODEL`: numerical validation therefore
does not validate physical overlap or the board's synchronization implementation.
