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

## Validated single-core candidates

All six families have FP16 serial and pipeline implementations. The original
15-case FP16 suite passes after the changes. Same-tile serial and depth-two
pipelines pass the target workloads above; attention also covers descending
maxima, nonuniform key weights, both mask modes and a separate multihead smoke
test. Outputs are bitwise equal to the same-tile serial version and satisfy
the original demo's independent-reference tolerances.

Depth-three pipelines with reversed independent load order pass all 17 smoke
cases. Explicit arrays are generated from each legalized loop by
`bind_explicit_schedule`, then revalidated during compilation. Split-K's two
loops have separate contracts; attention's online state remains ordered.

```bash
.risc/bin/python -m tpu_demo.pipeline.run --suite pipeline --stages 3 \
  --schedule reverse-loads --output research/artifacts/bm1690-pipelines/depth3-new
.risc/bin/python -m tpu_demo.pipeline.run --suite pipeline --case swiglu \
  --size target --reuse-swiglu-buffers --stages 3 --schedule reverse-loads \
  --output research/artifacts/bm1690-pipelines/swiglu-reuse-new
```

The SwiGLU experiment separately compares serial, pipeline, reuse-serial and
reuse-pipeline. Its FP32 arithmetic/rounding and exp scratch contract are
unchanged; target-size results are bitwise equal. Neither schedule candidates
nor storage reuse have a measured BM1690 speedup yet.

The runner rejects concurrent matrices from the same user and rejects a run
whose source or native-library hashes change during execution. Full C/TIR/logs
remain in the local artifact directories; portable summaries and schedule
reports are under `research/bm1690-pipelines/results/`.

## Workitems and local memory

`tpu_demo/pipeline/workitems.py` partitions independent output tasks as
`task % cores`. The launch count is a validated PrimFunc attribute carried
through to `kernel.cpp`: one argument structure per workitem, `group_num=1`,
`block_num=cores`, and the complete argument-array byte count. Typed workitem
queries are restricted to BM1690 TPU-Kernel. CModel topology remains eight
cores even when the launch uses one, two or four workitems.

Matmul output tiles, Split-K row groups and attention batch/head/query tiles
retain their complete reductions on one workitem. Elementwise, RoPE, SwiGLU
and normal RMSNorm stride across flat output tiles. These flat pipelines
currently require task counts divisible by the launch count and enough tasks
per workitem for a steady state; unsupported tails fail explicitly. Original
kernel-grid mapping handles uneven output-task counts with a final guard.
Multicore smoke shapes can grow to provide enough iterations; exact dimensions
are always recorded in each result.

The compiler reports allocator-derived sizes and address high water in bytes
**per lane**, against the 256 KiB per-lane address space. Before C generation,
it also verifies that future DMA destinations do not physically overlap current
compute buffers or another producer. This complements the descriptor/effect
checks and allocator lifetime extension.

CModel evidence includes 2/4/8-workitem ownership probes (11 tasks, exposing
uneven ownership), all 17 dual-core smoke cases, and 11 eight-core target cases
covering all six families and both attention mask modes. Every multicore test
compares single-core serial/pipeline and multicore serial/pipeline, checks the
independent reference, and requires bitwise equality among the four outputs.
No multicore speedup is inferred from CModel wall time.
An additional eight-case matrix passes with four cores, depth three and
regenerated reversed-load schedules. The final original FP16 regression is
15/15; all 22 compile/runner tests pass, including rejection of corrupted
physical buffer addresses. Logs and manifests preserve each tested source
hash even when the corresponding run preceded its final Git commit.

```bash
.risc/bin/python -m tpu_demo.pipeline.run --suite workitems --cores 2 \
  --output research/artifacts/bm1690-pipelines/abi-new
.risc/bin/python -m tpu_demo.pipeline.run --suite pipeline --cores 2 \
  --output research/artifacts/bm1690-pipelines/two-core-new
.risc/bin/python -m unittest discover -s testing/python/jit \
  -p 'test_bm1690_*unittest.py' -v
```

The two changed native compiler files require an incremental build using the
existing build tree before these commands. On this workstation:

```bash
taskset -c 0,1 nice -n 10 .venv/bin/cmake --build build-tpu --parallel 1
```

## Remote BM1690 handoff (pending)

The local source work and CModel checks above do not complete P8/P9. The remote
BM1690 is unavailable, its existing ChunkScan SDK/runtime paths are not yet
recorded, and GitHub HTTPS authentication is pending. Local commits are retained
on `feature/bm1690-pipelines`; `delivery` points to the user's new repository.
No source checkout, installed SDK, or existing environment was overwritten.

After the server returns, activate the known working ChunkScan environment and
run this **read-only** inventory script (a standalone stdlib Python file):

```bash
python /path/to/delivery/tpu_demo/pipeline/collect_environment.py \
  --repo /path/to/validated/ChunkScan --output bm1690-environment.json
```

If SDK/runtime variables are absent, it records missing values. Supply verified
paths via `--sdk` and `--pcie-runtime` when available. The script does not import
the vendor runtime, install dependencies, compile, or launch a TPU kernel. It
records hardware-query utility help and PCI inventory where available; the
actual BM1690 device id must also be recorded before any PCIe dispatch.

The next implementation step is a dedicated BM1690 PCIe entry using the already
chip-aware `LibraryGenerator.tpu_compile_pcie` and the validated workitem wrapper.
The existing demo profiler policy should not be relaxed globally: this first
round needs correctness and synchronous host-call latency, without requiring
TPUDNN profiling. The installed runtime must be distinguished from the SDK's
CModel runtime using `PPLLayout.pcie_runtime_lib`.

The timing entry will allocate/upload once, warm up, and record each invocation
of the wrapper containing launch plus `tpuRtStreamSynchronize` with
`steady_clock`. Download, allocation, compilation and host reference work stay
outside the timed interval. Start with one core, 5 warmups and 20 samples for
correctness/latency smoke; then use configurable 3 rounds of 10 warmups and 100
samples with a pause between rounds, recording median, IQR, p95 and raw samples.
These are planned sample counts, not measured results or performance thresholds.
Keep original demo, same-tile serial, pipeline-only, reuse and multicore variants
separate. Compare pipeline benefits at identical shape/dtype/tiling/core count;
compare multicore scaling separately. No prepacking has been introduced.

Begin remote validation with original single-core correctness, then same-tile
serial, then pipeline, and only then two cores. Retain the existing CPU/RSS/
timeout guards and fail-fast behavior. A failed hardware launch must stop the
matrix for manual inspection; killing a host worker is not proof that a board's
firmware recovered. Four/eight-core board runs follow only after the lower-core
checks and remote-desktop responsiveness are confirmed. Hardware overlap claims
require separate trace/profiler evidence and are outside the first acceptance.
