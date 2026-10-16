# BM1690 pipeline migration

Route A starts from `xwhzz/tilelang-tpu`, branch
`feature/sg2260e-rv-support`, commit
`e79e06c04c592014127d5eadd0b2f48777fe07ca`. The reference is
`arcflute/ChunkScan-2-TileLang-4-TPU-bm1690`, main commit
`683fb539cb14005ce3aba967d8c392a38fd1affe`.

The source owner reports that ChunkScan stages other than P10 have completed
validation. This project migrates those mechanisms; it does not restart that
project's environment bring-up. P10 is not complete performance evidence.

## Current test handoff (2026-10-16)

The user requested resumption after the P10-related deferral. At Oct 15 15:19:40 +0800,
the board still runs kernel 7.0.0-34-generic, sgcard is live with reference count
zero, both device nodes/PCI bindings exist, and privileged `fuser` reports no
holders (empty output, exit 1). This establishes no detected host device users,
not firmware recovery. The journal contains further FORCE_QUIT timeouts at
Oct 14 20:05:47 and Oct 15 14:33:33, with preceding task-wait and close-time
`load_lib_mutex` warnings. Follow-up source inspection shows that `sg_close`
unlocks after that warning and continues host cleanup even after FORCE_QUIT
fails. The normal `pci_exit`/`pci_remove` entries exist, and the existing module
hash/vermagic and `card_num=64` were reverified. The user subsequently completed
the gated, non-force unload/reload successfully: both commands returned zero,
and fresh logs show Chip 0 initialized at 16:53:49, Chip 1 at 16:53:59 and all
two devices probed successfully. Both nodes/PCI bindings were restored. The
subsequent original-only RoPE 1024 host diagnostic built all three variants,
verified device 0 / PCI 0000:01:00.0 and passed initialization, module loading,
four allocations and four uploads. It then timed out at 30 seconds after the
CALL marker for `tpuRtKernelLaunch`, without a RETURN marker or any subsequent
explicit stream synchronization. The supervisor reports failure / timeout /
worker -9; its host process-group cleanup completed. The fresh journal again
reports task-wait interruption and a FORCE_QUIT timeout. Thus driver
reinitialization did not resolve this target case, and its failure cannot be
explained solely by the earlier concurrent P10. No users were detected before
dispatch or at the 17:03:58 follow-up; this is not continuous exclusivity or
firmware health evidence. At that point further board dispatch awaited recovery assessment and
a bounded diagnostic, not an identical full-target retry. Evidence is
`results/p8-rope-1024-host-trace-board-user-reported.json`; the preceding
reinitialization remains recorded in `results/p8-driver-reload-20261015-user-reported.json`.

The first bounded case is `rope-target-window8`: full 1024x1024 allocations and
strides, with only eight original tiles computed. Its one-call CModel check and
149 regression checks passed. The user has now built all three variants at
revision `c3e7ee870d3254eabcf79b045e19a494425918b5`: 18 build steps, prepare exit
zero, runtime not loaded and zero kernel launches. The remote work directory is
`/home/bokai/bm1690-rope-window8-BTILGy/work`. At Oct 16 09:48:29 +0800 the kernel
and module hash still match, both bindings exist, sgcard is live with reference
count zero, and privileged fuser finds no users. The prior small RoPE runner
hash and saved artifact paths are intact. These checks do not establish firmware
health after the Oct 15 FORCE_QUIT timeout. The user then completed another
gated ordinary unload/reload, both returning zero. Fresh logs show Chip 0
initialized at Oct 16 09:56:04 and Chip 1 at 09:56:14, followed by all two devices
probed successfully. Both nodes/bindings were restored, with sgcard live and
reference count zero. The recovery log directory is
`/home/bokai/sgcard-reload-h7ezio`; evidence is
`results/p8-driver-reload-20261016-user-reported.json`. This establishes driver
reinitialization. A fresh 8x128 original smoke call subsequently passed at
`/home/bokai/bm1690-rope-recheck-4r1ONZ/run-original`: device 0 mapping verified,
host call returned, reference matched bitwise with zero error, and input/output
hashes matched the prior smoke. Worker and runner exited zero. The console's
1.095 ms is one host-call value, not a performance acceptance result. Evidence
is `results/p8-rope-smoke-post-reload-20261016-user-reported.json`. The subsequent
window8 original diagnostic passed on the board in
`/home/bokai/bm1690-window8-run-cUiK43/run-original`: all 22 traced runtime calls
returned, the 1024 active elements matched reference bitwise, and all 1047552
unwritten elements retained their sentinels. Input/output hashes matched CModel.
This is partial-window acceptance; full-target correctness and the cause of the
long-loop timeout remain unresolved. Evidence is
`results/p8-rope-1024-window8-board-user-reported.json`.

The subsequent `rope-target-window8-end` also passed on the board, at
`/home/bokai/bm1690-rope-end8-XmD4lQ/work/run-original`: 18 build steps, one
original call, zero reference error, unchanged outside sentinels and 22 matched
runtime CALL/RETURN pairs. Its input/output hashes match CModel. Evidence is
`results/p8-rope-1024-window8-end-board-user-reported.json`. Thus both small
beginning/end windows execute correctly; full-target correctness is still open.

The original-order prefixes of 128, 256 and 512 tiles have now passed on the
board at revision `368d7f5473a0a2df7c6f96267c865e6ab0eda60c`, under
`/home/bokai/bm1690-rope-prefix-AwTbR3/work`. Each returned all 22 traced runtime
calls, matched its reference bitwise and preserved every outside sentinel.
All three input/output hashes match CModel. The largest prefix covers the
first 64 full rows, 6.25% of the original target computation. This is no timing
or full-target acceptance; the original 8192-tile failure remains unexplained.
Evidence is `results/p8-rope-1024-prefixes-board-user-reported.json`.

The subsequent 1024/2048/4096 board sequence at revision
`5dd077018056a079cb0483f7c9416de150b56f23` passed its first two calls and
stopped at the third. Work is under
`/home/bokai/bm1690-rope-prefix-large-dVkaOi/work`. Prefixes 1024 and 2048
matched reference bitwise, preserved all outside sentinels, matched CModel
input/output hashes and returned all 22 runtime calls. They cover the first
128 and 256 rows: at most 25% of the complete original computation.

The 4096 case built successfully and passed initialization, module loading,
allocations and input copies. Its 13th CALL, `tpuRtKernelLaunch`, has no RETURN
marker; the runner reported failure and stopped the sequence. No explicit
stream synchronization, output/reference check or timing was reached. The
follow-up result.json confirms `reason=timeout`, worker `returncode=-9`, and
`cleanup_complete=true`, with a 30-second overall worker deadline. Peak group
RSS was 274128896 bytes, below the 4 GiB cap; the recorded stop reason was not
RSS. The partial worker checkpoint still says running because the process did
not complete the host call. Host cleanup does not establish device completion
or firmware recovery. At Oct 16 11:22:50 +0800, the user reported sgcard live with reference count
zero and empty privileged fuser output (exit 1) for both device nodes. The
journal reports task-wait interruption at 11:07:31 and FORCE_QUIT timeout at
11:08:34. No holders were detected, but firmware recovery is unverified.
Reuse the previously successful gated ordinary module reload before a fresh
small correctness call and any full tiled-variant dispatch. This runner does not save elapsed wall time, so the
previous request to extract successful-case durations from result.json is
withdrawn. The deadline is not a measured device execution time.
The original 2048/4096 sources differ only in bx's upper bound, 64 versus 128.
This narrows observed passing/failing cases without establishing a monotonic
threshold, counter overflow or synchronization defect. Evidence is
`results/p8-rope-1024-prefix-large-board-user-reported.json`.

All three larger cases had passed local CModel and 229 selected checks. Preserve
that evidence separately from board results; the CModel uses PPL 1.7 while the
board uses the inspected PPL 1.4 environment. The failed board case is not a
pipeline schedule: only the original variant was dispatched. Full original,
serial and pipeline target acceptance remains open.

Add/Sub/Mul/Div have completed their first board correctness and bounded
synchronous-call latency acceptance in the recorded FP16 1024x1024, one-core
cases. Div is the latest: all 18 build commands and six test processes passed.
All three Div outputs are finite and bitwise equal to each other; maximum
absolute error against the independent reference is 0.00390625, within the
original demo's `atol=rtol=0.01` contract. The work directory is
`/home/bokai/bm1690-div-UiXs7C/work`. Evidence and all 60 raw samples are in
`results/p8-div-1024-{build,three-variants-board,latency}-user-reported.json`.

Div median synchronous-call latencies are 132.6975 / 138.413 / 126.023 us for
original / same-tile serial / pipeline, with five warmups and 20 samples each.
The pipeline median is 8.951% below serial and 5.030% below original in this
single fixed-order round. Its 210.819 us maximum is retained; repeatability,
pure device execution time and physical overlap remain unverified. Earlier
Add/Sub/Mul results, including regressions, remain recorded separately.

The host now runs `7.0.0-34-generic`. A separately built sgcard 1.9.3 module with
SHA256 `0c66f5a553c903d8bfdf31061e06be7d20fbde7b617f6765dec5a8dadfb7ad65`
was temporarily loaded successfully. Both chips initialized, device nodes and
PCI bindings were restored, and all six Mul processes verified runtime/device
mapping before dispatch. The recovery record is
`results/p8-resume-kernel-change-user-reported.json`. Do not silently combine
these measurements with earlier `7.0.0-31-generic` measurements as a controlled
comparison. No permanent driver installation was performed.

Matmul has now completed its first board correctness and synchronous-call
latency acceptance at both M=N=K=128 and M=N=K=1024, FP16, one launch core.
For the 1024 target, all 18 build commands and all six test processes passed.
Maximum absolute error is 0.0009765625 with zero tolerance mismatches for each
variant; serial/pipeline are bitwise equal, and each variant's output hash
matches its CModel record. Evidence and 60 raw samples are in
`results/p8-matmul-1024-{build,three-variants-board,latency}-user-reported.json`.
The remote target directory is `/home/bokai/bm1690-matmul1024-yckfLu/work`.

Target original / same-tile serial / pipeline median synchronous-call latencies
are 445.92678 / 56.9566865 / 52.667097 ms. Pipeline is 7.531% lower than serial
in this one fixed-order round. Original uses 16-cubed tiles, versus 32-cubed for
serial/pipeline; original-to-candidate ratios include tiling changes. The smoke
round instead showed a 3.522% higher pipeline median. Retain both results;
repeatability and physical overlap remain unverified.

Ordinary RMSNorm has completed its first board smoke acceptance at FP16 16x128,
one TPU launch core. All 18 build commands and six correctness/timing processes
passed in `/home/bokai/bm1690-rmsnorm-AHShF4/work`. All three outputs match the
reference bitwise with zero absolute error; their input/output hashes also match
CModel. Original / same-tile serial / pipeline median synchronous-call latencies
are 84.4125 / 86.265 / 83.5965 us (five warmups, 20 samples each). Pipeline is
3.093% below serial in this one fixed-order smoke round, with a retained maximum
of 115.657 us. The median difference is smaller than the observed IQRs; stable
speedup and target-size performance remain unverified. Evidence and all 60 raw
samples are in `results/p8-rmsnorm-smoke-{build,three-variants-board,latency}-user-reported.json`.

Split-K RMSNorm has also completed board smoke acceptance at FP16 16x128,
with all 18 build commands and six test processes passed in
`/home/bokai/bm1690-rmsnorm-splitk-zr1ONP/work`. Outputs match the reference and
each other bitwise, maximum error zero. Median synchronous-call latencies are
172.5465 / 176.8495 / 173.623 us for original / serial / pipeline. Pipeline is
1.824% below serial in this one fixed-order round, but original and serial have
identical kernel sources at this smoke tiling. The original 1043.515 us and
pipeline 939.563 us maxima remain in the record; their causes are unverified.
This is correctness and timing acceptance, not an established stable speedup.
Evidence is `results/p8-rmsnorm-splitk-smoke-{build,three-variants-board,latency}-user-reported.json`.

Both 16x128 FP16 RMSNorm handoffs previously passed exact-source CModel checks.
All three variants in each case match the scalar reference and CPU PyTorch
bitwise, with zero absolute error. The input includes zero and small-magnitude
rows and nonuniform per-element weights. The original demo requires weights
with the same shape as the input, epsilon 1e-12, and FP16 rounding of the
normalized value before weight multiplication; these contracts are retained.
Evidence is `results/p8-rmsnorm-smoke-timing-abi-cmodel.json` and
`results/p8-rmsnorm-splitk-smoke-timing-abi-cmodel.json`.

Ordinary RMSNorm has now completed target board acceptance at FP16 1024x1024,
one TPU launch core, from revision `180b987e834b8cd6e3d13f8fa089a7513771dfb3`.
All 18 build commands and six correctness/timing processes passed in
`/home/bokai/bm1690-rmsnorm1024-XOF4Zd/work`. All three outputs are bitwise equal
to each other and match their CModel hashes, with maximum reference error
0.0009765625 and zero tolerance mismatches. Median synchronous-call latencies
are 1787.4105 / 273.9225 / 262.1675 us for original / same-tile serial / pipeline.
Pipeline is 4.291% below serial in this one fixed-order round. Original uses
4x1024 tiles and candidates use 32x1024, so the original-to-pipeline ratio of
6.818 includes tiling changes. Repeatability and physical overlap are unverified.
All 60 raw samples are retained in
`results/p8-rmsnorm-1024-latency-user-reported.json`.

Split-K RMSNorm has now also completed FP16 1024x1024 board acceptance in
`/home/bokai/bm1690-splitk1024-dfggi5/work`, from revision
`180b987e834b8cd6e3d13f8fa089a7513771dfb3`. All 18 build commands and six test
processes passed. Each output hash matches CModel; serial/pipeline are bitwise
equal, while original differs within tolerance because of different reduction
grouping. Maximum reference error is 0.0009765625, with zero tolerance mismatches.
Median synchronous-call latencies are 59026.27 / 1881.3245 / 1851.584 us.
Pipeline is 1.581% below same-tile serial in this one fixed-order round. The
31.879 original-to-pipeline ratio includes row/K tile changes from 4x32 to
32x128. Evidence and all 60 raw samples are in
`results/p8-rmsnorm-splitk-1024-{build,three-variants-board,latency}-user-reported.json`.

Three of the six operator families have completed their first single-core board
target correctness and synchronous-call latency acceptance: elementwise,
Matmul and RMSNorm (ordinary and Split-K). This is not completion of the entire
performance/multicore matrix. Next is RoPE smoke, followed by RoPE target,
SwiGLU and FlashAttention. Repeated varied-order rounds and multicore board
acceptance remain later work.

RoPE has completed its FP16 8x128 board smoke in
`/home/bokai/bm1690-rope-BMzRZC/work`, revision
`9a596d4f971dd124945a67db261e406dbd904a1a`. All 18 build commands and six test
processes passed. All outputs match the reference and each other bitwise,
maximum error zero, and input/output hashes match CModel. Original / same-tile
serial / pipeline median synchronous-call latencies are 995.9885 / 1025.3975 /
909.761 us (five warmups, 20 samples). Pipeline is 11.277% below serial in this
single fixed-order smoke round; the 1159.521 us maximum is retained. Stable
speedup and physical overlap remain unverified. All 60 raw samples are in
`results/p8-rope-smoke-latency-user-reported.json`.

Next is RoPE 1024x1024 board acceptance. Its three exact-source variants passed
CModel with output/reference bitwise equality and maximum error zero. Original
retains 4x32 tiles; serial/pipeline use the previously implemented 32x64 target
tiles. FP32 1024x512 sin/cos tables, interleaved pairing and the four-argument
ABI are preserved. All 141 relevant regression checks passed. Use a new pinned
`prepare_add_1024.py` with `--case rope-target`; the old smoke helper does not
support this case. All three target board builds subsequently passed, but the
original one-call execution timed out at the 90-second supervisor deadline.
Host cleanup is confirmed. A later kernel-journal report shows
`send_quit_request: FORCE_QUIT default timeout` at 18:12:20. Device task cleanup
and readiness remain unverified; inspect the driver error path and current
device users before further board dispatch. The original-only host-call
diagnostic is prepared and CModel-validated, but has not run on the board.
Follow-up source inspection confirms a 60-second exit-response wait. `fuser`
reports PID 2737445 (`pt_main_thread`) holding device 0, and sgcard's reference
count is 4. Process inspection subsequently identified a separate ChunkScan P10
run, started at 18:04:44 with core counts 1,2,4,8 and now waiting in
`rt_task_done_response_sync`. The user confirms simultaneous TPU testing.
The user chose to keep P10 and defer migration board testing. Preserve that
task and the prepared diagnostic; perform no process termination, driver unload
or reset. Resume after other TPU work ends and device availability/health is
checked. This does not establish concurrency as the sole RoPE timeout cause.
The runner/reference use standard Python. Keep one-call correctness before five
warmups/20 resident samples, with the explicit `--timeout-seconds 90` process
limit (default remains 30); CPU/RSS/core limits are unchanged. Stop on first
failure. The current failed attempt is retained at
`/home/bokai/bm1690-rope1024-ehiG4z/work/run-original`; do not rerun or advance
variants until device state has been assessed. The supervisor has confirmed
timeout and host cleanup; that is not evidence of TPU completion or reset. SwiGLU and FlashAttention follow RoPE target acceptance.

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
BM1690 is now reachable through the user's manual terminal workflow. GitHub
HTTPS authentication and repository write access are verified.
The delivery repository is
[`arcflute/tilelang-tpu-bm1690-pipelines`](https://github.com/arcflute/tilelang-tpu-bm1690-pipelines),
branch `main`. The local development branch is `feature/bm1690-pipelines`;
`delivery` is its delivery remote, while `origin` retains the upstream URL.
No source checkout, installed SDK, or existing environment was overwritten.

For a new environment, the inventory entry is a **read-only** standalone stdlib
Python file, run with the existing ChunkScan environment:

```bash
python /path/to/delivery/tpu_demo/pipeline/collect_environment.py \
  --repo /path/to/validated/ChunkScan --output bm1690-environment.json
```

If SDK/runtime variables are absent, it records missing values. Supply verified
paths via `--sdk` and `--pcie-runtime` when available. The script does not import
the vendor runtime, install dependencies, compile, or launch a TPU kernel. It
records hardware-query utility help and PCI inventory where available; the
actual BM1690 device id must also be recorded before any PCIe dispatch.

### Remote preflight evidence, 2026-09-30

User-supplied terminal output identifies the existing ChunkScan checkout as
main commit `f7df97ebb4b3bbb8b6fcd866a137a887e4710583`. Its TVM submodule
modifications have the same stable patch ID as its own `patches/tvm.patch`;
they are retained. The remote environment uses PPL
`v1.4.195-geb2acdd0-20250220`, Python 3.12.3, CPU PyTorch 2.3.1, NumPy
1.26.4, and installed TPUv7 runtime 1.9.3.

The existing driver 1.9.3 binary was built for kernel 6.17.0-35, while the
server runs 7.0.0-31. The user rebuilt the existing driver source in a separate
directory with the matching installed headers and GCC 13, retaining its prior
source modification. The new module was temporarily loaded: both PCI functions
bound to `sg-host-drv`, the module became live, and both chips completed AP/TP
firmware initialization. A single-shot SMI JSON query reports one MT00 card,
two chips, both Active. Chip1 temperature/voltage fields report the literal
string `F`; its meaning has not been established. No hardware serial numbers
or raw machine logs are published here. Driver persistence and the broken old
DKMS 1.2.7 entry have not been changed.

The driver evidence establishes initialization and management visibility. The
subsequent runtime enumeration used
[`probe_bm1690_runtime.cpp`](../tpu_demo/pipeline/probe_bm1690_runtime.cpp).
Compile it as a small host executable with the installed board runtime header,
then run it with the absolute board `libtpuv7_rt.so` path and a device-query
limit (2 for this observed topology). It initializes the runtime, queries the
actual device count, and checks candidate IDs through SetDevice/GetDevice,
properties and the borrowed device fd. It prints the actual loaded library,
raw PCI fields, and host sysfs paths, without assuming how the SDK encodes a
PCI function. A missing sysfs mapping remains unresolved. The utility performs
no TPU memory allocation, module loading, kernel launch, or synchronization.

Example commands, using already verified site paths:

```bash
c++ -std=c++17 -O0 -I"$BM1690_RUNTIME_ROOT/include" \
  tpu_demo/pipeline/probe_bm1690_runtime.cpp -ldl -o /path/to/new/probe
timeout -k 2s 15s taskset -c "$ALLOWED_TWO_CPUS" nice -n 10 \
  /path/to/new/probe "$BM1690_RUNTIME_ROOT/lib/libtpuv7_rt.so" 2
```

Use fresh output files, capture the exit status and both output streams, and
stop after any failed query. The host-tool tests in
`testing/python/jit/test_bm1690_runtime_probe_unittest.py` use an isolated fake
runtime to exercise early failures, bounded enumeration and fd ownership;
they are neither CModel nor board validation.

The user then compiled and ran the pinned probe from commit
`b21b279b49a647874ffd403348475a7f530081ea` against the installed runtime 1.9.3.
Compilation and execution both returned zero. The loaded library resolved to
the installed board runtime, the property structure size was 56 bytes, and
every queried API returned success. The actual mapping is:

| Runtime device | Character device | Canonical PCI function |
| --- | --- | --- |
| 0 | `/dev/sg-host-drv-0` | `0000:01:00.0` |
| 1 | `/dev/sg-host-drv-1` | `0000:01:00.1` |

Both report name `MT00` and 111132278776 bytes of global memory. Both raw PCI
property tuples are `(domain=0, bus=1, device=0)`; they do not distinguish the
functions. The fd/sysfs mappings above do. Initial operator validation will
use **device 0, one launch workitem**. Enumeration does not validate kernel
loading, arithmetic, latency, physical overlap, or persistent driver setup.

The PPL 1.4 paths, helper source, firmware archive, pipeline/workitem declarations
and board launch/synchronization declarations have been inspected. The existing
default `PPLLayout` continues to use PPL 1.7. P8 now adds an explicit
`TILELANG_TPU_PPL_PROFILE=ppl14-bm1690-pcie` profile, implemented in
`tilelang/jit/adapter/legacy_pcie.py` and selected by `LibraryGenerator`.
It requires the verified PPL root, absolute `TILELANG_TPU_PCIE_RUNTIME_PATH`
(the installed runtime's lib directory), and absolute
`TILELANG_TPU_PCIE_CROSS_GCC` (Linux RISC-V GCC). The legacy profile uses
`__bm1690__`, PPL 1.4 headers/helper and `libbm1690.a`, without the PPL 1.7
LTO flags. Its host header and linked runtime come from the same installed
runtime root. It rejects CModel, RV, SG2260E, and unvalidated legacy profiling.
No missing PPL 1.7 chip map silently selects the legacy ABI. Existing PCIe
load/device identity gates remain in force.

### P8.1 source-only compatibility handoff

`tpu_demo/pipeline/export_add_sources.py` exports three single-core FP16 Add
implementations at shape 8x128: original whole-block, 4x32 tiled serial, and
the same tiling with two pipeline stages. The bundle at
`research/bm1690-pipelines/handoff/add-smoke-sources.json` includes all generated
C/C++/header files, source hashes, native compiler hashes and lowered pipeline
reports. Its base commit plus dirty-source hashes identify the exact generating
tree; the containing delivery commit pins the final handoff. It contains no
vendor headers, SDK binaries or board-produced shared libraries.

The exact generated sources passed the existing PPL 1.7 CModel: all three
outputs are bitwise equal and match the independent FP32-add/FP16-rounding
reference. The portable report is
`research/bm1690-pipelines/results/p8-add-source-bundle-cmodel.json`.
34 stdlib regression tests pass, including six new legacy-profile/standalone
failure tests. An additional mocked-command check verifies JIT profile
selection and runtime identity; it is not a real legacy compilation.

On the remote machine, download the pinned `legacy_pcie.py` and source JSON,
verify both published SHA256 values, and run the standalone module with the
already recorded SDK/runtime/compiler environment variables:

```bash
timeout -k 5s 180s python3 legacy_pcie.py \
  --bundle add-smoke-sources.json --output /path/to/new/build-directory
```

This command imports only the Python standard library, limits itself and its
children to two allowed CPUs, reduces scheduling priority, and limits address
space to 4 GiB per process. Compilers run sequentially with a 60-second timeout
per process group. It checks the GCC target triple, validates all source
hashes, records actual SDK/header/library/compiler hashes, retains per-command
logs and partial result JSON, and stops at the first failure. It never calls
dlopen, allocates TPU memory, loads a module or launches a kernel. Existing
output directories are refused. The full handoff has an outer 180-second
timeout. No Python dependencies or new TVM build are required for this check.

The user reports **successful PPL 1.4 compilation/linking** for all three Add
variants from delivery commit `53f04f868aa51f08f370ae1faa61fbf1d46c47b0`:
all 18 compiler/linker commands completed, `BUILD_EXIT=0`, and
`BUILD_ONLY_OK variants=3 board_runtime_loaded=false kernel_launches=0`.
The builder recorded artifact/SDK hashes in the remote `build/result.json`;
those actual hash values have not been returned to this development host.
This establishes Add build compatibility, not successful device execution or
compatibility of every operation used by the other five families.

### P8.2 bounded original/serial/pipeline correctness entry

`tpu_demo/pipeline/run_add_pcie.py` is a standalone stdlib-only runner for the
**exact P8.1 build above**. Keep the downloaded `build.py`, `add.json` and
`build/` in their original location. Before loading a vendor library, it checks
the pinned builder/bundle hashes, all generated sources, both ELF artifacts
per variant, the full successful compilation recipe including embedded paths,
and the current SDK/runtime files against the recorded hashes. Relocated,
incomplete or changed builds fail. The generic JIT's refusal of arbitrary
prebuilt TPU artifacts and the existing demo profiler policy are unchanged.

An explicit `--allow-pcie --device-id 0 --expected-pci 0000:01:00.0` is required.
A fresh worker checks the loaded runtime's actual path, resolves the runtime's
borrowed fd through sysfs again, and binds the generated host module to that
same device before calling it. It rejects inherited loader overrides and
CModel topology settings. No TPUDNN profiling is enabled.

Each invocation executes one variant, one host call, one launch workitem,
FP16 shape 8x128, with no warmup or extra benchmark calls. Inputs are generated
by a deterministic integer sequence and converted to binary16; the independent
reference uses explicit FP32 addition followed by binary16 rounding. Output is
prefilled with NaNs; host staging buffers have boundary canaries. Results must
be finite and satisfy the original Add tolerances. Serial requires a passed
original receipt; pipeline requires a passed serial receipt, with matching
build/input/device identity and bitwise-equal output. These are host staging
checks, not proof that device input buffers are unmodified.

The parent monitors a private process group with a 30-second bound and a
conservative 4-GiB summed RSS bound. The execution worker uses at most two
allowed CPUs, reduced priority, and disabled core dumps. A separate Linux
parent-death guard cleans the group even if a vendor call blocks Python signal
handling. A same-user device-0 lock prevents concurrent runs of this entry.
Outputs must be new directories; `run.log`, checkpointed `worker.json`, final
`result.json`, and (on numerical success) `output.f16` are retained. An abnormal
exit is a failure even if a partial numeric result exists. No automatic retry,
driver reset, kernel replay or multicore selection is exposed.

Example after hash-verifying the runner, using the existing build location:

```bash
python3 run_add_pcie.py --build /path/to/build --output /path/to/new/original \
  --variant original --allow-pcie --device-id 0 --expected-pci 0000:01:00.0
```

Review the original board result before running serial with
`--variant serial --previous /path/to/passed/original`, then pipeline with
`--variant pipeline --previous /path/to/passed/serial`, each with a new output
directory. A host timeout does not prove firmware recovery; stop and inspect
any failure before proceeding.

Local evidence: 47 stdlib regressions pass, including 13 new manifest, host-ABI,
device-mapping and process-guard checks. The exact exported sources and this
entry's input/reference/staging functions pass all three variants on CModel;
outputs are bitwise equal and the stdlib reference matches CPU PyTorch. See
`research/bm1690-pipelines/results/p8-add-pcie-inputs-cmodel.json`. Fake-host
tests and CModel are explicitly separate from board validation. The generated
host template prints a single-call time; this entry
does **not** promote it to an accepted performance result. Synchronized-call
sampling, the six-family board matrix and P9 reporting remain incomplete.

The user has now run the **original Add** from the pinned P8.1 build using the
runner at `2d746196610c08010aa762f749fc4af2839521d1`. Device-0 mapping, source/
build/runtime validation, the host call and reference comparison all completed;
both worker and outer run returned zero. The 8x128 FP16 output is finite, with
zero mismatched elements and maximum absolute error 0, and is bitwise equal to
the independent reference. Its input/output SHA256 values also match the
recorded CModel run. The evidence is preserved in
`research/bm1690-pipelines/results/p8-add-original-board-user-reported.json`.
The printed 177-us host-wrapper time is one unwarmed diagnostic sample only.
The remote full result JSON and binary artifacts have not been copied here.
The subsequent user-supplied serial and pipeline runs also passed on the same
BM1690 device and build: reference error is zero, both previous-variant bitwise
comparisons are true, and all three output hashes agree with CModel. Both
workers and outer commands exited zero. The combined evidence is
`research/bm1690-pipelines/results/p8-add-three-variants-board-user-reported.json`.
The serial/pipeline printed 179/186 us are likewise unwarmed one-call diagnostics,
not an accepted performance comparison. This completes the 8x128 single-core
Add correctness smoke; larger shapes, other operators and overlap remain separate.

The new timing ABI allocates/uploads once and records each invocation of the
wrapper containing launch plus `tpuRtStreamSynchronize` with `steady_clock`.
Download, allocation, compilation and host reference work stay outside the timed
interval. The board entry uses one core, 5 warmups and 20 samples, with a required
matching one-call correctness receipt. The C++ ABI supports bounded counts;
configurable multi-round board sampling is still a later step (planned: 3 rounds,
10 warmups and 100 samples with a pause between rounds). Raw samples, median,
IQR and p95 are recorded. These sample counts are not performance thresholds.
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


### P8.3 1024x1024 Add and resident synchronous-call timing

The pinned `handoff/add-1024-sources.json` (schema v2) contains the original
whole-block Add, same-tile serial and two-stage pipeline at FP16 1024x1024,
launch cores 1. Serial and pipeline use 32x128 tiles (256 iterations). The
lowered schedule is order `[0,1,2,3]`, stage `[0,0,1,1]`; both input tiles
have two versions, output has one, and stores finish after the parallel region
before reuse. This is generated schedule evidence, not physical-overlap proof.
The existing 8x128 v1 bundle and old remote builds remain accepted unchanged.

`main_template.cpp::tilelang_tpu_run_timed` adds an explicit C ABI separate from
the original one-call function. It rejects invalid counts and profiling before
initialization, allocates/uploads once, synchronizes before warmup, and times
only synchronous `main_kernel` calls using a monotonic clock with double-valued
microseconds. It copies output back once and checks allocation, transfers,
launch/sync, free, module-unload and stream-destroy failures. A failed call stops
sampling and its measurements are rejected. This entry is for kernels such as
Add whose repeated execution uses immutable inputs; do not apply it unchanged
to an in-place or accumulating ABI. The original environment-driven diagnostic
benchmark is unchanged and is forced to zero by this runner.

`run_add_pcie.py` registers the exact v1/v2 bundle and builder hashes. Its new
`--measure --correctness /path/to/same-variant-correctness` mode requires a
passed receipt with matching build, inputs, shape, variant and mapped device,
checks the saved output, and verifies the final repeated-run output again.
Serial/pipeline also retain their `--previous` correctness-chain requirement.
No timing call runs by default. Same process-group, 2-CPU, nice-10, 4-GiB RSS,
30-second, device-lock and failure-stop limits apply to both modes.

Validation: 56 stdlib regressions pass, including the compiled fake-runtime
timing tests (25 calls, single allocation/upload, stop on warmup/sample and
cleanup failures, invalid-count/profile rejection). The frozen 53f04f8 builder
and v1 bundle also pass the new loader in a manifest-only fixture. Download
checks verify hashes before execution and reject altered bytes. These fixture
checks load no vendor library.

Local result `results/p8-add-1024-cmodel.json`: the exact exported sources pass
all three variants, all output bytes match the stdlib FP32-add/FP16-rounding
reference and CPU PyTorch, and maximum absolute error is zero. The timing ABI
was exercised with 1 warmup + 2 samples per variant, with unchanged correct
outputs; CModel timings are not reported as device-performance evidence.
The original whole-block 1024x1024 version compiled and executed on CModel;
all three versions also subsequently compiled against the older board SDK as
recorded below. All three versions have now passed 1024x1024 board correctness
and their first prewarmed synchronous-call latency sample set. That sample set
shows no speedup for the current tiled pipeline.

`prepare_add_1024.py` downloads and verifies the three pinned source/build/run
files into a **new** directory and compiles sequentially with the already
inspected bokai PPL 1.4, runtime 1.9.3 and Linux RISC-V GCC paths. It uses only
stdlib, refuses an existing directory, has no vendor-library loading or kernel
launch, and preserves download receipts and build logs on failure. Its own file
must be verified against the delivery SHA before execution. Downloads have a
45-second socket bound and up to three attempts for transient network errors;
hash/size failures and non-transient HTTP failures stop immediately. The compile
phase has a 180-second group bound and is never automatically retried. Wrap the
whole preparation in a 600-second timeout to bound slow downloads as well.

The first remote 1024 handoff at delivery `7c0efb8` stopped with a read timeout
while downloading `add.json`, after verifying `build.py`. The reported output
contains no compile commands; source ordering confirms compilation had not
started. At that point, large-shape board build compatibility was still pending.
The next delivery added `--resume` for that existing download directory:
it requires the same pinned source revision, verifies every retained file before
network access, fetches only missing files and saves a separate
`handoff-resume-*.json` receipt. The original `handoff.json` is preserved. A
directory containing `build/` or `build.log`, or a receipt recording a compile
attempt, cannot be resumed through this option. Resume does not change any
kernel, source bundle, builder or runtime entry. Six stdlib tests cover bounded
timeout recovery, retained-file integrity, preservation of the failure receipt,
download pins and refusal to repeat compilation.

The remote bootstrap download itself subsequently timed out before receiving
an HTTP status line, even with a 90-second socket timeout. No resume script was
written or executed. The cause (server, proxy, network path, etc.) is not yet
established. `--transport github-api` now selects the official Contents API
with `Accept: application/vnd.github.raw+json`, pinned by the same commit and
SHA256 values. It requests the bytes directly instead of following the JSON
`download_url` back to `raw.githubusercontent.com`. The ordinary github.com raw
URL also redirects back to that raw domain, so it is not a distinct transport.
See [GitHub's Contents API documentation](https://docs.github.com/en/rest/repos/contents#get-repository-content).
Development-host API retrieval is verified; reachability from the board host
was pending at that delivery and is confirmed by the subsequent handoff below.
No proxy/TLS settings or credentials are changed. Two
additional stdlib regressions check the API request, identical file hashes and
rejection of JSON metadata instead of blindly following its download URL.

The user subsequently completed the resumed **1024x1024 compilation** on the
BM1690 server. The helper from `f73c2c9` downloaded the pinned `7c0efb8` bundle
and runner through the GitHub API, reused the verified builder, and completed
all 18 compiler/linker commands for original, serial and pipeline. It reported
`RESUME_EXIT=0`, `board_runtime_loaded=false` and `kernel_launches=0`. The work
directory is `/home/bokai/bm1690-add1024-TPL9kW/work`; portable evidence is
`results/p8-add-1024-build-user-reported.json`. Full remote result JSON and ELF
artifacts have not been copied here. This confirms source/build compatibility
with the inspected old SDK, including compilation of the timing ABI. That build
receipt alone provides no large-shape execution or latency evidence.

The user then executed the **1024x1024 original Add** once on mapped device 0
(`0000:01:00.0`, `/dev/sg-host-drv-0`), FP16, launch cores 1. Every runner stage
completed, the worker and outer command returned zero, and the output is finite
with zero reference mismatches and maximum absolute error 0. Input and output
hashes match the retained CModel evidence. The portable record is
`results/p8-add-1024-original-board-user-reported.json`; full remote result JSON
has not been copied to this development host. The reported 219 us is one
unwarmed launch-plus-sync diagnostic, not an accepted latency measurement or
comparison with the earlier 8x128 cases. This run does not establish the larger
serial/pipeline kernels' correctness or hardware overlap.

The subsequent user-supplied **1024x1024 serial and pipeline** runs also passed
on the same mapped BM1690 device with one launch core. Both report finite
outputs, maximum absolute error 0, no reference mismatches and
`previous_variant_bitwise_equal=true`; all three input/output hashes agree
with CModel. Both workers and outer commands exited zero. The combined record
is `results/p8-add-1024-three-variants-board-user-reported.json`. This completes
the target-size single-core Add correctness check for these inputs and variants.
The printed serial/pipeline times, 777/719 us, are single unwarmed diagnostics.
Retain the original's 219-us diagnostic as well; do not infer a reproducible
speedup or an overall improvement from these three values. The comparison must
report both pipeline versus same-tile serial and each tiled version versus
original whole-block. No physical-overlap trace has been collected.

The user subsequently completed `--measure` on original, serial and pipeline,
in that order. Each invocation used 5 warmups and 20 measured synchronous calls
with resident device buffers, then verified the output again. All references
passed with zero error, outputs remained bitwise equal, and all worker and outer
commands returned zero. The complete raw sample arrays and correctness records
from the supplied terminal text are preserved in
`results/p8-add-1024-latency-user-reported.json`. Their statistics were separately
recomputed from the arrays; the full remote result JSON has not been copied here.

| Variant | Tile | Median (us) | IQR (us) | p95 (us) |
| --- | --- | ---: | ---: | ---: |
| Original whole-block | 1024x1024 | 115.302 | 2.141 | 117.628 |
| Same-tile serial | 32x128 | 632.039 | 8.275 | 643.917 |
| Pipeline, depth 2 | 32x128 | 654.429 | 17.556 | 678.482 |

This first measured sample set **does not show an optimization win**. Pipeline
median latency is 3.54% higher than same-tile serial, and 5.68 times the original
whole-block latency; serial is 5.48 times the original. These are ratios of the
sample medians, not claims of statistical significance or stable speedups.
All three variants share shape 1024x1024, FP16, the same inputs and one launch
core. The metric includes host launch plus stream synchronization, excluding
compilation, allocation, module loading, H2D, D2H and reference work. It is not
pure device time and provides no proof of physical DMA/compute overlap. One
fixed-order round cannot isolate host-load or time-order effects; repeated
order-varied rounds remain part of the later performance matrix.

The first Add correctness-plus-synchronous-latency acceptance is complete with
a negative performance result. It does not complete P8 for the other operations
or P9. Keep the original whole-block implementation as the faster observed
baseline for this case; no default dispatch is changed based on these samples.

### P8.4 bounded tile-size comparison

The measured baseline's loop iterates over 256 logical 32x128 tiles. Each tile
loads two inputs, performs an Add and stores an output; the pipeline adds its
parallel-scope synchronization. The original fits in local storage for this
case and executes whole-block operations. Extra tile-level work is a plausible
source of overhead, but timing alone does not establish its exact contribution.

`export_add_sources.py --target-size --coarse-tiles` exports a controlled
128x1024-tile comparison at the same global shape, dtype and launch count. The
serial and pipeline loops both have 8 tiles; the latter retains depth 2 and
stage `[0,0,1,1]`, two versions of each input and a single output. Generated C
has one initial prefetch, seven steady iterations, and the final drain. All
four original-baseline source hashes are identical to the previous bundle.
The compiler, allocation pass, runtime ABI, reference, inputs and timer are
unchanged. This is a candidate for measurement, not a promised speedup.

The exact bundle `handoff/add-1024-coarse-sources.json` passed the bounded local
BM1690 CModel: all three variants and their timing-ABI checks match the same
reference and output hash with error zero. Evidence is
`results/p8-add-1024-coarse-cmodel.json`.

The user subsequently completed the coarse bundle's compilation on the BM1690
server using the helper and sources pinned to `f7a4658`. The GitHub API downloads
and hash checks passed, followed by all 18 compiler/linker commands and
`PREPARE_EXIT=0`. The work directory is
`/home/bokai/bm1690-coarse-VRCQiY/work`; the supplied terminal output is recorded
in `results/p8-add-1024-coarse-build-user-reported.json`. The helper reported
`board_runtime_loaded=false` and `kernel_launches=0`. Full remote receipt JSON
and ELF hashes have not been copied here. That compile-only receipt establishes
neither board correctness nor latency.

The subsequent user-supplied original, serial and pipeline runs all passed on
BM1690 device 0 (`0000:01:00.0`, `/dev/sg-host-drv-0`) with FP16, shape 1024x1024
and one launch core. All outputs are finite and bitwise equal to the reference,
with zero mismatches and maximum absolute error 0. Serial and pipeline also
report `previous_variant_bitwise_equal=true`; input and output hashes match
each other and the coarse CModel record. All worker and outer exit codes are
zero. Evidence is
`results/p8-add-1024-coarse-three-variants-board-user-reported.json`, based on
terminal output; full remote result JSON has not been received here. This
completes coarse-bundle board correctness for the recorded inputs. The printed
217/250/242 us are single unwarmed diagnostics, not an accepted performance
comparison. That correctness-only run does not establish performance or overlap.

The user then completed one original/serial/pipeline timing round, with 5 warmups
and 20 resident synchronous calls per variant. All post-timing correctness
checks remained finite, bitwise equal and error-free; all exits were zero. Raw
arrays and independently recomputed statistics are retained in
`results/p8-add-1024-coarse-latency-user-reported.json` (user terminal evidence).

| Variant | Tile | Median (us) | IQR (us) | p95 (us) |
| --- | --- | ---: | ---: | ---: |
| Original whole-block | 1024x1024 | 114.588 | 3.694 | 122.458 |
| Same-tile serial | 128x1024 | 120.709 | 2.595 | 124.312 |
| Pipeline, depth 2 | 128x1024 | 120.596 | 8.674 | 136.146 |

The pipeline median is only 0.094% lower than serial, with larger IQR and p95;
this is no evidence of a stable pipeline speedup. It remains 5.24% above the
original whole-block median. Both coarse-tiled variants are much faster in this
round than their earlier fine-tiled measurements, consistent with tile overhead
being worth investigating; separate fixed-order rounds cannot establish a
causal breakdown or eliminate host-load effects. These remain launch-plus-sync
host timings, excluding compilation, allocation, module loading, H2D, D2H and
reference work. Physical overlap and pure device time are unverified. Retain
the faster observed original baseline and both tiled implementations. The
bounded Add tile comparison is complete; continue the other operations before
broader tuning or repeated/order-varied performance rounds.

The updated preparation helper selects this bundle with `--case coarse` and
continues to support `--transport github-api`. Use a new work directory; case
mixing on `--resume` is rejected. The new runner adds only the new bundle hash
to its allowlist and retains the previous builds. After compile-only success,
validate all three variants once and then collect the same 5+20 latency samples.
Run original again within the new build: the runner binds previous-variant
correctness receipts to the same bundle and build identity, even though the
original kernel sources are unchanged.
Retain both tile sizes, the whole-block baseline and every negative result.
Limit this to one controlled tile comparison before continuing the other five
families and remaining elementwise operations; repeated rounds and broader
tile/depth/multicore searches belong to the later performance matrix.

### P8.5 Sub/Mul/Div handoff

The existing original `build_elementwise` and `build_elementwise_tiled` now have
pinned PCIe source bundles for Sub, Mul and Div at FP16 1024x1024, one launch
core. Each preserves original whole-block, serial 128x1024 tiles and depth-2
pipeline 128x1024 tiles. Generated reports independently show eight iterations,
stage `[0,0,1,1]`, two versions each of inputs `a`/`b` and a single output `c`.
Generated C retains fill, steady loop, drain and parallel-scope synchronization;
the Div compute calls `tpu_bdc_fp_div(..., DT_FP16)`. No compiler pass, native
library, numerical kernel implementation or device launch ABI was changed.

The standalone legacy builder accepts a new operation-tagged schema. The
supervised runner obtains the operation from its hash-pinned bundle contract;
the caller cannot select a different reference for an existing binary. Original
Add bundle/build identities remain accepted. The historical script names
`export_add_sources.py`, `check_add_pcie_cmodel.py`, `prepare_add_1024.py` and
`run_add_pcie.py` are retained for existing handoffs; remote `add.json` and
`run_add.py` likewise remain wire filenames, with operation recorded in the
bundle and result. Select a new handoff using `--case sub`, `mul` or `div` and
a new work directory. Cross-case resume is rejected before download or compile.

The input generator uses deterministic FP16 inputs and a stdlib FP32-rounded
arithmetic reference, independently checked against Torch on the development
host. Div uses positive FP16 denominators in [0.5,2], following the demo's test
domain, and retains its `atol=rtol=1e-2`; Add/Sub/Mul retain 5e-3. Division timing
requires a matching passed correctness receipt whose output is checked again
under that same contract. Serial/pipeline outputs must still be bitwise equal
to the preceding variant. This does not cover division by zero, non-finite
inputs, arbitrary shapes or dtypes.

The exact three Sub/Mul/Div bundles and their timing ABIs passed bounded local
BM1690 CModel runs (one operation per process, two host CPUs, 120-second/4-GiB
limits). Sub and Mul match the reference bitwise with zero error. Div has maximum
absolute error 0.00390625, zero tolerance mismatches, and all three variants are
bitwise equal to each other. Each variant also passed one warmup plus two timing
ABI samples, whose CModel times are deliberately not used as board performance.
Records are `results/p8-{sub,mul,div}-1024-pcie-cmodel.json`; independent reference
checks and generated source hashes are included. Manifest, download, supervisor,
host ABI and timer tests passed. Validation details are in
`results/p8-elementwise-pcie-validation.json`.

The user subsequently completed Sub compilation against the inspected PPL 1.4
SDK on the BM1690 server. The helper and source files were pinned to `0a49d91`;
all GitHub API downloads and hash checks passed, followed by all 18 compiler/linker
commands for original, serial and pipeline. `PREPARE_EXIT=0` and
`SUB1024_BUILD_ONLY_OK kernel_launches=0` were reported. The work directory is
`/home/bokai/bm1690-sub-GvDYeE/work`, with receipt `handoff.json`. Evidence is
`results/p8-sub-1024-build-user-reported.json`, based on the user's terminal
output; full remote receipts and artifact hashes have not been copied here.
The helper loaded no board runtime and launched no kernels.

The user's subsequent Sub original, serial and pipeline runs all passed on
BM1690 device 0 (`0000:01:00.0`, `/dev/sg-host-drv-0`), FP16 1024x1024 and one
launch core. Outputs are finite and bitwise equal to the independent reference,
with zero tolerance mismatches and maximum absolute error 0. All three input
and output hashes agree with the Sub CModel record; serial and pipeline also
report `previous_variant_bitwise_equal=true`. Worker and outer exits are all
zero. Evidence is `results/p8-sub-1024-three-variants-board-user-reported.json`,
based on user-supplied terminal output; full remote receipts remain on the board
host. This completes Sub board correctness for the recorded inputs. The printed
226/223/248 us are single unwarmed diagnostics, not an accepted latency comparison.
That correctness-only run does not establish latency or physical overlap.

The user then completed a single original/serial/pipeline timing round with
5 warmups and 20 resident synchronous calls per variant. All post-timing outputs
remain finite, bitwise equal to the reference and to each other, with zero error;
all worker and outer exits are zero. The 60 raw samples, correctness records and
independently recomputed statistics are retained in
`results/p8-sub-1024-latency-user-reported.json`, based on terminal output. Full
remote result JSON has not been copied to this development host.

| Variant | Tile | Median (us) | IQR (us) | p95 (us) |
| --- | --- | ---: | ---: | ---: |
| Original whole-block | 1024x1024 | 113.564 | 4.599 | 117.195 |
| Same-tile serial | 128x1024 | 117.722 | 4.804 | 123.620 |
| Pipeline, depth 2 | 128x1024 | 119.352 | 4.862 | 141.784 |

The pipeline median is 1.38% higher than same-tile serial and 5.10% above the
original whole-block median. This round shows no pipeline speedup; its p95 is
also higher. Do not attribute the variation to a particular host or device cause
without further evidence. These are host launch-plus-stream-sync measurements,
excluding compilation, allocation, module loading, H2D, D2H and reference work.
One fixed-order round does not establish repeatability, pure device time or
physical overlap. Retain every variant and the negative result. Sub's first
correctness-plus-synchronous-latency acceptance is complete for this case.

The user subsequently compiled Mul original, serial and pipeline on the BM1690
server, reusing the hash-verified helper from the Sub handoff with `--case mul`
and source revision `0a49d91`. All three GitHub API downloads and hash checks
passed, followed by all 18 compiler/linker commands. The helper reported
`PREPARE_EXIT=0`, `board_runtime_loaded=false` and
`MUL1024_BUILD_ONLY_OK kernel_launches=0`. The work directory is
`/home/bokai/bm1690-mul-LNbaFw/work`, with receipt `handoff.json`. Evidence is
`results/p8-mul-1024-build-user-reported.json`, based on user terminal output;
full remote receipts and ELF hashes have not been copied here. That build-only
receipt establishes compilation, not board correctness or latency.

The user's subsequent Mul original, serial and pipeline runs all passed on
BM1690 device 0 (`0000:01:00.0`, `/dev/sg-host-drv-0`), FP16 1024x1024 and one
launch core. All outputs are finite and bitwise equal to the independent
reference, with zero tolerance mismatches and maximum absolute error 0. Their
input and output hashes match the recorded Mul CModel results; serial and
pipeline also report `previous_variant_bitwise_equal=true`. All worker and
outer exit codes are zero. Evidence is
`results/p8-mul-1024-three-variants-board-user-reported.json`, based on the user
terminal output; full remote result JSON remains on the board host. Mul board
correctness is complete for these recorded inputs. The reported 219/243/229 us
are single unwarmed diagnostics, not an accepted latency comparison.

After the host kernel changed to `7.0.0-34-generic` and its driver was rebuilt
and temporarily loaded, the user completed fresh one-call checks and 5+20
synchronous-call latency runs for each Mul variant. All six worker exits are
zero, all six processes report `MUL_PCIE_PASSED`, and the guarded sequence ends
with `MUL_EXIT=0`. Every output remains finite and bitwise equal to the reference
with maximum absolute error 0; serial and pipeline match the preceding variant.
Evidence is `results/p8-mul-1024-latency-user-reported.json`. It preserves all 60
raw samples and both correctness and timing receipt paths under
`/home/bokai/bm1690-mul-LNbaFw/work/resume-k34-5FAebL`.

| Mul FP16 1024x1024, one TPU core | median us | IQR us | p95 us |
| --- | ---: | ---: | ---: |
| original whole-block | 114.013 | 2.685 | 123.55790 |
| same-tile serial 128x1024 | 118.763 | 2.05025 | 120.27950 |
| pipeline 128x1024, depth 2 | 121.1325 | 3.482 | 126.25795 |

Statistics were independently recomputed from the pasted raw samples. Pipeline
median is 1.995% higher than same-tile serial and 6.244% higher than original.
This is one fixed-order round of host launch-plus-stream-sync measurements,
excluding compilation, allocation, module loading, H2D, D2H and reference work.
It establishes neither a speedup nor physical overlap. Preserve this negative
result; later repeated rounds with varied execution order will assess
reproducibility. Mul's first correctness-plus-synchronous-latency acceptance is
complete for this case.

Div subsequently completed its three-variant build and all six correctness/timing
processes using the same restored session, source revision `0a49d91`, and
`/home/bokai/bm1690-div-UiXs7C/work`. All exits are zero. Each variant's maximum
absolute error is 0.00390625 with no tolerance mismatches (`atol=rtol=0.01`).
Outputs are bitwise equal across variants, but not to independently rounded FP32
division, as permitted by the existing demo contract. Input/output hashes match
the Div CModel evidence. The terminal evidence is archived in
`results/p8-div-1024-build-user-reported.json`,
`results/p8-div-1024-three-variants-board-user-reported.json`, and
`results/p8-div-1024-latency-user-reported.json`; full remote receipts remain on
the board host.

| Div FP16 1024x1024, one TPU core | median us | IQR us | p95 us |
| --- | ---: | ---: | ---: |
| original whole-block | 132.6975 | 4.4685 | 139.02905 |
| same-tile serial 128x1024 | 138.413 | 1.9015 | 141.30445 |
| pipeline 128x1024, depth 2 | 126.023 | 3.9515 | 140.10385 |

The pipeline median is 8.951% lower than serial and 5.030% lower than original
in this fixed-order round. All 60 raw samples and recomputed statistics are
preserved, including the pipeline's 210.819 us maximum. Timing includes launch
and stream synchronization, excluding compilation, allocation, module loading,
H2D, D2H and reference work. Varied-order repetitions are still needed to assess
stability; neither pure device time nor physical overlap has been measured.

## P8.6 Matmul smoke handoff

`export_add_sources.py --operation matmul` exports the existing low-precision
`build_matmul` implementation with original 16-cubed tiles and serial/pipeline
32-cubed tiles. `matmul-smoke-sources.json` fixes M=N=K=128, FP16, row-major,
TPU-Kernel and one launch core. No prepacking is introduced. The separate FP32
`local.matrix` path is outside this FP16 handoff.

The pipeline report records K extent 4, stage `[0,0,1]`, order `[0,1,2]`, and two
physical buffer versions each for A and B. Generated C fills the first tiles,
then runs three prefetch/compute iterations inside paired parallel scopes,
consumes the final tiles, casts the ordered single-version FP32 accumulator and
stores FP16 output. CModel disables the parallel scopes; it does not prove
hardware overlap. The board smoke compiled successfully with the existing PPL 1.4
headers/archive, which have not been replaced by the local PPL 1.7 installation.

The exact source bundle, deterministic dense signed inputs, independent scalar
FP32 reference, CPU PyTorch check, host buffer canaries, one-call ABI and timed
ABI (one warmup plus two samples locally) passed under the existing bounded
CModel supervisor. Serial/pipeline output is bitwise equal. Original/serial uses
the existing 0.01 tolerance because its K grouping differs; timing admission
also revalidates the saved output against the reference. Remote tests still use
five warmups and 20 samples. The scalar reference is intentionally bounded to
128-cubed; the 1024-cubed handoff uses the NumPy reference described below.

`prepare_add_1024.py --case matmul-smoke` downloads pinned sources, builder and
runner through the same official GitHub transport and compiles only. Old
Add/Sub/Mul/Div bundles and builder hashes remain accepted. No dependencies,
native compiler passes or runtime library installations changed for this step.
On any failure retain the source bundle, logs and completed results; do not retry
failed board kernels automatically. Resume larger Matmul and the other families
after recording the smoke results.


The user completed the smoke in `/home/bokai/bm1690-matmul-eHJxZ9/work` from
source revision `12bfcbf163400bd2b25efb1523f7f623b79fa690`. All 18 build commands
passed, as did each variant's one-call correctness test and five-warmup/20-sample
resident timing test. Worker exits and `MATMUL_EXIT` are zero. Each output hash
agrees with the corresponding CModel record. Serial and pipeline are bitwise
equal; original/serial differ within the original `atol=rtol=0.01` contract.
The maximum absolute error against the scalar FP32 reference is 0.000244140625
for all three variants, with zero tolerance mismatches and all outputs finite.

| Matmul FP16 M=N=K=128, one TPU core | median us | IQR us | p95 us |
| --- | ---: | ---: | ---: |
| original 16x16x16 tiles | 1092.2855 | 21.124 | 1111.4176 |
| same-tile serial 32x32x32 | 186.5785 | 5.570 | 195.67165 |
| pipeline 32x32x32, depth 2 | 193.149 | 2.93625 | 197.12635 |

The three user-reported result files preserve all 60 raw samples and independently
recomputed statistics. The 3.522% higher pipeline median establishes no pipeline
speedup in this round. Original/serial differ in tile sizes, so their ratio must
not be attributed to pipelining. Timing covers host launch plus stream
synchronization and excludes allocation, transfers, compilation, module loading
and reference evaluation. This is one fixed-order round, without a device trace.

## P8.7 Matmul 1024 handoff

The target bundle extends the already board-validated smoke to M=N=K=1024,
retaining the original 16-cubed baseline and the 32-cubed serial/pipeline pair.
Its pipeline K loop has extent 32, stage `[0,0,1]`, order `[0,1,2]`, alternating
A/B buffer versions, and one ordered FP32 accumulator. The original and both
candidate implementations remain separate. No prepacking or multicore changes
are included.

The target runner uses the existing NumPy installation only for the CPU reference:
FP16 input bytes become row-major FP32 matrices, CPU GEMM produces FP32 output,
and the final conversion is FP16. Dense deterministic signed inputs and original
tolerances remain unchanged. Reference evaluation occurs before board runtime
loading and outside timing. Each remote worker records the actual NumPy version
and path, and requests one BLAS thread after restricting CPU affinity to two
logical CPUs. Elementwise and Matmul smoke remain standard-library-only paths.

Regression checks compare this NumPy reference against the scalar smoke reference
and sample target dot products using independent high-precision sums, reject
non-finite/corrupted output, and preserve all earlier bundle registrations.
The exact-source CModel check independently uses CPU PyTorch. Each target variant
runs in its own 120-second, 4096-MiB supervised process, first once and then through
the timed ABI with one warmup and one sample. CModel durations are diagnostics
only; the board measurement still uses five warmups and 20 samples.


The complete target CModel check passed all three supervised workers and retained
per-variant output bytes, source identities and logs. The serial/pipeline outputs
are bitwise equal. Original/serial, and each variant against the NumPy reference,
have maximum absolute difference 0.0009765625 with no tolerance mismatches. The
NumPy versus CPU PyTorch comparison also passes the same contract; those two CPU
references are not required to be bitwise equal. This evidence is stored in
`results/p8-matmul-1024-timing-abi-cmodel.json`. The subsequent remote validation is recorded below.


The user subsequently completed all target builds and six correctness/timing
processes from revision `6c83e0b79bb578d9eec0a85eca142aef2f37afc3`, ending with
`MATMUL1024_EXIT=0`. All outputs are finite and within `atol=rtol=0.01`, with
maximum absolute error 0.0009765625. Serial/pipeline are bitwise equal, while the
original differs within tolerance because its K tiling differs. All three output
hashes agree with the existing CModel evidence. Full remote result JSON remains
on the board host; the archived evidence is from the user-provided terminal log.

| Matmul FP16 M=N=K=1024, one TPU core | median ms | IQR ms | p95 ms |
| --- | ---: | ---: | ---: |
| original 16x16x16 | 445.926780 | 5.326600 | 450.594007 |
| same-tile serial 32x32x32 | 56.956687 | 0.605835 | 57.444230 |
| pipeline 32x32x32, depth 2 | 52.667097 | 0.159819 | 52.946247 |

All 60 raw samples and recomputed statistics are retained in
`results/p8-matmul-1024-latency-user-reported.json`. Five warmups and 20 samples
were used per variant. Pipeline median is 7.531% lower than same-tile serial in
this fixed-order round. The boundary includes host launch and stream synchronization,
excluding compilation, allocation, module loading, H2D, D2H and reference work.
This completes first-round target acceptance, not repeatability, broad-shape,
multicore or physical-overlap validation.

## P8.8 RMSNorm ordinary and Split-K smoke handoffs

Both immutable source bundles use FP16 16x128 matrices, four rows per tile,
per-element weights of the same shape, epsilon 1e-12 and one TPU launch core.
The ordinary variant reduces the full width and pipelines independent output
row tiles using `pipeline_kernel_tiles`. Its lowered report contains 12
statements; input and weight each have two physical buffer versions, while the
FP32 reduction state remains ordered. The output store follows the parallel
scope before the output buffer is reused.

Split-K uses block_k=32 and pipelines each of its two K passes separately.
The first loop has five lowered statements and two input buffer versions; the
second has seven statements and two input and weight versions. The generated
first-pass drain and final accumulation precede `tpu_bdc_fp_rsqrt`, which precedes
the second loop. The second pass retains the original reverse-K traversal.
No pass crosses the dependency on the completed sum of squares. Split-K original
and same-tile serial sources are identical at this smoke tiling and are retained
under separate labels; no distinct optimization is claimed for that pair.

The scalar FP32 reference includes the epsilon addition, reciprocal square root,
FP16 normalization boundary and subsequent weight multiplication. Regression
fixtures distinguish it from an incorrectly fused FP32 expression and from
broadcast weights. Zero rows and small rows exercise epsilon behavior. CPU
PyTorch independently checks the same mathematical/precision contract.

Exact exported sources, host canaries, immutable inputs, one-call execution and
one-warmup/two-sample timed ABI all passed locally for both RMSNorm forms.
Their outputs agree bitwise across original/serial/pipeline and with their
references (maximum error zero for these inputs). Local PPL 1.7 CModel still
serializes parallel scopes, so this evidence does not establish physical overlap.
Existing Add/Matmul bundle identities, precision contracts and prior results
remain intact. No compiler-pass changes or dependency installation were needed
for this handoff.

The user subsequently completed ordinary RMSNorm's PPL 1.4 board build and all
six correctness/timing processes from revision
`78a3f2b9d22315d4c3998a9ec538d421ba3d6c9e`, ending with `RMSNORM_EXIT=0`.
All three variants are bitwise equal to the independent reference and to each
other, with zero absolute error; all input/output hashes match CModel. Each
worker verified device 0, `/dev/sg-host-drv-0`, PCI `0000:01:00.0` before launch.
The evidence is the user-supplied terminal log; full remote result JSON and the
contents of `environment.txt` have not been received at the development host.

| Ordinary RMSNorm FP16 16x128, one TPU core | median us | IQR us | p95 us |
| --- | ---: | ---: | ---: |
| original 4x128 tiles | 84.4125 | 4.66600 | 87.34455 |
| same-tile serial 4x128 | 86.2650 | 5.68875 | 95.10865 |
| pipeline 4x128, depth 2 | 83.5965 | 5.34125 | 92.59955 |

All 60 raw samples are retained and their statistics independently recomputed in
`results/p8-rmsnorm-smoke-latency-user-reported.json`. Five warmups and 20 resident
samples were used per variant in original/serial/pipeline order. Timing includes
host launch plus stream synchronization and excludes compilation, allocation,
module loading, H2D, D2H and reference work. The pipeline median is 3.093% below
serial and 0.967% below original in this round, but its 115.657 us maximum is
retained and the median differences are smaller than the observed IQRs. This
small smoke establishes first-round correctness and synchronous-call timing,
not stable speedup, target-size performance, pure device time or physical overlap.
The user subsequently completed Split-K smoke with `SPLITK_EXIT=0`, all 18 build
commands and six correctness/timing processes passed. All variants match each
other, the scalar reference and CModel output hashes bitwise, maximum error zero.

| Split-K RMSNorm FP16 16x128, one TPU core | median us | IQR us | p95 us |
| --- | ---: | ---: | ---: |
| original 4x32 tiles | 172.5465 | 9.56625 | 229.18830 |
| same-tile serial 4x32 | 176.8495 | 7.60875 | 185.06785 |
| pipeline 4x32, depth 2 | 173.6230 | 9.39800 | 231.49855 |

Five warmups and 20 resident samples per variant use the same host-call timing
boundary described above. All 60 raw samples and recomputed statistics are in
`results/p8-rmsnorm-splitk-smoke-latency-user-reported.json`. Pipeline median is
1.824% below serial and 0.624% above original; identical original/serial sources
also produce different measured medians. Preserve the original 1043.515 us and
pipeline 939.563 us maxima without guessing their cause. This one fixed-order
round establishes no stable performance improvement or physical overlap.

## P8.9 RMSNorm 1024 target handoffs

Both target bundles preserve the original mathematical and FP16 rounding
contracts, epsilon 1e-12, 1024x1024 per-element weights and one TPU launch core.
Their candidate tiling reuses the existing P4 CModel target configuration:

| Operation | Original tiles | Serial/pipeline tiles | Pipelined loop |
| --- | --- | --- | --- |
| Ordinary RMSNorm | 4x1024 | 32x1024 | 32 independent output-row tiles |
| Split-K RMSNorm | 4x32 | 32x128 | two separately pipelined K passes, each extent 8 |

The ordinary depth-2 schedule has 12 lowered statements, two input versions and
two weight versions. Split-K's first pass has five statements and two input
versions; its second has seven statements and two input/weight versions. The
generated first-pass drain/final sum precedes reciprocal square root and the
reverse-K output pass. All versions retain single ordered reduction state and
stores after the parallel scope before output-buffer reuse. The exported source
contains distinct local buffer addresses and explicit parallel start/end calls;
CModel serializes those calls, so hardware overlap remains unverified.

The board runner extends its bounded standard-library FP32 scalar reference to
1024x1024. Inputs include a full zero row, a small-magnitude row and nonuniform
weights; all reference evaluation remains outside the measured interval. CPU
PyTorch independently checks the precision contract in CModel, and regression
tests sample target rows using high-precision scalar sums. The FP16 boundary
before weight multiplication remains unchanged.

Each target CModel variant ran in a fresh worker, limited to two host CPUs,
4096 MiB group RSS and 120 seconds, first once and then one warmup/one timed
sample to validate the host ABI. All six passed with maximum absolute error
0.0009765625 and no tolerance mismatches; repeated outputs are unchanged.
Each pipeline is bitwise equal to its same-tile serial. Ordinary original is
also bitwise equal; Split-K original differs within 0.01 tolerance due to its
different reduction grouping. Only target Split-K original-to-serial permits
that difference; smoke and every serial-to-pipeline comparison remain strict.
The scalar-reference/PyTorch comparison passes tolerance without requiring
bitwise equality. Emulator durations are not device performance measurements.

All 118 manifest, precision-reference, ABI, supervisor and download-handoff
checks passed. Existing source bundles and their historical builder hashes remain
accepted. No kernel arithmetic, compiler passes, native libraries or installed
dependencies changed. On the board, run ordinary then Split-K, each with fresh
one-call correctness followed by five warmups/20 resident synchronous calls per
variant. Compare pipeline against its same-tile serial; original-to-candidate
ratios also include tiling changes. The subsequent ordinary target board results
are recorded below, followed by Split-K target board results.

The user completed ordinary RMSNorm 1024x1024 in
`/home/bokai/bm1690-rmsnorm1024-XOF4Zd/work` from revision
`180b987e834b8cd6e3d13f8fa089a7513771dfb3`. All 18 build commands and six
correctness/timing processes passed, ending with `RMSNORM1024_EXIT=0`.
All outputs are finite and bitwise equal across original/serial/pipeline, and
each input/output hash matches CModel. Maximum reference error is 0.0009765625
with zero mismatches under `atol=rtol=0.01`; outputs are not bitwise equal to
the independent reference. Every worker verified device 0, `/dev/sg-host-drv-0`,
PCI `0000:01:00.0` before dispatch.

| Ordinary RMSNorm FP16 1024x1024, one TPU core | median us | IQR us | p95 us |
| --- | ---: | ---: | ---: |
| original 4x1024 tiles | 1787.4105 | 19.07450 | 1851.05175 |
| same-tile serial 32x1024 | 273.9225 | 4.40850 | 280.95345 |
| pipeline 32x1024, depth 2 | 262.1675 | 6.17225 | 268.89950 |

Five warmups and 20 resident samples per variant were executed in fixed
original/serial/pipeline order. All 60 raw samples and independently recomputed
statistics are in `results/p8-rmsnorm-1024-latency-user-reported.json`.
Pipeline median is 4.291% below same-tile serial in this round; the 6.818
original-to-pipeline ratio also includes the row tile change from 4 to 32.
The original's 2446.887 us maximum is retained without attributing its cause.
Timing includes host launch plus stream synchronization, excluding compilation,
allocation, module loading, H2D, D2H and reference work. Evidence is the supplied
terminal output; full remote result JSON and `environment.txt` contents have not
been received. This completes first-round ordinary RMSNorm target acceptance.
Repeated varied-order rounds, pure device timing and physical overlap remain
unverified. Split-K target board results follow.


The user completed Split-K RMSNorm 1024x1024 in
`/home/bokai/bm1690-splitk1024-dfggi5/work` from the same target handoff revision
`180b987e834b8cd6e3d13f8fa089a7513771dfb3`. All 18 build commands and six
correctness/timing processes passed, ending with `SPLITK1024_EXIT=0`.
All outputs are finite and within `atol=rtol=0.01` of the independent reference,
maximum absolute error 0.0009765625. Input/output hashes match CModel.
Serial/pipeline outputs are bitwise equal; original differs within tolerance
because the K tiles change from 32 to 128. This Split-K is two ordered K passes
on one TPU core; it is not a multicore reduction.

| Split-K RMSNorm FP16 1024x1024, one TPU core | median us | IQR us | p95 us |
| --- | ---: | ---: | ---: |
| original 4x32 tiles | 59026.2700 | 278.72900 | 59362.99875 |
| same-tile serial 32x128 | 1881.3245 | 26.45975 | 1935.65540 |
| pipeline 32x128, depth 2 | 1851.5840 | 16.16750 | 1871.29705 |

Five warmups and 20 resident samples per variant ran in fixed original/serial/
pipeline order. All 60 raw samples and recomputed statistics are retained in
`results/p8-rmsnorm-splitk-1024-latency-user-reported.json`. Pipeline median is
1.581% lower than same-tile serial in this round. The 31.879 original-to-pipeline
ratio includes row/K tiling changes and must not be attributed to pipelining.
The boundary is host launch plus stream synchronization, excluding compilation,
allocation, module loading, H2D, D2H and reference. Full remote JSON receipts and
`environment.txt` contents have not been received; this is supplied terminal
output evidence. Stable gains, pure device time and physical overlap remain
unverified. Both RMSNorm target variants now have first-round board acceptance.

## P8.10 RoPE smoke handoff

`build_rope` retains interleaved even/odd pairs, FP16 input/output (8x128) and
FP32 cosine/sine tables (8x64). All variants use 4x32 tiles. The generated ABI is
`source, cosine, sine, destination`, with output index 3 and 2048 bytes per
argument. The equal byte counts do not imply equal dtypes. The standard-library
reference rounds each product and add/sub to FP32 before FP16 output; the CPU
PyTorch demo reference independently agrees bitwise for the supplied inputs.
Generated tables differ from the demo's Torch tables by at most 5.96046448e-8;
all variants/reference use the same recorded table bytes.

The flattened eight-tile loop has five lowered statements: three loads, the
16-pair compute loop, and one store. Its depth-2 stage array is [0,0,0,1,1], with
order [0,1,2,3,4]. Generated local addresses are x 256/640, cos 320/704 and sin
384/768 for the two versions. The source contains a fill, seven steady steps
that prefetch the next tile in a parallel scope with current computation, and a
final drain. Output and compute temporaries remain single-version; each store
follows parallel end and completes before reuse. The inner pair loop preserves
the original FP32 arithmetic and output placement. No compiler/native changes
were needed for this handoff. These are generated-schedule facts, not proof of
physical hardware overlap.

All three exact source sets compiled and ran under the existing PPL 1.7 BM1690
CModel, including one warmup/two sample calls to check the timed ABI. Outputs
are bitwise equal across variants and reference, maximum error zero. The worker
was bounded to two host CPUs, 4096 MiB group RSS and 120 seconds. CModel timing
samples are not retained as device performance data. Evidence is
`results/p8-rope-smoke-timing-abi-cmodel.json`; the source bundle is
`handoff/rope-smoke-sources.json`.

The board runner now checks four host buffers, all three inputs' immutability,
canaries and NaN-prefilled output. Host-only tests cover mixed-dtype argument
ordering, swapped equal-size tables, partial output, overwritten inputs,
canary corruption, reference/timing receipts, pairing, and strict
serial/pipeline equality. All 118 PCIe regression checks and 13 download-handoff
checks passed. Existing source bundles and historical builders remain accepted.
Board smoke compilation/execution subsequently passed, as recorded below;
1024 target board acceptance follows.


The RoPE smoke board handoff completed in
`/home/bokai/bm1690-rope-BMzRZC/work` from revision
`9a596d4f971dd124945a67db261e406dbd904a1a`, ending with `ROPE_EXIT=0`.
All 18 build commands and six correctness/timing processes passed. All variants
are finite and bitwise equal to the independent reference and one another;
maximum error is zero. Input/output hashes match CModel. Each worker verified
device 0 -> `/dev/sg-host-drv-0` -> PCI `0000:01:00.0` before dispatch.

| RoPE FP16 8x128, FP32 tables, one TPU core | median us | IQR us | p95 us |
| --- | ---: | ---: | ---: |
| original 4x32 tiles | 995.9885 | 17.56650 | 1019.68805 |
| same-tile serial 4x32 | 1025.3975 | 24.59925 | 1049.90185 |
| pipeline 4x32, depth 2 | 909.7610 | 24.19225 | 945.75770 |

Five warmups and 20 resident samples per variant were run in fixed original /
serial / pipeline order. Pipeline median is 11.277% below serial and 8.657% below
original in this round. The pipeline maximum 1159.521 us is retained without
attributing its cause. All 60 raw samples and independently recomputed statistics
are in `results/p8-rope-smoke-latency-user-reported.json`, with companion build
and correctness records. Timing includes host launch plus stream synchronization,
excluding compilation, allocation, module loading, H2D, D2H and reference work.
Evidence is user-supplied terminal output; full remote JSON receipts and the
contents of `environment.txt` have not been received. Stable speedup, pure device
time and physical overlap remain unverified.

## P8.11 RoPE 1024 target handoff

The target preserves FP16 1024x1024 source/destination, FP32 1024x512 cosine/sine
tables, interleaved even/odd pairs, and FP32 arithmetic before FP16 output. All
four ABI arguments are 2097152 bytes. Original retains the demo's 4x32 tiles
(8192 tiles); serial/pipeline use the existing P4 target configuration, 32x64
(512 tiles). Original-to-candidate ratios will therefore include tiling changes.
Pipeline-only gains must be compared against the same-tile serial version.

The depth-2 candidate loop has 512 independent tiles and five lowered statements:
three loads, one inner 32-pair compute loop, one store. Stage/order remain
[0,0,0,1,1] / [0,1,2,3,4]. The generated input local addresses are x 256/768,
cos 384/896 and sin 512/1024. Fill loads the first tile; 511 steady iterations
prefetch the next while current computation sits in the parallel scope; the
final tile drains afterward. Output and compute temporaries remain single-version,
with each store after parallel end before reuse. No original arithmetic,
compiler pass, native library or installed dependency changed for this extension.

All three variants passed exact-source CModel checks in separate bounded workers
(two host CPUs, 4096 MiB group RSS, 120 seconds each). Each ran one correctness
call and a one-warmup/one-sample timed ABI check. All three outputs match one
another and the independent scalar reference bitwise, maximum error zero. CPU
PyTorch independently agrees with that reference on the identical input/table
bytes. The generated sin/cos tables differ from the demo's Torch generation by
at most 3.0517578125e-5; FP32 power rounding differences grow with the position
multiply. Table-generation comparison uses a row-scaled FP32 bound, recorded
separately; the kernel's original `atol=rtol=0.005` contract is unchanged.
Evidence is `results/p8-rope-1024-timing-abi-cmodel.json`. CModel durations are
not board performance measurements.

All 128 PCIe regression checks and 13 download-handoff checks passed. New tests
cover target table shape/dtype, original/candidate tiles, exact quarter turns
across target rows/pairs, preserved smoke input bytes and deadline propagation.
The board runner offers bounded deadlines of 30 seconds (unchanged default) or
90 seconds for the larger batch, still with two host CPUs, nice +10, 4096 MiB
RSS, one TPU launch core, group cleanup, no automatic kernel retry. The original
has 1024 times as many tiles as the smoke; the larger watchdog budget is not a
prediction of actual duration. Board target compilation and execution remain
pending. Use a fresh directory and correctness before timing for every variant.


The first RoPE target board attempt used revision
`2610cc6dcd06b7fb1fa094a024961c611e97fbfe` in
`/home/bokai/bm1690-rope1024-ehiG4z/work`. All 18 build commands passed and the
helper/runner checksums matched. Original one-call execution reached
`STAGE tilelang_tpu_run_once`, then the supervisor reported `ROPE_PCIE_FAILED`
and the sequence ended with `ROPE1024_EXIT=1`. No reference comparison, timing,
serial or pipeline run completed/started in the supplied output. The checkpoint
precedes the host ABI call, which also performs allocation, module loading and
transfers; it alone cannot establish successful kernel launch or completion.
The supervisor reason/return code/cleanup flag and full runtime log have not
been received. Timeout, resource exhaustion, native process failure and device
state are not established causes. Preserve the failed directory and inspect
read-only receipts/process/device state before retrying. No retry or kernel
change was performed in response to this report. Evidence is
`results/p8-rope-1024-board-attempt-user-reported.json`; prior CModel and smoke
passes remain valid for their recorded cases, not proof of target board success.


## P8.12 RoPE target timeout investigation

The remote supervisor receipt confirms `reason=timeout`, a 90-second deadline,
`returncode=-9` and `cleanup_complete=true`. Peak group RSS was 245784576 bytes,
below the 4 GiB limit. The checkpoint remained `tilelang_tpu_run_once`; no
reference result or timing was produced. Kernel 7.0.0-34-generic, live sgcard,
the device node and PCI binding are confirmed after failure. The supplied
process search found no runner processes. These observations establish host
cleanup and driver presence, not idle/completed/reset firmware. Evidence is
`results/p8-rope-1024-timeout-diagnostics-user-reported.json`. Kernel logs and
the observed board SDK's poll/synchronization declarations were initially requested;
the follow-up findings are below. Do not simply extend the timeout or substitute a
candidate for the failed original baseline.

A separate diagnostic bundle is derived reproducibly by
`tpu_demo/pipeline/export_rope_trace.py` from the frozen target bundle. Only
original `main.cpp` and `kernel.cpp` change: flushed stderr CALL/RETURN markers
wrap runtime init/device selection, stream/module creation, allocation, H2D,
kernel launch, synchronization, D2H and cleanup. `kernel.c`, ABI header and the
other variants remain byte-identical. These host call-boundary logs are not a
hardware profiler trace or overlap evidence. The diagnostic bundle is separately
hash-pinned; the runner rejects timing and non-original variants before loading
a vendor runtime. Its receipts cannot replace normal-bundle correctness receipts.

One bounded original CModel call passed with the same input/output hashes as the
normal target. All 22 calls returned: four initialization calls, four allocations,
four uploads, one launch, two synchronizations, one download, four frees and two
cleanup calls. The initial postprocessing expectation of 23 calls was corrected
to the 22 shown by source and log without rerunning the kernel; the preserved
record distinguishes that bookkeeping correction from kernel validation.
All 135 PCIe, 13 handoff and two source-derivation regression tests passed.
Evidence is `results/p8-rope-1024-host-trace-cmodel.json`. Native libraries,
compiler passes, device kernels, SDK and dependencies remain unchanged.
At that point diagnostic board build/execution was pending. The subsequent
post-reload board attempt, described below, located the unreturned host runtime
API without an uninstrumented retry or timing batch.

The subsequent user-supplied journal contains a driver error at 18:12:20:
`[host drv-0, send_quit_request, 379]:FORCE_QUIT default timeout!`
with request ID `0x2348`, PID field `0x2a446b` and `stream_num: 2`.
The failed run log's reported mtime is 18:09:49.956059557 +0800; the PID has not
yet been correlated to the worker. The exact condition and recovery semantics
require inspection of the installed driver's source. Host process-group cleanup
does not establish device task cleanup. Keep board dispatch held while checking
that source, the full failure-window journal and users of both device nodes.
The 15:41 module-signature warning and initial channel-state values are earlier
load-time observations, not a current firmware health check.

The board PPL 1.4 header confirms declarations for `tpu_poll`,
`tpu_poll_descriptor`, `tpu_poll_empty` and the `tpu_sync_*` family. Their presence
alone neither identifies the timeout cause nor justifies inserting additional
polls or synchronization into the kernel. No such implementation change,
driver reset/unload or new board execution was performed for this follow-up.
Evidence is `results/p8-rope-1024-force-quit-user-reported.json`.

The next source excerpt identifies `sgdrv_fops.c::send_quit_request` at lines
361-385. It sends the request to the message FIFO, then waits up to 60000 ms on
`STREAM_DESTROY_RESPONSE` for `find_api_count_response` to satisfy the requested
ID/stream-count condition. A zero wait result produces the error and returns
-1; this function contains no reset. Its callers and the response predicate
have not yet been inspected. The full 18:09-18:15 journal also shows
`rt_task_done_response_sync: wait task done force exit` and an ioctl error for
request `0x2346` at 18:11:18. The triggering branch and mapping to a specific
runtime API remain to be read; these messages do not establish the original
kernel timeout's root cause.

The same follow-up reports PID 2737445 (`pt_main_thread`, user bokai) holding
`/dev/sg-host-drv-0` and module reference count 4. No holder for device 1 is
printed, but the `fuser` exit status was not supplied. The error's PID field,
`0x2a446b` (2770027), differs from the holder PID; their relationship is unknown.
The earlier runner-name search did not inspect every device user. The local
supervisor's `cleanup_complete` checks that the worker leader was reaped and
its tracked process group's RSS sum is zero, not absence of other device users
or firmware work. Do not attribute the holder to a leaked runner without its
command line, parent/group/session, start time and thread-state evidence.
Board dispatch and driver unload remain held while collecting those facts.
Evidence is `results/p8-rope-1024-driver-exit-and-holder-user-reported.json`.

Further process inspection identifies the holder as a separate ChunkScan P10
invocation in `/home/bokai/ChunkScan-bm1690-test-86add2f`, using
`test_chunk_scan_multicore_benchmark_p10.py --stages s1,s3,p6 --shapes r3
--core-counts 1,2,4,8 --quick`. It started at 18:04:44, before the RoPE target
attempt, and has PPID/PGID 2737430. Its only reported thread waits in
`rt_task_done_response_sync`. The user explicitly confirms running other TPU
operator tests concurrently. The wrapper's requested host affinity of CPUs 2,4
does not establish TPU resource separation from the RoPE worker on CPUs 0,1.

The supplied driver source at lines 2407-2425 waits for `TASK_DONE_RESPONSE`
using `wait_event_interruptible` and emits the force-exit message when its
result is negative. This wait call has no timeout argument. The source and
process identity narrow the investigation, but do not prove concurrent physical
kernel execution or that P10 caused the original RoPE failure. In particular,
the failed run lacks a verified exclusive-device baseline. Preserve prior
results and correlate their timestamps before deciding which measurements need
repetition; do not discard all earlier acceptance evidence indiscriminately.
The existing runner lock coordinates participating invocations of that runner,
not arbitrary other TPU tests. Obtain the user's test-priority choice and the
P10 launcher/log location before stopping the separate task. No task was stopped
and no driver reset or board retry was performed in this follow-up.
The parent is a `timeout -k 10s 7200s` launcher with
`CHUNKSCAN_BENCH_MAX_RUNS=1000`. Stdout/stderr both target `pipe:[46552045]`;
the final log file is not identified. The user explicitly chose to keep P10
running and defer pipeline-migration board testing. Honor that choice without
further board dispatch, task termination or driver recovery actions. Retain the
prepared diagnostic for use after other TPU testing ends and device availability
and health have been checked. Overlapping measurements from either project must
not be presented as verified exclusive-device performance.
Evidence is `results/p8-rope-1024-concurrent-p10-user-reported.json`.

## P8.13 Resume preflight after concurrent testing

The Oct 15 user-requested resume starts from fresh driver/device observations,
not an assumption that P10 completed successfully. `fuser` prints no holder and
returns 1; sgcard's reference count is zero. The kernel and device bindings are
unchanged from the last verified 7.0.0-34-generic setup. No successful task
completion or device reset has been reported after the earlier failure.

Additional journal entries show task-wait interruption, an ioctl error and the
close-time `load_lib_mutex` warning at Oct 14 20:04:45, followed by FORCE_QUIT
timeout at 20:05:47. The PID field `0x29c525` equals 2737445, the previously
identified P10 process. Similar messages recur at Oct 15 14:32:32/14:33:33 for
PID field `0x243bc3`, whose workload identity is not established. The log alone
does not establish that the mutex remains locked after `sg_close` returns.
Read its handling and the driver unload path before a recovery action; verify
the existing matching module rather than rebuild or use the incompatible
installed module. No board dispatch, driver unload/reset, dependency change or
termination of another workload was performed for this preflight.
Evidence is `results/p8-resume-20261015-driver-state-user-reported.json`.

The supplied `sg_close` excerpt subsequently confirms an explicit unlock after
the `load_lib_mutex` warning. It logs the FORCE_QUIT result, then frees stream
contexts, memory and the host handle regardless of the timeout. Successful host
close therefore does not imply successful device task termination. The driver
registers `pci_exit` as its module exit: it unregisters the PCI driver, destroys
the class and removes proc entries. `pci_remove` releases core resources and
frees the device object; the body of `release_core_resources` was not included
in the excerpt. These observations identify the ordinary unload path, not a
guarantee of recovery from the observed firmware/response state.

The existing module's SHA256 remains
`0c66f5a553c903d8bfdf31061e06be7d20fbde7b617f6765dec5a8dadfb7ad65`, with
vermagic 7.0.0-34-generic and current `card_num=64`. Prepare a normal bounded
unload/reload using this module: recheck holders/refcount immediately before
unload, capture logs, and reload only after successful unload and disappearance
of the module and both PCI bindings. Use no force option or automatic retry.
On failure preserve state/output and stop; on success inspect fresh chip
initialization before any diagnostic kernel. No rebuild or dependency change is
needed for this planned attempt, and no board recovery has yet been reported.
Evidence is `results/p8-driver-reload-plan-20261015.json`.

The user completed that procedure in `/home/bokai/sgcard-reload-1hsmI5`.
The module hash/kernel check and both preflight checks passed. `fuser` had empty
output and exit 1 before unload; module refcount was zero. Ordinary `rmmod`
returned zero, and the module and both PCI driver links disappeared before
`insmod` of the verified module with `card_num=64` returned zero. The new journal
shows successful AP/TP firmware loading and both chips initialized, ending in
`All 2 devices probed successfully` at Oct 15 16:53:59. The supplied fresh
initialization excerpt contains no FORCE_QUIT error. Device nodes and bindings
are restored. This establishes driver reinitialization, not yet successful
post-reload runtime/mapping or operator correctness.

Proceed to a new directory using pinned revision
`e30dade9ef20b13131100c6fecb9c2afc63a71a1`, case `rope-target-trace`. The previously
CModel-validated diagnostic permits only original one-call execution and rejects
timing. Recheck that both device nodes have no users immediately before dispatch;
use one launch core, two host CPUs, 4 GiB group RSS and a 30-second supervisor
deadline. Its call-boundary logging will identify an unreturned runtime call if
the original target fails again. A diagnostic pass will not replace the normal
bundle's correctness receipt or count as a performance result. Existing failed
attempts remain preserved. Evidence for this recovery is
`results/p8-driver-reload-20261015-user-reported.json`; the subsequent diagnostic
result is below.

### P8.14: post-reload RoPE host launch timeout

The original-only diagnostic in `/home/bokai/bm1690-rope-trace-lT2aoZ/work`
passed all 18 build commands and checksum checks. Before dispatch, privileged
`fuser` reported empty output / exit 1. Runtime mapping again identified device
0, `/dev/sg-host-drv-0`, PCI `0000:01:00.0`. Host tracing recorded 13 CALL and 12
RETURN markers: init, selection, stream creation, module loading, four
allocations and four uploads returned successfully; `tpuRtKernelLaunch` at
generated `kernel.cpp:79` did not return within the 30-second supervisor limit.
No explicit `tpuRtStreamSynchronize`, output download or reference comparison
was reached. The installed runtime header declares separate launch and async
launch APIs (lines 566 and 584), but does not specify their internal completion
behavior. The observed boundary cannot locate a particular device instruction
or even establish that the device kernel entry was reached.

The supplied result reports `status=failed`, `reason=timeout`, worker return
code -9 and peak group RSS 273035264 bytes. `numeric.status=running` is the last
worker checkpoint, not successful completion. `cleanup_complete=true` refers
only to the supervised host group. Fresh driver messages at 17:01:53 show task
0x4 / stream 0x2 interrupted; 17:02:55 shows FORCE_QUIT request 0x6 / PID
0x33be2e timing out. At 17:03:58 both device nodes had no detected holders and
module refcount was zero. Those observations do not show firmware recovery.
The original device library hash matches the first uninstrumented target
attempt; only host call tracing was added. Neither serial nor pipeline nor any
timing test was started. This failure therefore is not a pipeline-only result.

Source inspection of `build_rope` and frozen original `kernel.c` identifies a
scale hypothesis to investigate. The 4x32 local tile and 16-pair arithmetic are
unchanged between smoke and target; global shapes/strides/offset constants and
outer loops differ. Smoke executes 2x4 = 8 tiles; target executes 256x32 = 8192.
Each pair contains 16 BDC API calls (six copies, four casts, four multiplies,
one subtract, one add); each tile has three S2L and one L2S calls. If all target
loops complete, this yields 2097152 BDC and 32768 GDMA API invocations, with
`tpu_poll` explicitly placed only at the end of the generated wrapper. These
are static source-derived counts, not measured descriptors or evidence of a
specific command-limit overflow. The installed PPL 1.4 implementation still
needs inspection; local PPL 1.7 CModel success does not establish that behavior.

Next, inspect the installed SDK's synchronization/command handling read-only
and prepare a small-work diagnostic that separates iteration count from global
stride/layout. Any such derived kernel must have its own identity, local
correctness checks and explicit diagnostic scope. Preserve the failing normal
original baseline. Before another board dispatch, recover device health and
check a previously passed small case. Do not extend the full-target timeout,
insert unvalidated polling as a production fix, or replace original results
with candidate results. This turn changed evidence and documentation only;
no new kernel tests or performance claims are included. Evidence:
`results/p8-rope-1024-host-trace-board-user-reported.json`.

The follow-up installed PPL 1.4 header declares `tpu_initialize`, `tpu_poll`,
`tpu_poll_descriptor` and `tpu_poll_empty` (lines 226–234). Reading the actual
RISC-V device library's symbol table confirms text definitions for initialize,
poll, poll-empty and the enable/disable-id-check functions; the exact
poll-descriptor name was not printed by that symbol filter. Declarations alone
are therefore not treated as proof of a callable implementation. The supplied
disassembly shows initialize clearing several `id_node` fields. Poll checks
byte offset 12, takes a self-branch if it is nonzero, otherwise calls
`poll_all_engine_done_internal`. This is a static code path, not evidence that
the failed launch reached it. Inspect that internal wait and the meaning of the
state byte before proposing a polling change. The library was read with
`objdump`; no runtime was loaded or new kernel dispatched in this inspection.

The first name-based dump of the internal wait resolved to 0x1ff0a0; the symbol
table contains several local definitions with that name. A subsequent address
range dump verified the actual call target 0x21cdd0. Its BDC/GDMA waits read
target counts from offsets 0/4 and repeatedly compare mapped completion
registers using an unsigned-less-than branch. These two function bodies have
no timeout. This does not show that the failed kernel reached them, or establish
hardware counter width, command overflow or a particular engine stall.
`tpu_is_parallel_state` confirms byte offset 12 is the parallel flag. The
installed `runtime/kernel/common.h` instead places its parallel member directly
after two unsigned-int command IDs, and `ppl_helper.c` reads that header member
when selecting the set/get-id helper. The frozen generated RoPE does not
construct a `CMD_ID_NODE` or call those helpers. Preserve this compatibility
concern without treating it as this failure's proven cause or rewriting the SDK.
At 17:31:56 +0800 the driver was live, refcount zero and both device nodes had no
detected users; firmware recovery remained unverified. Evidence:
`results/p8-rope-legacy-wait-audit-user-reported.json`.

### P8.15: eight-tile diagnostic with target allocation and strides

`export_rope_window.py` derives a separately pinned bundle from the frozen host
trace bundle. Only two original `kernel.c` loop bounds change: bx 256 to 2 and
by 32 to 4. Full 1024x1024 shapes, table strides, argument bytes, local addresses,
pair arithmetic, host call tracing and final poll remain unchanged. The
diagnostic computes rows [0,8), columns [0,128), covering eight 4x32 tiles. No
new poll, ID reset, SDK change or production-kernel change is introduced.

`run_add_pcie.py::check_rope_window_output` compares the 1024 computed FP16
elements to the existing independent reference and checks every other output
byte for the original FP16 NaN sentinel. The receipt explicitly limits its
reference scope to the active window. The source hash registry admits only
original one-call diagnostic execution; timing, serial and pipeline are
rejected before loading a runtime. Diagnostic receipts cannot be used as
ordinary source-bundle correctness receipts.

One bounded BM1690 CModel worker passed using the existing PPL 1.7 installation:
two host CPUs, 4 GiB group RSS cap and 120-second deadline. All computed values
matched the reference bitwise (zero error), and all 1047552 uncomputed elements
retained their sentinel bytes. All 22 host calls returned, with exactly one
launch. Source and native-library identities remained unchanged during the
run. The unit suites passed all 149 checks in the existing `.risc` environment.
An initial system-Python run had one Matmul-fixture failure and three skips
because NumPy was absent; the error was reproduced without a vendor load, then
resolved by reusing `.risc`, without dependency installation or weakening tests.
The initial and successful logs are retained with the CModel artifacts.

Prepare case `rope-target-window8` with the new pinned helper in a new remote
directory; compilation alone never loads the board runtime. Before execution,
perform the existing gated ordinary driver recovery and one previously passed
smoke check. Run the window diagnostic once and stop on failure. A window pass
would establish this low-address window with the target strides, not all target
addresses or long-loop behavior. If it passes, separately investigate a window
near the matrix end and increasing tile counts. If a recovered known smoke
fails, investigate the runtime/device state before another RoPE target call.
The user subsequently reported successful remote compilation of all three
variants (18 build steps, no runtime load or kernel launches), plus the Oct 16
read-only preflight. The Oct 16 ordinary reload and both-chip initialization
subsequently passed, followed by a fresh small original smoke pass with zero
reference error and the prior output hash. The window8 diagnostic then passed
with zero active-window error, all sentinels preserved and 22 matched runtime
CALL/RETURN pairs. The smoke evidence is
`results/p8-rope-smoke-post-reload-20261016-user-reported.json`.
Evidence: `results/p8-rope-window8-build-and-preflight-user-reported.json`,
`results/p8-driver-reload-20261016-user-reported.json` and
`results/p8-rope-1024-window8-{cmodel,handoff-validation,board-user-reported}.json`.

### P8.16: eight-tile end-window diagnostic

`export_rope_window_end.py` derives a separate bundle from the frozen, now
board-validated beginning-window bundle. Only the original kernel's two tile
loop headers change: bx [0,2) to [254,256), by [0,4) to [28,32). Eight 4x32 tiles,
full target allocations and strides, local addresses, pair arithmetic, host
traces and final poll are preserved. The active output region is rows
[1016,1024), columns [896,1024). Every preceding output byte and row gap must
retain its original sentinel; the last output element is now actively checked.

At this stage the runner admitted only the two explicitly registered diagnostic windows;
bundle hashes determine their coverage. Serial/pipeline and timing requests are
rejected before runtime loading. Existing bundles and the previously validated
beginning-window exporter are unchanged. Tests check source derivation,
manifest separation, blocked dispatch modes, missing/incorrect first and last
writes, and corruption outside the end window.

All 172 selected regression checks passed in the existing `.risc` environment.
One BM1690 CModel call passed with two host CPUs, a 4 GiB group RSS limit and a
120-second deadline: zero active-window error, all 1047552 outside elements
unchanged, and 22 returned runtime calls including one launch. Input hash is
unchanged from the target; output SHA256 is
`6455bb71bc61bc96afb2eab980bc7ee260a131f4215f9a06bcd456487d140757`.
Native libraries and source identity stayed unchanged during the run. Evidence:
`results/p8-rope-1024-window8-end-{cmodel,handoff-validation}.json`.

The board subsequently passed the one original call with the same 30-second
supervisor bound: zero active error, all outside sentinels preserved, and all
22 runtime calls returned. Both CModel hashes matched. This establishes two
small windows at different matrix locations, not complete address coverage or
long-loop correctness. Evidence is
`results/p8-rope-1024-window8-end-board-user-reported.json`. No counter width, overflow,
SDK incompatibility or missing synchronization has been established as the
timeout's cause; do not alter the production synchronization based on this
partial evidence.

### P8.17: bounded original-loop prefixes

`export_rope_prefix.py` starts from the frozen full-target host-trace bundle.
For 128/256/512 tiles, it changes only bx's upper bound from 256 to 4/8/16;
the full 32-iteration by loop and pair loop remain unchanged. These are actual
prefixes of the original iteration order. They cover 16384/32768/65536 elements
while retaining the original 1024x1024 allocation, input bytes, local buffers,
arithmetic and synchronization. Separate source hashes bind each diagnostic to
its exact reference window; every remaining output byte must retain its NaN
sentinel. All cases remain original-only with no timing acceptance.

The existing Python environment passed 202 selected tests, including source
derivation, manifest binding, boundary corruption, and shell sequencing checks
with fake download/build/launch tools. The sequencing checks verify ordered
execution and no next-case build/launch after a failed call, a wrong receipt,
device occupation or an occupation-check error. They do not launch a TPU.
Three sequential BM1690 CModel workers then passed with two host CPUs, 4 GiB
group RSS cap and 120 seconds per case. Each produced zero active reference
error, preserved all outside bytes and returned all 22 traced runtime calls.
Source/native identities remained unchanged. Evidence is
`results/p8-rope-1024-prefixes-{cmodel,handoff-validation}.json`.

The board sequence requires a new output directory and an explicit
`--allow-pcie` invocation. It uses the observed SDK/runtime through the pinned
prepare helper, checks privileged fuser immediately before each launch, and
requires a passed receipt with the expected prefix coverage. Permission errors
or nonempty fuser output stop the sequence. It neither unloads the driver nor
terminates another task. Per-case host-call timeouts remain 30 seconds with a
40-second outer bound; no failed case is retried automatically. The maximum
workload is 512 of the full target's 8192 tiles. These diagnostic bounds are
workload choices, not claims about hardware command-counter width or overflow.

### P8.18: larger prefixes following three board passes

The 128/256/512 board sequence completed all 54 build steps and three diagnostic
calls. Active reference errors were zero, all outside sentinel bytes were
preserved, and input/output hashes matched local CModel in every case. The
receipt and console output establish only these bounded original prefixes.

`export_rope_prefix_large.py` derives 1024/2048/4096-tile bundles from the frozen,
board-validated 512-tile bundle. Only bx's upper bound changes from 16 to
32/64/128. The by and pair loops, global allocations/strides, local buffers,
arithmetic, host ABI and synchronization are preserved. Old source bundles and
the old prefix exporter remain byte-identical. The runner registers distinct
hashes and windows for each new case; serial/pipeline dispatch and timing are
still blocked for diagnostic bundles. No counter-overflow or SDK explanation
has been established by these workload choices.

The existing environment passed 229 selected checks and three sequential,
one-launch BM1690 CModel workers, with two host CPUs, 4 GiB RSS cap and 120
seconds per worker. Each matched reference bitwise, preserved all unwritten
bytes and returned all 22 traced runtime calls. Source and native library
identities stayed unchanged. These emulator results do not establish board
correctness or performance. Board results subsequently passed 1024/2048 and failed at the unreturned
launch of 4096; see P8.19. The local results remain CModel-only evidence.

The sequence script's explicit `--larger` selector preserves the previous
small default. Tests cover both orders and stop before the next case after a
failed call or incorrect receipt. The larger board sequence remains single-core,
one call per case, 30-second supervisor and 40-second outer run bound, with
privileged occupation checks before each dispatch. It has no automatic retry,
driver operation or performance measurement. Maximum work is 4096/8192 tiles;
only a future complete-target reference match can close full-target correctness.

### P8.19: board passes through 2048 tiles; 4096 launch does not return

The user reported 54 successful build steps, two complete original calls and a
failure on the third original call. The first two returned all runtime APIs,
checked 131072/262144 active elements with zero error and preserved all outside
bytes. The 4096 trace stops after the launch CALL: 13 calls, 12 returns. The
sequence exited one and did not proceed to another case. A subsequent full
failed-case result.json confirms timeout at the 30-second worker limit,
returncode -9 and completed host process cleanup. Its runner/bundle hashes
match the pinned handoff and its input hash matches the passing cases. The later
11:22:50 device check found no holders, but the journal confirms a FORCE_QUIT
timeout at 11:08:34. This requires recovery assessment; host cleanup alone
did not establish firmware recovery. The code's
supervisor measures its deadline over the complete worker, not exclusively
the kernel, and does not store an elapsed-time field. Do not infer device
latency or recovery from these receipt fields.

The generated 2048 and 4096 device sources differ only in outer-loop bound.
There are sixteen pair iterations per tile, each containing sixteen BDC API
calls, plus four GDMA API calls per tile. These are static source-level API
counts, not measured emitted descriptors or hardware counter values. Neither
the inspected local SDK headers nor the provided board disassembly establish
a counter limit or the device-side instruction where this call stopped. Keep
production synchronization, SDK files and preserved baselines unchanged while
collecting the remaining device-state and recovery evidence. The handoff script issues no additional kernel
launch or driver operation after its failure.

The next correctness path will reuse the existing full-target serial and
pipeline builds after ordinary driver recovery and one known small smoke.
Both process all 1024x1024 elements with 32x64 tiling and have prior local
CModel evidence; neither has yet passed the full target on this board. Their
checks use the independent reference even though the original full-target
baseline remains failed. Pipeline must additionally match serial. No full
three-variant performance or original correctness completion can be claimed
from these checks. The failed original artifacts and diagnostics remain
preserved for root-cause work; this path does not replace their status.
