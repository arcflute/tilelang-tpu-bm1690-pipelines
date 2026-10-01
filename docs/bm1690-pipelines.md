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

Mul/Div board compilation remains pending. Next compile Mul in a fresh work
directory using the already verified helper from `0a49d91` with `--case mul`;
the frozen helper can be reused without another bootstrap download. Validate
all three variants once, then collect their 5+20 latency samples. Repeat for Div
without changing the environment, then continue Matmul,
RMSNorm (including Split-K), RoPE, SwiGLU and FlashAttention. A compiler/runtime
failure retains the immutable source bundle, logs and prior Add results. Do not
rebuild native dependencies or retry failed board kernels automatically.
