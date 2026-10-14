"""Derive a host-call diagnostic from the frozen RoPE target; never load a runtime."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/bm1690-pipelines/handoff/rope-1024-sources.json"
BASE_SHA256 = "c45a7e3a2676f59be6808d940938a4618bab1f7e78f5fb5ceb58112b5ed8eb87"
APIS = ("tpuRtInit", "tpuRtSetDevice", "tpuRtStreamCreate", "tpuRtStreamDestroy",
        "tpuRtKernelLoadModuleFile", "tpuRtKernelUnloadModule", "tpuRtMalloc",
        "tpuRtFree", "tpuRtMemcpyS2D", "tpuRtMemcpyD2S", "tpuRtKernelLaunch",
        "tpuRtStreamSynchronize")

TRACE_HELPER = r'''
// Diagnostic only: flush before entering any vendor call, including cleanup.
#include <cstdio>
#include <type_traits>
template <typename Call>
static auto tilelang_trace_runtime(const char* name, const char* file, int line, Call call) {
  std::fprintf(stderr, "TPU_RUNTIME_TRACE CALL %s %s:%d\n", name, file, line);
  std::fflush(stderr);
  auto result = call();
  if constexpr (std::is_pointer_v<decltype(result)>) {
    std::fprintf(stderr, "TPU_RUNTIME_TRACE RETURN %s nonnull=%d\n", name, result != nullptr);
  } else {
    std::fprintf(stderr, "TPU_RUNTIME_TRACE RETURN %s rc=%d\n", name, static_cast<int>(result));
  }
  std::fflush(stderr);
  return result;
}
'''


def instrument(source, anchor):
    if source.count(anchor) != 1 or "TPU_RUNTIME_TRACE" in source:
        raise ValueError("Unexpected host source or already instrumented")
    macros = "\n".join(
        f'#define {api}(...) tilelang_trace_runtime("{api}", __FILE__, __LINE__, '
        f'[&]() {{ return ({api})(__VA_ARGS__); }})' for api in APIS)
    # After every original include: never rewrite vendor/header declarations.
    return source.replace(anchor, TRACE_HELPER + macros + "\n" + anchor)


def export():
    payload = BASE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen target bundle changed")
    original = json.loads(payload)
    bundle = copy.deepcopy(original)
    variant = bundle["variants"]["original"]
    for filename, anchor in (("main.cpp", "tpuRtStream_t stream = nullptr;"),
                             ("kernel.cpp", "extern tpuRtStream_t stream;")):
        variant["sources"][filename] = instrument(variant["sources"][filename], anchor)
    # The host's normal duration print would include tracing overhead.
    duration_print = ('    std::cout << "Single kernel execution time: " << elapsed_time_ms << " ms ("\n'
                      '              << duration.count() << " us)\\n";')
    if variant["sources"]["main.cpp"].count(duration_print) != 1:
        raise ValueError("Unexpected single-call timing output")
    variant["sources"]["main.cpp"] = variant["sources"]["main.cpp"].replace(
        duration_print, '    std::cout << "Diagnostic trace: no performance measurement\\n";')
    variant["sha256"] = {name: hashlib.sha256(source.encode()).hexdigest()
                         for name, source in variant["sources"].items()}
    bundle["host_diagnostic"] = {
        "kind": "flushed-runtime-call-boundaries", "allowed_variant": "original",
        "base_bundle_sha256": BASE_SHA256,
        "derivation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modified_files": ["original/main.cpp", "original/kernel.cpp"],
        "device_kernel_unchanged": True, "timing_allowed": False,
        "apis": list(APIS), "board_execution_performed": False}
    assert variant["sources"]["kernel.c"] == original["variants"]["original"]["sources"]["kernel.c"]
    return bundle


if __name__ == "__main__":
    destination = ROOT / "research/bm1690-pipelines/handoff/rope-1024-host-trace-sources.json"
    with destination.open("x") as output:
        output.write(json.dumps(export(), indent=2, sort_keys=True) + "\n")
