#!/usr/bin/env bash
# Sequential FP16 smoke/target checks, followed by bounded resident timing.
set -euo pipefail
if [ "$#" != 3 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Usage: bash run_flashattn_board.sh --allow-pcie FULL_REVISION NEW_OUTPUT_DIR" >&2
  exit 2
fi
fa_revision=$2
fa_root=$3
fa_stage=download
trap 'fa_rc=$?; echo "FLASHATTN_STOP stage=$fa_stage exit=$fa_rc RESULT_ROOT=$fa_root" >&2; exit "$fa_rc"' ERR
for fa_tool in python3 timeout sha256sum sudo fuser uname taskset nice; do
  command -v "$fa_tool" >/dev/null
done
mkdir -- "$fa_root"
fa_root=$(cd -- "$fa_root" && pwd -P)
echo "FLASHATTN_RESULT_ROOT=$fa_root"
timeout -k 2s 60s python3 - "$fa_revision" "$fa_root/prepare.py" <<'PY'
import pathlib, sys, urllib.request
base = 'https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents'
url = base + '/tpu_demo/pipeline/prepare_add_1024.py?ref=' + sys.argv[1]
request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github.raw+json'})
with urllib.request.urlopen(request, timeout=45) as response:
    payload = response.read(1048577)
with pathlib.Path(sys.argv[2]).open('xb') as output:
    output.write(payload)
PY
printf '%s  %s\n' bd6b9174f96c50a3388cfd2ebbf25a7634a7ffc7c08d9ad4094f4fd804f65745 "$fa_root/prepare.py" | sha256sum -c -
fa_python=${BM1690_REFERENCE_PYTHON:-/home/bokai/ChunkScan-bm1690-test-86add2f/.venv/bin/python}
test -x "$fa_python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
timeout -k 2s 10s taskset -c 0,1 nice -n 10 "$fa_python" -c 'import sys,numpy; print("REFERENCE_PYTHON",sys.executable,"NUMPY",numpy.__version__)'
for fa_size in smoke target; do
 for fa_mode in noncausal causal; do
  fa_operation=flashattn
  [ "$fa_mode" != causal ] || fa_operation=flashattn-causal
  fa_stage="$fa_size:$fa_mode:build"
  fa_work="$fa_root/$fa_size-$fa_mode"
  timeout -k 5s 600s python3 "$fa_root/prepare.py" --case "$fa_operation-$fa_size" \
    --transport github-api --revision "$fa_revision" --output "$fa_work"
  printf '%s  %s\n' b46d84d8a9c8ba789de383991d8db542f1d1c7e2a3fc7b6d1f987bed78ab51a1 "$fa_work/run_add.py" | sha256sum -c -
  uname -a > "$fa_work/environment.txt"
  for fa_phase in run timing; do
    fa_previous=()
    for fa_variant in original serial pipeline; do
      fa_stage="$fa_size:$fa_mode:$fa_phase:$fa_variant"
      fa_measure=()
      if [ "$fa_phase" = timing ]; then
        fa_measure=(--measure --correctness "$fa_work/run-$fa_variant")
      fi
      echo "FLASHATTN_BEGIN stage=$fa_stage"
      fa_users=0
      sudo -n timeout -k 2s 5s fuser -v /dev/sg-host-drv-0 /dev/sg-host-drv-1 \
        > "$fa_work/users-$fa_phase-$fa_variant.log" 2>&1 || fa_users=$?
      cat "$fa_work/users-$fa_phase-$fa_variant.log"
      if [ "$fa_users" != 1 ] || [ -s "$fa_work/users-$fa_phase-$fa_variant.log" ]; then
        echo "STOP: device users or occupation check error" >&2
        exit 2
      fi
      timeout -k 3s 40s "$fa_python" "$fa_work/run_add.py" --build "$fa_work/build" \
        --output "$fa_work/$fa_phase-$fa_variant" --variant "$fa_variant" \
        "${fa_previous[@]}" "${fa_measure[@]}" --allow-pcie --device-id 0 \
        --expected-pci 0000:01:00.0 --timeout-seconds 30
      python3 - "$fa_work/$fa_phase-$fa_variant/result.json" "$fa_size" "$fa_variant" "$fa_phase" "$fa_operation" <<'PY'
import json, math, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
size, variant, phase, operation = sys.argv[2:]
n = report.get('numeric') or {}
valid = (report.get('status') == n.get('status') == 'passed' and
         n.get('operation') == operation and n.get('variant') == variant and
         n.get('shape') == ([1,64,1,16] if size == 'smoke' else [1,1024,1,64]) and
         n.get('dtype') == 'float16' and n.get('accumulation_dtype') == 'float32' and
         n.get('probability_dtype') == 'float16' and n.get('layout') == 'BSHD' and
         n.get('mask_dtype') == 'float32' and n.get('output_index') == 4 and
         n.get('is_causal') is operation.endswith('-causal') and n.get('launch_cores') == 1 and
         not n.get('diagnostic_only') and not n.get('original_baseline_failure') and
         n.get('reference', {}).get('passed') is True)
if variant == 'pipeline':
    valid = valid and n.get('previous_variant_bitwise_equal') is True
elif variant == 'serial':
    valid = valid and n.get('previous_variant_comparison', {}).get('passed') is True
timing = n.get('timing') or {}
if phase == 'timing':
    raw = timing.get('raw_us') or []
    valid = (valid and timing.get('warmups') == 5 and timing.get('samples') == len(raw) == 20 and
             timing.get('pure_device_time') is False and
             all(math.isfinite(v) and v > 0 for v in raw))
else:
    valid = valid and not timing
if not valid:
    raise SystemExit('STOP: missing or mismatched FlashAttention receipt')
PY
      fa_previous=(--previous "$fa_work/run-$fa_variant")
      echo "FLASHATTN_CASE_PASSED stage=$fa_stage"
    done
  done
 done
done
echo "FLASHATTN_SUITE_PASSED RESULT_ROOT=$fa_root"
