#!/usr/bin/env bash
# Sequential FP16 smoke/target checks, followed by bounded resident timing.
set -euo pipefail
if [ "$#" != 3 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Usage: bash run_swiglu_board.sh --allow-pcie FULL_REVISION NEW_OUTPUT_DIR" >&2
  exit 2
fi
sg_revision=$2
sg_root=$3
sg_stage=download
trap 'sg_rc=$?; echo "SWIGLU_STOP stage=$sg_stage exit=$sg_rc RESULT_ROOT=$sg_root" >&2; exit "$sg_rc"' ERR
for sg_tool in python3 timeout sha256sum sudo fuser uname; do
  command -v "$sg_tool" >/dev/null
done
mkdir -- "$sg_root"
sg_root=$(cd -- "$sg_root" && pwd -P)
echo "SWIGLU_RESULT_ROOT=$sg_root"
timeout -k 2s 60s python3 - "$sg_revision" "$sg_root/prepare.py" <<'PY'
import pathlib, sys, urllib.request
base = 'https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents'
url = base + '/tpu_demo/pipeline/prepare_add_1024.py?ref=' + sys.argv[1]
request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github.raw+json'})
with urllib.request.urlopen(request, timeout=45) as response:
    payload = response.read(1048577)
with pathlib.Path(sys.argv[2]).open('xb') as output:
    output.write(payload)
PY
printf '%s  %s\n' bd6b9174f96c50a3388cfd2ebbf25a7634a7ffc7c08d9ad4094f4fd804f65745 "$sg_root/prepare.py" | sha256sum -c -
for sg_size in smoke target; do
  sg_stage="$sg_size:build"
  sg_work="$sg_root/$sg_size"
  timeout -k 5s 600s python3 "$sg_root/prepare.py" --case "swiglu-$sg_size" \
    --transport github-api --revision "$sg_revision" --output "$sg_work"
  printf '%s  %s\n' b46d84d8a9c8ba789de383991d8db542f1d1c7e2a3fc7b6d1f987bed78ab51a1 "$sg_work/run_add.py" | sha256sum -c -
  uname -a > "$sg_work/environment.txt"
  for sg_phase in run timing; do
    sg_previous=()
    for sg_variant in original serial pipeline; do
      sg_stage="$sg_size:$sg_phase:$sg_variant"
      sg_measure=()
      if [ "$sg_phase" = timing ]; then
        sg_measure=(--measure --correctness "$sg_work/run-$sg_variant")
      fi
      echo "SWIGLU_BEGIN stage=$sg_stage"
      sg_users=0
      sudo -n timeout -k 2s 5s fuser -v /dev/sg-host-drv-0 /dev/sg-host-drv-1 \
        > "$sg_work/users-$sg_phase-$sg_variant.log" 2>&1 || sg_users=$?
      cat "$sg_work/users-$sg_phase-$sg_variant.log"
      if [ "$sg_users" != 1 ] || [ -s "$sg_work/users-$sg_phase-$sg_variant.log" ]; then
        echo "STOP: device users or occupation check error" >&2
        exit 2
      fi
      timeout -k 3s 40s python3 "$sg_work/run_add.py" --build "$sg_work/build" \
        --output "$sg_work/$sg_phase-$sg_variant" --variant "$sg_variant" \
        "${sg_previous[@]}" "${sg_measure[@]}" --allow-pcie --device-id 0 \
        --expected-pci 0000:01:00.0 --timeout-seconds 30
      python3 - "$sg_work/$sg_phase-$sg_variant/result.json" "$sg_size" "$sg_variant" "$sg_phase" <<'PY'
import json, math, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
size, variant, phase = sys.argv[2:]
n = report.get('numeric') or {}
valid = (report.get('status') == n.get('status') == 'passed' and
         n.get('operation') == 'swiglu' and n.get('variant') == variant and
         n.get('shape') == ([8,128] if size == 'smoke' else [1024,1024]) and
         n.get('dtype') == 'float16' and n.get('intermediate_dtype') == 'float32' and
         n.get('buffer_reuse') is False and n.get('launch_cores') == 1 and
         not n.get('diagnostic_only') and not n.get('original_baseline_failure') and
         n.get('reference', {}).get('passed') is True)
if variant != 'original':
    valid = valid and n.get('previous_variant_bitwise_equal') is True
timing = n.get('timing') or {}
if phase == 'timing':
    raw = timing.get('raw_us') or []
    valid = (valid and timing.get('warmups') == 5 and timing.get('samples') == len(raw) == 20 and
             timing.get('pure_device_time') is False and
             all(math.isfinite(v) and v > 0 for v in raw))
else:
    valid = valid and not timing
if not valid:
    raise SystemExit('STOP: missing or mismatched SwiGLU receipt')
PY
      sg_previous=(--previous "$sg_work/run-$sg_variant")
      echo "SWIGLU_CASE_PASSED stage=$sg_stage"
    done
  done
done
echo "SWIGLU_SUITE_PASSED RESULT_ROOT=$sg_root"
