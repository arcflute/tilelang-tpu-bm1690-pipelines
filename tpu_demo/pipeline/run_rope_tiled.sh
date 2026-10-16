#!/usr/bin/env bash
# Reuse existing builds: recovery smoke, full serial, full pipeline. No rebuilds.
set -euo pipefail
if [ "$#" != 5 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Usage: bash run_rope_tiled.sh --allow-pcie FULL_REVISION NEW_OUTPUT_DIR SMOKE_WORK TARGET_WORK" >&2
  exit 2
fi
rope_revision=$2
rope_root=$3
rope_smoke=$4
rope_target=$5
rope_case=download
trap 'rope_rc=$?; echo "ROPE_TILED_STOP case=$rope_case exit=$rope_rc RESULT_ROOT=$rope_root" >&2; exit "$rope_rc"' ERR
for rope_tool in python3 timeout sha256sum sudo fuser uname; do
  command -v "$rope_tool" >/dev/null
done
test -f "$rope_smoke/build/result.json"
test -f "$rope_target/build/result.json"
test -f "$rope_target/run-original/result.json"
mkdir -- "$rope_root"
rope_root=$(cd -- "$rope_root" && pwd -P)
echo "ROPE_TILED_RESULT_ROOT=$rope_root"
timeout -k 2s 60s python3 - "$rope_revision" "$rope_root/run_add.py" <<'PY'
import pathlib, sys, urllib.request
base = 'https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents'
url = base + '/tpu_demo/pipeline/run_add_pcie.py?ref=' + sys.argv[1]
request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github.raw+json'})
with urllib.request.urlopen(request, timeout=45) as response:
    payload = response.read(1048577)
with pathlib.Path(sys.argv[2]).open('xb') as output:
    output.write(payload)
PY
printf '%s  %s\n' d11dac8c7d8d2689ace4df8318bd641eb02389666b135bccd3f7908ee1df3b9c "$rope_root/run_add.py" | sha256sum -c -
uname -a > "$rope_root/environment.txt"
for rope_case in smoke serial pipeline; do
  rope_variant=$rope_case
  rope_build="$rope_target/build"
  rope_comparison=()
  case "$rope_case" in
    smoke) rope_variant=original; rope_build="$rope_smoke/build";;
    serial) rope_comparison=(--original-failure "$rope_target/run-original");;
    pipeline) rope_comparison=(--previous "$rope_root/serial");;
  esac
  echo "ROPE_TILED_BEGIN case=$rope_case"
  rope_users=0
  sudo -n timeout -k 2s 5s fuser -v /dev/sg-host-drv-0 /dev/sg-host-drv-1 \
    > "$rope_root/users-$rope_case.log" 2>&1 || rope_users=$?
  cat "$rope_root/users-$rope_case.log"
  if [ "$rope_users" != 1 ] || [ -s "$rope_root/users-$rope_case.log" ]; then
    echo "STOP: device users or occupation check error" >&2
    exit 2
  fi
  timeout -k 3s 40s python3 "$rope_root/run_add.py" --build "$rope_build" \
    --output "$rope_root/$rope_case" --variant "$rope_variant" "${rope_comparison[@]}" \
    --allow-pcie --device-id 0 --expected-pci 0000:01:00.0 --timeout-seconds 30
  python3 - "$rope_root/$rope_case/result.json" "$rope_case" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
case = sys.argv[2]
n = report.get('numeric') or {}
valid = (report.get('status') == n.get('status') == 'passed' and
         n.get('operation') == 'rope' and n.get('dtype') == 'float16' and
         n.get('launch_cores') == 1 and not n.get('diagnostic_only') and not n.get('timing') and
         n.get('reference', {}).get('passed') is True and
         n.get('variant') == ('original' if case == 'smoke' else case) and
         n.get('shape') == ([8, 128] if case == 'smoke' else [1024, 1024]))
if case != 'smoke':
    valid = (valid and n.get('original_baseline_status') == 'failed' and
             n.get('three_variant_acceptance_complete') is False and
             n.get('original_baseline_failure', {}).get('reason') == 'timeout')
if case == 'pipeline':
    valid = valid and n.get('previous_variant_bitwise_equal') is True
if not valid:
    raise SystemExit('STOP: missing or mismatched full correctness receipt')
PY
  echo "ROPE_TILED_CASE_PASSED case=$rope_case"
done
echo "ROPE_TILED_CORRECTNESS_PASSED original_full=still_failed RESULT_ROOT=$rope_root"
