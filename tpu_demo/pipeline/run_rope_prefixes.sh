#!/usr/bin/env bash
# Three bounded original-only diagnostics; no retries or driver changes.
set -euo pipefail
if { [ "$#" != 3 ] && [ "$#" != 4 ]; } || [ "${1:-}" != --allow-pcie ] ||
   [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]] || { [ "$#" = 4 ] && [ "$4" != --larger ]; }; then
  echo "Usage: bash run_rope_prefixes.sh --allow-pcie FULL_REVISION NEW_OUTPUT_DIR [--larger]" >&2
  exit 2
fi
prefix_revision=$2
prefix_root=$3
prefix_case=bootstrap
prefix_counts=(128 256 512)
if [ "${4:-}" = --larger ]; then
  prefix_counts=(1024 2048 4096)
fi
trap 'prefix_rc=$?; echo "PREFIX_STOP case=$prefix_case exit=$prefix_rc RESULT_ROOT=$prefix_root" >&2; exit "$prefix_rc"' ERR
for prefix_tool in python3 timeout sha256sum sudo fuser uname; do
  command -v "$prefix_tool" >/dev/null
done
mkdir -- "$prefix_root"
prefix_root=$(cd -- "$prefix_root" && pwd -P)
echo "PREFIX_RESULT_ROOT=$prefix_root"
timeout -k 2s 60s python3 - "$prefix_revision" "$prefix_root/prepare.py" <<'PY'
import pathlib, sys, urllib.request
base = 'https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents'
url = base + '/tpu_demo/pipeline/prepare_add_1024.py?ref=' + sys.argv[1]
request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github.raw+json'})
with urllib.request.urlopen(request, timeout=45) as response:
    payload = response.read(1048577)
with pathlib.Path(sys.argv[2]).open('xb') as output:
    output.write(payload)
PY
printf '%s  %s\n' 1a7b88e70f8026d84539986794b8cde61acb2db15596bec05ef4811c0a1cc6ae "$prefix_root/prepare.py" | sha256sum -c -
for prefix_tiles in "${prefix_counts[@]}"; do
  prefix_case="rope-target-prefix$prefix_tiles"
  prefix_work="$prefix_root/$prefix_tiles"
  echo "PREFIX_BEGIN tiles=$prefix_tiles"
  timeout -k 5s 600s python3 "$prefix_root/prepare.py" --case "$prefix_case" \
    --transport github-api --revision "$prefix_revision" --output "$prefix_work"
  printf '%s  %s\n' 06baf81deeae3249c5c9249c2067381798396c7fe9bc62869579564750d7a135 "$prefix_work/run_add.py" | sha256sum -c -
  prefix_users=0
  sudo -n timeout -k 2s 5s fuser -v /dev/sg-host-drv-0 /dev/sg-host-drv-1 \
    > "$prefix_work/users.log" 2>&1 || prefix_users=$?
  cat "$prefix_work/users.log"
  if [ "$prefix_users" != 1 ] || [ -s "$prefix_work/users.log" ]; then
    echo "STOP: device users or occupation check error; inspect $prefix_work/users.log" >&2
    exit 2
  fi
  uname -a > "$prefix_work/environment.txt"
  timeout -k 3s 40s python3 "$prefix_work/run_add.py" --build "$prefix_work/build" \
    --output "$prefix_work/run-original" --variant original --allow-pcie \
    --device-id 0 --expected-pci 0000:01:00.0 --timeout-seconds 30
  python3 - "$prefix_work/run-original/result.json" "$prefix_tiles" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
tiles = int(sys.argv[2])
numeric = report.get('numeric') or {}
reference = numeric.get('reference') or {}
if not (report.get('status') == numeric.get('status') == 'passed' and
        numeric.get('operation') == 'rope' and numeric.get('variant') == 'original' and
        numeric.get('diagnostic_only') is True and not numeric.get('timing') and
        reference.get('passed') is True and reference.get('scope') == 'active_window_only' and
        reference.get('window') == [0, tiles // 8, 0, 1024] and
        reference.get('checked_elements') == tiles * 128 and
        reference.get('untouched_elements') == 1024 * 1024 - tiles * 128 and
        reference.get('untouched_sentinel_preserved') is True):
    raise SystemExit('STOP: missing or mismatched prefix correctness receipt')
PY
  echo "PREFIX_CASE_PASSED tiles=$prefix_tiles"
done
echo "PREFIX_SUITE_PASSED RESULT_ROOT=$prefix_root"
