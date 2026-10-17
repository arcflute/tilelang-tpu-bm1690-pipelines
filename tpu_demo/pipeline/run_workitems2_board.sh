#!/usr/bin/env bash
# One bounded two-workitem ownership check, reusing the passed board toolchain.
set -euo pipefail
if [ "$#" != 4 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Usage: bash run_workitems2_board.sh --allow-pcie REVISION NEW_OUTPUT PASSED_FLASH_TARGET_DIR' >&2
  exit 2
fi
wi_revision=$2
wi_root=$3
wi_previous=$4
wi_stage=prepare
trap 'wi_rc=$?; echo "WORKITEM_SUITE_STOP stage=$wi_stage exit=$wi_rc RESULT_ROOT=$wi_root" >&2; exit "$wi_rc"' ERR
mkdir -- "$wi_root"
wi_root=$(cd -- "$wi_root" && pwd -P)
echo "WORKITEM_RESULT_ROOT=$wi_root"
timeout -k 5s 300s python3 - "$wi_revision" "$wi_root" "$wi_previous" <<'PY'
import hashlib, json, os, pathlib, shutil, subprocess, sys, urllib.request
revision, root, old = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]).resolve()
def check(path, sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise SystemExit('Checksum mismatch: '+str(path))
for name, sha in (
    ('build.py','595fca52044098d9ea319f17b5ed86362d6700ce6e65c419e3876ece63e18ff2'),
    ('run_add.py','b46d84d8a9c8ba789de383991d8db542f1d1c7e2a3fc7b6d1f987bed78ab51a1')):
    check(old/name,sha)
    shutil.copyfile(old/name,root/name)
    check(root/name,sha)
    print('REUSED_VERIFIED '+name,flush=True)
base='https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents/'
for path, name, sha in (
    ('tpu_demo/pipeline/workitem_probe_pcie.py','probe.py','d239e30415374aec2ff4d4cc239c8528eb42b0e8423b443b980205d94d69578a'),
    ('research/bm1690-pipelines/handoff/workitems-2-sources.json','probe.json','cb193033e73d7a77eb6574ee1a1844a9ebd523f666b265b7281c0dacde74b4c8')):
    request=urllib.request.Request(base+path+'?ref='+revision,headers={'Accept':'application/vnd.github.raw+json'})
    print('DOWNLOAD '+name,flush=True)
    with urllib.request.urlopen(request,timeout=45) as response:
        data=response.read(1048577)
    if len(data)>1048576:
        raise SystemExit('Downloaded file exceeds size limit')
    with (root/name).open('xb') as output:
        output.write(data)
    check(root/name,sha)
    print('VERIFIED '+name,flush=True)
# Take paths from the user's passed build receipt, then check its SDK hashes.
manifest=json.loads((old/'build/result.json').read_text())
if manifest.get('status')!='compile_only_passed':
    raise SystemExit('Existing FlashAttention build did not pass')
for path,sha in manifest['sdk_inputs'].items():
    check(pathlib.Path(path),sha)
cross=[p for p in manifest['sdk_inputs'] if pathlib.Path(p).name=='riscv64-unknown-linux-gnu-gcc']
if len(cross)!=1:
    raise SystemExit('Missing compiler identity')
env=dict(os.environ,PPL_PROJECT_ROOT=manifest['runtime_identity'][0],
         TILELANG_TPU_PCIE_RUNTIME_PATH=manifest['runtime_identity'][1],TILELANG_TPU_PCIE_CROSS_GCC=cross[0])
subprocess.run(['timeout','-k','5s','180s',sys.executable,str(root/'probe.py'),'--build-only','--work',str(root)],env=env,check=True)
PY
wi_stage=ownership
uname -a > "$wi_root/environment.txt"
timeout -k 3s 40s python3 "$wi_root/probe.py" --run-only --allow-pcie --work "$wi_root" --output "$wi_root/run"
python3 - "$wi_root/run/result.json" <<'PY'
import json,pathlib,sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text());n=r.get('numeric') or {}
if not (r.get('status')=='passed' and r.get('cleanup_complete') is True and n.get('status')=='passed' and
        n.get('launch_cores')==2 and n.get('tasks')==11 and n.get('ownership_counts')==[6,5] and
        n.get('checked_elements')==704 and n.get('exact_ownership') is True and
        n.get('requested_kernel_calls')==1 and n.get('mapping',{}).get('pci_function')=='0000:01:00.0' and
        n.get('output_sha256')=='e28614fe1296504598a309802c398b69b6e2fc0612f030ac915754f193882290'):
    raise SystemExit('STOP: two-workitem receipt differs from the checked CModel contract')
PY
echo "WORKITEM_SUITE_PASSED RESULT=$wi_root/run/result.json"
