#!/usr/bin/env bash
# Fixed Add matrix: original, serial/pipeline at one/two launched workitems.
set -euo pipefail
if [ "$#" != 4 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Usage: bash run_add_multicore_board.sh --allow-pcie REVISION NEW_OUTPUT PASSED_WORKITEM_DIR' >&2
  exit 2
fi
mc_revision=$2
mc_root=$3
mc_previous=$4
mc_stage=prepare
trap 'mc_rc=$?; echo "ADD_MULTICORE_STOP stage=$mc_stage exit=$mc_rc RESULT_ROOT=$mc_root" >&2; exit "$mc_rc"' ERR
mkdir -- "$mc_root"
mc_root=$(cd -- "$mc_root" && pwd -P)
echo "ADD_MULTICORE_RESULT_ROOT=$mc_root"
timeout -k 5s 600s python3 - "$mc_revision" "$mc_root" "$mc_previous" <<'PY'
import hashlib, importlib.util, json, os, pathlib, shutil, subprocess, sys, urllib.request
revision, root, old = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]).resolve()
def check(path, sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise SystemExit('Checksum mismatch: '+str(path))
# Validate the passed two-workitem gate and derive all SDK paths from its build.
check(old/'probe.py','d239e30415374aec2ff4d4cc239c8528eb42b0e8423b443b980205d94d69578a')
spec=importlib.util.spec_from_file_location('checked_probe',old/'probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
bundle,builder,support=probe.assets(old)
manifest,runtime=probe.validate_build(old,bundle,builder)
receipt=json.loads((old/'run/result.json').read_text());numeric=receipt.get('numeric') or {}
required=dict(status='passed',launch_cores=2,tasks=11,ownership_counts=[6,5],checked_elements=704,
              exact_ownership=True,bundle_sha256=probe.BUNDLE_SHA,artifacts=manifest['artifacts'],
              build_manifest_sha256=probe.digest(old/'build/result.json'),runtime_sha256=probe.digest(runtime),
              output_sha256='e28614fe1296504598a309802c398b69b6e2fc0612f030ac915754f193882290')
if receipt.get('status')!='passed' or receipt.get('cleanup_complete') is not True or any(numeric.get(k)!=v for k,v in required.items()):
    raise SystemExit('Missing passed two-workitem board gate')
check(old/'run/output.f16',required['output_sha256'])
support.save(root/'workitem-gate.json',dict(source=str(old),receipt_sha256=probe.digest(old/'run/result.json'),numeric=numeric))
for name,sha in (('build.py',probe.BUILDER_SHA),('run_add.py',probe.SUPPORT_SHA)):
    check(old/name,sha);shutil.copyfile(old/name,root/name);check(root/name,sha)
    print('REUSED_VERIFIED '+name,flush=True)
base='https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents/'
for path,name,sha in (
    ('tpu_demo/pipeline/add_multicore_pcie.py','multicore.py','6c7065ae3e87f85e5cbab183c7c43cfe96b29bebdd8007ecffaad5b45bb039d5'),
    ('research/bm1690-pipelines/handoff/add-multicore-sources.json','add-multicore.json','da283c0dcc371eacbd45544f803dc819fdb0c753ff7c7344be9811007b817fd5')):
    request=urllib.request.Request(base+path+'?ref='+revision,headers={'Accept':'application/vnd.github.raw+json'})
    print('DOWNLOAD '+name,flush=True)
    with urllib.request.urlopen(request,timeout=45) as response:
        data=response.read(1048577)
    if len(data)>1048576:
        raise SystemExit('Download exceeds size limit')
    with (root/name).open('xb') as output:
        output.write(data)
    check(root/name,sha);print('VERIFIED '+name,flush=True)
cross=[p for p in manifest['sdk_inputs'] if pathlib.Path(p).name=='riscv64-unknown-linux-gnu-gcc']
if len(cross)!=1:
    raise SystemExit('Missing compiler identity')
env=dict(os.environ,PPL_PROJECT_ROOT=manifest['runtime_identity'][0],
         TILELANG_TPU_PCIE_RUNTIME_PATH=manifest['runtime_identity'][1],TILELANG_TPU_PCIE_CROSS_GCC=cross[0])
for variant in ('original','serial_1','pipeline_1','serial_2','pipeline_2'):
    subprocess.run(['timeout','-k','5s','180s',sys.executable,str(root/'multicore.py'),
                    '--build-only','--work',str(root),'--variant',variant],env=env,check=True)
PY
uname -a > "$mc_root/environment.txt"
# Each worker independently checks both device nodes and acquires the device lock.
for mc_variant in original serial_1 pipeline_1 serial_2 pipeline_2; do
  mc_stage="correctness-$mc_variant"
  echo "ADD_MULTICORE_BEGIN $mc_stage"
  timeout -k 3s 40s python3 "$mc_root/multicore.py" --run-only --allow-pcie \
    --work "$mc_root" --variant "$mc_variant" --output "$mc_root/run-$mc_variant"
  mc_stage="timing-$mc_variant"
  echo "ADD_MULTICORE_BEGIN $mc_stage"
  timeout -k 3s 40s python3 "$mc_root/multicore.py" --run-only --allow-pcie \
    --work "$mc_root" --variant "$mc_variant" --output "$mc_root/timing-$mc_variant" \
    --correctness "$mc_root/run-$mc_variant"
done
mc_stage=summary
python3 - "$mc_root" <<'PY'
import importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('add_multicore_summary',root/'multicore.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.summarize_suite(root)
PY
echo "ADD_MULTICORE_SUITE_PASSED RESULT=$mc_root/summary.json"
