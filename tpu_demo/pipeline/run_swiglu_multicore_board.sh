#!/usr/bin/env bash
# One sequential batch: SwiGLU, original plus serial/pipeline at one/two cores.
set -euo pipefail
if [ "$#" != 4 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Usage: bash run_swiglu_multicore_board.sh --allow-pcie REVISION NEW_OUTPUT PASSED_ADD_DIR' >&2
  exit 2
fi
sg_revision=$2
sg_root=$3
sg_previous=$4
sg_stage=prepare
trap 'sg_rc=$?; echo "SWIGLU_MULTICORE_STOP stage=$sg_stage exit=$sg_rc RESULT_ROOT=$sg_root" >&2; exit "$sg_rc"' ERR
mkdir -- "$sg_root"
sg_root=$(cd -- "$sg_root" && pwd -P)
echo "SWIGLU_MULTICORE_RESULT_ROOT=$sg_root"
timeout -k 5s 900s python3 - "$sg_revision" "$sg_root" "$sg_previous" <<'PY'
import hashlib,importlib.util,json,os,pathlib,shutil,subprocess,sys,urllib.request
revision,root,old=sys.argv[1],pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3]).resolve()
def check(path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
        raise SystemExit('Checksum mismatch: '+str(path))
check(old/'multicore.py','6c7065ae3e87f85e5cbab183c7c43cfe96b29bebdd8007ecffaad5b45bb039d5')
spec=importlib.util.spec_from_file_location('passed_add',old/'multicore.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
bundle,builder,support=base.assets(old)
gate={}
for variant in base.VARIANTS:
    manifest,runtime=base.validate_build(old,bundle,builder,variant)
    path=old/('timing-'+variant)
    receipt=json.loads((path/'result.json').read_text());n=receipt.get('numeric') or {}
    expected=dict(status='passed',operation='add',variant=variant,**base.contract(variant),
        measurement='timing',requested_kernel_calls=25,shape=[1024,1024],dtype='float16',
        bundle_sha256=base.BUNDLE_SHA,build_manifest_sha256=base.digest(old/'build'/variant/'result.json'),
        artifacts=manifest['artifacts'],runtime_sha256=base.digest(runtime),
        input_sha256='171cadcff2c82f6478fc59a9f95e713d8d7b7379a48796aebdc31ea94d22f5f1',
        output_sha256='014b884c7bac8e0ca6d6da60329da9bdf043cfecb4f709b301fe60b64a0f16e0')
    if (receipt.get('status')!='passed' or receipt.get('cleanup_complete') is not True or
            any(n.get(k)!=v for k,v in expected.items()) or not n.get('reference',{}).get('bitwise_equal_to_reference') or
            n.get('mapping',{}).get('pci_function')!='0000:01:00.0' or n.get('mapping',{}).get('device_id')!=0):
        raise SystemExit('Missing passed matched Add gate: '+variant)
    prior=base.validate_correctness(old/('run-'+variant),n)
    check(path/'output.f16',expected['output_sha256'])
    if (path/'output.f16').read_bytes()!=prior:
        raise SystemExit('Add timing output differs from correctness')
    t=n['sync_latency']
    if (t.get('warmups')!=5 or t.get('samples')!=20 or len(t['raw_us'])!=20 or
            any(t.get(k)!=v for k,v in support.summarize_samples(t['raw_us']).items())):
        raise SystemExit('Incomplete Add timing evidence')
    gate[variant]=base.digest(path/'result.json')
support.save(root/'add-gate.json',dict(source=str(old),timing_receipt_sha256=gate))
print('REUSED_VERIFIED Add five-variant gate',flush=True)
api='https://api.github.com/repos/arcflute/tilelang-tpu-bm1690-pipelines/contents/'
def download(source,dest,sha):
    print('DOWNLOAD '+str(dest.relative_to(root)),flush=True)
    request=urllib.request.Request(api+source+'?ref='+revision,headers={'Accept':'application/vnd.github.raw+json'})
    with urllib.request.urlopen(request,timeout=45) as response:
        data=response.read(1048577)
    if len(data)>1048576:
        raise SystemExit('Download exceeds size limit')
    with dest.open('xb') as out:out.write(data)
    check(dest,sha);print('VERIFIED '+str(dest.relative_to(root)),flush=True)
download('tpu_demo/pipeline/swiglu_multicore_pcie.py',root/'swiglu.py','aa987f43ea9d332ff782e86c11154f6395d6496ebc6c7f97bce36127068cbb71')
for op,sha in (('swiglu', '26c7e140dac693ad1005bb1dd0a495c73863f3d6547ad8edf77783a31dc42f11'),):
    case=root/op;case.mkdir()
    for name,pin in (('build.py',base.BUILDER_SHA),('run_add.py',base.SUPPORT_SHA)):
        check(old/name,pin);shutil.copyfile(old/name,case/name);check(case/name,pin)
    download('research/bm1690-pipelines/handoff/'+op+'-multicore-sources.json',case/'swiglu-multicore.json',sha)
cross=[p for p in manifest['sdk_inputs'] if pathlib.Path(p).name=='riscv64-unknown-linux-gnu-gcc']
if len(cross)!=1:raise SystemExit('Missing compiler identity')
env=dict(os.environ,PPL_PROJECT_ROOT=manifest['runtime_identity'][0],
    TILELANG_TPU_PCIE_RUNTIME_PATH=manifest['runtime_identity'][1],TILELANG_TPU_PCIE_CROSS_GCC=cross[0])
for op in ('swiglu',):
    for variant in base.VARIANTS:
        print('SWIGLU_MULTICORE_COMPILE operation='+op+' variant='+variant,flush=True)
        subprocess.run(['timeout','-k','5s','180s',sys.executable,str(root/'swiglu.py'),
            '--build-only','--work',str(root/op),'--variant',variant],env=env,check=True)
PY
uname -a > "$sg_root/environment.txt"
for sg_op in swiglu; do
  sg_work="$sg_root/$sg_op"
  for sg_variant in original serial_1 pipeline_1 serial_2 pipeline_2; do
    sg_stage="$sg_op-correctness-$sg_variant"
    echo "SWIGLU_MULTICORE_BEGIN $sg_stage"
    timeout -k 3s 40s python3 "$sg_root/swiglu.py" --run-only --allow-pcie \
      --work "$sg_work" --variant "$sg_variant" --output "$sg_work/run-$sg_variant"
    sg_stage="$sg_op-timing-$sg_variant"
    echo "SWIGLU_MULTICORE_BEGIN $sg_stage"
    timeout -k 3s 40s python3 "$sg_root/swiglu.py" --run-only --allow-pcie \
      --work "$sg_work" --variant "$sg_variant" --output "$sg_work/timing-$sg_variant" \
      --correctness "$sg_work/run-$sg_variant"
  done
  sg_stage="$sg_op-summary"
  python3 - "$sg_root" "$sg_op" <<'PY'
import importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('sg_summary',root/'swiglu.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.summarize_suite(root/sys.argv[2])
PY
done
echo "SWIGLU_MULTICORE_SUITE_PASSED operation=swiglu RESULT_ROOT=$sg_root"
