#!/usr/bin/env bash
# One sequential batch: ordinary/Split-K RMSNorm, original plus serial/pipeline at one/two cores.
set -euo pipefail
if [ "$#" != 4 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Usage: bash run_rmsnorm_multicore_board.sh --allow-pcie REVISION NEW_OUTPUT PASSED_ADD_DIR' >&2
  exit 2
fi
rn_revision=$2
rn_root=$3
rn_previous=$4
rn_stage=prepare
trap 'rn_rc=$?; echo "RMSNORM_MULTICORE_STOP stage=$rn_stage exit=$rn_rc RESULT_ROOT=$rn_root" >&2; exit "$rn_rc"' ERR
mkdir -- "$rn_root"
rn_root=$(cd -- "$rn_root" && pwd -P)
echo "RMSNORM_MULTICORE_RESULT_ROOT=$rn_root"
timeout -k 5s 900s python3 - "$rn_revision" "$rn_root" "$rn_previous" <<'PY'
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
download('tpu_demo/pipeline/rmsnorm_multicore_pcie.py',root/'rmsnorm.py','1db87fcd80ec8f81860878908c795ae69a3c5ae1a05ce32d9ec234075011e79e')
for op,sha in (('rmsnorm', '7b67b3740b01498098dac1e3a548941e9ce10e435b4f6c3215c313d2aaf5d82a'), ('rmsnorm-splitk', '917970da9d4496e6a11702d240f8ae68a76aa94215a4d9f115c63b456b559e10')):
    case=root/op;case.mkdir()
    for name,pin in (('build.py',base.BUILDER_SHA),('run_add.py',base.SUPPORT_SHA)):
        check(old/name,pin);shutil.copyfile(old/name,case/name);check(case/name,pin)
    download('research/bm1690-pipelines/handoff/'+op+'-multicore-sources.json',case/'rmsnorm-multicore.json',sha)
cross=[p for p in manifest['sdk_inputs'] if pathlib.Path(p).name=='riscv64-unknown-linux-gnu-gcc']
if len(cross)!=1:raise SystemExit('Missing compiler identity')
env=dict(os.environ,PPL_PROJECT_ROOT=manifest['runtime_identity'][0],
    TILELANG_TPU_PCIE_RUNTIME_PATH=manifest['runtime_identity'][1],TILELANG_TPU_PCIE_CROSS_GCC=cross[0])
for op in ('rmsnorm','rmsnorm-splitk'):
    for variant in base.VARIANTS:
        print('RMSNORM_MULTICORE_COMPILE operation='+op+' variant='+variant,flush=True)
        subprocess.run(['timeout','-k','5s','180s',sys.executable,str(root/'rmsnorm.py'),
            '--build-only','--work',str(root/op),'--variant',variant],env=env,check=True)
PY
uname -a > "$rn_root/environment.txt"
for rn_op in rmsnorm rmsnorm-splitk; do
  rn_work="$rn_root/$rn_op"
  for rn_variant in original serial_1 pipeline_1 serial_2 pipeline_2; do
    rn_stage="$rn_op-correctness-$rn_variant"
    echo "RMSNORM_MULTICORE_BEGIN $rn_stage"
    timeout -k 3s 40s python3 "$rn_root/rmsnorm.py" --run-only --allow-pcie \
      --work "$rn_work" --variant "$rn_variant" --output "$rn_work/run-$rn_variant"
    rn_stage="$rn_op-timing-$rn_variant"
    echo "RMSNORM_MULTICORE_BEGIN $rn_stage"
    timeout -k 3s 40s python3 "$rn_root/rmsnorm.py" --run-only --allow-pcie \
      --work "$rn_work" --variant "$rn_variant" --output "$rn_work/timing-$rn_variant" \
      --correctness "$rn_work/run-$rn_variant"
  done
  rn_stage="$rn_op-summary"
  python3 - "$rn_root" "$rn_op" <<'PY'
import importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('rn_summary',root/'rmsnorm.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.summarize_suite(root/sys.argv[2])
PY
done
echo "RMSNORM_MULTICORE_SUITE_PASSED operations=rmsnorm,rmsnorm-splitk RESULT_ROOT=$rn_root"
