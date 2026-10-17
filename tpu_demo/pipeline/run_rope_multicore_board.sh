#!/usr/bin/env bash
# One sequential batch: RoPE serial/pipeline at one/two cores; retain the failed full original.
set -euo pipefail
if [ "$#" != 5 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Usage: bash run_rope_multicore_board.sh --allow-pcie REVISION NEW_OUTPUT PASSED_ADD_DIR ORIGINAL_FAILURE_DIR' >&2
  exit 2
fi
rp_revision=$2
rp_root=$3
rp_previous=$4
rp_failure=$5
rp_stage=prepare
trap 'rp_rc=$?; echo "ROPE_MULTICORE_STOP stage=$rp_stage exit=$rp_rc RESULT_ROOT=$rp_root" >&2; exit "$rp_rc"' ERR
mkdir -- "$rp_root"
rp_root=$(cd -- "$rp_root" && pwd -P)
echo "ROPE_MULTICORE_RESULT_ROOT=$rp_root"
timeout -k 5s 900s python3 - "$rp_revision" "$rp_root" "$rp_previous" "$rp_failure" <<'PY'
import hashlib,importlib.util,json,os,pathlib,shutil,subprocess,sys,urllib.request
revision,root,old=sys.argv[1],pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3]).resolve()
failure=pathlib.Path(sys.argv[4]).resolve()/'result.json'
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
download('tpu_demo/pipeline/rope_multicore_pcie.py',root/'rope.py','316731f3968a2ac4885870a08d9ee132f0710f4a16d66e247c14fa782bd9942a')
for name,pin in (('build.py',base.BUILDER_SHA),('run_add.py',base.SUPPORT_SHA)):
    check(old/name,pin);shutil.copyfile(old/name,root/name);check(root/name,pin)
download('research/bm1690-pipelines/handoff/rope-multicore-sources.json',root/'rope-multicore.json','a32a3f62dd09eaf41bbd1ecb3239e4a690e8b356d6a63f967bbd451d9de6d1d9')
shutil.copyfile(failure,root/'original-failure.json')
spec=importlib.util.spec_from_file_location('rope_batch',root/'rope.py')
rope=importlib.util.module_from_spec(spec);spec.loader.exec_module(rope)
context=rope.original_failure(root,base.digest(runtime))
support.save(root/'original-context.json',dict(source=str(failure),**context))
print('RETAINED_ORIGINAL_TIMEOUT no original dispatch',flush=True)
cross=[p for p in manifest['sdk_inputs'] if pathlib.Path(p).name=='riscv64-unknown-linux-gnu-gcc']
if len(cross)!=1:raise SystemExit('Missing compiler identity')
env=dict(os.environ,PPL_PROJECT_ROOT=manifest['runtime_identity'][0],
    TILELANG_TPU_PCIE_RUNTIME_PATH=manifest['runtime_identity'][1],TILELANG_TPU_PCIE_CROSS_GCC=cross[0])
for variant in ('serial_1','pipeline_1','serial_2','pipeline_2'):
    print('ROPE_MULTICORE_COMPILE variant='+variant,flush=True)
    subprocess.run(['timeout','-k','5s','180s',sys.executable,str(root/'rope.py'),
        '--build-only','--work',str(root),'--variant',variant],env=env,check=True)
PY
uname -a > "$rp_root/environment.txt"
for rp_variant in serial_1 pipeline_1 serial_2 pipeline_2; do
  rp_stage="correctness-$rp_variant"
  echo "ROPE_MULTICORE_BEGIN $rp_stage"
  timeout -k 3s 40s python3 "$rp_root/rope.py" --run-only --allow-pcie \
    --work "$rp_root" --variant "$rp_variant" --output "$rp_root/run-$rp_variant"
  rp_stage="timing-$rp_variant"
  echo "ROPE_MULTICORE_BEGIN $rp_stage"
  timeout -k 3s 40s python3 "$rp_root/rope.py" --run-only --allow-pcie \
    --work "$rp_root" --variant "$rp_variant" --output "$rp_root/timing-$rp_variant" \
    --correctness "$rp_root/run-$rp_variant"
done
rp_stage=summary
python3 - "$rp_root" <<'PYEND'
import importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('rp_summary',root/'rope.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.summarize_suite(root)
PYEND
echo "ROPE_MULTICORE_SUITE_PASSED original_full=still_failed RESULT_ROOT=$rp_root"
