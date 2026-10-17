#!/usr/bin/env bash
# One sequential batch: FP16 Matmul, original plus serial/pipeline at one/two cores.
set -euo pipefail
if [ "$#" != 5 ] || [ "${1:-}" != --allow-pcie ] || [[ ! "${2:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Usage: bash run_matmul_multicore_board.sh --allow-pcie REVISION NEW_OUTPUT PASSED_ADD_DIR NUMPY_PYTHON' >&2
  exit 2
fi
mm_revision=$2
mm_root=$3
mm_previous=$4
mm_python=$5
test -x "$mm_python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mm_stage=prepare
trap 'mm_rc=$?; echo "MATMUL_MULTICORE_STOP stage=$mm_stage exit=$mm_rc RESULT_ROOT=$mm_root" >&2; exit "$mm_rc"' ERR
mkdir -- "$mm_root"
mm_root=$(cd -- "$mm_root" && pwd -P)
echo "MATMUL_MULTICORE_RESULT_ROOT=$mm_root"
timeout -k 2s 10s taskset -c 0,1 nice -n 10 "$mm_python" -c 'import sys,numpy; print("REFERENCE_PYTHON="+sys.executable+" NUMPY="+numpy.__version__)' > "$mm_root/python.txt"
cat "$mm_root/python.txt"
timeout -k 5s 900s python3 - "$mm_revision" "$mm_root" "$mm_previous" <<'PY'
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
download('tpu_demo/pipeline/matmul_multicore_pcie.py',root/'matmul.py','4b2b0d40144ec839061b37ae97c5eaa1885687bdd2ae8ffda9068e89a7593c55')
for name,pin in (('build.py',base.BUILDER_SHA),('run_add.py',base.SUPPORT_SHA)):
    check(old/name,pin);shutil.copyfile(old/name,root/name);check(root/name,pin)
download('research/bm1690-pipelines/handoff/matmul-multicore-sources.json',root/'matmul-multicore.json','d48950d2dedc7624112d19c630d8e02322e80f716b2d6785eb4fde7993c6f39b')
cross=[p for p in manifest['sdk_inputs'] if pathlib.Path(p).name=='riscv64-unknown-linux-gnu-gcc']
if len(cross)!=1:raise SystemExit('Missing compiler identity')
env=dict(os.environ,PPL_PROJECT_ROOT=manifest['runtime_identity'][0],
    TILELANG_TPU_PCIE_RUNTIME_PATH=manifest['runtime_identity'][1],TILELANG_TPU_PCIE_CROSS_GCC=cross[0])
for variant in base.VARIANTS:
    print('MATMUL_MULTICORE_COMPILE variant='+variant,flush=True)
    subprocess.run(['timeout','-k','5s','180s',sys.executable,str(root/'matmul.py'),
        '--build-only','--work',str(root),'--variant',variant],env=env,check=True)
PY
uname -a > "$mm_root/environment.txt"
for mm_variant in original serial_1 pipeline_1 serial_2 pipeline_2; do
  mm_stage="correctness-$mm_variant"
  echo "MATMUL_MULTICORE_BEGIN $mm_stage"
  timeout -k 3s 40s "$mm_python" "$mm_root/matmul.py" --run-only --allow-pcie \
    --work "$mm_root" --variant "$mm_variant" --output "$mm_root/run-$mm_variant"
  mm_stage="timing-$mm_variant"
  echo "MATMUL_MULTICORE_BEGIN $mm_stage"
  timeout -k 3s 40s "$mm_python" "$mm_root/matmul.py" --run-only --allow-pcie \
    --work "$mm_root" --variant "$mm_variant" --output "$mm_root/timing-$mm_variant" \
    --correctness "$mm_root/run-$mm_variant"
done
mm_stage=summary
python3 - "$mm_root" <<'PY'
import importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('mm_summary',root/'matmul.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.summarize_suite(root)
PY
echo "MATMUL_MULTICORE_SUITE_PASSED RESULT_ROOT=$mm_root"
