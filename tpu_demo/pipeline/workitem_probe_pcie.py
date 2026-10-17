#!/usr/bin/env python3
"""Build or run one fixed two-workitem ABI/ownership check, without dependencies."""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys

BUNDLE_SHA = "cb193033e73d7a77eb6574ee1a1844a9ebd523f666b265b7281c0dacde74b4c8"
BUILDER_SHA = "595fca52044098d9ea319f17b5ed86362d6700ce6e65c419e3876ece63e18ff2"
SUPPORT_SHA = "b46d84d8a9c8ba789de383991d8db542f1d1c7e2a3fc7b6d1f987bed78ab51a1"
TARGET = dict(chip="bm1690", programming_model="tpukernel", launch_cores=2, tasks=11,
              width=32, input_shape=[11,32], output_shape=[2,11,32], dtype="float16")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def verified_module(path, expected, name):
    checked(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assets(work):
    checked(work / "probe.json", BUNDLE_SHA)
    bundle = json.loads((work / "probe.json").read_text())
    validate_bundle(bundle)
    builder = verified_module(work / "build.py", BUILDER_SHA, "probe_legacy_builder")
    support = verified_module(work / "run_add.py", SUPPORT_SHA, "probe_runtime_support")
    return bundle, builder, support


def validate_bundle(bundle):
    if bundle.get("schema") != "bm1690-workitem-probe-v1" or bundle.get("target") != TARGET:
        raise ValueError("Expected the fixed two-workitem FP16 ownership probe")
    names = {"kernel.c", "kernel.cpp", "kernel.h", "main.cpp"}
    if set(bundle["sources"]) != names or set(bundle["sha256"]) != names:
        raise ValueError("Incomplete exported source bundle")
    for name, source in bundle["sources"].items():
        if hashlib.sha256(source.encode()).hexdigest() != bundle["sha256"][name]:
            raise ValueError("Source checksum mismatch")
    device, host = bundle["sources"]["kernel.c"], bundle["sources"]["kernel.cpp"]
    for token in ("tpu_workitem_num() == 2", "tpu_workitem_index()", "task < 11", "tpu_initialize();", "tpu_poll();"):
        if token not in device:
            raise ValueError("Missing device ownership/synchronization contract")
    for token in ("constexpr int core_num = 2;", "int group_num = 1;", "int block_num = core_num;",
                  "apis[core_num]", "&apis, sizeof(apis)", "tpuRtStreamSynchronize(stream)"):
        if token not in host:
            raise ValueError("Missing host workitem argument-array contract")


def vectors():
    source = [i % 17 + 1 for i in range(11 * 32)]
    expected = [source[t*32+i] if t % 2 == c else 0 for c in range(2) for t in range(11) for i in range(32)]
    return struct.pack('<352e', *source), struct.pack('<704e', *expected)


def sdk_inputs(layout):
    return (layout.firmware_archive, layout.ppl_helper_source, layout.kernel_include / "tpu_kernel.h",
            layout.runtime_include / "tpuv7_rt.h", layout.board_lib / "libtpuv7_rt.so", layout.cross_gcc)


def commands(builder, layout, build):
    # The shared profile verifies these variables; keep its existing contract.
    env = {"TILELANG_TPU_PCIE_RUNTIME_PATH": str(layout.board_lib),
           "TILELANG_TPU_PCIE_CROSS_GCC": str(layout.cross_gcc)}
    previous = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        return [dict(label=label, command=cmd, log=str(build / f'{i}.log'))
                for i, (label, cmd) in enumerate(builder.legacy_pcie_commands(layout, build))]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compile_only(work):
    bundle, builder, support = assets(work)
    layout = builder.resolve_legacy_pcie(builder._explicit_path(os.environ, "PPL_PROJECT_ROOT"), "bm1690")
    build = work / "build"
    build.mkdir()
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min([4*1024**3] + [v for v in (soft,hard) if v != resource.RLIM_INFINITY])
    resource.setrlimit(resource.RLIMIT_AS, (cap,cap))
    report = dict(schema="bm1690-workitem-build-v1", status="failed", bundle_sha256=BUNDLE_SHA,
                  builder_sha256=BUILDER_SHA, support_sha256=SUPPORT_SHA, launch_cores=2,
                  probe_builder_sha256=digest(Path(__file__)), runtime_identity=layout.runtime_identity_for("pcie"),
                  sdk_inputs={str(p):digest(p) for p in sdk_inputs(layout)},
                  board_runtime_loaded=False, kernel_launches=0, completed_commands=[])
    environment = dict(os.environ)
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH",
                "GCC_EXEC_PREFIX", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    try:
        machine = subprocess.check_output([str(layout.cross_gcc), '-dumpmachine'], env=environment, timeout=10, text=True).strip()
        if machine != 'riscv64-unknown-linux-gnu':
            raise ValueError("Wrong device compiler target")
        for name, source in bundle['sources'].items():
            (build / name).write_text(source)
        for row in commands(builder, layout, build):
            print('WORKITEM_BUILD ' + row['label'], flush=True)
            report['active_command'] = row
            support.save(build / 'result.json', report)
            with Path(row['log']).open('x') as log:
                subprocess.run(['timeout','-k','2s','60s',*row['command']], env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            report['completed_commands'].append(row)
            report.pop('active_command')
        report.update(status='compile_only_passed', artifacts={n:digest(build/n) for n in ('main.so','libkernel.so')})
    except Exception as error:
        report['error'] = str(error)
        log = report.get('active_command',{}).get('log')
        if log and Path(log).exists():
            print(Path(log).read_text(errors='replace')[-8000:], flush=True)
        raise
    finally:
        support.save(build / 'result.json', report)
    print('WORKITEM_BUILD_ONLY_OK launch_cores=2 kernel_launches=0', flush=True)


def validate_build(work, bundle, builder):
    build = work / 'build'
    report = json.loads((build / 'result.json').read_text())
    required = dict(schema='bm1690-workitem-build-v1',status='compile_only_passed',bundle_sha256=BUNDLE_SHA,
                    builder_sha256=BUILDER_SHA,support_sha256=SUPPORT_SHA,launch_cores=2,
                    probe_builder_sha256=digest(Path(__file__)),board_runtime_loaded=False,kernel_launches=0)
    if any(report.get(k) != v for k,v in required.items()) or 'active_command' in report:
        raise ValueError('Missing or mismatched two-core build receipt')
    if set(report.get('artifacts',{})) != {'main.so','libkernel.so'}:
        raise ValueError('Incomplete device/host artifacts')
    sdk, board, backend = map(Path, report['runtime_identity'])
    cross = [Path(p) for p in report['sdk_inputs'] if Path(p).name == 'riscv64-unknown-linux-gnu-gcc']
    if len(cross) != 1:
        raise ValueError('Missing device compiler identity')
    environment = dict(TILELANG_TPU_PCIE_RUNTIME_PATH=str(board), TILELANG_TPU_PCIE_CROSS_GCC=str(cross[0]))
    layout = builder.resolve_legacy_pcie(sdk, 'bm1690', environment=environment)
    if (list(layout.runtime_identity_for('pcie',environment)) != report['runtime_identity'] or
            set(report['sdk_inputs']) != {str(p) for p in sdk_inputs(layout)} or
            report['completed_commands'] != commands(builder,layout,build)):
        raise ValueError('SDK identity or absolute build location changed')
    for path in sdk_inputs(layout):
        checked(path, report['sdk_inputs'][str(path)])
    for name, sha in bundle['sha256'].items():
        checked(build/name,sha)
    for name,machine in (('main.so',62),('libkernel.so',243)):
        checked(build/name,report['artifacts'][name])
        header=(build/name).read_bytes()[:20]
        if header[:6] != b'\x7fELF\x02\x01' or struct.unpack('<HH',header[16:20]) != (3,machine):
            raise ValueError('Unexpected ELF architecture')
    return report, (board/'libtpuv7_rt.so').resolve()


def worker(args, bundle, builder, support):
    support.arm_parent_death(args.worker_parent)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    result = dict(status='running',operation='workitems',launch_cores=2,tasks=11,requested_kernel_calls=1,
                  dispatch_attempted=False,hardware_overlap_verified=False,device_performance_measured=False)
    def checkpoint(stage):
        result['last_stage']=stage
        support.save(args.output/'worker.json',result)
        print('STAGE '+stage,flush=True)
    try:
        checkpoint('validate_build')
        for key in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','TPU_RT_CORE_NUM'):
            if os.environ.get(key):
                raise ValueError('Fresh board worker requires '+key+' unset')
        if support.loaded_vendor_libraries():
            raise ValueError('Vendor runtime already loaded')
        receipt,runtime_path=validate_build(args.work,bundle,builder)
        result.update(bundle_sha256=BUNDLE_SHA,build_manifest_sha256=digest(args.work/'build/result.json'),
                      runtime_library=str(runtime_path),runtime_sha256=digest(runtime_path),artifacts=receipt['artifacts'])
        for key in tuple(os.environ):
            if key.startswith('TILELANG_TPU_PROFILE_') or key in ('BMLIB_ENABLE_ALL_PROFILE','TILELANG_TPU_ALLOW_PCIE_PROFILE','FILE_DUMP_CMD'):
                os.environ.pop(key)
        os.environ.update(TILELANG_TPU_ALLOW_PCIE_LOAD='1',TILELANG_TPU_DEVICE_ID='0',TILELANG_TPU_BENCHMARK_RUNS='0')
        checkpoint('load_verified_board_runtime')
        runtime=ctypes.CDLL(str(runtime_path),mode=os.RTLD_NOW|os.RTLD_LOCAL)
        globals()['_vendor_handles']=(runtime,)
        if support.loaded_vendor_libraries() != {str(runtime_path)}:
            raise ValueError('Unexpected runtime mapping')
        result['mapping']=support.verify_device(runtime,0,'0000:01:00.0',checkpoint)
        checkpoint('load_verified_host_module')
        for name,sha in receipt['artifacts'].items():
            checked(args.work/'build'/name,sha)
        library=ctypes.CDLL(str(args.work/'build/main.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL)
        globals()['_vendor_handles']=(runtime,library)
        if support.loaded_vendor_libraries() != {str(runtime_path)}:
            raise ValueError('Host module resolved a different runtime')
        bind=library.tilelang_tpu_bind_device
        bind.argtypes,bind.restype=[ctypes.c_int],ctypes.c_int
        if bind(0) != 0:
            raise RuntimeError('Device binding failed')
        source,expected=vectors()
        result['dispatch_attempted']=True
        checkpoint('two_workitem_launch')
        actual=support._call_host_buffers(library,(source,),len(expected))
        checkpoint('compare_exact_ownership')
        if actual != expected:
            raise ValueError('Ownership/output mismatch (includes missing, swapped or duplicated workitems)')
        (args.output/'output.f16').write_bytes(actual)
        result.update(status='passed',ownership_counts=[6,5],checked_elements=704,exact_ownership=True,
                      input_sha256=hashlib.sha256(source).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest())
        checkpoint('completed')
        print('WORKITEM_CORRECTNESS='+json.dumps(result,sort_keys=True),flush=True)
        return 0
    except Exception as error:
        result.update(status='failed',error=str(error))
        support.save(args.output/'worker.json',result)
        print('WORKITEM_STOP '+str(error),flush=True)
        return 1


def child(args):
    return [sys.executable,str(Path(__file__).resolve()),'--run-only','--allow-pcie',
            '--work',str(args.work),'--output',str(args.output)]


def guard(args, support):
    if os.getpgrp() != os.getpid():
        raise RuntimeError('Guard requires private process group')
    def stop(*_):
        os.killpg(os.getpgrp(),signal.SIGKILL)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,stop)
    support.arm_parent_death(args.guard_parent,signal.SIGTERM)
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM})
    if os.getppid() != args.guard_parent:
        stop()
    process=subprocess.Popen(child(args)+['--worker-parent',str(os.getpid())],pass_fds=(args.lock_fd,))
    rc=process.wait()
    print('WORKER_EXIT='+str(rc),flush=True)
    return 128-rc if rc < 0 else rc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--build-only',action='store_true')
    action.add_argument('--run-only',action='store_true')
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-pcie',action='store_true')
    parser.add_argument('--guard-parent',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--worker-parent',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.work=args.work.resolve()
    if args.build_only:
        compile_only(args.work)
        return 0
    if not args.allow_pcie or not args.output:
        parser.error('Run requires --allow-pcie and a new --output directory')
    args.output=args.output.resolve()
    bundle,builder,support=assets(args.work)
    if args.worker_parent:
        return worker(args,bundle,builder,support)
    if args.guard_parent:
        return guard(args,support)
    with open(f'/tmp/tilelang-bm1690-pcie-{os.getuid()}-device-0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.output.mkdir()
        users=subprocess.run(['sudo','-n','timeout','-k','2s','5s','fuser','-v',
                              '/dev/sg-host-drv-0','/dev/sg-host-drv-1'],capture_output=True,text=True,timeout=10)
        (args.output/'users.log').write_text(users.stdout+users.stderr)
        if users.returncode != 1 or users.stdout or users.stderr:
            raise RuntimeError('Device occupied or occupation check failed; no launch')
        command=child(args)+['--guard-parent',str(os.getpid()),'--lock-fd',str(lock.fileno())]
        return support.supervise(command,args.output,timeout_s=30,lock_fd=lock.fileno())


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        raise SystemExit('WORKITEM_STOP '+str(error)) from error
