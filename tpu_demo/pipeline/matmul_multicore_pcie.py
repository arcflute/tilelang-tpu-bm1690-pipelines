#!/usr/bin/env python3
"""Build and compare matched single/dual-core FP16 Matmul using the existing NumPy environment."""
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

BUNDLE_SHAS = {'matmul': 'd48950d2dedc7624112d19c630d8e02322e80f716b2d6785eb4fde7993c6f39b'}
INPUT_SHAS = {'matmul': '03c8ccda05f09345f20c7df68b2c1f58764bbb84071b747bb7567fd1bc7de052'}
BUILDER_SHA = "595fca52044098d9ea319f17b5ed86362d6700ce6e65c419e3876ece63e18ff2"
SUPPORT_SHA = "b46d84d8a9c8ba789de383991d8db542f1d1c7e2a3fc7b6d1f987bed78ab51a1"
VARIANTS = ('original', 'serial_1', 'pipeline_1', 'serial_2', 'pipeline_2')
COUNT = 1024 * 1024


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
    bundle = json.loads((work / "matmul-multicore.json").read_text())
    validate_bundle(bundle)
    checked(work / "matmul-multicore.json", BUNDLE_SHAS[bundle["operation"]])
    builder = verified_module(work / "build.py", BUILDER_SHA, "mm_mc_legacy_builder")
    support = verified_module(work / "run_add.py", SUPPORT_SHA, "mm_mc_runtime_support")
    return bundle, builder, support


def contract(variant):
    return dict(launch_cores=2 if variant.endswith('_2') else 1,
                num_stages=2 if variant.startswith('pipeline') else 0,
                tiling=[16,16,16] if variant == 'original' else [32,32,32],
                matrix_dimensions=[1024,1024,1024],layout='row-major',accumulator_dtype='float32')


def validate_bundle(bundle):
    if bundle.get('operation') not in BUNDLE_SHAS:
        raise ValueError('Expected matmul')
    required = dict(schema='bm1690-matmul-multicore-v1', chip='bm1690', programming_model='tpukernel',
                    operation='matmul', dtype='float16', shape=[1024, 1024],
                    matrix_dimensions=[1024,1024,1024], layout='row-major', accumulator_dtype='float32')
    if any(bundle.get(k) != v for k, v in required.items()) or set(bundle['variants']) != set(VARIANTS):
        raise ValueError('Expected the five matched matmul variants')
    for variant in VARIANTS:
        item, expected = bundle['variants'][variant], contract(variant)
        if any(item.get(k) != expected[k] for k in ('launch_cores','num_stages','tiling')):
            raise ValueError('Mismatched core count, precision or tiling')
        cores = expected['launch_cores']
        if item.get('tasks_per_core') != ((1024 // expected['tiling'][0])**2 // cores):
            raise ValueError('Wrong output task partition')
        names = {'kernel.c', 'kernel.cpp', 'kernel.h', 'main.cpp'}
        if set(item['sources']) != names or set(item['sha256']) != names:
            raise ValueError('Incomplete exported source bundle')
        for name, source in item['sources'].items():
            if hashlib.sha256(source.encode()).hexdigest() != item['sha256'][name]:
                raise ValueError('Source checksum mismatch')
        device, host = item['sources']['kernel.c'], item['sources']['kernel.cpp']
        required_tokens = ['tpu_initialize();', 'tpu_poll();']
        if cores == 2:
            required_tokens += ['tpu_workitem_num() == 2', 'tpu_workitem_index()']
            if item['mapping'] != dict(mode='strided-kernel-grid', cores=2, output_tasks=1024,
                                      assignment='task % cores', reduction_split_across_cores=False):
                raise ValueError('Wrong workitem mapping')
        if any(t not in device for t in required_tokens):
            raise ValueError('Missing device mapping/synchronization contract')
        if any(t not in host for t in (f'constexpr int core_num = {cores};', 'int group_num = 1;',
               'int block_num = core_num;', 'apis[core_num]', '&apis, sizeof(apis)',
               'tpuRtStreamSynchronize(stream)')):
            raise ValueError('Missing host argument-array contract')
        if expected['num_stages'] and (not item['pipeline_reports'] or 'tpu_parallel_start();' not in device):
            raise ValueError('Missing generated pipeline evidence')


def validate_correctness(path, expected):
    receipt = json.loads((path/'result.json').read_text())
    numeric = receipt.get('numeric') or {}
    if (receipt.get('status') != 'passed' or receipt.get('cleanup_complete') is not True or
            numeric.get('status') != 'passed' or numeric.get('measurement') != 'correctness' or
            not numeric.get('reference', {}).get('passed')):
        raise ValueError('Missing successful correctness receipt')
    for key in ('bundle_sha256', 'build_manifest_sha256', 'variant', 'launch_cores', 'num_stages',
                'operation', 'tiling', 'matrix_dimensions', 'layout', 'accumulator_dtype', 'shape', 'dtype', 'input_sha256', 'runtime_sha256', 'artifacts', 'mapping'):
        if numeric.get(key) != expected.get(key):
            raise ValueError('Correctness receipt identity differs: '+key)
    checked(path/'output.f16', numeric['output_sha256'])
    return (path/'output.f16').read_bytes()


def serial_output(work, bundle, builder, expected):
    """Same-tile baseline: K-block reassociation may differ from block-16 original."""
    receipt, runtime = validate_build(work, bundle, builder, 'serial_1')
    identity = dict(expected, variant='serial_1', **contract('serial_1'))
    identity.update(build_manifest_sha256=digest(work/'build/serial_1/result.json'),
                    artifacts=receipt['artifacts'], runtime_sha256=digest(runtime))
    return validate_correctness(work/'run-serial_1', identity)


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


def compile_only(work, variant):
    bundle, builder, support = assets(work)
    layout = builder.resolve_legacy_pcie(builder._explicit_path(os.environ, "PPL_PROJECT_ROOT"), "bm1690")
    build = work / "build" / variant
    build.mkdir(parents=True)
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min([4*1024**3] + [v for v in (soft,hard) if v != resource.RLIM_INFINITY])
    resource.setrlimit(resource.RLIMIT_AS, (cap,cap))
    report = dict(schema="bm1690-matmul-multicore-build-v1", status="failed", bundle_sha256=BUNDLE_SHAS[bundle['operation']],operation=bundle['operation'],
                  builder_sha256=BUILDER_SHA, support_sha256=SUPPORT_SHA, variant=variant, **contract(variant),
                  runner_sha256=digest(Path(__file__)), runtime_identity=layout.runtime_identity_for("pcie"),
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
        for name, source in bundle['variants'][variant]['sources'].items():
            (build / name).write_text(source)
        for row in commands(builder, layout, build):
            print('MATMUL_MULTICORE_BUILD '+variant+': '+row['label'], flush=True)
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
    print(f'MATMUL_MULTICORE_BUILD_ONLY_OK variant={variant} kernel_launches=0', flush=True)


def validate_build(work, bundle, builder, variant):
    build = work / 'build' / variant
    report = json.loads((build / 'result.json').read_text())
    required = dict(schema='bm1690-matmul-multicore-build-v1',status='compile_only_passed',bundle_sha256=BUNDLE_SHAS[bundle['operation']],operation=bundle['operation'],
                    builder_sha256=BUILDER_SHA,support_sha256=SUPPORT_SHA,variant=variant,**contract(variant),
                    runner_sha256=digest(Path(__file__)),board_runtime_loaded=False,kernel_launches=0)
    if any(report.get(k) != v for k,v in required.items()) or 'active_command' in report:
        raise ValueError('Missing or mismatched matmul build receipt')
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
    for name, sha in bundle['variants'][variant]['sha256'].items():
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
    result = dict(status='running',operation=bundle['operation'],variant=args.variant,**contract(args.variant),
                  shape=[1024,1024],dtype='float16',measurement='timing' if args.correctness else 'correctness',
                  requested_kernel_calls=25 if args.correctness else 1,
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
        receipt,runtime_path=validate_build(args.work,bundle,builder,args.variant)
        result.update(bundle_sha256=BUNDLE_SHAS[bundle['operation']],operation=bundle['operation'],build_manifest_sha256=digest(args.work/'build'/args.variant/'result.json'),
                      runtime_library=str(runtime_path),runtime_sha256=digest(runtime_path),artifacts=receipt['artifacts'])
        for key in tuple(os.environ):
            if key.startswith('TILELANG_TPU_PROFILE_') or key in ('BMLIB_ENABLE_ALL_PROFILE','TILELANG_TPU_ALLOW_PCIE_PROFILE','FILE_DUMP_CMD'):
                os.environ.pop(key)
        os.environ.update(TILELANG_TPU_ALLOW_PCIE_LOAD='1',TILELANG_TPU_DEVICE_ID='0',TILELANG_TPU_BENCHMARK_RUNS='0')
        lhs, rhs, expected = support.test_vectors(COUNT, bundle['operation'])
        result['input_sha256'] = hashlib.sha256(lhs+rhs).hexdigest()
        import numpy as np
        result['reference_backend'] = dict(name='numpy.matmul', version=np.__version__,
                                           accumulation_dtype='float32', output_dtype='float16')
        # Mapping is verified against the live runtime below, before dispatch.
        checkpoint('load_verified_board_runtime')
        runtime=ctypes.CDLL(str(runtime_path),mode=os.RTLD_NOW|os.RTLD_LOCAL)
        globals()['_vendor_handles']=(runtime,)
        if support.loaded_vendor_libraries() != {str(runtime_path)}:
            raise ValueError('Unexpected runtime mapping')
        result['mapping']=support.verify_device(runtime,0,'0000:01:00.0',checkpoint)
        checkpoint('load_verified_host_module')
        for name,sha in receipt['artifacts'].items():
            checked(args.work/'build'/args.variant/name,sha)
        library=ctypes.CDLL(str(args.work/'build'/args.variant/'main.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL)
        globals()['_vendor_handles']=(runtime,library)
        if support.loaded_vendor_libraries() != {str(runtime_path)}:
            raise ValueError('Host module resolved a different runtime')
        bind=library.tilelang_tpu_bind_device
        bind.argtypes,bind.restype=[ctypes.c_int],ctypes.c_int
        if bind(0) != 0:
            raise RuntimeError('Device binding failed')
        baseline = serial_output(args.work, bundle, builder, result) if args.variant not in ('original','serial_1') else None
        prior = validate_correctness(args.correctness, result) if args.correctness else None
        timing = dict(warmups=5, samples=20) if args.correctness else None
        result['dispatch_attempted'] = True
        checkpoint('matmul_launch_timed' if timing else 'matmul_launch_once')
        actual = support.call_host(library, lhs, rhs, timing=timing)
        checkpoint('compare_reference')
        result['reference'] = support.check_output(actual, expected, COUNT, bundle['operation'])
        if ((baseline is not None and actual != baseline) or
                (prior is not None and actual != prior)):
            raise ValueError('Output differs from same-tile serial or saved correctness')
        (args.output/'output.f16').write_bytes(actual)
        result.update(status='passed',output_sha256=hashlib.sha256(actual).hexdigest(),
                      serial_variant_bitwise_equal=actual == baseline if baseline is not None else None)
        if timing:
            result['sync_latency'] = timing
            print('MATMUL_MULTICORE_SYNC_LATENCY='+json.dumps(dict(operation=bundle['operation'],variant=args.variant,**timing),sort_keys=True),flush=True)
        checkpoint('completed')
        print('MATMUL_MULTICORE_CORRECTNESS='+json.dumps(result,sort_keys=True),flush=True)

        return 0
    except Exception as error:
        result.update(status='failed',error=str(error))
        support.save(args.output/'worker.json',result)
        print('MATMUL_MULTICORE_STOP '+str(error),flush=True)
        return 1


def child(args):
    command = [sys.executable,str(Path(__file__).resolve()),'--run-only','--allow-pcie',
               '--work',str(args.work),'--output',str(args.output),'--variant',args.variant]
    if args.correctness:
        command += ['--correctness',str(args.correctness)]
    return command


def summarize_suite(work):
    """Keep pipeline comparisons within one core count; scaling within one variant."""
    bundle, builder, support = assets(work)
    rows = {}
    for variant in VARIANTS:
        build, runtime = validate_build(work, bundle, builder, variant)
        timing_dir = work / ('timing-' + variant)
        receipt = json.loads((timing_dir / 'result.json').read_text())
        numeric = receipt.get('numeric') or {}
        expected = dict(variant=variant, **contract(variant), shape=[1024, 1024], dtype='float16',
                        measurement='timing', requested_kernel_calls=25, bundle_sha256=BUNDLE_SHAS[bundle['operation']],operation=bundle['operation'],
                        build_manifest_sha256=digest(work/'build'/variant/'result.json'),
                        artifacts=build['artifacts'], runtime_sha256=digest(runtime),
                        input_sha256=INPUT_SHAS[bundle['operation']])
        if (receipt.get('status') != 'passed' or receipt.get('cleanup_complete') is not True or
                numeric.get('status') != 'passed' or any(numeric.get(k) != v for k,v in expected.items()) or
                not numeric.get('reference', {}).get('passed')):
            raise ValueError('Incomplete matched timing receipt: '+variant)
        mapping = numeric.get('mapping') or {}
        if (mapping.get('device_id'), mapping.get('pci_function'), mapping.get('character_device')) != (
                0, '0000:01:00.0', '/dev/sg-host-drv-0'):
            raise ValueError('Wrong board mapping')
        prior = validate_correctness(work/('run-'+variant), numeric)
        checked(timing_dir/'output.f16', numeric['output_sha256'])
        if (timing_dir/'output.f16').read_bytes() != prior:
            raise ValueError('Repeated output differs from correctness output')
        if variant not in ('original','serial_1') and prior != serial_output(work,bundle,builder,numeric):
            raise ValueError('Cross-variant output mismatch')
        timing = numeric['sync_latency']
        if timing.get('warmups') != 5 or timing.get('samples') != 20 or len(timing['raw_us']) != 20:
            raise ValueError('Incorrect timing sample count')
        recomputed = support.summarize_samples(timing['raw_us'])
        if any(timing.get(k) != v for k,v in recomputed.items()):
            raise ValueError('Timing statistics / boundary do not match samples')
        rows[variant] = dict(**contract(variant), sync_latency=timing, reference=numeric['reference'],
                             input_sha256=numeric['input_sha256'], output_sha256=numeric['output_sha256'])
    comparisons = {}
    for label, before, after in (
            ('pipeline_at_1_core', 'serial_1', 'pipeline_1'),
            ('pipeline_at_2_cores', 'serial_2', 'pipeline_2'),
            ('serial_1_to_2_cores', 'serial_1', 'serial_2'),
            ('pipeline_1_to_2_cores', 'pipeline_1', 'pipeline_2')):
        a, b = (rows[n]['sync_latency']['median_us'] for n in (before, after))
        comparisons[label] = dict(before=before, after=after, median_ratio=a/b,
                                  median_reduction_percent=100*(a-b)/a)
    result = dict(status='passed', operation=bundle['operation'], shape=[1024,1024], dtype='float16',
                  matrix_dimensions=[1024,1024,1024],layout='row-major',accumulator_dtype='float32',
                  variants=rows, comparisons=comparisons, timing_workers=5, correctness_workers=5,
                  raw_samples=100, repeatability_established=False, hardware_overlap_verified=False,
                  pure_device_time=False)
    support.save(work/'summary.json', result)
    print('MATMUL_MULTICORE_SUMMARY='+json.dumps(result,sort_keys=True),flush=True)
    return result


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
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--correctness',type=Path)
    parser.add_argument('--allow-pcie',action='store_true')
    parser.add_argument('--guard-parent',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--worker-parent',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.work=args.work.resolve()
    if args.build_only:
        compile_only(args.work,args.variant)
        return 0
    if not args.allow_pcie or not args.output:
        parser.error('Run requires --allow-pcie and a new --output directory')
    args.output=args.output.resolve()
    if args.correctness:
        args.correctness=args.correctness.resolve(strict=True)
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
        raise SystemExit('MATMUL_MULTICORE_STOP '+str(error)) from error
