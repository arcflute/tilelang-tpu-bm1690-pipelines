"""Export matched one/two-workitem FP16 Matmul sources and verify each in CModel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

VARIANTS = ('original', 'serial_1', 'pipeline_1', 'serial_2', 'pipeline_2')


def export():
    operation = "matmul"
    import tilelang
    from tilelang import tvm
    from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper
    from tilelang.jit.adapter.legacy_pcie import SOURCE_NAMES
    from tpu_demo.matmul.matmul import build_matmul
    from tpu_demo.pipeline.workitems import map_workitems, MAPPING_ATTR
    from tpu_demo.pipeline.run import source_identity
    target = tvm.target.Target('tpu -mcpu=bm1690 -tpu-programming-model=tpukernel')
    bundle = dict(schema='bm1690-matmul-multicore-v1', shape=[1024, 1024], matrix_dimensions=[1024, 1024, 1024],
                  layout='row-major', accumulator_dtype='float32', dtype='float16',
                  operation=operation, chip='bm1690', programming_model='tpukernel',
                  generator=source_identity(), variants={},
                  evidence=dict(board_executed=False, hardware_overlap_verified=False))
    for name in VARIANTS:
        cores = 2 if name.endswith('_2') else 1
        stages = 2 if name.startswith('pipeline') else 0
        block = 16 if name == 'original' else 32
        function = build_matmul(m=1024, n=1024, k=1024, block_m=block, block_n=block,
                                block_k=block, dtype='float16', num_stages=stages)
        if cores == 2:
            function = map_workitems(function, cores)
        artifact = tilelang.lower(function, target=target, runtime_mode='pcie')
        reports, lmem = [], []
        for mod in (artifact.host_mod, artifact.device_mod):
            for lowered in mod.functions.values():
                attrs = lowered.attrs
                if attrs and attrs.get('tilelang.tpu.pipeline_report') is not None:
                    reports.extend(json.loads(str(attrs['tilelang.tpu.pipeline_report'])))
                if attrs and attrs.get('tilelang.tpu.lmem.high_water_bytes') is not None:
                    lmem.append(int(attrs['tilelang.tpu.lmem.high_water_bytes']))
        if stages and (not reports or 'tpu_parallel_start();' not in artifact.kernel_source):
            raise AssertionError('Missing lowered schedule / parallel scope')
        with tempfile.TemporaryDirectory(prefix='bm1690-matmul-mc-') as directory:
            TLTPUSourceWrapper(tvm.IRModule({'main': function}), artifact.kernel_source, target,
                               output_indices=[2], output_dir=directory)
            sources = {n: (Path(directory) / n).read_text() for n in SOURCE_NAMES}
        bundle['variants'][name] = dict(launch_cores=cores, num_stages=stages,
            tiling=[block, block, block],
            tasks_per_core=(1024 // block)**2 // cores,
            mapping=json.loads(str(function.attrs[MAPPING_ATTR])) if cores == 2 else None,
            pipeline_reports=reports, local_memory_high_water_bytes=lmem, sources=sources,
            sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()})
    return bundle


def verify(path, variant):
    import torch
    from tilelang import tvm
    from tilelang.engine.tpu_config import TPUTargetSpec, TPURuntimeConfig
    from tilelang.jit.adapter.libgen import LibraryGenerator
    from tpu_demo.common import configure_runtime
    from tpu_demo.pipeline.run_add_pcie import test_vectors, call_host, check_output
    from tpu_demo.pipeline.matmul_multicore_pcie import validate_bundle
    os.sched_setaffinity(0, {int(c) for c in os.environ['BM1690_PIPELINE_CPUS'].split(',')})
    torch.set_num_threads(1)
    configure_runtime('cmodel', False, None, chip='bm1690')
    bundle = json.loads(path.read_text())
    validate_bundle(bundle)
    operation = bundle['operation']
    item = bundle['variants'][variant]
    target = tvm.target.Target('tpu -mcpu=bm1690 -tpu-programming-model=tpukernel')
    build = LibraryGenerator(target, tpu_target=TPUTargetSpec('bm1690', 'tpukernel'),
                             tpu_runtime=TPURuntimeConfig('cmodel'))
    for name, source in item['sources'].items():
        (Path(build.tpu_workspace_dir) / name).write_text(source)
    build.compile_lib(timeout=60)
    library = build.load_lib()
    lhs, rhs, expected = test_vectors(1024 * 1024, operation)
    actual = call_host(library, lhs, rhs)
    metrics = check_output(actual, expected, 1024 * 1024, operation)
    timing = dict(warmups=1, samples=1)
    repeated = call_host(library, lhs, rhs, timing=timing)
    if repeated != actual:
        raise AssertionError('Repeated-launch output differs')
    print('BM1690_PIPELINE_RESULT=' + json.dumps(dict(status='passed', operation=operation, variant=variant,
        bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), launch_cores=item['launch_cores'],
        input_sha256=hashlib.sha256(lhs+rhs).hexdigest(), output_sha256=hashlib.sha256(actual).hexdigest(),
        reference=metrics, timed_abi_checked=True, requested_kernel_calls=3,
        device_performance_measured=False, hardware_overlap_verified=False), sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--verify', type=Path)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    if args.output:
        with args.output.open('x') as out:
            out.write(json.dumps(export(), indent=2, sort_keys=True) + '\n')
    elif not args.variant:
        parser.error('--verify requires --variant')
    else:
        verify(args.verify, args.variant)
