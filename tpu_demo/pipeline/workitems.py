"""Independent output-task mapping for the BM1690 TPU-Kernel launch ABI."""

import json

from tvm import tir, arith
import tilelang.language as T

LAUNCH_ATTR = "tilelang.tpu.launch_cores"
MAPPING_ATTR = "tilelang.tpu.task_mapping"


def map_workitems(function, cores, *, tile_loop=None):
    """Partition output tiles, keeping all reductions local to one workitem.

    Flat output pipelines currently require equal task counts; reject tails
    instead of introducing a conditional DMA producer. Original kernel grids
    support uneven counts with a guarded last task. Launch count is part of
    the function, so compilation and host argument-array sizing agree.
    """
    if isinstance(cores,bool) or cores not in (1,2,4,8):
        raise ValueError("cores must be 1, 2, 4 or 8")
    root = function.body
    if not isinstance(root,tir.BlockRealize):
        raise ValueError("workitem mapping requires a root block")
    # The modulo gives the bounds prover a closed interval. The independent
    # coverage probe checks the runtime's actual index contract.
    core = T.tpu_workitem_index() % cores
    total = None
    if tile_loop:
        found = []
        def rewrite(node):
            nonlocal total
            if not isinstance(node,tir.For) or node.loop_var.name != tile_loop:
                return None
            found.append(node)
            if node.kind != tir.ForKind.SERIAL or not isinstance(node.extent,tir.IntImm) or int(node.min) != 0:
                raise ValueError("output tile loop must have a static zero-based serial extent")
            total = int(node.extent)
            if total % cores:
                raise ValueError("flat output pipelines currently require task count divisible by cores")
            count = total // cores
            stages = int(node.annotations.get("num_stages",0))
            if stages and count < stages:
                raise ValueError("each workitem needs at least num_stages output tasks")
            body = tir.stmt_functor.substitute(node.body,{node.loop_var:node.loop_var*cores+core})
            return tir.For(node.loop_var,0,count,node.kind,body,node.thread_binding,node.annotations)
        body = tir.stmt_functor.ir_transform(root.block.body,None,rewrite,["tir.For"])
        if len(found) != 1:
            raise ValueError("workitem mapping requires one named output tile loop")
        mode = "strided-flat-output-tiles"
    else:
        node = root.block.body
        grid = []
        while isinstance(node,tir.For):
            extent = arith.Analyzer().simplify(node.extent)
            if node.kind != tir.ForKind.SERIAL or not isinstance(extent,tir.IntImm) or int(extent) <= 0:
                raise ValueError("workitem mapping requires a positive static kernel grid")
            grid.append((node.loop_var,node.min,int(extent)))
            node = node.body
        if not grid or not isinstance(node,tir.BlockRealize) or not node.block.annotations.get("tilelang.is_cpu_kernel_frame"):
            raise ValueError("workitem mapping requires the original CPU-kernel grid")
        iteration = tir.Var("workitem_task","int32")
        global_task = iteration*cores+core
        total = 1
        mapping = {}
        for var,minimum,extent in reversed(grid):
            mapping[var] = minimum+(global_task//total)%extent
            total *= extent
        body = tir.stmt_functor.substitute(node,mapping)
        if total % cores:
            body = tir.IfThenElse(global_task < total,body,None)
        body = tir.For(iteration,0,(total+cores-1)//cores,tir.ForKind.SERIAL,body)
        mode = "strided-kernel-grid"
    # A launch-ABI mismatch must produce an observable unwritten output in the
    # NaN-initialized tests, rather than silently duplicating work.
    body = tir.IfThenElse(T.tpu_workitem_num() == cores,body,None)
    block = tir.Block(root.block.iter_vars,[],[],root.block.name_hint,body,
                      root.block.init,root.block.alloc_buffers,root.block.match_buffers,root.block.annotations)
    return function.with_body(tir.BlockRealize(root.iter_values,root.predicate,block)).with_attr(
        LAUNCH_ATTR,cores).with_attr(MAPPING_ATTR,json.dumps({
            "mode":mode,"cores":cores,"output_tasks":total,"assignment":"task % cores",
            "reduction_split_across_cores":False},sort_keys=True))


def build_workitem_probe(cores, tasks=11):
    """Every workitem writes its entire row; owned task slots carry a marker."""
    if cores not in (1,2,4,8):
        raise ValueError("cores must be 1, 2, 4 or 8")
    @T.prim_func
    def workitem_probe(source: T.Tensor((tasks,32),"float16"),
                       ownership: T.Tensor((cores,tasks,32),"float16")):
        with T.Kernel(1,is_cpu=True) as (_bx,):
            local = T.alloc_shared((1,32),"float16")
            if T.tpu_workitem_num() == cores:
                for task in T.serial(tasks):
                    T.ppl_fill(local,T.float16(0))
                    if task % cores == T.tpu_workitem_index():
                        T.ppl_copy(source[task,0],local)
                    T.ppl_copy(local,ownership[T.tpu_workitem_index()%cores,task,0])
    return workitem_probe.with_attr(LAUNCH_ATTR,cores)
