# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Conservative BM1690 software pipelines over typed TPU tensor operations.

Unlike the legacy GPU/PPL injector, this pass never creates descriptor aliases
or an extra tensor dimension. A pipeline consists of independent global loads,
an ordered local compute chain (including recurrences), and trailing stores.
Only the loads move across iterations. Independent Allocate-owned buffers hold
their versions; stores complete outside the parallel scope before reuse.

This is intentionally a restricted scheduling contract. Unknown effects,
conditional producers and unmodelled cross-iteration hazards are rejected.
"""

from dataclasses import dataclass, field
import hashlib
import json

from tvm import arith, tir, IRModule

PARALLEL_SCOPE = "tilelang.tpu.pipeline_parallel"
REPORT_ATTR = "tilelang.tpu.pipeline_report"

# Canonical tensor argument access, matching AddressAssign and TPU codegen.
# Reduction pads its source; exp mutates all four buffers, including coeff.
_EFFECTS = {
    "tl.tpu.copy": (1, 2), "tl.tpu.fill": (2,),
    "tl.tpu.add": (2, 1, 1), "tl.tpu.sub": (2, 1, 1),
    "tl.tpu.mul": (2, 1, 1), "tl.tpu.div": (2, 1, 1), "tl.tpu.max": (2, 1, 1),
    "tl.tpu.mul_scalar": (2, 1), "tl.tpu.add_scalar": (2, 1), "tl.tpu.rsqrt": (2, 1),
    "tl.tpu.reduce_sum": (3, 2, 3), "tl.tpu.reduce_max": (3, 2, 3),
    "tl.tpu.exp": (3, 3, 3, 3),
}


def _error(message):
    raise ValueError("BM1690 pipeline: " + message)


def _integer(value, description):
    value = arith.Analyzer().simplify(value)
    if not isinstance(value, tir.IntImm):
        _error(f"{description} must be a static integer")
    return int(value)


def _seq(statements):
    return statements[0] if len(statements) == 1 else tir.SeqStmt(statements) if statements else tir.Evaluate(0)


def _statements(statement):
    if isinstance(statement, tir.BlockRealize):
        block = statement.block
        if block.match_buffers or block.iter_vars or block.init is not None:
            _error("pipeline-body block must not own aliases, iterators or init")
        if not arith.Analyzer().can_prove(statement.predicate):
            _error("conditional pipeline-body block is unsupported")
        if block.alloc_buffers:
            # An opaque local compute macro (e.g. reduce) is one statement;
            # its private scratch ownership must survive schedule expansion.
            return [statement]
        return _statements(block.body)
    if isinstance(statement, tir.SeqStmt):
        result = []
        for child in statement.seq:
            result.extend(_statements(child))
        return result
    return [statement]


def _extern(statement):
    if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
        call = statement.value
        if getattr(call.op, "name", None) == "tir.call_extern" and call.args:
            return call
    return None


def _operand(call, position):
    region = call.args[position]
    if not isinstance(region, tir.Call) or getattr(region.op, "name", None) != "tl.region":
        _error("tensor operands must be canonical tl.region descriptors")
    if not isinstance(region.args[0], tir.BufferLoad):
        _error("region must identify a buffer")
    return region.args[0].buffer


@dataclass
class Effects:
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    buffers: dict = field(default_factory=dict)

    def add(self, buffer, mode):
        self.buffers[buffer.data] = buffer
        if mode & 1:
            self.reads.add(buffer.data)
        if mode & 2:
            self.writes.add(buffer.data)

    def merge(self, other):
        self.reads.update(other.reads)
        self.writes.update(other.writes)
        self.buffers.update(other.buffers)
        return self


def effects(statement):
    """Collect whole-buffer effects conservatively, including nested local loops."""
    result = Effects()
    def visit(node):
        if isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child)
        elif isinstance(node, tir.For):
            if node.kind not in (tir.ForKind.SERIAL, tir.ForKind.UNROLLED):
                _error("compute loops must be serial or unrolled")
            if "num_stages" in node.annotations:
                _error("nested pipelines are unsupported")
            visit(node.body)
        elif isinstance(node, tir.BlockRealize):
            if node.block.match_buffers or node.block.init is not None:
                _error("compute block aliases/init are unsupported")
            visit(node.block.body)
        elif isinstance(node, (tir.Allocate, tir.DeclBuffer)):
            visit(node.body)
        elif isinstance(node, tir.IfThenElse):
            # Outside-loop task guards and local branches conservatively
            # touch both arms. Conditional prefetch producers remain rejected
            # by _copy_kind/full-tile checks.
            visit(node.then_case)
            if node.else_case is not None:
                visit(node.else_case)
        elif isinstance(node, tir.Evaluate):
            call = _extern(node)
            if call is None:
                if isinstance(node.value, tir.IntImm) and int(node.value) == 0:
                    return
                _error("only typed TPU operations are schedulable")
            name = call.args[0].value
            modes = _EFFECTS.get(name)
            if name == "tl.tpu.gemm":
                if len(call.args) != 10 or not isinstance(call.args[9], tir.IntImm):
                    _error("GEMM requires its explicit constant accumulate operand")
                modes = (1, 1, 3 if int(call.args[9]) else 2)
            if modes is None:
                _error(f"unmodelled operation {name}")
            for index, mode in enumerate(modes, 1):
                result.add(_operand(call, index), mode)
        else:
            _error(f"unsupported statement {type(node).__name__}; make its effects explicit first")
    visit(statement)
    return result


def _copy_kind(statement):
    call = _extern(statement)
    if call is None or call.args[0].value != "tl.tpu.copy":
        return "compute"
    source, destination = _operand(call, 1), _operand(call, 2)
    if source.scope() == "global" and destination.scope() != "global":
        return "load"
    if source.scope() != "global" and destination.scope() == "global":
        return "store"
    return "compute"


def validate_parallel_scope(node):
    """Recheck effects after final lowering, including scopes supplied as IR."""
    if not isinstance(node.node, tir.IntImm) or int(node.node) != 0:
        _error("parallel scope requires the compiler's constant node marker")
    statements = _statements(node.body)
    producers, consumers = Effects(), Effects()
    compute_started = False
    for statement in statements:
        kind = _copy_kind(statement)
        if kind == "load" and not compute_started:
            producers.merge(effects(statement))
        else:
            compute_started = True
            current = effects(statement)
            if any(buffer.scope() == "global" for buffer in current.buffers.values()):
                _error("parallel compute must use local buffers, with stores outside the scope")
            consumers.merge(current)
    if not producers.writes or not consumers.buffers:
        _error("parallel scope needs independent prefetch and compute")
    if producers.writes & (consumers.reads | consumers.writes):
        _error("parallel prefetch overwrites a concurrent compute operand")


def validate_pipeline_storage(module):
    """Check assigned byte ranges as well as descriptor identity before C codegen."""
    for function in module.functions.values():
        if not isinstance(function,tir.PrimFunc):
            continue
        def check(node):
            if not isinstance(node,tir.AttrStmt) or node.attr_key != PARALLEL_SCOPE:
                return
            producers, consumers = Effects(), Effects()
            for statement in _statements(node.body):
                (producers if _copy_kind(statement) == "load" else consumers).merge(effects(statement))
            def interval(buffer):
                name = buffer.data.name
                address = function.attrs.get("tilelang.tpu.lmem.address."+name)
                size = function.attrs.get("tilelang.tpu.lmem.bytes."+name)
                if address is None or size is None:
                    _error(f"missing assigned storage metadata for {name}; rebuild the TPU compiler")
                return int(address),int(address)+int(size)
            for written in producers.writes:
                a = interval(producers.buffers[written])
                others = dict(consumers.buffers)
                others.update({var:producers.buffers[var] for var in producers.writes if var != written})
                for buffer in others.values():
                    b = interval(buffer)
                    if a[0] < b[1] and b[0] < a[1]:
                        _error(f"physical storage overlap between prefetch {written.name} and {buffer.data.name}")
        tir.stmt_functor.post_order_visit(function.body,check)


def _replace(statement, loop_var, iteration, buffers):
    statement = tir.stmt_functor.substitute(statement, {loop_var: iteration})
    buffers = dict(buffers)
    def fresh_private_allocations(node):
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                buffers[buffer.data] = tir.decl_buffer(
                    buffer.shape, buffer.dtype, name=buffer.name, scope=buffer.scope(),
                    elem_offset=buffer.elem_offset, strides=buffer.strides,
                    data_alignment=buffer.data_alignment, offset_factor=buffer.offset_factor)
    tir.stmt_functor.post_order_visit(statement, fresh_private_allocations)
    def rewrite(node):
        if isinstance(node, tir.BufferLoad) and node.buffer.data in buffers:
            return tir.BufferLoad(buffers[node.buffer.data], node.indices, node.span)
        if isinstance(node, tir.Block):
            def regions(items):
                return [tir.BufferRegion(buffers.get(item.buffer.data, item.buffer), item.region) for item in items]
            return tir.Block(node.iter_vars, regions(node.reads), regions(node.writes), node.name_hint,
                             node.body, node.init,
                             [buffers.get(buffer.data, buffer) for buffer in node.alloc_buffers], node.match_buffers,
                             node.annotations)
        return None
    return tir.stmt_functor.ir_transform(statement, None, rewrite, ["tir.BufferLoad", "tir.Block"])


def _lower_function(function):
    loops = []
    allocated = {}
    def find(node):
        if isinstance(node, tir.For) and "num_stages" in node.annotations:
            loops.append(node)
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                allocated[buffer.data] = buffer
    tir.stmt_functor.post_order_visit(function.body, find)
    if not loops:
        return function
    target = function.attrs["target"]
    if str(target.attrs["mcpu"]) != "bm1690" or str(target.attrs["tpu-programming-model"]) != "tpukernel":
        _error("currently requires bm1690 + tpukernel")

    def omit_pipeline(node):
        if isinstance(node, tir.For) and "num_stages" in node.annotations:
            return tir.Evaluate(0)
        return None
    outside = effects(tir.stmt_functor.ir_transform(function.body, omit_pipeline, None, ["tir.For"]))
    all_effects = Effects().merge(outside)
    for loop in loops:
        all_effects.merge(effects(loop.body))
    globally_written = {var for var in all_effects.writes if all_effects.buffers[var].scope() == "global"}
    additions = {}
    reports = []

    def lower_loop(loop):
        depth = _integer(loop.annotations["num_stages"], "num_stages")
        if depth not in (2, 3):
            _error("supported num_stages candidates are 2 and 3")
        delay = depth - 1
        minimum = _integer(loop.min, "loop minimum")
        count = _integer(loop.extent, "loop extent")
        if count <= delay:
            _error("loop extent must exceed prefetch distance to have a steady state")
        if loop.kind != tir.ForKind.SERIAL:
            _error("pipeline loop must be serial before scheduling")
        unsupported = set(map(str, loop.annotations)) - {"num_stages", "tl_pipeline_order", "tl_pipeline_stage", "tl_pipeline_fingerprint"}
        if unsupported:
            _error(f"unmodelled pipeline annotations: {sorted(unsupported)}")
        statements = _statements(loop.body)
        kinds = [_copy_kind(statement) for statement in statements]
        accesses = [effects(statement) for statement in statements]
        loads = [i for i, kind in enumerate(kinds) if kind == "load"]
        compute = [i for i, kind in enumerate(kinds) if kind == "compute"]
        stores = [i for i, kind in enumerate(kinds) if kind == "store"]
        if not loads or not compute:
            _error("requires global-to-local producers and local compute consumers")
        if stores and max(compute) > min(stores):
            _error("stores must trail the compute chain")
        inputs = []
        for index in loads:
            call = _extern(statements[index])
            source, buffer = _operand(call, 1), _operand(call, 2)
            if source.data in globally_written:
                _error("prefetched global inputs must be read-only across the function")
            if buffer.data not in allocated or buffer.scope() not in ("shared", "shared.dyn"):
                _error("prefetch destinations require enclosing shared-buffer allocations")
            if buffer.strides or _integer(buffer.elem_offset, "buffer offset"):
                _error("prefetch destinations require canonical contiguous descriptors")
            region = call.args[2]
            if len(region.args) - 2 != len(buffer.shape) or \
                    any(_integer(value, "destination offset") != 0 for value in region.args[0].indices) or \
                    any(not arith.Analyzer().can_prove_equal(a, b) for a, b in zip(region.args[2:], buffer.shape)):
                _error("each prefetch must define its entire local destination")
            if buffer.data in inputs or any(buffer.data in access.reads for access in accesses[:index]):
                _error("prefetched buffer has multiple producers or a read before its definition")
            if buffer.data in outside.reads:
                _error("prefetched buffer escapes the pipelined loops")
            for other, access in enumerate(accesses):
                if other != index and buffer.data in access.writes:
                    _error("prefetched buffer must remain read-only during compute/store")
            if not any(buffer.data in accesses[i].reads for i in compute):
                _error("prefetch has no compute consumer")
            inputs.append(buffer.data)
        for index in compute:
            if any(buffer.scope() == "global" for buffer in accesses[index].buffers.values()):
                _error("compute chain must use local buffers; global transfers must be explicit copies")

        # P6-style schedules are bound to this lowered body, not a fixed length
        # from a different kernel. Whole-buffer RAW/WAR/WAW guards are conservative.
        fingerprint = hashlib.sha256(loop.body.script().encode()).hexdigest()
        order = loop.annotations.get("tl_pipeline_order")
        stage = loop.annotations.get("tl_pipeline_stage")
        if (order is None) != (stage is None):
            _error("explicit order and stage must be supplied together")
        sequence = loads + compute + stores
        stages = [0 if kind == "load" else delay for kind in kinds]
        if order is not None:
            order, stage = [int(x) for x in order], [int(x) for x in stage]
            if len(order) != len(statements) or sorted(order) != list(range(len(statements))) or stage != stages:
                _error("explicit schedule must be a permutation with load stage 0 and compute/store at prefetch distance")
            expected_fingerprint = loop.annotations.get("tl_pipeline_fingerprint", "")
            if getattr(expected_fingerprint, "value", expected_fingerprint) != fingerprint:
                _error("explicit schedule fingerprint does not match the lowered statement list")
            sequence = sorted(range(len(statements)), key=lambda i: order[i])
            classes = [{"load": 0, "compute": 1, "store": 2}[kinds[i]] for i in sequence]
            if classes != sorted(classes):
                _error("explicit schedule must preserve producer/compute/store boundaries")
            for left in range(len(statements)):
                for right in range(left + 1, len(statements)):
                    a, b = accesses[left], accesses[right]
                    if (a.writes & (b.reads | b.writes) or a.reads & b.writes) and order[left] > order[right]:
                        _error(f"explicit schedule violates RAW/WAR/WAW dependency {left}->{right}")
            loads = [i for i in sequence if kinds[i] == "load"]
            compute = [i for i in sequence if kinds[i] == "compute"]
            stores = [i for i in sequence if kinds[i] == "store"]

        versions = {}
        for var in inputs:
            buffer = allocated[var]
            clones = additions.setdefault(var, [])
            while len(clones) < depth - 1:
                clones.append(tir.decl_buffer(buffer.shape, buffer.dtype,
                    name=f"{buffer.name}_pipeline_v{len(clones) + 1}", scope=buffer.scope(),
                    elem_offset=0, data_alignment=buffer.data_alignment,
                    offset_factor=buffer.offset_factor))
            versions[var] = [buffer, *clones[:depth - 1]]

        def group(indices, iteration, slot):
            return _seq([_replace(statements[i], loop.loop_var, iteration,
                                 {var: buffers[slot] for var, buffers in versions.items()}) for i in indices])

        prologue = [group(loads, tir.IntImm(loop.loop_var.dtype, minimum + i), i) for i in range(delay)]
        iteration = tir.Var(loop.loop_var.name + "_pipeline", loop.loop_var.dtype)
        branches = []
        for slot in range(depth):
            prefetch = group(loads, iteration + minimum + delay, (slot + delay) % depth)
            body = group(compute, iteration + minimum, slot)
            parallel = tir.AttrStmt(tir.IntImm("int32", 0), PARALLEL_SCOPE, 1, _seq([prefetch, body]))
            branches.append(_seq([parallel, group(stores, iteration + minimum, slot)]))
        selected = branches[-1]
        for slot in reversed(range(depth - 1)):
            selected = tir.IfThenElse(iteration % depth == slot, branches[slot], selected)
        steady = tir.For(iteration, 0, count - delay, tir.ForKind.SERIAL, selected)
        epilogue = [group(compute + stores, tir.IntImm(loop.loop_var.dtype, minimum + i), i % depth)
                    for i in range(count - delay, count)]
        reports.append({
            "loop": loop.loop_var.name, "extent": count, "num_stages": depth,
            "prefetch_distance": delay, "fingerprint": fingerprint,
            "explicit": order is not None, "stage": stages,
            "order": [sequence.index(i) for i in range(len(statements))],
            "statements": [{"index": i, "kind": kinds[i], "tir": statement.script(),
                            "reads": sorted(accesses[i].buffers[v].name for v in accesses[i].reads),
                            "writes": sorted(accesses[i].buffers[v].name for v in accesses[i].writes)}
                           for i, statement in enumerate(statements)],
            "buffer_versions": {allocated[v].name: [b.name for b in buffers] for v, buffers in versions.items()},
            "compute_state": "single-version ordered recurrence",
            "stores": "after parallel end; completed before output reuse",
            "cmodel_parallel_execution": False,
        })
        return _seq([*prologue, steady, *epilogue])

    def transform(node):
        if isinstance(node, tir.For) and "num_stages" in node.annotations:
            return lower_loop(node)
        if isinstance(node, tir.Block):
            buffers = list(node.alloc_buffers)
            for buffer in node.alloc_buffers:
                buffers.extend(additions.get(buffer.data, []))
            if len(buffers) != len(node.alloc_buffers):
                return tir.Block(node.iter_vars, node.reads, node.writes, node.name_hint, node.body,
                                 node.init, buffers, node.match_buffers, node.annotations)
        return None
    body = tir.stmt_functor.ir_transform(function.body, None, transform, ["tir.For", "tir.Block"])
    return function.with_body(body).with_attr(REPORT_ATTR, json.dumps(reports, sort_keys=True))


def lower_tpu_pipelines(module):
    """Run before allocation placement, while Buffer ownership is still explicit."""
    result = IRModule({gv: _lower_function(f) if isinstance(f, tir.PrimFunc) else f
                       for gv, f in module.functions.items()}, attrs=module.attrs)
    # Prologue/steady branches/epilogue can duplicate nested compute loops.
    # Renew their bound scalar Vars before any pass relies on SSA identity.
    if any(isinstance(f, tir.PrimFunc) and f.attrs and REPORT_ATTR in f.attrs
           for f in result.functions.values()):
        result = tir.transform.ConvertSSA()(result)
    return result


def bind_explicit_schedule(function, target, *, reverse_loads=False):
    """Generate a P6 candidate from this function's legalized statements.

    Returns legalized IR with a fingerprint-bound contract. The normal compiler
    still validates that contract; this helper does not bypass effect checks.
    Reversing independent loads is a candidate, not a performance assertion.
    """
    from tilelang.engine.phase import LowerAndLegalize
    legalized = LowerAndLegalize(IRModule({"main": function}), target)["main"]
    planned = _lower_function(legalized)
    if not planned.attrs or REPORT_ATTR not in planned.attrs:
        _error("explicit schedule needs at least one pipelined loop")
    reports = iter(json.loads(str(planned.attrs[REPORT_ATTR])))

    def annotate(node):
        if not isinstance(node, tir.For) or "num_stages" not in node.annotations:
            return None
        report = next(reports)
        order = report["order"]
        if reverse_loads:
            loads = [item["index"] for item in report["statements"] if item["kind"] == "load"]
            positions = sorted(order[index] for index in loads)
            for index, position in zip(reversed(loads), positions):
                order[index] = position
        annotations = dict(node.annotations)
        annotations.update(tl_pipeline_order=order, tl_pipeline_stage=report["stage"],
                           tl_pipeline_fingerprint=report["fingerprint"])
        return tir.For(node.loop_var, node.min, node.extent, node.kind, node.body,
                       node.thread_binding, annotations)

    return legalized.with_body(tir.stmt_functor.ir_transform(legalized.body, None, annotate, ["tir.For"]))
