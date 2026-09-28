"""Same-tile serial and pipelined variants; original demos remain available."""

import tilelang.language as T
from tpu_demo.common import validate_dimensions, validate_exact_tiling


def pipeline_kernel_tiles(function, num_stages=0):
    """Make the original serial CPU-kernel grid an explicit flat tile loop.

    Preserve grid order and kernel arithmetic. Buffers owned by the CPU-kernel
    block move outside that loop, enabling independent input versions. This
    adapter is for independent output tiles (RoPE/SwiGLU/normal RMSNorm), not
    reduction loops or hardware workitem mapping.
    """
    from tvm import tir, arith
    if isinstance(num_stages, bool) or num_stages not in (0, 2, 3):
        raise ValueError("num_stages must be 0 (serial), 2 or 3")
    root = function.body
    if not isinstance(root, tir.BlockRealize) or root.block.iter_vars or root.block.init is not None:
        raise ValueError("tile adapter requires a plain root block")
    node = root.block.body
    grid = []
    while isinstance(node, tir.For):
        extent = arith.Analyzer().simplify(node.extent)
        if node.kind != tir.ForKind.SERIAL or not isinstance(extent, tir.IntImm) or int(extent) <= 0:
            raise ValueError("tile adapter requires a positive static serial grid")
        grid.append((node.loop_var, node.min, int(extent)))
        node = node.body
    if not grid or not isinstance(node, tir.BlockRealize) or not node.block.annotations.get("tilelang.is_cpu_kernel_frame"):
        raise ValueError("tile adapter requires the explicit CPU-kernel block")
    block = node.block
    if block.iter_vars or block.match_buffers or block.init is not None or not arith.Analyzer().can_prove(node.predicate):
        raise ValueError("tile adapter does not support conditional or aliased kernel blocks")
    tile = tir.Var("output_tile", "int32")
    count = 1
    mapping = {}
    for var, minimum, extent in reversed(grid):
        mapping[var] = minimum + (tile // count) % extent
        count *= extent
    body = tir.stmt_functor.substitute(block.body, mapping)
    loop = tir.For(tile, 0, count, tir.ForKind.SERIAL, body,
                   annotations={"num_stages": num_stages} if num_stages else {})
    kernel = tir.Block([], [], [], block.name_hint, loop, alloc_buffers=block.alloc_buffers,
                       annotations=block.annotations)
    outer = tir.Block(root.block.iter_vars, [], [], root.block.name_hint,
                      tir.BlockRealize([], True, kernel), alloc_buffers=root.block.alloc_buffers,
                      annotations=root.block.annotations)
    return function.with_body(tir.BlockRealize(root.iter_values, root.predicate, outer))


def build_elementwise_tiled(operation="add", *, rows=8, width=128,
                            block_rows=4, block_width=32, num_stages=0):
    validate_dimensions("elementwise-tiled", rows=rows, width=width,
                        block_rows=block_rows, block_width=block_width)
    validate_exact_tiling("elementwise-tiled", ("rows", rows, block_rows), ("width", width, block_width))
    if num_stages not in (0, 2, 3):
        raise ValueError("num_stages must be 0 (serial), 2 or 3")
    operations = {"add": T.ppl_add, "sub": T.ppl_subtract, "mul": T.ppl_mul, "div": T.ppl_div}
    if operation not in operations:
        raise ValueError(f"unsupported elementwise operation {operation}")
    calculate = operations[operation]
    columns = width // block_width
    tiles = (rows // block_rows) * columns

    @T.prim_func
    def elementwise_tiled(lhs: T.Tensor((rows, width), "float16"),
                          rhs: T.Tensor((rows, width), "float16"),
                          output: T.Tensor((rows, width), "float16")):
        with T.Kernel(1, is_cpu=True) as (_bx,):
            a = T.alloc_shared((block_rows, block_width), "float16")
            b = T.alloc_shared((block_rows, block_width), "float16")
            c = T.alloc_shared((block_rows, block_width), "float16")
            for tile in T.Pipelined(tiles, num_stages=num_stages):
                row = tile // columns * block_rows
                col = tile % columns * block_width
                T.ppl_copy(lhs[row, col], a)
                T.ppl_copy(rhs[row, col], b)
                calculate(c, a, b)
                T.ppl_copy(c, output[row, col])
    return elementwise_tiled
