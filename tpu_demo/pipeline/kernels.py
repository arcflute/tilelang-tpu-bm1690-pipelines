"""Same-tile serial and pipelined variants; original demos remain available."""

import tilelang.language as T
from tpu_demo.common import validate_dimensions, validate_exact_tiling


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
