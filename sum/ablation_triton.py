"""Public Triton equivalents for the historical direct-TTIR transformations.

These are separate expressibility controls, never replacements for historical IR.
"""

import triton
import triton.language as tl


@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    I32: tl.constexpr = False,
    FOLD: tl.constexpr = False,
    CACHE: tl.constexpr = "",
    TAIL: tl.constexpr = False,
):
    if I32:
        pid = tl.program_id(0)
    else:
        pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if TAIL:
        if (pid + 1) * BLOCK_SIZE <= M:
            values = tl.load(inp + offsets, cache_modifier=CACHE)
        else:
            values = tl.load(inp + offsets, offsets < M, other=0, cache_modifier=CACHE)
    else:
        values = tl.load(inp + offsets, offsets < M, other=0, cache_modifier=CACHE)
    if FOLD and BLOCK_SIZE >= 512:
        partial = tl.sum(
            tl.reshape(values, (4, BLOCK_SIZE // 4), can_reorder=True), axis=0
        )
        result = tl.sum(partial, axis=0)
    else:
        result = tl.sum(values, axis=0)
    tl.store(mid + pid, result)


@triton.jit
def sum_kernel_2(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
    FOLD: tl.constexpr = False,
    CACHE: tl.constexpr = "",
):
    offsets = tl.arange(0, BLOCK_MID)
    values = tl.load(mid + offsets, offsets < mid_size, other=0, cache_modifier=CACHE)
    if FOLD and BLOCK_MID >= 512:
        partial = tl.sum(
            tl.reshape(values, (4, BLOCK_MID // 4), can_reorder=True), axis=0
        )
        result = tl.sum(partial, axis=0)
    else:
        result = tl.sum(values, axis=0)
    tl.store(out, result)
