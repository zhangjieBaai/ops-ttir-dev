import triton
import triton.language as tl

import frozen_extension as ext


@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)
