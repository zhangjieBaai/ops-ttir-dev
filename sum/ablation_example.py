"""Runnable public Triton host wrapper for the measured FP32 ablation scope.

This is an experimental example, not a replacement for FlagGems dispatch.
"""

import torch
import triton
from ablation_triton import sum_kernel_1, sum_kernel_2
from common import OPTIONS, PERF_SIZES, geometry

VARIANTS = {
    "baseline",
    "i32",
    "fold",
    "i32_fold",
    "cg",
    "tail",
    "wide_two",
    "single",
    "single_tail",
}


def sum_fp32(x, variant="single"):
    """Allocate and launch the public kernels; no PyTorch computation in this wrapper."""
    if variant not in VARIANTS:
        raise ValueError("Unsupported variant; public tl.load does not accept .cs")
    if (
        x.dtype != torch.float32
        or not x.is_cuda
        or x.ndim != 1
        or not x.is_contiguous()
    ):
        raise ValueError("Expected contiguous 1D CUDA FP32 input")
    n = x.numel()
    if n not in PERF_SIZES or x.data_ptr() % 16:
        raise ValueError("Input outside the 30 frozen shapes or pointer alignment")
    block, size, tail = geometry(n)
    out = torch.empty((), dtype=x.dtype, device=x.device)
    cache = ".cg" if variant == "cg" else ""
    folded = variant in {"fold", "i32_fold"}
    single = variant in {"single", "single_tail"} and n <= 16384
    wide_two = variant == "wide_two" and n <= 16384
    if single:
        sum_kernel_1[(1, 1, 1)](x, out, n, triton.next_power_of_2(n), **OPTIONS)
    else:
        if wide_two:
            block, size, tail = triton.next_power_of_2(n), 1, 1
        mid = torch.empty((size,), dtype=x.dtype, device=x.device)
        sum_kernel_1[(size, 1, 1)](
            x,
            mid,
            n,
            block,
            I32=variant in {"i32", "i32_fold"},
            FOLD=folded,
            CACHE=cache,
            TAIL=variant in {"tail", "single_tail"},
            **OPTIONS,
        )
        sum_kernel_2[(1, 1, 1)](
            mid, out, size, tail, FOLD=folded, CACHE=cache, **OPTIONS
        )
    return out
