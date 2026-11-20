"""Upstream Triton equivalents, isolated from direct-TTIR performance experiments."""

import triton
import triton.language as tl


@triton.jit
def centered_loop(X, N, correction, BLOCK: tl.constexpr):
    sums = tl.zeros((BLOCK,), tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        values = tl.load(X + cols, cols < N, other=0.0)
        sums += values
    mean = tl.sum(sums, 0) / N
    squares = tl.zeros((BLOCK,), tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        values = tl.load(X + cols, cols < N, other=0.0)
        diff = tl.where(cols < N, values - mean, 0.0)
        squares += diff * diff
    variance = tl.sum(squares, 0) / (N - correction)
    return variance, mean


@triton.jit
def centered_tile(X, N, correction, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    values = tl.load(X + cols, cols < N, other=0.0)
    mean = tl.sum(values, 0) / N
    diff = tl.where(cols < N, values - mean, 0.0)
    variance = tl.sum(diff * diff, 0) / (N - correction)
    return variance, mean


@triton.jit(do_not_specialize=["M", "N", "correction"])
def source_variant(
    X, Var, Mean, M, N, correction, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    stride64 = N.to(tl.int64)
    if C:
        stride64 = tl.multiple_of(stride64, 16)
    row = X + pid * stride64
    if A:
        if N <= 128:
            variance, mean = centered_tile(row, N, correction, 128)
        elif N <= 512:
            variance, mean = centered_tile(row, N, correction, 512)
        elif N <= 1024:
            variance, mean = centered_tile(row, N, correction, 1024)
        else:
            if B:
                variance, mean = centered_loop(row, N, correction, 256)
            else:
                variance, mean = centered_loop(row, N, correction, 1024)
    elif B:
        if N <= 1024:
            variance, mean = centered_loop(row, N, correction, 1024)
        else:
            variance, mean = centered_loop(row, N, correction, 256)
    else:
        variance, mean = centered_loop(row, N, correction, 1024)
    tl.store(Var + pid, variance)
    tl.store(Mean + pid, mean)
