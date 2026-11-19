"""Frozen row-kernel extraction; see upstream/provenance.json."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["correction", "M", "N"])
def var_mean_welford_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_N: tl.constexpr,
):
    # One row per program to avoid autotune correctness issues on some backends.
    pid = tl.program_id(0).to(tl.int64)
    X = X + pid * N
    Var = Var + pid
    Mean = Mean + pid

    _sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum) / N

    _acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        _acc += diff * diff
    var = tl.sum(_acc) / (N - correction)
    # Write mean / var
    tl.store(Mean, mean)
    tl.store(Var, var)
