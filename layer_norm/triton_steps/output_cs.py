"""Step 5: keep step 4 and mark only output Y stores .cs."""

import triton
import triton.language as tl

BLOCK_SIZE = 2048


@triton.jit
def _dynamic_body(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0, cache_modifier=".ca").to(
            tl.float32
        )
        _mean += a
    inv_n = 1.0 / N
    mean = tl.sum(_mean, axis=0) * inv_n
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) * inv_n
    rstd = tl.rsqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask, cache_modifier=".cs")


@triton.jit
def _static_body(
    X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr, K: tl.constexpr
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block in tl.static_range(K):
        cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0, cache_modifier=".ca").to(
            tl.float32
        )
        _mean += a
    inv_n = 1.0 / N
    mean = tl.sum(_mean, axis=0) * inv_n
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block in tl.static_range(K):
        cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) * inv_n
    rstd = tl.rsqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for block in tl.static_range(K):
        cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask, cache_modifier=".cs")


@triton.jit
def kernel(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    if N <= 1024:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 1)
    elif N <= 2048:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 1)
    elif N <= 3072:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 2)
    elif N <= 4096:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 2)
    elif N <= 6144:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 3)
    elif N <= 8192:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 4)
    elif N <= 10240:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 5)
    elif N <= 12288:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 6)
    elif N <= 14336:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 7)
    elif N <= 16384:
        _static_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048, 8)
    else:
        _dynamic_body(X, Y, W, B, Mean, Rstd, stride, N, eps, 2048)
