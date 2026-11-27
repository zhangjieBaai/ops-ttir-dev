"""Frozen, fully fused research baseline; provenance in optimization_plan.md."""

import triton
import triton.language as tl


@triton.jit
def residual_rms_int8(X, R, W, Z, Q, S, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=0).to(tl.float32)
    r = tl.load(R + row * N + col, mask, other=0).to(tl.float32)
    w = tl.load(W + col, mask, other=0).to(tl.float32)
    z = x + r
    u = z * w
    ss = tl.sum(z * z, 0)
    rho = tl.rsqrt(ss / N + 1.0e-5)
    y = u * rho
    amax = tl.max(tl.abs(y), 0)
    scale = tl.where(amax > 0, amax / 127.0, 1.0)
    q = tl.div_rn(y, scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    q = tl.extra.cuda.libdevice.nearbyint(q).to(tl.int8)
    tl.store(Z + row * N + col, z, mask)
    tl.store(Q + row * N + col, q, mask)
    tl.store(S + row, scale)
