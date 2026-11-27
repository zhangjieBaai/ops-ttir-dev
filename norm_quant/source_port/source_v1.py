"""Public Triton source port of stream_v1, frozen after formatting."""

import triton
import triton.language as tl


@triton.jit
def _sum_max(sum_l, max_l, sum_r, max_r):
    return sum_l + sum_r, tl.maximum(max_l, max_r)


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
    ss, maxu = tl.reduce((z * z, tl.abs(u)), axis=0, combine_fn=_sum_max)
    rho = tl.rsqrt(ss / N + 1.0e-5)
    y = u * rho
    amax = maxu * rho
    scale = tl.where(amax > 0, amax / 127.0, 1.0)
    q = tl.div_rn(y, scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    q = tl.extra.cuda.libdevice.nearbyint(q).to(tl.int8)
    tl.store(Z + row * N + col, z, mask, cache_modifier=".cs")
    tl.store(Q + row * N + col, q, mask, cache_modifier=".cs")
    tl.store(S + row, scale, cache_modifier=".cs")
