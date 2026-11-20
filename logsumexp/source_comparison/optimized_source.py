"""Public Triton Python port of the selected r4_focus TTIR algorithms."""

import triton
import triton.language as tl


@triton.jit
def maximum_half(a, b):
    return tl.maximum(a, b)


@triton.jit
def inner_tile(
    output_ptr,
    input_ptr,
    N,
    WIDTH: tl.constexpr,
    HALF_MAX: tl.constexpr,
    INDEX32: tl.constexpr,
):
    pid = tl.program_id(0)
    if not INDEX32:
        pid = pid.to(tl.int64)
    cols = tl.arange(0, WIDTH)
    x16 = tl.load(input_ptr + pid * N + cols, cols < N, other=-float("inf"))
    if HALF_MAX:
        maximum = tl.reduce(x16, 0, maximum_half).to(tl.float32)
        x = x16.to(tl.float32)
    else:
        x = x16.to(tl.float32)
        maximum = tl.max(x, 0)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    total = tl.sum(tl.exp(x - safe), 0)
    out = tl.where(maximum == -float("inf"), maximum, safe + tl.log(total))
    tl.store(output_ptr + pid, out)


@triton.jit
def small_rows(output_ptr, input_ptr, M, N, WIDTH: tl.constexpr):
    pid = tl.program_id(0)
    if pid % 8 == 0:
        rows = pid + tl.arange(0, 8)
        cols = tl.arange(0, WIDTH)
        x = tl.load(
            input_ptr + rows[:, None] * N + cols[None, :],
            (rows[:, None] < M) & (cols[None, :] < N),
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.max(x, 1)
        safe = tl.where(maximum == -float("inf"), 0.0, maximum)
        total = tl.sum(tl.exp(x - safe[:, None]), 1)
        out = tl.where(maximum == -float("inf"), maximum, safe + tl.log(total))
        tl.store(output_ptr + rows, out, rows < M)


@triton.jit
def lane32(output_ptr, input_ptr, N):
    cols = tl.arange(0, 32)
    x0 = tl.load(input_ptr + cols, cols < N, other=-float("inf"))
    x1 = tl.load(input_ptr + cols + 32, cols + 32 < N, other=-float("inf"))
    x2 = tl.load(input_ptr + cols + 64, cols + 64 < N, other=-float("inf"))
    x3 = tl.load(input_ptr + cols + 96, cols + 96 < N, other=-float("inf"))
    maxima = tl.maximum(x3, tl.maximum(x2, tl.maximum(x1, x0)))
    maximum = tl.reduce(maxima, 0, maximum_half).to(tl.float32)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    e0 = tl.exp(x0.to(tl.float32) - safe)
    e1 = tl.exp(x1.to(tl.float32) - safe)
    e2 = tl.exp(x2.to(tl.float32) - safe)
    e3 = tl.exp(x3.to(tl.float32) - safe)
    total = tl.sum(e3 + (e2 + (e1 + e0)), 0)
    out = tl.where(maximum == -float("inf"), maximum, safe + tl.log(total))
    tl.store(output_ptr, out)


@triton.jit
def online_inner(output_ptr, input_ptr, N, WIDTH: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    m = tl.full((WIDTH,), -float("inf"), tl.float32)
    z = tl.full((WIDTH,), 0.0, tl.float32)
    for start in range(0, N, WIDTH):
        offsets = start + tl.arange(0, WIDTH)
        x = tl.load(input_ptr + pid * N + offsets, offsets < N, other=-float("inf")).to(
            tl.float32
        )
        new_m = tl.maximum(m, x)
        q = tl.exp(-tl.abs(x - m))
        updated = tl.where(x > m, z * q + 1.0, z + q)
        z = tl.where(new_m == -float("inf"), z, updated)
        m = new_m
    maximum = tl.max(m, 0)
    total = tl.sum(z * tl.exp(m - maximum), 0)
    tl.store(
        output_ptr + pid,
        tl.where(maximum == -float("inf"), maximum, maximum + tl.log(total)),
    )


@triton.jit
def logsumexp_source_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
    SINGLE_ROW: tl.constexpr,
):
    if ONE_TILE_PER_CTA:
        if TILE_N <= 128:
            if SINGLE_ROW:
                lane32(output_ptr, input_ptr, N)
            else:
                small_rows(output_ptr, input_ptr, M, N, TILE_N)
        else:
            inner_tile(output_ptr, input_ptr, N, TILE_N, TILE_N >= 16384, False)
    else:
        if N <= 65536:
            inner_tile(output_ptr, input_ptr, N, 65536, True, True)
        else:
            online_inner(output_ptr, input_ptr, N, TILE_N)


@triton.jit
def noninner_tile(
    output_ptr, input_ptr, N, K, WIDTH: tl.constexpr, TILE_K: tl.constexpr
):
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    cols = pk * TILE_K + tl.arange(0, TILE_K)[None, :]
    rows = tl.arange(0, WIDTH)[:, None]
    x = tl.load(
        input_ptr + pm * N * K + rows * K + cols,
        (rows < N) & (cols < K),
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(x, 0, keep_dims=True)
    safe = tl.where(maximum == -float("inf"), tl.zeros_like(maximum), maximum)
    total = tl.sum(tl.exp(x - safe), 0, keep_dims=True)
    out = tl.where(maximum == -float("inf"), maximum, safe + tl.log(total))
    tl.store(output_ptr + pm * K + cols, out, cols < K)


@triton.jit
def logsumexp_source_noninner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if ONE_TILE_PER_CTA:
        if tl.constexpr(TILE_N > 128):
            if N <= 128:
                noninner_tile(output_ptr, input_ptr, N, K, 128, TILE_K)
            else:
                if tl.constexpr(TILE_N > 512):
                    if N <= 512:
                        noninner_tile(output_ptr, input_ptr, N, K, 512, TILE_K)
                    else:
                        if tl.constexpr(TILE_N > 2048):
                            if N <= 2048:
                                noninner_tile(output_ptr, input_ptr, N, K, 2048, TILE_K)
                            else:
                                if tl.constexpr(TILE_N > 4096):
                                    if N <= 4096:
                                        noninner_tile(
                                            output_ptr, input_ptr, N, K, 4096, TILE_K
                                        )
                                    else:
                                        noninner_tile(
                                            output_ptr, input_ptr, N, K, TILE_N, TILE_K
                                        )
                                else:
                                    noninner_tile(
                                        output_ptr, input_ptr, N, K, TILE_N, TILE_K
                                    )
                        else:
                            noninner_tile(output_ptr, input_ptr, N, K, TILE_N, TILE_K)
                else:
                    noninner_tile(output_ptr, input_ptr, N, K, TILE_N, TILE_K)
        else:
            noninner_tile(output_ptr, input_ptr, N, K, TILE_N, TILE_K)
    else:
        pm = tl.program_id(0)
        pk = tl.program_id(1)
        cols = pk * TILE_K + tl.arange(0, TILE_K)[None, :]
        m = tl.full((TILE_N, TILE_K), -float("inf"), tl.float32)
        z = tl.full((TILE_N, TILE_K), 0.0, tl.float32)
        for start in range(0, N, TILE_N):
            rows = start + tl.arange(0, TILE_N)[:, None]
            x = tl.load(
                input_ptr + pm * N * K + rows * K + cols,
                (rows < N) & (cols < K),
                other=-float("inf"),
            ).to(tl.float32)
            new_m = tl.maximum(m, x)
            q = tl.exp(-tl.abs(x - m))
            updated = tl.where(x > m, z * q + 1.0, z + q)
            z = tl.where(new_m == -float("inf"), z, updated)
            m = new_m
        maximum = tl.max(m, 0, keep_dims=True)
        total = tl.sum(z * tl.exp(m - maximum), 0, keep_dims=True)
        out = tl.where(maximum == -float("inf"), maximum, maximum + tl.log(total))
        tl.store(output_ptr + pm * K + cols, out, cols < K)
