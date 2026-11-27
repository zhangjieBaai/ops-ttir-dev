"""Public Triton-language witnesses; separate from the TTIR performance experiment."""

import triton
import triton.language as tl


@triton.jit
def public_body(
    output_ptr,
    input_ptr,
    N,
    K,
    FLAT: tl.constexpr,
    INDEX64: tl.constexpr,
    UNROLL: tl.constexpr,
    RANK2: tl.constexpr,
):
    index_type = tl.int64 if INDEX64 else tl.int32
    pm = tl.program_id(0).to(index_type)
    pk = tl.program_id(1).to(index_type)
    if FLAT:
        ko = pk * 1024 + tl.arange(0, 1024).to(index_type)
        if RANK2:
            ko = ko[None, :]
    else:
        ko = (
            pk * 1024
            + tl.arange(0, 128)[:, None].to(index_type) * 8
            + tl.arange(0, 8)[None, :].to(index_type)
        )
    acc = tl.full(ko.shape, 0, tl.float32)
    base = pm * tl.cast(N, index_type) * tl.cast(K, index_type)
    for n in range(0, N, UNROLL):
        values = ()
        for j in tl.static_range(UNROLL):
            ni = n + j
            values += (
                tl.load(
                    input_ptr + base + ni.to(index_type) * tl.cast(K, index_type) + ko,
                    (ni < N) & (ko < K),
                    other=0,
                ).to(tl.float32),
            )
        for j in tl.static_range(UNROLL):
            acc = acc + values[j]
    if RANK2:
        result = tl.sum(acc, axis=0)
        out_ko = tl.reshape(ko, (1024,))
    else:
        result = acc
        out_ko = ko
    tl.store(output_ptr + pm * tl.cast(K, index_type) + out_ko, result / N, out_ko < K)


@triton.jit
def public_mean(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    FLAT: tl.constexpr,
    INDEX64: tl.constexpr,
    UNROLL: tl.constexpr,
    RANK2: tl.constexpr,
    GUARD: tl.constexpr,
    TAIL: tl.constexpr,
):
    if TAIL and K - tl.program_id(1) * 1024 == 1:
        col = tl.program_id(1) * 1024
        rows = tl.arange(0, 256)
        acc = tl.full((256,), 0, tl.float32)
        for start in range(0, N, 256):
            ni = start + rows
            offset = tl.program_id(0) * N * K + ni * K + col
            value = tl.load(input_ptr + offset, ni < N, other=0).to(tl.float32)
            acc += value
        result = tl.sum(acc, 0) / N
        tl.store(output_ptr + tl.program_id(0) * K + col, result)
    else:
        if GUARD and N < 8:
            public_body(output_ptr, input_ptr, N, K, True, False, 1, True)
        else:
            public_body(output_ptr, input_ptr, N, K, FLAT, INDEX64, UNROLL, RANK2)
