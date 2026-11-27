"""Public Triton source port of ordered_carry_v3, frozen after formatting."""

import triton
import triton.language as tl


@triton.jit
def partial_select(
    X,
    I,
    V,
    O,
    N: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    CHUNKS: tl.constexpr,
    MERGE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    col = chunk * B + tl.arange(0, B)
    mask = col < N
    values = tl.load(X + row * N + col, mask, other=-float("inf"))
    if MERGE:
        ids = tl.load(I + row * N + col, mask, other=2147483647)
    else:
        ids = tl.where(mask, col, 2147483647)
    values = tl.where(values == 0.0, 0.0, values)
    bits = values.to(tl.int32, bitcast=True)
    flip = (bits >> 31) | -2147483648
    keys = (bits ^ flip).to(tl.uint32)
    for step in range(K):
        best_key = tl.max(keys, 0)
        positive = best_key.to(tl.int32) < 0
        undo = tl.where(positive, -2147483648, -1).to(tl.uint32)
        best = (best_key ^ undo).to(tl.float32, bitcast=True)
        best_id = tl.min(tl.where(keys == best_key, ids, 2147483647), 0)
        tl.store(V + (row * CHUNKS + chunk) * K + step, best)
        tl.store(O + (row * CHUNKS + chunk) * K + step, best_id)
        selected = ids == best_id
        keys = tl.where(selected, tl.full((), 8388607, tl.uint32), keys)
        ids = tl.where(selected, 2147483647, ids)
