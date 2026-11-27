"""Frozen two-stage partial selection research baseline (not a full sort)."""

import triton
import triton.language as tl


@triton.jit
def partial_select(X, I, V, O, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr, CHUNKS: tl.constexpr, MERGE: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    col = chunk * B + tl.arange(0, B)
    mask = col < N
    values = tl.load(X + row * N + col, mask, other=-float("inf"))
    if MERGE:
        ids = tl.load(I + row * N + col, mask, other=2147483647)
    else:
        ids = tl.where(mask, col, 2147483647)
    for step in range(K):
        best = tl.max(values, 0)
        matches = values == best
        best_id = tl.min(tl.where(matches, ids, 2147483647), 0)
        tl.store(V + (row * CHUNKS + chunk) * K + step, best)
        tl.store(O + (row * CHUNKS + chunk) * K + step, best_id)
        selected = ids == best_id
        values = tl.where(selected, -float("inf"), values)
        ids = tl.where(selected, 2147483647, ids)
