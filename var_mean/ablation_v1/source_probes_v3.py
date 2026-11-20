"""Keep N runtime, while permitting Triton JIT's automatic alignment specialization."""

import triton
from source_probes import source_variant as unaligned_variant

# All legal N in this experiment are >=16 and divisible by16. They share the
# same alignment key; N remains a runtime argument rather than a constexpr.
aligned_variant = triton.jit(
    unaligned_variant.fn, do_not_specialize=["M", "correction"]
)


def select_variant(alignment):
    return aligned_variant if alignment else unaligned_variant
