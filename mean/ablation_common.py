"""Frozen factorial contrasts; every comparison uses samples from the same block."""

import itertools

from make_ablation import NAMES


def edges():
    contrasts = [
        ("I_at_F0", "baseline", "ab_f0i1"),
        ("I_at_F1", "ab_f1i0", "ab_f1i1"),
        ("F_at_I0", "baseline", "ab_f1i0"),
        ("F_at_I1", "ab_f0i1", "ab_f1i1"),
        ("FI_combined", "baseline", "ab_f1i1"),
        ("round1_bridge", "ab_f1i1", "optimized"),
        ("round1_total", "baseline", "optimized"),
        ("round2_scaffold", "optimized", "ab_u0g0t0"),
        ("round2_total", "optimized", "ab_u1g1t1"),
        ("all_total", "baseline", "ab_u1g1t1"),
    ]
    for factor, pos in [("U", 0), ("G", 1), ("T", 2)]:
        for others in itertools.product((0, 1), repeat=2):
            bits = list(others)
            bits.insert(pos, 0)
            reference = "ab_u%dg%dt%d" % tuple(bits)
            bits[pos] = 1
            candidate = "ab_u%dg%dt%d" % tuple(bits)
            context = reference.removeprefix("ab_")
            contrasts.append((f"{factor}_at_{context}", reference, candidate))
    return contrasts


EDGES = edges()
assert len(NAMES) == 13 and len(EDGES) == 22
