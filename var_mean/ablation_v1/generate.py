"""Generate an exact 2^3 factor design directly from the frozen TTIR."""

import re
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, digest  # noqa: E402
from make_candidates import record_generation, strip_locations, write  # noqa: E402
from make_dispatch import tile  # noqa: E402
from make_hybrid import with_alignment  # noqa: E402


def main():
    base = strip_locations((ROOT / "ir/baseline.ttir").read_text())
    for num, name in [(4, "row_ptr"), (5, "var_ptr"), (6, "mean_ptr")]:
        base = re.sub(rf"%{num}(?![0-9])", f"%{name}", base)
    prefix, tail = base.split("    %_sum = scf.for", 1)
    loops, suffix = tail.split("    tt.return", 1)
    loops = "    %_sum = scf.for" + loops
    records = {}
    for a, b, c in product([0, 1], repeat=3):
        bits = f"{a}{b}{c}"
        long_body = loops.replace("1024", "256") if b else loops
        if a:
            body = ""
            for width in (128, 512, 1024):
                body += f"""%limit{width} = arith.constant {width} : i32
%short{width} = arith.cmpi sle, %N, %limit{width} : i32
scf.if %short{width} {{
{tile(width)}
}} else {{
"""
            body += long_body + "}\n" * 3
        elif b:
            body = f"""%long_bound = arith.constant 1024 : i32
%is_short = arith.cmpi sle, %N, %long_bound : i32
scf.if %is_short {{
{loops}
}} else {{
{long_body}
}}
"""
        else:
            body = loops
        active_prefix = prefix.replace("1024", "256") if a and b else prefix
        if b and not a:
            body = body.replace(
                "} else {\n" + long_body,
                "} else {\n%c256_i32 = arith.constant 256 : i32\n"
                "%zero_long = arith.constant dense<0.0> : tensor<256xf32>\n"
                + long_body.replace("%cst", "%zero_long"),
            )
        text = active_prefix + body + "tt.return" + suffix
        if c:
            text = with_alignment(text)
        name = f"ab4_{bits}"
        write(name, text)
        records[bits] = dict(
            name=name,
            A=bool(a),
            B=bool(b),
            C=bool(c),
            sha256=digest(ROOT / f"ir/{name}.ttir"),
        )
    assert (ROOT / "ir/ab4_111.ttir").read_bytes() == (
        ROOT / "ir/optimized.ttir"
    ).read_bytes()
    record_generation(
        ROOT / "ablation_v1/manifest.json",
        dict(
            contract_sha256=digest(ROOT / "contract.json"),
            plan_sha256=digest(ROOT / "ablation_v1/plan.md"),
            baseline_sha256=digest(ROOT / "ir/baseline.ttir"),
            optimized_sha256=digest(ROOT / "ir/optimized.ttir"),
            variants=records,
        ),
    )


if __name__ == "__main__":
    main()
