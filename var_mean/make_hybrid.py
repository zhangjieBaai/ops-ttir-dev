"""Combine short-row TTIR dispatch with the narrow loop for longer rows."""

from common import ROOT, digest
from make_candidates import record_generation, write
from make_dispatch import tile


def with_alignment(text):
    # The frozen contract guarantees N % 16 == 0. Every row remains aligned.
    assert text.count("%N: i32") == 1
    return text.replace("%N: i32", "%N: i32 {tt.divisibility = 16 : i32}")


def main():
    narrow = (ROOT / "ir/r2_tile256_v2.ttir").read_text()
    prefix, loop = narrow.split("    %_sum = scf.for", 1)
    body, suffix = loop.split("    tt.return", 1)
    body = "    %_sum = scf.for" + body
    hybrid = prefix
    for width in (128, 512, 1024):
        hybrid += f"""%limit{width} = arith.constant {width} : i32
%short{width} = arith.cmpi sle, %N, %limit{width} : i32
scf.if %short{width} {{
{tile(width)}
}} else {{
"""
    hybrid += body + "}\n" * 3 + "tt.return" + suffix
    versions = {
        "r2_short_hybrid_v1": hybrid,
        "r2_short_hybrid_aligned_v1": with_alignment(hybrid),
        "r2_tile256_aligned_v1": with_alignment(narrow),
    }
    for name, text in versions.items():
        write(name, text)
    record_generation(
        ROOT / "results/runs/r2/candidates_hybrid.json",
        dict(
            parent_sha256=digest(ROOT / "ir/r2_tile256_v2.ttir"),
            candidates={name: digest(ROOT / f"ir/{name}.ttir") for name in versions},
            hypothesis="Cap register pressure using short single tiles and exploit proven row divisibility.",
        ),
    )


if __name__ == "__main__":
    main()
