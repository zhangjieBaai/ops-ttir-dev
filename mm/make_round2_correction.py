"""Keep efficient pointer recurrences while simplifying the scalar K induction variable."""

from common import ROOT, frozen, save_json, sha256
from make_round2 import FAMILY, once


def main():
    config = frozen()["families"][FAMILY]
    source = ROOT / config["ttir_path"]
    text = source.read_text()
    text = once(
        text,
        "scf.for %k = %c0_i32 to %6 step %c1_i32",
        "scf.for %k = %c0_i32 to %K step %c64_i32",
    )
    text = once(text, "%a = arith.muli %k, %c64_i32", "%a = arith.addi %k, %c0_i32")
    path = source.with_name("r2_k_offset.ttir")
    if path.exists():
        assert path.read_text() == text
    else:
        with path.open("x") as f:
            f.write(text)
    save_json(
        ROOT / "results/runs/mm_opt_r2_20261119/correction_candidate.json",
        dict(
            name="r2_k_offset",
            ttir_path=str(path.relative_to(ROOT)),
            ttir_sha256=sha256(path),
            parent_sha256=sha256(source),
            reason=(
                "Absolute addressing regressed. Preserve pointer iter_args but carry actual K offset in step64 loop; "
                "same ceil(K/64) dots and masks. Eighth/final functional candidate."
            ),
        ),
    )


if __name__ == "__main__":
    main()
