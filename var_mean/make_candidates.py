"""Deterministic direct TTIR edits; never re-export modified Python kernels."""

import json
import re

from common import ROOT, digest, save


def strip_locations(s):
    s = re.sub(r"^#loc.*\n", "", s, flags=re.M)
    s = re.sub(r" loc\([^()]*\)", "", s)
    return s


def record_generation(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value
    else:
        save(path, value)


def write(name, s):
    p = ROOT / f"ir/{name}.ttir"
    if p.exists():
        assert p.read_text() == s
    else:
        p.write_text(s)


def main():
    original = (ROOT / "ir/baseline.ttir").read_text()
    base = strip_locations(original)
    for num, name in [(4, "row_ptr"), (5, "var_ptr"), (6, "mean_ptr")]:
        base = re.sub(rf"%{num}(?![0-9])", f"%{name}", base)
    for width in [256, 4096, 8192]:
        write(f"r2_tile{width}_v2", base.replace("1024", str(width)))
    record_generation(
        ROOT / "results/runs/r2/candidates_v2.json",
        dict(
            baseline_sha256=digest(ROOT / "ir/baseline.ttir"),
            edits="Replace all tensor/range/step widths consistently; no Python kernel change.",
            names=["r2_tile256_v2", "r2_tile4096_v2", "r2_tile8192_v2"],
        ),
    )


if __name__ == "__main__":
    main()
