"""Archive static SASS resources and the two diagnostic NCU collections."""

import csv
import io
import json
import re
import subprocess
from collections import Counter

from common import ROOT, digest, save

R = ROOT / "results/round3"


def profile(source, target, shapes, candidate):
    text = (R / source).read_text()
    rows = list(csv.DictReader(io.StringIO(text[text.index('"ID"') :])))
    units = rows[0]
    rows = [r for r in rows if str(r.get("ID", "")).isdigit()]
    assert len(rows) == len(shapes) * 3
    out = []
    for i, row in enumerate(rows):
        rec = dict(N=shapes[i // 3], kernel=["baseline", "r3_parent", candidate][i % 3])
        for key, value in row.items():
            if key.startswith(
                ("gpu__", "dram__", "sm__", "smsp__", "launch__registers")
            ):
                rec[key] = dict(value=float(value.replace(",", "")), unit=units[key])
        out.append(rec)
    save(R / target, out)


def main():
    resources = {}
    paths = list(ROOT.glob("artifacts/b*/r3_*/metadata.json"))
    paths += list(ROOT.glob("artifacts/b*/baseline/metadata.json"))
    for path in paths:
        m = json.load(open(path))
        name, block = path.parent.name, path.parent.parent.name
        cubin = path.parent / "kernel.cubin"
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", str(cubin)], text=True
        )
        ops = Counter(
            re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]*)", sass)
        )
        resources[f"{name}/{block}"] = dict(
            n_regs=m["n_regs"],
            n_spills=m["n_spills"],
            shared=m["shared"],
            ttir_sha256=digest(ROOT / f"ir/{block}/{name}.ttir"),
            cubin_sha256=digest(cubin),
            static_instruction_counts=dict(ops),
            note="Static counts; loop body counted once, not runtime executed instructions",
        )
    save(R / "binary_analysis.json", resources)
    profile("ncu.csv", "ncu_summary.json", [8192, 8704, 15872], "r3_group_center2048")
    profile(
        "ncu_stream.csv", "ncu_stream_summary.json", [1024, 2048], "r3_centered_stream"
    )


if __name__ == "__main__":
    main()
