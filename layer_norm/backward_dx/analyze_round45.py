"""Archive static and profiler evidence for rounds four and five."""

import csv
import io
import json
import re
import subprocess
from collections import Counter

from common import ROOT, digest, save


def main():
    r = ROOT / "results/round4"
    out = {}
    paths = sorted(ROOT.glob("artifacts/b*/r[45]_*/metadata.json"))
    for path in paths:
        name, group = path.parent.name, path.parent.parent.name
        metadata = json.load(open(path))
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", str(path.parent / "kernel.cubin")], text=True
        )
        ops = Counter(
            re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]*)", sass)
        )
        out[f"{name}/{group}"] = dict(
            n_regs=metadata["n_regs"],
            n_spills=metadata["n_spills"],
            shared=metadata["shared"],
            ttir_sha256=digest(ROOT / f"ir/{group}/{name}.ttir"),
            cubin_sha256=digest(path.parent / "kernel.cubin"),
            static_instruction_counts=dict(ops),
            note="Static counts include loop body once, not runtime instruction totals",
        )
    save(r / "binary_analysis.json", out)
    text = (r / "ncu.csv").read_text()
    rows = list(csv.DictReader(io.StringIO(text[text.index('"ID"') :])))
    units = rows[0]
    rows = [v for v in rows if str(v.get("ID", "")).isdigit()]
    assert len(rows) == 12
    result = []
    for i, v in enumerate(rows):
        record = dict(
            N=[3072, 8192, 15872][i // 4],
            kernel=["baseline", "r4_parent", "r4_evict", "r5_rows8_tile256"][i % 4],
        )
        for key in v:
            if key.startswith(
                ("gpu__", "dram__", "sm__", "smsp__", "launch__registers")
            ):
                record[key] = dict(
                    value=float(v[key].replace(",", "")), unit=units[key]
                )
        result.append(record)
    save(r / "ncu_summary.json", result)


if __name__ == "__main__":
    main()
