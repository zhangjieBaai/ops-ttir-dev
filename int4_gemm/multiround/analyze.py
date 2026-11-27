"""Static code/resource evidence and identity counts; no GPU launches."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    records = []
    identities = {}
    for stage in ("r3", "r4", "r5"):
        folder = ROOT / stage
        data = json.loads((folder / "results/screen/results.json").read_text())
        for name in data["valid_names"]:
            if name in ("baseline", "anchor", "parent"):
                continue
            identities[stage + "/" + name] = sum(
                r["kernels"][name]["elf_text"] == r["kernels"]["parent"]["elf_text"]
                for r in data["records"]
            )
        for row in data["records"]:
            for name, meta in row["kernels"].items():
                ptx = (
                    folder / "artifacts" / row["case"] / name / "kernel.ptx"
                ).read_text()
                records.append(
                    {
                        "round": stage,
                        "case": row["case"],
                        "name": name,
                        "n_regs": meta["n_regs"],
                        "n_spills": meta["n_spills"],
                        "shared": meta["metadata"]["shared"],
                        "elf_text": meta["elf_text"],
                        "static_ptx_counts": {
                            p: len(re.findall(p, ptx))
                            for p in (
                                r"div\.u32",
                                r"div\.s32",
                                r"shr\.b32",
                                r"rcp\.",
                                r"ld\.global",
                                r"st\.local",
                                r"ld\.local",
                                r"bar\.sync",
                                r"mma\.sync",
                            )
                        },
                    }
                )
    result = {
        "evidence_level": "C, static only, not dynamic/profiler counts",
        "identical_to_parent_points": identities,
        "matrix_points_per_round": 30,
        "records": records,
    }
    (ROOT / "static_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(identities)


if __name__ == "__main__":
    main()
