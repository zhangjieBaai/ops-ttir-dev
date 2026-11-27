"""Freeze a cumulative chain and final-version removals without changing winners."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OP = ROOT.parent
sys.path.insert(0, str(OP))
import common as c  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    r2 = load("int4_round2_candidates", OP / "round2/make_candidates.py")
    r3 = load("int4_multiround_candidates", OP / "multiround/generate.py")
    original = OP / "round2/expanded"
    manifest = json.loads((original / "manifest.json").read_text())
    records = []
    for case in manifest["cases"]:
        folder = original / "ir" / case["id"]
        parent = (folder / "parent.ttir").read_text()
        narrow = r2.narrow(parent, case["config"][2], case["config"][1])
        full = r2.full_k(narrow)
        pair = r2.group_pair(full, case["config"][2])
        final = r3.group128(pair, case)
        selected = OP / "multiround/r6/ir" / case["id"] / "optimized.ttir"
        assert final == selected.read_text(), selected
        no_narrow_pair = r2.full_k(parent).replace(
            "%group_scalar = arith.divui %k0, %qGroupSize : i32",
            "%abl_shift7 = arith.constant 7 : i32\n"
            "      %group_scalar = arith.shrui %k0, %abl_shift7 : i32",
        )
        variants = {
            "baseline": (folder / "baseline.ttir").read_text(),
            "parent": parent,
            "narrow": narrow,
            "full_k": r2.full_k(parent),
            "narrow_full_k": full,
            "group_pair": pair,
            "final": final,
            "no_full_k": r3.group128(r2.group_pair(narrow, case["config"][2]), case),
            "no_pair": r3.group128(full, case),
            "no_narrow_pair": no_narrow_pair,
        }
        for name, content in variants.items():
            path = ROOT / "frozen/ir" / case["id"] / f"{name}.ttir"
            c.write_once(path, content)
            records.append({"case": case["id"], "name": name, "sha256": c.sha(path)})
    c.write_once(ROOT / "frozen/manifest.json", json.dumps(manifest, indent=2) + "\n")
    c.write_once(
        ROOT / "frozen/candidates.json",
        json.dumps({"names": list(variants), "records": records}, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
