"""Replace max+min-index dataflow with a stable tuple argmax in frozen TTIR."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from complex_suite.support import digest, freeze, save  # noqa: E402


def joint(base):
    begin = base.index('      %best = "tt.reduce"')
    end = base.index("      %0 =", begin)
    value, b = re.search(
        r'%best = "tt.reduce"\((%[\w]+)\).*?\(tensor<(\d+)xf32>\) -> f32',
        base[begin:end],
        re.S,
    ).groups()
    ids = re.search(r"%best_id = arith.select %\w+, (%\w+),", base[begin:end]).group(1)
    best_id = re.search(r'(%best_id_\d+) = "tt.reduce"', base[begin:end]).group(1)
    region = f"""      %pair:2 = "tt.reduce"({value}, {ids}) <{{axis = 0 : i32}}> ({{
      ^bb0(%lv: f32, %li: i32, %rv: f32, %ri: i32):
        %gt = arith.cmpf ogt, %lv, %rv : f32
        %eq = arith.cmpf oeq, %lv, %rv : f32
        %lt = arith.cmpi slt, %li, %ri : i32
        %tie = arith.andi %eq, %lt : i1
        %left = arith.ori %gt, %tie : i1
        %v = arith.select %left, %lv, %rv : f32
        %i = arith.select %left, %li, %ri : i32
        tt.reduce.return %v, %i : f32, i32
      }}) : (tensor<{b}xf32>, tensor<{b}xi32>) -> (f32, i32)
"""
    tail = base[end:]
    tail = re.sub(re.escape(best_id) + r"\b", "%pair#1", tail)
    tail = re.sub(r"%best\b", "%pair#0", tail)
    return base[:begin] + region + tail


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    records = []
    for row in manifest["records"]:
        for stage in ("local", "merge"):
            folder = ROOT / "ir" / row["group"] / stage
            base = (folder / "baseline.ttir").read_text()
            text = joint(base)
            cg = re.sub(r"(tt.load [^\n]+?) :", r"\1 cacheModifier = cg :", text)
            for name, value in {"joint_v1": text, "joint_cg_v1": cg}.items():
                path = folder / f"{name}.ttir"
                freeze(path, value)
                records.append(
                    {
                        "group": row["group"],
                        "stage": stage,
                        "name": name,
                        "baseline_sha256": digest(folder / "baseline.ttir"),
                        "sha256": digest(path),
                    }
                )
    save(ROOT / "candidates.json", records)


if __name__ == "__main__":
    main()
