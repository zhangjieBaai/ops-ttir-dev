"""Follow-up to tuple regression: expose unsigned score max reduction in TTIR."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from complex_suite.support import digest, freeze, save  # noqa: E402


def ordered(base):
    begin = base.index('      %best = "tt.reduce"')
    end = base.index("      %matches =", begin)
    value, b = re.search(
        r'%best = "tt.reduce"\((%[\w]+)\).*?\(tensor<(\d+)xf32>\) -> f32',
        base[begin:end],
        re.S,
    ).groups()
    region = f"""      %bits = tt.bitcast {value} : tensor<{b}xf32> -> tensor<{b}xi32>
      %shift = arith.constant dense<31> : tensor<{b}xi32>
      %signbit = arith.constant dense<-2147483648> : tensor<{b}xi32>
      %sign = arith.shrsi %bits, %shift : tensor<{b}xi32>
      %flip = arith.ori %sign, %signbit : tensor<{b}xi32>
      %key = arith.xori %bits, %flip : tensor<{b}xi32>
      %maxkey = "tt.reduce"(%key) <{{axis = 0 : i32}}> ({{
      ^bb0(%lkey: i32, %rkey: i32):
        %mkey = arith.maxui %lkey, %rkey : i32
        tt.reduce.return %mkey : i32
      }}) : (tensor<{b}xi32>) -> i32
      %izero = arith.constant 0 : i32
      %iall = arith.constant -1 : i32
      %isign = arith.constant -2147483648 : i32
      %positive = arith.cmpi slt, %maxkey, %izero : i32
      %undo = arith.select %positive, %isign, %iall : i32
      %bestbits = arith.xori %maxkey, %undo : i32
      %best = tt.bitcast %bestbits : i32 -> f32
"""
    return base[:begin] + region + base[end:]


def main():
    records = []
    for row in json.loads((ROOT / "manifest.json").read_text())["records"]:
        for stage in ("local", "merge"):
            folder = ROOT / "ir" / row["group"] / stage
            path = folder / "ordered_max_v2.ttir"
            freeze(path, ordered((folder / "baseline.ttir").read_text()))
            records.append(
                {
                    "group": row["group"],
                    "stage": stage,
                    "baseline_sha256": digest(folder / "baseline.ttir"),
                    "sha256": digest(path),
                }
            )
    save(ROOT / "ordered_candidates.json", records)


if __name__ == "__main__":
    main()
