"""Direct immutable TTIR rewrites; never compile a candidate Python kernel."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from complex_suite.support import digest, freeze, save  # noqa: E402


def candidates(base):
    b = re.search(r"%ss = arith.mulf %z, %z : tensor<(\d+)xf32>", base).group(1)
    ty = f"tensor<{b}xf32>"
    start = base.index("    %ss_17 =")
    end = base.index("    %scale_23 =")
    absline = f"    %absu = math.absf %u : {ty}\n"
    maxreduce = f"""    %maxu = "tt.reduce"(%absu) <{{axis = 0 : i32}}> ({{
    ^bb0(%ml: f32, %mr: f32):
      %mv = arith.maxnumf %ml, %mr : f32
      tt.reduce.return %mv : f32
    }}) : ({ty}) -> f32
"""
    sumregion = base[start : base.index("    %rho_18 =", start)]
    rhoregion = base[
        base.index("    %rho_18 =", start) : base.index("    %amax =", start)
    ]
    tail = "    %amax_22 = arith.mulf %maxu, %rho_20 : f32\n"
    factor = (
        base[:start] + absline + sumregion + maxreduce + rhoregion + tail + base[end:]
    )
    jointreduce = f"""    %stats:2 = "tt.reduce"(%ss, %absu) <{{axis = 0 : i32}}> ({{
    ^bb0(%sl: f32, %ml: f32, %sr: f32, %mr: f32):
      %sv = arith.addf %sl, %sr : f32
      %mv = arith.maxnumf %ml, %mr : f32
      tt.reduce.return %sv, %mv : f32, f32
    }}) : ({ty}, {ty}) -> (f32, f32)
"""
    joint = (
        base[:start]
        + absline
        + jointreduce
        + rhoregion.replace("%ss_17", "%stats#0")
        + tail.replace("%maxu", "%stats#1")
        + base[end:]
    )
    cancel = joint.replace(
        "    %q_26 =",
        "    %uscale = arith.divf %stats#1, %cst_2 : f32\n"
        "    %usafe = arith.select %scale_23, %uscale, %cst_1 : f32\n"
        "    %q_26 =",
        1,
    )
    cancel = cancel.replace("tt.splat %scale_25 :", "tt.splat %usafe :").replace(
        "tt.precise_divf %y_21, %q_26", "tt.precise_divf %u, %q_26"
    )
    stream = re.sub(r"(tt.store %[^\n]+?) :", r"\1 cacheModifier = cs :", joint)
    return {
        "factor_v1": factor,
        "joint_v1": joint,
        "joint_cancel_v1": cancel,
        "stream_v1": stream,
    }


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    records = []
    for row in manifest["records"]:
        folder = ROOT / "ir" / row["group"]
        base = (folder / "baseline.ttir").read_text()
        for name, text in candidates(base).items():
            path = folder / f"{name}.ttir"
            freeze(path, text)
            records.append(
                {
                    "group": row["group"],
                    "candidate": name,
                    "parent_sha256": digest(folder / "baseline.ttir"),
                    "sha256": digest(path),
                }
            )
    save(ROOT / "candidates.json", records)


if __name__ == "__main__":
    main()
