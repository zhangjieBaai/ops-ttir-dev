"""Second-round immutable direct TTIR dataflow rewrites."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from complex_suite.support import digest, freeze, save  # noqa: E402


def reciprocal(text):
    insert = (
        "    %r2_inv = tt.precise_divf %cst_1, %scale_25 : f32\n"
        "    %r2_invvec = tt.splat %r2_inv : f32 -> "
        + re.search(r"%q_26 = tt.splat .* -> (tensor<\d+xf32>)", text)[1]
        + "\n"
    )
    text = text.replace("    %q_26 =", insert + "    %q_26 =", 1)
    return text.replace("tt.precise_divf %y_21, %q_26", "arith.mulf %y_21, %r2_invvec")


def early_z(text):
    begin = text.index("    %0 = tt.addptr %Z")
    end = text.index("    %3 = tt.addptr %Q", begin)
    code = text[begin:end]
    text = text[:begin] + text[end:]
    return text.replace("    %stats:2 =", code + "    %stats:2 =", 1)


def fold(text):
    b = int(re.search(r"%ss = arith.mulf.*tensor<(\d+)xf32>", text)[1])
    rows = b // 256
    begin = text.index('    %stats:2 = "tt.reduce"')
    end = text.index("    %rho_18 =", begin)
    body = f"""    %r2_ss = tt.reshape %ss : tensor<{b}xf32> -> tensor<{rows}x256xf32>
    %r2_am = tt.reshape %absu : tensor<{b}xf32> -> tensor<{rows}x256xf32>
    %r2_fold:2 = "tt.reduce"(%r2_ss, %r2_am) <{{axis = 0 : i32}}> ({{
    ^bb0(%sl: f32, %ml: f32, %sr: f32, %mr: f32):
      %s = arith.addf %sl, %sr : f32
      %m = arith.maxnumf %ml, %mr : f32
      tt.reduce.return %s, %m : f32, f32
    }}) : (tensor<{rows}x256xf32>, tensor<{rows}x256xf32>) -> (tensor<256xf32>, tensor<256xf32>)
    %stats:2 = "tt.reduce"(%r2_fold#0, %r2_fold#1) <{{axis = 0 : i32}}> ({{
    ^bb0(%sl: f32, %ml: f32, %sr: f32, %mr: f32):
      %s = arith.addf %sl, %sr : f32
      %m = arith.maxnumf %ml, %mr : f32
      tt.reduce.return %s, %m : f32, f32
    }}) : (tensor<256xf32>, tensor<256xf32>) -> (f32, f32)
"""
    return text[:begin] + body + text[end:]


def main(expanded=False):
    records = []
    output_root = ROOT / "expanded" if expanded else ROOT
    manifest_path = (
        output_root / "manifest.json" if expanded else ROOT.parent / "manifest.json"
    )
    original = json.loads(manifest_path.read_text())
    for row in original["records"]:
        folder = output_root / "ir" / row["group"]
        parent_path = (
            folder / "parent.ttir"
            if expanded
            else ROOT.parent / "ir" / row["group"] / "stream_v1.ttir"
        )
        parent = parent_path.read_text()
        freeze(folder / "parent.ttir", parent)
        variants = {
            "reciprocal": reciprocal(parent),
            "early_z": early_z(parent),
            "reciprocal_early": early_z(reciprocal(parent)),
            "fold256": fold(early_z(reciprocal(parent))),
            "div_approx": parent.replace("tt.precise_divf", "arith.divf"),
        }
        for name, text in variants.items():
            p = folder / f"{name}.ttir"
            freeze(p, text)
            records.append(
                {
                    "group": row["group"],
                    "name": name,
                    "parent_sha256": digest(parent_path),
                    "candidate_sha256": digest(p),
                }
            )
    save(output_root / "candidates.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", action="store_true")
    main(parser.parse_args().expanded)
