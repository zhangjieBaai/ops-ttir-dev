"""Quantization conversion and four-way statistics from frozen r5 TTIR."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from complex_suite.support import digest, freeze, save  # noqa: E402


def convert(text, integer_clamp=False):
    b = int(re.search(r"%q_27 = arith.mulf.*tensor<(\d+)xf32>", text)[1])
    start = text.index("    %q_28 =")
    end = text.index("    %3 = tt.addptr %Q", start)
    if integer_clamp:
        prefix = ""
        arg = "%q_27"
    else:
        prefix = text[start : text.index("    %q_30 =", start)]
        arg = "%q_29"
    code = (
        f"    %r6_round = tt.extern_elementwise {arg} "
        '{libname = "", libpath = "", pure = true, symbol = "__nv_float2int_rn"} '
        f": (tensor<{b}xf32>) -> tensor<{b}xi32>\n"
    )
    result = "%r6_round"
    if integer_clamp:
        code += f"""    %r6_lo = arith.constant dense<-127> : tensor<{b}xi32>
    %r6_hi = arith.constant dense<127> : tensor<{b}xi32>
    %r6_lower = arith.maxsi %r6_round, %r6_lo : tensor<{b}xi32>
    %r6_upper = arith.minsi %r6_lower, %r6_hi : tensor<{b}xi32>
"""
        result = "%r6_upper"
    code += f"    %q_31 = arith.trunci {result} : tensor<{b}xi32> to tensor<{b}xi8>\n"
    return text[:start] + prefix + code + text[end:]


def fold(text):
    b = int(re.search(r"%ss = arith.mulf.*tensor<(\d+)xf32>", text)[1])
    start = text.index('    %stats:2 = "tt.reduce"')
    end = text.index("    %rho_18 =", start)
    t = b // 4
    body = f"""    %r6_ss = tt.reshape %ss : tensor<{b}xf32> -> tensor<4x{t}xf32>
    %r6_abs = tt.reshape %absu : tensor<{b}xf32> -> tensor<4x{t}xf32>
    %r6_part:2 = "tt.reduce"(%r6_ss, %r6_abs) <{{axis = 1 : i32}}> ({{
    ^bb0(%r6_sl: f32, %r6_ml: f32, %r6_sr: f32, %r6_mr: f32):
      %r6_sv = arith.addf %r6_sl, %r6_sr : f32
      %r6_mv = arith.maxnumf %r6_ml, %r6_mr : f32
      tt.reduce.return %r6_sv, %r6_mv : f32, f32
    }}) : (tensor<4x{t}xf32>, tensor<4x{t}xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    %stats:2 = "tt.reduce"(%r6_part#0, %r6_part#1) <{{axis = 0 : i32}}> ({{
    ^bb0(%r6_gl: f32, %r6_al: f32, %r6_gr: f32, %r6_ar: f32):
      %r6_gv = arith.addf %r6_gl, %r6_gr : f32
      %r6_av = arith.maxnumf %r6_al, %r6_ar : f32
      tt.reduce.return %r6_gv, %r6_av : f32, f32
    }}) : (tensor<4xf32>, tensor<4xf32>) -> (f32, f32)
"""
    return text[:start] + body + text[end:]


def main():
    records = []
    manifest = json.loads((ROOT.parent / "round2/expanded/manifest.json").read_text())
    for item in manifest["records"]:
        group = item["group"]
        parent = ROOT / "r5/ir" / group / "optimized.ttir"
        text = parent.read_text()
        folder = ROOT / "r6/ir" / group
        freeze(folder / "parent.ttir", text)
        variants = {
            "convert_rn": convert(text),
            "integer_clamp": convert(text, True),
            "fold4": fold(text),
        }
        for name, value in variants.items():
            path = folder / f"{name}.ttir"
            freeze(path, value)
            records.append(
                {
                    "group": group,
                    "name": name,
                    "sha256": digest(path),
                    "parent_sha256": digest(parent),
                }
            )
    save(ROOT / "r6/candidates.json", records)


if __name__ == "__main__":
    main()
