"""Round-specific direct TTIR transformations; immutable accepted parent chain."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from complex_suite.support import digest, freeze, save  # noqa: E402


def gain(text, cancel=False):
    b = re.search(r"%q_27 = arith.mulf.*tensor<(\d+)xf32>", text)[1]
    if cancel:
        code = "    %m3_safe = arith.select %scale_23, %stats#1, %cst_1 : f32\n"
        code += "    %m3_gain = tt.precise_divf %cst_2, %m3_safe : f32\n"
    else:
        code = "    %m3_gain = arith.mulf %rho_20, %r2_inv : f32\n"
    code += f"    %m3_vec = tt.splat %m3_gain : f32 -> tensor<{b}xf32>\n"
    text = text.replace("    %q_27 =", code + "    %q_27 =", 1)
    return text.replace("arith.mulf %y_21, %r2_invvec", "arith.mulf %u, %m3_vec")


def statistics_form(text, mode):
    b = re.search(r"%ss = arith.mulf.*tensor<(\d+)xf32>", text)[1]
    start = text.index('    %stats:2 = "tt.reduce"')
    end = text.index("    %rho_18 =", start)
    uint = mode != "separate_float"
    joint = mode == "joint_uint"
    pre = (
        f"    %m4_bits = tt.bitcast %absu : tensor<{b}xf32> -> tensor<{b}xi32>\n"
        if uint
        else ""
    )
    if joint:
        body = f"""    %stats:2 = "tt.reduce"(%ss, %m4_bits) <{{axis = 0 : i32}}> ({{
    ^bb0(%sl: f32, %ml: i32, %sr: f32, %mr: i32):
      %s = arith.addf %sl, %sr : f32
      %m = arith.maxui %ml, %mr : i32
      tt.reduce.return %s, %m : f32, i32
    }}) : (tensor<{b}xf32>, tensor<{b}xi32>) -> (f32, i32)
    %m4_max = tt.bitcast %stats#1 : i32 -> f32
"""
        tail = text[end:].replace("%stats#1", "%m4_max")
    else:
        ty = "i32" if uint else "f32"
        inp = "%m4_bits" if uint else "%absu"
        op = "arith.maxui" if uint else "arith.maxnumf"
        body = f"""    %m4_sum = "tt.reduce"(%ss) <{{axis = 0 : i32}}> ({{
    ^bb0(%sl: f32, %sr: f32):
      %s = arith.addf %sl, %sr : f32
      tt.reduce.return %s : f32
    }}) : (tensor<{b}xf32>) -> f32
    %m4_am = "tt.reduce"({inp}) <{{axis = 0 : i32}}> ({{
    ^bb0(%ml: {ty}, %mr: {ty}):
      %m = {op} %ml, %mr : {ty}
      tt.reduce.return %m : {ty}
    }}) : (tensor<{b}x{ty}>) -> {ty}
"""
        if uint:
            body += "    %m4_max = tt.bitcast %m4_am : i32 -> f32\n"
        tail = (
            text[end:]
            .replace("%stats#0", "%m4_sum")
            .replace("%stats#1", "%m4_max" if uint else "%m4_am")
        )
    return text[:start] + pre + body + tail


def reload_u(text):
    b = re.search(r"%u = arith.mulf.*tensor<(\d+)xf32>", text)[1]
    code = f"""    %m5_x = tt.load %x_7, %mask_4, %cst cacheModifier = ca : tensor<{b}x!tt.ptr<f16>>
    %m5_r = tt.load %r_11, %mask_4, %cst cacheModifier = ca : tensor<{b}x!tt.ptr<f16>>
    %m5_w = tt.load %w_14, %mask_4, %cst cacheModifier = ca : tensor<{b}x!tt.ptr<f16>>
    %m5_xf = arith.extf %m5_x : tensor<{b}xf16> to tensor<{b}xf32>
    %m5_rf = arith.extf %m5_r : tensor<{b}xf16> to tensor<{b}xf32>
    %m5_wf = arith.extf %m5_w : tensor<{b}xf16> to tensor<{b}xf32>
    %m5_z = arith.addf %m5_xf, %m5_rf : tensor<{b}xf32>
    %m5_u = arith.mulf %m5_z, %m5_wf : tensor<{b}xf32>
    %m5_y = arith.mulf %m5_u, %y : tensor<{b}xf32>
"""
    text = text.replace("    %q_27 =", code + "    %q_27 =", 1)
    return text.replace("%q_27 = arith.mulf %u,", "%q_27 = arith.mulf %m5_u,").replace(
        "%q_27 = arith.mulf %y_21,", "%q_27 = arith.mulf %m5_y,"
    )


def streaming(text, n):
    b = int(re.search(r"%q_27 = arith.mulf.*tensor<(\d+)xf32>", text)[1])
    t = min(1024, b)
    lhs, rhs = re.search(r"%q_27 = arith.mulf (%\w+), (%\w+)", text).groups()
    scalar = re.search(re.escape(rhs) + r" = tt.splat (%\w+) : f32", text)[1]
    begin = text.index("    %q_27 =")
    end = text.index("    %6 = tt.addptr %S", begin)
    rho = (
        f"      %m5_rr = tt.splat %rho_20 : f32 -> tensor<{t}xf32>\n"
        f"      %m5_y = arith.mulf %m5_u, %m5_rr : tensor<{t}xf32>\n"
        if lhs == "%y_21"
        else ""
    )
    u = "%m5_y" if rho else "%m5_u"
    body = f"""    %m5_zero = arith.constant 0 : i32
    %m5_n = arith.constant {n} : i32
    %m5_step = arith.constant {t} : i32
    %m5_zeros = arith.constant dense<0.0> : tensor<{t}xf16>
    %m5_maxq = arith.constant dense<127.0> : tensor<{t}xf32>
    %m5_minq = arith.constant dense<-127.0> : tensor<{t}xf32>
    %m5_nvec = tt.splat %m5_n : i32 -> tensor<{t}xi32>
    %m5_range = tt.make_range {{end = {t} : i32, start = 0 : i32}} : tensor<{t}xi32>
    %m5_xp = tt.splat %x_5 : !tt.ptr<f16> -> tensor<{t}x!tt.ptr<f16>>
    %m5_rp = tt.splat %r : !tt.ptr<f16> -> tensor<{t}x!tt.ptr<f16>>
    %m5_wp = tt.splat %W : !tt.ptr<f16> -> tensor<{t}x!tt.ptr<f16>>
    %m5_qbase = tt.addptr %Q, %x : !tt.ptr<i8>, i32
    %m5_qp = tt.splat %m5_qbase : !tt.ptr<i8> -> tensor<{t}x!tt.ptr<i8>>
    %m5_gain = tt.splat {scalar} : f32 -> tensor<{t}xf32>
    scf.for %m5_off = %m5_zero to %m5_n step %m5_step : i32 {{
      %m5_offvec = tt.splat %m5_off : i32 -> tensor<{t}xi32>
      %m5_col = arith.addi %m5_range, %m5_offvec : tensor<{t}xi32>
      %m5_mask = arith.cmpi slt, %m5_col, %m5_nvec : tensor<{t}xi32>
      %m5_px = tt.addptr %m5_xp, %m5_col : tensor<{t}x!tt.ptr<f16>>, tensor<{t}xi32>
      %m5_pr = tt.addptr %m5_rp, %m5_col : tensor<{t}x!tt.ptr<f16>>, tensor<{t}xi32>
      %m5_pw = tt.addptr %m5_wp, %m5_col : tensor<{t}x!tt.ptr<f16>>, tensor<{t}xi32>
      %m5_xv = tt.load %m5_px, %m5_mask, %m5_zeros cacheModifier = ca : tensor<{t}x!tt.ptr<f16>>
      %m5_rv = tt.load %m5_pr, %m5_mask, %m5_zeros cacheModifier = ca : tensor<{t}x!tt.ptr<f16>>
      %m5_wv = tt.load %m5_pw, %m5_mask, %m5_zeros cacheModifier = ca : tensor<{t}x!tt.ptr<f16>>
      %m5_xf = arith.extf %m5_xv : tensor<{t}xf16> to tensor<{t}xf32>
      %m5_rf = arith.extf %m5_rv : tensor<{t}xf16> to tensor<{t}xf32>
      %m5_wf = arith.extf %m5_wv : tensor<{t}xf16> to tensor<{t}xf32>
      %m5_z = arith.addf %m5_xf, %m5_rf : tensor<{t}xf32>
      %m5_u = arith.mulf %m5_z, %m5_wf : tensor<{t}xf32>
{rho}      %m5_q = arith.mulf {u}, %m5_gain : tensor<{t}xf32>
      %m5_qlo = arith.maxnumf %m5_q, %m5_minq : tensor<{t}xf32>
      %m5_qhi = arith.minnumf %m5_qlo, %m5_maxq : tensor<{t}xf32>
      %m5_round = tt.extern_elementwise %m5_qhi
        {{libname = "", libpath = "", pure = true, symbol = "__nv_nearbyintf"}}
        : (tensor<{t}xf32>) -> tensor<{t}xf32>
      %m5_qi = arith.fptosi %m5_round : tensor<{t}xf32> to tensor<{t}xi8>
      %m5_out = tt.addptr %m5_qp, %m5_col : tensor<{t}x!tt.ptr<i8>>, tensor<{t}xi32>
      tt.store %m5_out, %m5_qi, %m5_mask cacheModifier = cs : tensor<{t}x!tt.ptr<i8>>
      scf.yield
    }}
"""
    return text[:begin] + body + text[end:]


def main(round_id):
    manifest = json.loads((ROOT.parent / "round2/expanded/manifest.json").read_text())
    round_root = ROOT / f"r{round_id}"
    records = []
    for item in manifest["records"]:
        group = item["group"]
        anchor = ROOT.parent / "round2/expanded/ir" / group / "optimized.ttir"
        parent = (
            anchor
            if round_id == 3
            else ROOT / f"r{round_id-1}/ir" / group / "optimized.ttir"
        )
        text = parent.read_text()
        freeze(round_root / "ir" / group / "parent.ttir", text)
        if round_id == 3:
            variants = {"gain_reassociate": gain(text), "gain_cancel": gain(text, True)}
        elif round_id == 4:
            variants = {
                name: statistics_form(text, name)
                for name in ("joint_uint", "separate_uint", "separate_float")
            }
        else:
            stream = streaming(text, item["shape"][1])
            variants = {
                "reload_u": reload_u(text),
                "stream1024": stream,
                "stream1024_large": (
                    stream
                    if item["shape"][0] >= 512 and item["shape"][1] >= 4096
                    else text
                ),
            }
        for name, value in variants.items():
            p = round_root / "ir" / group / f"{name}.ttir"
            freeze(p, value)
            records.append(
                {
                    "group": group,
                    "name": name,
                    "sha256": digest(p),
                    "parent_sha256": digest(parent),
                    "anchor_sha256": digest(anchor),
                }
            )
    save(round_root / "candidates.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=[3, 4, 5], required=True)
    main(parser.parse_args().round)
