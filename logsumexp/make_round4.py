"""Direct TTIR row grouping and logical-axis experiments under a fixed launch."""

import re

from common import ROOT, ir_path, manifest, save_json, sha256
from make_candidates import narrow_indices
from make_round3 import header, reduce


def row_group(text, config, rows, max_n):
    meta = config["meta"]
    if "TILE_K" in meta or not meta["ONE_TILE_PER_CTA"]:
        return text
    width = meta["TILE_N"]
    if width > max_n or "%M:" not in text:
        return text

    def t(ty):
        return f"tensor<{rows}x{width}x{ty}>"

    def rt(ty):
        return f"tensor<{rows}x{ty}>"

    def row(ty):
        return f"tensor<{rows}x1x{ty}>"

    def col(ty):
        return f"tensor<1x{width}x{ty}>"

    body = f"""    %r4_r = arith.constant {rows} : i32
    %r4_zero_i = arith.constant 0 : i32
    %r4_pid = tt.get_program_id x : i32
    %r4_remainder = arith.remui %r4_pid, %r4_r : i32
    %r4_active = arith.cmpi eq, %r4_remainder, %r4_zero_i : i32
    scf.if %r4_active {{
    %r4_neg = arith.constant dense<0xFC00> : {t('f16')}
    %r4_inf = arith.constant dense<0xFF800000> : {rt('f32')}
    %r4_zero = arith.constant dense<0.000000e+00> : {rt('f32')}
    %r4_rows = tt.make_range {{end = {rows} : i32, start = 0 : i32}} : {rt('i32')}
    %r4_pidv = tt.splat %r4_pid : i32 -> {rt('i32')}
    %r4_mids = arith.addi %r4_pidv, %r4_rows : {rt('i32')}
    %r4_mv = tt.splat %M : i32 -> {rt('i32')}
    %r4_mmask = arith.cmpi slt, %r4_mids, %r4_mv : {rt('i32')}
    %r4_mexpand = tt.expand_dims %r4_mids {{axis = 1 : i32}} : {rt('i32')} -> {row('i32')}
    %r4_mmaskex = tt.expand_dims %r4_mmask {{axis = 1 : i32}} : {rt('i1')} -> {row('i1')}
    %r4_nv = tt.splat %N : i32 -> {row('i32')}
    %r4_base = arith.muli %r4_mexpand, %r4_nv : {row('i32')}
    %r4_cols = tt.make_range {{end = {width} : i32, start = 0 : i32}} : tensor<{width}xi32>
    %r4_cex = tt.expand_dims %r4_cols {{axis = 0 : i32}} : tensor<{width}xi32> -> {col('i32')}
    %r4_ncols = tt.splat %N : i32 -> {col('i32')}
    %r4_nmask = arith.cmpi slt, %r4_cex, %r4_ncols : {col('i32')}
    %r4_baseb = tt.broadcast %r4_base : {row('i32')} -> {t('i32')}
    %r4_cb = tt.broadcast %r4_cex : {col('i32')} -> {t('i32')}
    %r4_offsets = arith.addi %r4_baseb, %r4_cb : {t('i32')}
    %r4_mb = tt.broadcast %r4_mmaskex : {row('i1')} -> {t('i1')}
    %r4_nb = tt.broadcast %r4_nmask : {col('i1')} -> {t('i1')}
    %r4_mask = arith.andi %r4_mb, %r4_nb : {t('i1')}
    %r4_ptr = tt.splat %input_ptr : !tt.ptr<f16> -> {t('!tt.ptr<f16>')}
    %r4_ptrs = tt.addptr %r4_ptr, %r4_offsets : {t('!tt.ptr<f16>')}, {t('i32')}
    %r4_x16 = tt.load %r4_ptrs, %r4_mask, %r4_neg : {t('!tt.ptr<f16>')}
    %r4_x = arith.extf %r4_x16 : {t('f16')} to {t('f32')}
"""
    body += reduce("%r4_x", t("f32"), rt("f32"), "r4_max", "maxnumf", axis=1)
    body += f"""    %r4_allinf = arith.cmpf oeq, %r4_max, %r4_inf : {rt('f32')}
    %r4_safe = arith.select %r4_allinf, %r4_zero, %r4_max : {rt('i1')}, {rt('f32')}
    %r4_maxex = tt.expand_dims %r4_safe {{axis = 1 : i32}} : {rt('f32')} -> {row('f32')}
    %r4_maxb = tt.broadcast %r4_maxex : {row('f32')} -> {t('f32')}
    %r4_delta = arith.subf %r4_x, %r4_maxb : {t('f32')}
    %r4_exp = math.exp %r4_delta : {t('f32')}
"""
    body += reduce("%r4_exp", t("f32"), rt("f32"), "r4_sum", "addf", axis=1)
    body += f"""    %r4_log = math.log %r4_sum : {rt('f32')}
    %r4_value = arith.addf %r4_safe, %r4_log : {rt('f32')}
    %r4_out = arith.select %r4_allinf, %r4_max, %r4_value : {rt('i1')}, {rt('f32')}
    %r4_y16 = arith.truncf %r4_out : {rt('f32')} to {rt('f16')}
    %r4_yptr = tt.splat %output_ptr : !tt.ptr<f16> -> {rt('!tt.ptr<f16>')}
    %r4_yptrs = tt.addptr %r4_yptr, %r4_mids : {rt('!tt.ptr<f16>')}, {rt('i32')}
    tt.store %r4_yptrs, %r4_y16, %r4_mmask : {rt('!tt.ptr<f16>')}
    }}
    tt.return
  }}
}}
"""
    return header(text) + body


def transpose_noninner(text, config):
    if "TILE_K" not in config["meta"]:
        return text
    text = re.sub(r"tensor<(\d+)x(\d+)x", lambda m: f"tensor<{m[2]}x{m[1]}x", text)
    return re.sub(r"axis = ([01]) : i32", lambda m: f"axis = {1-int(m[1])} : i32", text)


def main():
    variants = {
        "r4_rows2": lambda t, c: row_group(t, c, 2, 1024),
        "r4_rows4": lambda t, c: row_group(t, c, 4, 1024),
        "r4_rows8": lambda t, c: row_group(t, c, 8, 1024),
        "r4_rows4wide": lambda t, c: row_group(t, c, 4, 8192),
        "r4_transpose": transpose_noninner,
        "r4_i32non": lambda t, c: narrow_indices(t) if "TILE_K" in c["meta"] else t,
    }
    output = dict(
        generator_sha256=sha256(__file__),
        round_protocol_sha256=sha256(ROOT / "optimization_plan_round4.md"),
        groups={},
    )
    for group, config in manifest()["groups"].items():
        parent = ir_path(group, "r3_mix").read_text()
        output["groups"][group] = {}
        for name, fn in variants.items():
            path = ir_path(group, name)
            with path.open("x") as stream:
                stream.write(fn(parent, config))
            output["groups"][group][name] = dict(
                sha256=sha256(path), parent_sha256=sha256(ir_path(group, "r3_mix"))
            )
    save_json(ROOT / "results/runs/r4_setup/candidates.json", output)


if __name__ == "__main__":
    main()
