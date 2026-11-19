"""Round-three deterministic TTIR-only tensor/loop specialization experiments."""

import re

from common import ROOT, ir_path, manifest, save_json, sha256


def header(text):
    return "\n".join(text.splitlines()[:2]) + "\n"


def reduce(value, input_type, output_type, result, op, axis=0):
    return (
        f'    %{result} = "tt.reduce"({value}) <{{axis = {axis} : i32}}> ({{\n'
        f"    ^bb0(%{result}_a: f32, %{result}_b: f32):\n"
        f"      %{result}_v = arith.{op} %{result}_a, %{result}_b : f32\n"
        f"      tt.reduce.return %{result}_v : f32\n"
        f"    }}) : ({input_type}) -> {output_type}\n"
    )


def full_inner_body(width):
    def t(ty):
        return f"tensor<{width}x{ty}>"

    text = f"""    %r3_neg = arith.constant dense<0xFC00> : {t('f16')}
    %r3_inf = arith.constant 0xFF800000 : f32
    %r3_zero = arith.constant 0.000000e+00 : f32
    %r3_pid = tt.get_program_id x : i32
    %r3_base = arith.muli %r3_pid, %N : i32
    %r3_cols = tt.make_range {{end = {width} : i32, start = 0 : i32}} : {t('i32')}
    %r3_row = tt.splat %r3_base : i32 -> {t('i32')}
    %r3_offsets = arith.addi %r3_row, %r3_cols : {t('i32')}
    %r3_n = tt.splat %N : i32 -> {t('i32')}
    %r3_mask = arith.cmpi slt, %r3_cols, %r3_n : {t('i32')}
    %r3_ptr = tt.splat %input_ptr : !tt.ptr<f16> -> {t('!tt.ptr<f16>')}
    %r3_ptrs = tt.addptr %r3_ptr, %r3_offsets : {t('!tt.ptr<f16>')}, {t('i32')}
    %r3_x16 = tt.load %r3_ptrs, %r3_mask, %r3_neg : {t('!tt.ptr<f16>')}
    %r3_x = arith.extf %r3_x16 : {t('f16')} to {t('f32')}
"""
    text += reduce("%r3_x", t("f32"), "f32", "r3_max", "maxnumf")
    text += f"""    %r3_allinf = arith.cmpf oeq, %r3_max, %r3_inf : f32
    %r3_safe = arith.select %r3_allinf, %r3_zero, %r3_max : f32
    %r3_bmax = tt.splat %r3_safe : f32 -> {t('f32')}
    %r3_delta = arith.subf %r3_x, %r3_bmax : {t('f32')}
    %r3_exp = math.exp %r3_delta : {t('f32')}
"""
    text += reduce("%r3_exp", t("f32"), "f32", "r3_sum", "addf")
    text += """    %r3_log = math.log %r3_sum : f32
    %r3_value = arith.addf %r3_safe, %r3_log : f32
    %r3_out = arith.select %r3_allinf, %r3_max, %r3_value : f32
    %r3_out16 = arith.truncf %r3_out : f32 to f16
    %r3_yptr = tt.addptr %output_ptr, %r3_pid : !tt.ptr<f16>, i32
    tt.store %r3_yptr, %r3_out16 : !tt.ptr<f16>
"""
    return text


def full_inner(text, config):
    if "TILE_K" in config["meta"] or config["meta"]["ONE_TILE_PER_CTA"]:
        return text
    return (
        header(text) + "    %r3_limit = arith.constant 131072 : i32\n"
        "    %r3_fits = arith.cmpi sle, %N, %r3_limit : i32\n"
        "    scf.if %r3_fits {\n"
        + full_inner_body(131072)
        + "    } else {\n"
        + "".join(text.splitlines(keepends=True)[2:-2]).replace("    tt.return\n", "")
        + "    }\n    tt.return\n  }\n}\n"
    )


def full_inner65536(text, config):
    if "TILE_K" in config["meta"] or config["meta"]["ONE_TILE_PER_CTA"]:
        return text
    return (
        header(text) + "    %r3_limit = arith.constant 65536 : i32\n"
        "    %r3_fits = arith.cmpi sle, %N, %r3_limit : i32\n"
        "    scf.if %r3_fits {\n"
        + full_inner_body(65536)
        + "    } else {\n"
        + "".join(text.splitlines(keepends=True)[2:-2]).replace("    tt.return\n", "")
        + "    }\n    tt.return\n  }\n}\n"
    )


def narrow_noninner(text, config, width):
    old = config["meta"]["TILE_N"]
    if "scf.for" not in text or config["meta"].get("TILE_K") != 64:
        return text
    text = text.replace(f"tensor<{old}x", f"tensor<{width}x").replace(
        f"end = {old} :", f"end = {width} :"
    )
    return text.replace(
        f"%c{old}_i32 = arith.constant {old} :",
        f"%c{old}_i32 = arith.constant {width} :",
    )


def cache_load(text):
    return re.sub(r"(= tt.load [^\n]*?)( : tensor)", r"\1 cacheModifier = cg\2", text)


def split_reduce(text, config):
    if "TILE_K" in config["meta"] or not config["meta"]["ONE_TILE_PER_CTA"]:
        return text
    width = config["meta"]["TILE_N"]
    if width < 4096:
        return text
    # Group contiguous halves, first reduce lane-local pairs along axis0.
    pattern = re.compile(
        r'    (%\w+) = "tt.reduce"\((%\w+)\) <\{axis = 0 : i32\}> \(\{\n(.*?)\n    \}\) : \(tensor<'
        + str(width)
        + r"xf32>\) -> f32",
        re.S,
    )
    index = 0

    def replace(match):
        nonlocal index
        index += 1
        result, value, region = match.groups()
        op = "maxnumf" if "maxnumf" in region else "addf"
        part = f"r3_fold{index}"
        return (
            f"    %{part}_shape = tt.reshape {value} : tensor<{width}xf32> -> tensor<2x{width//2}xf32>\n"
            + reduce(
                "%" + part + "_shape",
                f"tensor<2x{width//2}xf32>",
                f"tensor<{width//2}xf32>",
                part,
                op,
            )
            + f'    {result} = "tt.reduce"(%{part}) <{{axis = 0 : i32}}> ({{\n{region}\n'
            + f"    }}) : (tensor<{width//2}xf32>) -> f32"
        )

    output, count = pattern.subn(replace, text)
    assert count == 2
    return output


def main():
    variants = {
        "r3_full64": full_inner65536,
        "r3_full128": full_inner,
        "r3_narrow32": lambda t, c: narrow_noninner(t, c, 32),
        "r3_narrow64": lambda t, c: narrow_noninner(t, c, 64),
        "r3_cachecg": lambda t, c: cache_load(t),
        "r3_fold": split_reduce,
    }
    result = dict(
        protocol_sha256=sha256(ROOT / "optimization_plan_round3.md"),
        generator_sha256=sha256(__file__),
        groups={},
    )
    for group, config in manifest()["groups"].items():
        parent = ir_path(group, "r2_select_after").read_text()
        result["groups"][group] = {}
        for name, fn in variants.items():
            path = ir_path(group, name)
            with path.open("x") as stream:
                stream.write(fn(parent, config))
            result["groups"][group][name] = dict(
                sha256=sha256(path),
                parent_sha256=sha256(ir_path(group, "r2_select_after")),
            )
    save_json(ROOT / "results/runs/r3_setup/candidates.json", result)


if __name__ == "__main__":
    main()
