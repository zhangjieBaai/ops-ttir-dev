"""Direct TTIR online-recurrence rewrites under the unchanged round-one ABI."""

import re

from common import ROOT, ir_path, manifest, save_json, sha256


def single_exp(text, form="fma"):
    if "scf.for" not in text:
        return text
    pattern = re.compile(
        r"(?P<indent> +)%m_new = arith.maxnumf %m, (?P<x>%\w+) : (?P<t>tensor<[^>]+>)\n"
        r"(?P<body>.*?)"
        r"(?P=indent)(?P<result>%z_\d+) = arith.select %all_neg_inf, (?P<z>%z_\d+), %z_\d+ : [^\n]+\n",
        re.S,
    )
    match = pattern.search(text)
    assert match and text.count("%m_new =") == 1
    indent, x, dtype, result, z = (
        match[g] for g in ("indent", "x", "t", "result", "z")
    )
    booltype = dtype.replace("f32", "i1")
    guard = match["body"].splitlines()[0].strip()
    lines = [
        f"%m_new = arith.maxnumf %m, {x} : {dtype}",
        guard,
        f"%r2_one = arith.constant dense<1.000000e+00> : {dtype}",
        f"%r2_zero = arith.constant dense<0.000000e+00> : {dtype}",
        f"%r2_delta = arith.subf {x}, %m : {dtype}",
        f"%r2_abs = math.absf %r2_delta : {dtype}",
        f"%r2_neg = arith.subf %r2_zero, %r2_abs : {dtype}",
        f"%r2_ratio = math.exp %r2_neg : {dtype}",
        f"%r2_newmax = arith.cmpf ogt, {x}, %m : {dtype}",
    ]
    if form == "fma":
        lines += [
            f"%r2_scale = arith.select %r2_newmax, %r2_ratio, %r2_one : {booltype}, {dtype}",
            f"%r2_add = arith.select %r2_newmax, %r2_one, %r2_ratio : {booltype}, {dtype}",
            f"%r2_mul = arith.mulf {z}, %r2_scale : {dtype}",
            f"%r2_sum = arith.addf %r2_mul, %r2_add : {dtype}",
        ]
    else:
        lines += [
            f"%r2_mul = arith.mulf {z}, %r2_ratio : {dtype}",
            f"%r2_high = arith.addf %r2_mul, %r2_one : {dtype}",
            f"%r2_low = arith.addf {z}, %r2_ratio : {dtype}",
            f"%r2_sum = arith.select %r2_newmax, %r2_high, %r2_low : {booltype}, {dtype}",
        ]
    lines += [
        f"{result} = arith.select %all_neg_inf, {z}, %r2_sum : {booltype}, {dtype}"
    ]
    return (
        text[: match.start()]
        + "\n".join(indent + line for line in lines)
        + "\n"
        + text[match.end() :]
    )


def widen(text, config, width):
    old = config["meta"]["TILE_N"]
    if "scf.for" not in text or config["meta"].get("TILE_K") != 64:
        return text
    text = text.replace(f"tensor<{old}x", f"tensor<{width}x")
    text = text.replace(f"end = {old} :", f"end = {width} :")
    return text.replace(
        f"%c{old}_i32 = arith.constant {old} :",
        f"%c{old}_i32 = arith.constant {width} :",
    )


def unroll(text):
    if "scf.for" not in text:
        return text
    return text.replace(
        "    }\n    %m_reduced",
        "    } {tt.loop_unroll_factor = 4 : i32}\n    %m_reduced",
    )


def main():
    variants = {
        "r2_single_exp": lambda t, c: single_exp(t),
        "r2_select_after": lambda t, c: single_exp(t, "after"),
        "r2_wide256": lambda t, c: widen(single_exp(t), c, 256),
        "r2_wide512": lambda t, c: widen(single_exp(t), c, 512),
        "r2_unroll4": lambda t, c: unroll(single_exp(t)),
    }
    result = dict(
        protocol_sha256=sha256(ROOT / "optimization_plan_round2.md"),
        generator_sha256=sha256(__file__),
        groups={},
    )
    for group, config in manifest()["groups"].items():
        parent = ir_path(group, "r1_trim").read_text()
        result["groups"][group] = {}
        for name, transform in variants.items():
            text = transform(parent, config)
            path = ir_path(group, name)
            with path.open("x") as stream:
                stream.write(text)
            result["groups"][group][name] = dict(
                sha256=sha256(path), parent_sha256=sha256(ir_path(group, "r1_trim"))
            )
    save_json(ROOT / "results/runs/r2_setup/candidates.json", result)


if __name__ == "__main__":
    main()
