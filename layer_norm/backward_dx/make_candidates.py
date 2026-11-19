"""Direct edits of frozen dX TTIR: joint reduction, reciprocal and output transforms."""

import re

from common import BLOCKS, ROOT, digest, save
from grouped_ir import grouped


def without_locations(text):
    text = (
        "\n".join(line for line in text.splitlines() if not line.startswith("#loc"))
        + "\n"
    )
    while " loc(" in text:
        start = text.index(" loc(")
        end = start + 5
        depth = 1
        while depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        text = text[:start] + text[end:]
    return text


def reciprocal(text):
    old = "    %c1_24 = arith.divf %c1_22, %c1_23 : f32"
    assert old in text
    return text.replace(
        old,
        "    %one = arith.constant 1.0 : f32\n"
        "    %inv = arith.divf %one, %c1_23 : f32\n"
        "    %c1_24 = arith.mulf %c1_22, %inv : f32",
    ).replace("%c2_25 = arith.divf %c2, %c1_23", "%c2_25 = arith.mulf %c2, %inv")


def joint(text, block):
    first = text.index('    %c1_22 = "tt.reduce"')
    cast = text.index("    %c1_23 =", first)
    second = text.index('    %c2 = "tt.reduce"', cast)
    end = text.index("    %c2_25 =", second)
    pair = f"""    %pair:2 = "tt.reduce"(%c1, %wdy_21) <{{axis = 0 : i32}}> ({{
    ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
      %s0 = arith.addf %a0, %b0 : f32
      %s1 = arith.addf %a1, %b1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }}) : (tensor<{block}xf32>, tensor<{block}xf32>) -> (f32, f32)
"""
    result = text[:first] + pair + text[cast:second] + text[end:]
    result = re.sub(r"%c1_22\b", "%pair#0", result)
    result = re.sub(r"%c2\b", "%pair#1", result)
    return result


def coefficients(text, block):
    begin = text.index("    %dx = tt.splat")
    end = text.index("    %0 = tt.splat", begin)
    ty = f"tensor<{block}xf32>"
    body = f"""    %alpha = arith.mulf %c1_24, %rstd_16 : f32
    %beta = arith.mulf %c2_25, %rstd_16 : f32
    %alphas = tt.splat %alpha : f32 -> {ty}
    %betas = tt.splat %beta : f32 -> {ty}
    %scaled = arith.mulf %wdy_21, %xhat_18 : {ty}
    %center = arith.subf %scaled, %betas : {ty}
    %term = arith.mulf %xhat_20, %alphas : {ty}
    %dx_30 = arith.subf %center, %term : {ty}
"""
    return text[:begin] + body + text[end:]


def stream(text):
    return text.replace(
        "tt.store %1, %2, %mask_1 :", "tt.store %1, %2, %mask_1 cacheModifier = cs :"
    )


def main():
    manifest = {"families": {}}
    for block in BLOCKS:
        base = without_locations((ROOT / f"ir/b{block}/baseline.ttir").read_text())
        variants = {
            "canonical": base,
            "reciprocal": reciprocal(base),
            "joint": joint(base, block),
        }
        variants["joint_reciprocal"] = joint(variants["reciprocal"], block)
        variants["joint_coefficients"] = coefficients(
            variants["joint_reciprocal"], block
        )
        variants["joint_stream"] = stream(variants["joint_reciprocal"])
        variants["coefficients_stream"] = stream(variants["joint_coefficients"])
        variants["grouped2048"] = grouped(base, block)
        manifest["families"][str(block)] = {}
        for name, text in variants.items():
            path = ROOT / f"ir/b{block}/{name}.ttir"
            if path.exists():
                assert path.read_text() == text, name
            else:
                path.write_text(text)
            manifest["families"][str(block)][name] = digest(path)
    save(ROOT / "results/candidates.json", manifest)


if __name__ == "__main__":
    main()
