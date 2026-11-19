"""Third-round direct TTIR algebra/layout candidates; no source kernel or downstream edits."""

from common import BLOCKS, ROOT, digest, save


def centered(text, block, raw=False):
    ty = f"tensor<{block}xf32>"
    # Keep existing xhat definitions until downstream DCE; the new graph removes their uses.
    text = text.replace(
        "%c1 = arith.mulf %xhat_20, %wdy_21",
        f'%c1 = arith.mulf {"%x_8" if raw else "%masked_center"}, %wdy_21',
    )
    insert = f"    %masked_center = arith.select %mask_1, %xhat_17, %cst_0 : tensor<{block}xi1>, {ty}\n"
    text = text.replace("    %c1 =", insert + "    %c1 =", 1)
    begin = text.index("    %alpha =")
    end = text.index("    %0 = tt.splat", begin)
    extra = ""
    c1 = "%c1_24"
    if raw:
        extra = (
            "    %minusone = arith.constant -1.0 : f32\n"
            "    %negmean = arith.mulf %mean_15, %minusone : f32\n"
            "    %centered_c1 = math.fma %negmean, %c2_25, %c1_24 : f32\n"
        )
        c1 = "%centered_c1"
    output = f"""{extra}    %rsq = arith.mulf %rstd_16, %rstd_16 : f32
    %rcube = arith.mulf %rsq, %rstd_16 : f32
    %alpha = arith.mulf {c1}, %rcube : f32
    %beta = arith.mulf %c2_25, %rstd_16 : f32
    %alphas = tt.splat %alpha : f32 -> {ty}
    %betas = tt.splat %beta : f32 -> {ty}
    %scaled = arith.mulf %wdy_21, %xhat_18 : {ty}
    %center = arith.subf %scaled, %betas : {ty}
    %term = arith.mulf %masked_center, %alphas : {ty}
    %dx_30 = arith.subf %center, %term : {ty}
"""
    return text[:begin] + output + text[end:]


def joined(text, block):
    begin = text.index("    %pair:2 =")
    end = text.index("    %c1_23 =", begin)
    body = f"""    %stacked = tt.join %c1, %wdy_21 : tensor<{block}xf32> -> tensor<{block}x2xf32>
    %sums = "tt.reduce"(%stacked) <{{axis = 0 : i32}}> ({{
    ^bb0(%l: f32, %r: f32):
      %s = arith.addf %l, %r : f32
      tt.reduce.return %s : f32
    }}) : (tensor<{block}x2xf32>) -> tensor<2xf32>
    %shaped = tt.reshape %sums : tensor<2xf32> -> tensor<1x2xf32>
    %split:2 = tt.split %shaped : tensor<1x2xf32> -> tensor<1xf32>
    %scalar0 = "tt.reduce"(%split#0) <{{axis = 0 : i32}}> ({{
    ^bb0(%l: f32, %r: f32):
      %s = arith.addf %l, %r : f32
      tt.reduce.return %s : f32
    }}) : (tensor<1xf32>) -> f32
    %scalar1 = "tt.reduce"(%split#1) <{{axis = 0 : i32}}> ({{
    ^bb0(%l: f32, %r: f32):
      %s = arith.addf %l, %r : f32
      tt.reduce.return %s : f32
    }}) : (tensor<1xf32>) -> f32
"""
    result = text[:begin] + body + text[end:]
    return result.replace("%pair#0", "%scalar0").replace("%pair#1", "%scalar1")


def main():
    manifest = {}
    for block in BLOCKS:
        base = (ROOT / f"ir/b{block}/joint_coefficients.ttir").read_text()
        variants = {
            "r3_centered": centered(base, block),
            "r3_rawmoment": centered(base, block, True),
            "r3_joined": joined(base, block),
        }
        variants["r3_centered_stream"] = variants["r3_centered"].replace(
            "tt.store %1, %2, %mask_1 :",
            "tt.store %1, %2, %mask_1 cacheModifier = cs :",
        )
        manifest[str(block)] = {}
        for name, text in variants.items():
            p = ROOT / f"ir/b{block}/{name}.ttir"
            if p.exists():
                assert p.read_text() == text
            else:
                p.write_text(text)
            manifest[str(block)][name] = digest(p)
    save(ROOT / "results/round3/candidates.json", manifest)


if __name__ == "__main__":
    main()
