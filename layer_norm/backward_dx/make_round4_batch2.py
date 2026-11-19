"""Conservative layout hints, cache eviction and centered affine output TTIR."""

import re

from common import BLOCKS, ROOT, digest, save


def affine(text, block):
    begin = text.index("    %alpha =")
    end = text.index("    %0 = tt.splat", begin)
    ty = f"tensor<{block}xf32>"
    body = f"""    %squared_r = arith.mulf %rstd_16, %rstd_16 : f32
    %slope = arith.mulf %c1_24, %squared_r : f32
    %minusone = arith.constant -1.0 : f32
    %negative_slope = arith.mulf %slope, %minusone : f32
    %scaled_c2 = arith.mulf %c2_25, %rstd_16 : f32
    %negative_c2 = arith.mulf %scaled_c2, %minusone : f32
    %intercept = math.fma %mean_15, %slope, %negative_c2 : f32
    %slopes = tt.splat %negative_slope : f32 -> {ty}
    %intercepts = tt.splat %intercept : f32 -> {ty}
    %shifted = math.fma %x_8, %slopes, %intercepts : {ty}
    %dx_30 = math.fma %wdy_21, %xhat_18, %shifted : {ty}
"""
    return text[:begin] + body + text[end:]


def main():
    manifest = {}
    for block in BLOCKS:
        parent = (ROOT / f"ir/b{block}/r4_parent.ttir").read_text()
        pair = (ROOT / f"ir/b{block}/joint_coefficients.ttir").read_text()
        evict = re.sub(
            r"(tt.load [^\n]+)( : tensor<)",
            r"\1 evictionPolicy = evict_first\2",
            parent,
        )
        # Keep W reusable in L1 while streaming X/DY.
        evict = evict.replace(
            "%w_12, %mask_1 evictionPolicy = evict_first",
            "%w_12, %mask_1 evictionPolicy = evict_last",
        )
        variants = {"r4_evict": evict, "r4_affine": affine(pair, block)}
        for width in [2, 4]:
            value = f"start = 0 : i32, tt.contiguity = dense<{width}> : tensor<1xi32>"
            variants[f"r4_contig{width}"] = parent.replace("start = 0 : i32", value)
        for name, text in variants.items():
            path = ROOT / f"ir/b{block}/{name}.ttir"
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
            manifest[f"{name}/b{block}"] = digest(path)
    save(ROOT / "results/round4/candidates_batch2.json", manifest)


if __name__ == "__main__":
    main()
