"""Eliminate redundant masked intermediates while keeping masked memory accesses."""

from common import BLOCKS, ROOT, digest, save


def maskfold(text, block):
    # Inactive W and DY become zero, making the inactive product exactly zero.
    text = text.replace("tt.load %w_12, %mask_1 :", "tt.load %w_12, %mask_1, %cst :")
    text = text.replace(
        f"    %xhat_20 = arith.select %mask_1, %xhat_19, %cst_0 : tensor<{block}xi1>, tensor<{block}xf32>\n",
        "",
    )
    text = text.replace(
        f"    %wdy_21 = arith.select %mask_1, %wdy, %cst_0 : tensor<{block}xi1>, tensor<{block}xf32>\n",
        "",
    )
    text = text.replace(
        f"    %masked_center = arith.select %mask_1, %xhat_17, %cst_0 : tensor<{block}xi1>, tensor<{block}xf32>\n",
        "",
    )
    return (
        text.replace("%xhat_20", "%xhat_19")
        .replace("%wdy_21", "%wdy")
        .replace("%masked_center", "%xhat_17")
    )


def main():
    manifest = {}
    for block in BLOCKS:
        for name, source in [
            ("r4_maskfold", "joint_coefficients"),
            ("r4_center_maskfold", "r3_centered_stream"),
        ]:
            text = maskfold((ROOT / f"ir/b{block}/{source}.ttir").read_text(), block)
            path = ROOT / f"ir/b{block}/{name}.ttir"
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
            manifest[f"{name}/b{block}"] = digest(path)
    save(ROOT / "results/round4/candidates_batch3.json", manifest)


if __name__ == "__main__":
    main()
