"""Keep original FP16 values live across maximum reduction, then widen for exp."""

import re

from common import ROOT, ir_path, manifest, save_json, sha256


def half_max(text):
    # Only one-tile inner bodies have scalar max results; non-inner untouched.
    if "@logsumexp_kernel_inner" not in text:
        return text
    pattern = re.compile(
        r"(?P<cast>    (?P<x32>%\w+) = arith.extf (?P<x16>%\w+) : "
        r"(?P<half>tensor<\d+xf16>) to (?P<wide>tensor<\d+xf32>)\n)"
        r'    (?P<max>%\w+) = "tt.reduce"\((?P=x32)\) <\{axis = 0 : i32\}> \(\{\n'
        r"(?P<region>.*?)\n    \}\) : \((?P=wide)\) -> f32\n",
        re.S,
    )

    def replace(m):
        region = m["region"].replace("f32", "f16")
        value = m["max"]
        return (
            f'    {value}_half = "tt.reduce"({m["x16"]}) <{{axis = 0 : i32}}> ({{\n'
            f'{region}\n    }}) : ({m["half"]}) -> f16\n'
            f"    {value} = arith.extf {value}_half : f16 to f32\n" + m["cast"]
        )

    return pattern.sub(replace, text)


def main():
    result = {}
    for group in manifest()["groups"]:
        result[group] = {}
        for name, parent in [
            ("r3_halfmax", "r2_select_after"),
            ("r3_full64_half", "r3_full64"),
        ]:
            path = ir_path(group, name)
            with path.open("x") as stream:
                stream.write(half_max(ir_path(group, parent).read_text()))
            result[group][name] = dict(
                sha256=sha256(path),
                parent=parent,
                parent_sha256=sha256(ir_path(group, parent)),
            )
    save_json(
        ROOT / "results/runs/r3_setup/candidates_half.json",
        dict(generator_sha256=sha256(__file__), groups=result),
    )


if __name__ == "__main__":
    main()
