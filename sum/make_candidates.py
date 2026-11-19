"""Deterministic direct TTIR rewrites; never regenerate candidate Python kernels."""

import re

from common import ROOT, frozen, save_json, sha256, write_once


def narrow_indices(text):
    # For the frozen mapping every program*block + lane < 2**31.
    # First eliminate extensions and substitute their operands, then narrow i64.
    aliases = {}
    lines = []
    for line in text.splitlines():
        match = re.match(
            r"\s*(%\w+) = arith.extsi (%\w+) : (?:i32 to i64|tensor<\d+xi32> to tensor<\d+xi64>)",
            line,
        )
        if match:
            aliases[match[1]] = match[2]
        else:
            lines.append(line)
    text = "\n".join(lines) + "\n"
    for name, replacement in aliases.items():
        text = re.sub(re.escape(name) + r"(?![\w])", replacement, text)
    return text.replace("i64", "i32")


def cached(text, modifier):
    return re.sub(
        r"(= tt.load [^\n]+?)( : tensor<)", rf"\1 cacheModifier = {modifier}\2", text
    )


def fold(text):
    # Reshape a wide 1D sum into [4,width/4], reduce outer then inner.
    pattern = (
        r'(%\w+) = "tt.reduce"\((%\w+)\) <\{axis = 0 : i32\}> '
        r"\(\{(.*?)\}\) : \(tensor<(\d+)xf32>\) -> f32([^\n]*)"
    )

    def replace(match):
        out, operand, body, width, loc = match.groups()
        width = int(width)
        if width < 512:
            return match[0]
        columns = width // 4
        return (
            f"%fold_input = tt.reshape {operand} allow_reorder : tensor<{width}xf32> -> tensor<4x{columns}xf32>\n"
            f'%fold_partial = "tt.reduce"(%fold_input) <{{axis = 0 : i32}}> ({{{body}}}) : '
            f"(tensor<4x{columns}xf32>) -> tensor<{columns}xf32>\n"
            f'{out} = "tt.reduce"(%fold_partial) <{{axis = 0 : i32}}> ({{{body}}}) : '
            f"(tensor<{columns}xf32>) -> f32{loc}"
        )

    return re.sub(pattern, replace, text, flags=re.S)


def main():
    manifest = frozen()
    rewrites = {
        "r1_i32": narrow_indices,
        "r1_cg": lambda text: cached(text, "cg"),
        "r1_cs": lambda text: cached(text, "cs"),
        "r1_fold": fold,
        "r1_i32_fold": lambda text: fold(narrow_indices(text)),
    }
    result = {}
    for name, rewrite in rewrites.items():
        result[name] = {}
        for group, config in manifest["groups"].items():
            baseline = ROOT / config["path"]
            text = baseline.read_text()
            path = baseline.with_name(f"{name}.ttir")
            write_once(path, rewrite(text))
            result[name][group] = dict(
                path=str(path.relative_to(ROOT)),
                sha256=sha256(path),
                changed=path.read_text() != text,
            )
    save_json(ROOT / "results/candidates_r1.json", result)


if __name__ == "__main__":
    main()
