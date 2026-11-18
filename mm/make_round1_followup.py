"""Repair lost contiguous lanes in wrap_select, optionally combine unsigned indexing."""

import re

from common import ROOT, frozen, save_json, sha256


def wrap16(text):
    pattern = r"^(    )(%[\w]+) = arith.remsi (%offs_[\w]+), (%offs_[\w]+) : tensor<(\d+)xi32>(.*)$"
    count = 0

    def replace(match):
        nonlocal count
        indent, result, index, limit, width, loc = match.groups()
        count += 1
        p = f"%r1_wrap16_{count}"
        # Every input dimension is a positive multiple of 16. For masked-out
        # rows/columns, 0..15 is a safe fallback preserving contiguous chunks.
        # The validity predicate switches only on a 16-element boundary.
        return (
            f"{indent}{p}_ok = arith.cmpi slt, {index}, {limit} : tensor<{width}xi32>{loc}\n"
            f"{indent}{p}_mask = arith.constant dense<15> : tensor<{width}xi32>{loc}\n"
            f"{indent}{p}_safe = arith.andi {index}, {p}_mask : tensor<{width}xi32>{loc}\n"
            f"{indent}{result} = arith.select {p}_ok, {index}, {p}_safe "
            f"{{tt.contiguity = dense<16> : tensor<1xi32>}} : tensor<{width}xi1>, tensor<{width}xi32>{loc}"
        )

    value = re.sub(pattern, replace, text, flags=re.M)
    assert count == 2
    return value


def main():
    rows = []
    for family, config in frozen()["families"].items():
        src = ROOT / config["ttir_path"]
        value = wrap16(src.read_text())
        for name in ["r1_wrap16", "r1_wrap16_unsigned"]:
            content = value
            if name.endswith("_unsigned"):
                content = content.replace("arith.divsi", "arith.divui").replace(
                    "arith.remsi", "arith.remui"
                )
            dest = src.with_name(f"{name}.ttir")
            if dest.exists():
                assert dest.read_text() == content
            else:
                with dest.open("x") as f:
                    f.write(content)
            rows.append(
                dict(
                    family=family,
                    name=name,
                    path=str(dest.relative_to(ROOT)),
                    ttir_sha256=sha256(dest),
                    parent_sha256=sha256(src),
                )
            )
    save_json(ROOT / "results/round1_followup_candidates.json", dict(candidates=rows))


if __name__ == "__main__":
    main()
