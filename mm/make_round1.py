"""Deterministic direct TTIR changes; baseline and all named candidates are immutable."""

import re

from common import ROOT, frozen, save_json, sha256

NAMES = ["canonical", "unsigned", "wrap_select", "k_aligned", "cache_cg", "unroll2"]


def canonical(text):
    lines = []
    for line in text.splitlines():
        if line.startswith("#loc"):
            continue
        line = re.sub(r' loc\("[^"\n]*"\(#loc\w*\)\)', "", line)
        line = re.sub(r" loc\(#loc\w*\)", "", line)
        lines.append(line)
    return "\n".join(lines) + "\n"


def wrap_select(text):
    pattern = r"^(    )(%[\w]+) = arith.remsi (%offs_[\w]+), (%offs_[\w]+) : tensor<(\d+)xi32>(.*)$"
    count = 0

    def replace(match):
        nonlocal count
        indent, result, index, limit, width, loc = match.groups()
        # Only offsets of A rows and B columns, not program-ID group mapping.
        count += 1
        prefix = f"%r1_wrap{count}"
        return (
            f"{indent}{prefix}_ok = arith.cmpi slt, {index}, {limit} : tensor<{width}xi32>{loc}\n"
            f"{indent}{prefix}_zero = arith.constant dense<0> : tensor<{width}xi32>{loc}\n"
            f"{indent}{result} = arith.select {prefix}_ok, {index}, {prefix}_zero "
            f": tensor<{width}xi1>, tensor<{width}xi32>{loc}"
        )

    value = re.sub(pattern, replace, text, flags=re.M)
    assert count == 2
    return value


def k_aligned(text, bk):
    lines = text.splitlines()
    begin = next(i for i, line in enumerate(lines) if " = scf.for " in line)
    end = next(
        i for i in range(begin + 1, len(lines)) if lines[i].startswith("    } loc(")
    )
    loop = "\n".join(lines[begin : end + 1])
    result = re.search(r"(%\w+):3 = scf.for", loop)[1]
    types = re.search(r" -> \((.*)\)  : i32 \{", lines[begin])[1]
    fast = loop.replace(f"{result}:3 =", "%r1_fast:3 =", 1)
    fast, count = re.subn(r"(tt.load %\w+), %\w+, %\w+( : tensor<)", r"\1\2", fast)
    assert count == 2
    slow = loop.replace(f"{result}:3 =", "%r1_slow:3 =", 1)
    branch = (
        f"    %r1_bk = arith.constant {bk} : i32\n"
        f"    %r1_rem = arith.remsi %K, %r1_bk : i32\n"
        f"    %r1_aligned = arith.cmpi eq, %r1_rem, %c0_i32 : i32\n"
        f"    {result}:3 = scf.if %r1_aligned -> ({types}) {{\n"
        + "\n".join("  " + line for line in fast.splitlines())
        + f"\n      scf.yield %r1_fast#0, %r1_fast#1, %r1_fast#2 : {types}\n"
        + "    } else {\n"
        + "\n".join("  " + line for line in slow.splitlines())
        + f"\n      scf.yield %r1_slow#0, %r1_slow#1, %r1_slow#2 : {types}\n    }}"
    )
    return (
        "\n".join(lines[:begin])
        + "\n"
        + branch
        + "\n"
        + "\n".join(lines[end + 1 :])
        + "\n"
    )


def transform(text, name, config):
    if name == "canonical":
        return canonical(text)
    if name == "unsigned":
        return text.replace("arith.divsi", "arith.divui").replace(
            "arith.remsi", "arith.remui"
        )
    if name == "wrap_select":
        return wrap_select(text)
    if name == "k_aligned":
        return k_aligned(text, config["meta"]["BLOCK_SIZE_K"])
    if name == "cache_cg":
        result, count = re.subn(
            r"(tt.load %\w+, %\w+, %\w+) :", r"\1 cacheModifier = cg :", text
        )
        assert count == 2
        return result
    if name == "unroll2":
        result, count = re.subn(
            r"(    \}) loc\(#loc\d+\)",
            r"\1 {tt.loop_unroll_factor = 2 : i32}",
            text,
            count=1,
        )
        assert count == 1
        return result
    raise ValueError(name)


def main():
    manifest = frozen()
    rows = []
    for family, config in manifest["families"].items():
        baseline = ROOT / config["ttir_path"]
        text = baseline.read_text()
        for name in NAMES:
            path = baseline.with_name(f"r1_{name}.ttir")
            value = transform(text, name, config)
            assert value != text
            if path.exists():
                assert path.read_text() == value, f"Never overwrite {path}"
            else:
                with path.open("x") as f:
                    f.write(value)
            rows.append(
                dict(
                    family=family,
                    name=f"r1_{name}",
                    path=str(path.relative_to(ROOT)),
                    parent_sha256=sha256(baseline),
                    ttir_sha256=sha256(path),
                )
            )
    output = ROOT / "results/round1_candidates.json"
    save_json(
        output,
        dict(plan_sha256=sha256(ROOT / "optimization_round1.md"), candidates=rows),
    )
    print(f"Generated {len(rows)} direct TTIR candidates")


if __name__ == "__main__":
    main()
