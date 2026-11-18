"""Deterministic TTIR-only edits. Tutorial source and baseline are never modified."""

import re

from common import ROOT, save_json, sha256


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


def arithmetic(text):
    old = "    %mean_5 = arith.divf %mean, %mean_4 : f32"
    new = "    %recip_n = arith.divf %rstd, %mean_4 : f32\n    %mean_5 = arith.mulf %mean, %recip_n : f32"
    assert text.count(old) == 1
    text = text.replace(old, new)
    text = text.replace(
        "%var_6 = arith.divf %var, %mean_4", "%var_6 = arith.mulf %var, %recip_n"
    )
    old = "    %rstd_8 = math.sqrt %rstd_7 : f32\n    %rstd_9 = arith.divf %rstd, %rstd_8 : f32"
    assert old in text
    return text.replace(old, "    %rstd_9 = math.rsqrt %rstd_7 : f32")


LOOP = re.compile(
    r"^    (?:(%[\w]+) = )?scf.for %off = .*?\{\n(.*?)^    }", re.M | re.S
)
TOKEN = re.compile(r"%[\w]+")


def unroll(text, count):
    """Inline exactly count masked tiles for all three tutorial passes."""
    found = 0

    def replace(match):
        nonlocal found
        found += 1
        result, body = match.groups()
        yielded = re.search(r"^      scf.yield (%\w+) :", body, re.M)
        local = set(re.findall(r"^      (%\w+) =", body, re.M))
        carry = "%cst_0"
        out = []
        for tile in range(count):
            mapping = {name: f"%p{found}t{tile}_{name[1:]}" for name in local}
            mapping["%off"] = "%c0_i32" if tile == 0 else f"%offset_{found}_{tile}"
            mapping["%_mean_10"] = carry
            mapping["%_var_10"] = carry
            if tile:
                out.append(
                    f"    {mapping['%off']} = arith.constant {tile * 8192} : i32\n"
                )
            if yielded and tile == count - 1:
                mapping[yielded[1]] = result
            copied = re.sub(r"^      scf.yield .*\n", "", body, flags=re.M)
            copied = TOKEN.sub(lambda token: mapping.get(token[0], token[0]), copied)
            copied = re.sub(r"^  ", "", copied, flags=re.M)
            out.append(copied)
            if yielded:
                carry = mapping[yielded[1]]
        return "".join(out).rstrip("\n")

    text = LOOP.sub(replace, text)
    assert found == 3, found
    return text


def dispatch(text, branches):
    begin = text.index("    %row =")
    end = text.index("    tt.return")
    body = text[begin:end]
    prefix, suffix = text[:begin], text[end:]

    def nest(index):
        if index == len(branches):
            return body
        limit, block, count = branches[index]
        specialized = unroll(body, count)
        # Offsets for count>1 deliberately retain the original 8192 tile width.
        if block != 8192:
            assert count == 1
            specialized = specialized.replace("8192", str(block))
            specialized = TOKEN.sub(
                lambda t: {"%cst": "%tile_zero16", "%cst_0": "%tile_zero32"}.get(
                    t[0], t[0]
                ),
                specialized,
            )
            specialized = (
                f"    %tile_zero16 = arith.constant dense<0.0> : tensor<{block}xf16>\n"
                f"    %tile_zero32 = arith.constant dense<0.0> : tensor<{block}xf32>\n"
                + specialized
            )
        condition = (
            f"    %limit{index} = arith.constant {limit} : i32\n"
            f"    %fits{index} = arith.cmpi sle, %N, %limit{index} : i32\n"
            f"    scf.if %fits{index} {{\n"
        )
        indent = (
            lambda s: "\n".join("  " + line for line in s.rstrip().splitlines()) + "\n"
        )
        return (
            condition
            + indent(specialized)
            + "    } else {\n"
            + indent(nest(index + 1))
            + "    }\n"
        )

    return prefix + nest(0) + suffix


def main():
    baseline = ROOT / "ir/baseline.ttir"
    clean = without_locations(baseline.read_text())
    candidates = {
        "canonical": clean,
        "arithmetic": arithmetic(clean),
        "single_tile": dispatch(clean, [(8192, 8192, 1)]),
        "tile_dispatch_v2": dispatch(
            clean,
            [
                (1024, 1024, 1),
                (2048, 2048, 1),
                (4096, 4096, 1),
                (8192, 8192, 1),
                (16384, 8192, 2),
            ],
        ),
    }
    candidates["tile_dispatch_math_v2"] = dispatch(
        arithmetic(clean),
        [
            (1024, 1024, 1),
            (2048, 2048, 1),
            (4096, 4096, 1),
            (8192, 8192, 1),
            (16384, 8192, 2),
        ],
    )
    manifest = dict(baseline_sha256=sha256(baseline), candidates={})
    for name, text in candidates.items():
        path = ROOT / "ir" / f"{name}.ttir"
        if path.exists():
            assert (
                path.read_text() == text
            ), f"Immutable candidate changed: {name}; choose a new name"
        else:
            path.write_text(text)
        manifest["candidates"][name] = dict(
            sha256=sha256(path), bytes=path.stat().st_size
        )
    save_json(ROOT / "results/candidates.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
