"""Deterministic textual SSA rewrites of frozen TTIR only."""

import re

from common import ROOT, manifest, save, sha, write_once


def replace_uses(text, old, new):
    return re.sub(re.escape(old) + r"(?![\w])", new, text)


def share(text):
    low_scale = re.search(r"(%scale_low_\d+) = arith.extf", text)[1]
    low_zero = re.search(r"(%zero_low_\d+) = arith.extf", text)[1]
    high_scale = re.search(r"(%scale_high_\d+) = arith.extf", text)[1]
    high_zero = re.search(r"(%zero_high_\d+) = arith.extf", text)[1]
    start = text.index("      %group_high =")
    end = text.index("      %high =", start)
    text = text[:start] + text[end:]
    return replace_uses(replace_uses(text, high_scale, low_scale), high_zero, low_zero)


def scalar(text):
    text = share(text)
    pattern = r"(      )(%group_low_\d+) = arith.divsi %group_low, %group_low_\d+ : (tensor<\d+x1xi32>)([^\n]*)"
    m = re.search(pattern, text)
    assert m
    replacement = (
        f"{m[1]}%group_scalar = arith.divsi %k0, %qGroupSize : i32\n"
        f"{m[1]}{m[2]} = tt.splat %group_scalar : i32 -> {m[3]}{m[4]}"
    )
    text = text[: m.start()] + replacement + text[m.end() :]
    text, count = re.subn(
        r"(%scale_low_\d+) = arith.cmpi slt, %group_low, %scale_low : (tensor<\d+x1xi32>)",
        lambda m: f"{m[1]} = arith.constant dense<true> : {m[2].replace('xi32', 'xi1')}",
        text,
    )
    assert count == 1
    return text


def unsigned(text):
    return text.replace("arith.divsi", "arith.divui").replace(
        "arith.remsi", "arith.remui"
    )


def cast_after(text):
    low = re.search(r"      (%low_\d+) = arith.truncf (%low_\d+) : [^\n]+\n", text)
    high = re.search(r"      (%high_\d+) = arith.truncf (%high_\d+) : [^\n]+\n", text)
    assert low and high
    text = text.replace(low[0], "").replace(high[0], "")
    start = text.index("      %w = tt.join")
    end = text.index("      %acc_", start)
    segment = text[start:end].replace("xf16", "xf32")
    segment = replace_uses(replace_uses(segment, low[1], low[2]), high[1], high[2])
    last = segment.splitlines()[-1]
    m = re.match(r"      (%w_\d+) = tt.reshape .* -> (tensor<\d+x\d+xf32>)(.*)", last)
    assert m
    segment = replace_uses(segment, m[1], m[1] + "_wide")
    segment += f"      {m[1]} = arith.truncf {m[1]}_wide : {m[2]} to {m[2].replace('xf32', 'xf16')}{m[3]}\n"
    return text[:start] + segment + text[end:]


def main():
    entries = []
    functions = {
        "shared_scale_v1": share,
        "group_scalar_v1": scalar,
        "unsigned_v1": unsigned,
        "scalar_unsigned_v1": lambda t: unsigned(scalar(t)),
        "cast_after_v1": cast_after,
    }
    for case in manifest()["cases"]:
        path = ROOT / "ir" / case["id"] / "baseline.ttir"
        source = path.read_text()
        for name, fn in functions.items():
            dest = path.with_name(name + ".ttir")
            write_once(dest, fn(source))
            entries.append(
                dict(
                    case=case["id"],
                    name=name,
                    parent_sha256=sha(path),
                    sha256=sha(dest),
                )
            )
    save(ROOT / "results/candidate_manifest.json", entries)


if __name__ == "__main__":
    main()
