"""Three rounds of immutable direct TTIR transformations."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OP = ROOT.parent
OLD = OP / "round2/expanded"
sys.path.insert(0, str(OP))
import common as c  # noqa: E402


def uses(text, before, after):
    return re.sub(re.escape(before) + r"(?![\w])", after, text)


def group128(text, _case):
    needle = "      %r2_group = arith.divui %k0, %qGroupSize : i32"
    assert needle in text
    return text.replace(
        needle,
        "      %r3_shift7 = arith.constant 7 : i32\n      %r2_group = arith.shrui %k0, %r3_shift7 : i32",
        1,
    )


def contiguous(text, _case):
    # Preserve the runtime ABI; these equalities follow from the existing contract.
    i = text.index("\n", text.index("  tt.func public")) + 1
    body = text[i:]
    for old, new in (
        ("%stride_am", "%K"),
        ("%stride_mn", "%K2"),
        ("%stride_sg", "%N"),
        ("%stride_zg", "%N"),
        ("%stride_cm", "%N"),
    ):
        body = uses(body, old, new)
    return text[:i] + body


def pid_group(text, case, gm):
    bm = case["config"][0]
    if (case["shape"][0] + bm - 1) // bm == 1:
        return text
    match = re.search(
        r"    %pid_m = arith.divui %pid, (%\w+) : i32[^\n]*\n    %pid_n = arith.remui %pid, %\w+ : i32[^\n]*\n",
        text,
    )
    assert match
    gn = match[1]
    assert (
        "%M: i32"
        in text.splitlines()[
            next(i for i, x in enumerate(text.splitlines()) if "tt.func public" in x)
        ]
    )
    body = f"""    %r4_bm = arith.constant {bm} : i32
    %r4_bm_less1 = arith.constant {bm-1} : i32
    %r4_gm = arith.constant {gm} : i32
    %r4_mceil = arith.addi %M, %r4_bm_less1 : i32
    %r4_num_m = arith.divui %r4_mceil, %r4_bm : i32
    %r4_span = arith.muli %r4_gm, {gn} : i32
    %r4_group = arith.divui %pid, %r4_span : i32
    %r4_first = arith.muli %r4_group, %r4_gm : i32
    %r4_remaining = arith.subi %r4_num_m, %r4_first : i32
    %r4_size = arith.minui %r4_remaining, %r4_gm : i32
    %r4_local = arith.remui %pid, %r4_span : i32
    %r4_m = arith.remui %r4_local, %r4_size : i32
    %pid_m = arith.addi %r4_first, %r4_m : i32
    %pid_n = arith.divui %r4_local, %r4_size : i32
"""
    return text[: match.start()] + body + text[match.end() :]


def preload_pair(text, case):
    if case["config"][2] == 128:
        return text
    begin = text.index("      %r2_second_k =")
    load = re.search(r"      %packed_\d+_pair = tt.load[^\n]*\n", text[begin:])
    assert load
    end = begin + load.end()
    segment = text[begin:end]
    text = text[:begin] + text[end:]
    first_dot = re.search(r"      %acc_\d+ = tt.dot", text).start()
    return text[:first_dot] + segment + text[first_dot:]


def early_scale(text, _case):
    start = text.index("      %r2_group =")
    # A previous round may introduce a local constant just before the group expression.
    if "      %r3_shift7 =" in text[:start]:
        start = text.index("      %r3_shift7 =")
    end = text.index("      %low =", start)
    segment = text[start:end]
    first = text[:start]
    definitions = []
    for name in ("%packed",):
        m = re.search(r"^      " + re.escape(name) + r" =[^\n]*\n", first, re.M)
        assert m
        definitions.append(m[0])
    mask = re.search(
        r"^      (%packed_\d+) = arith.cmpi slt, %packed, (%packed_\d+) :[^\n]*\n",
        first,
        re.M,
    )
    assert mask
    splat = re.search(r"^      " + re.escape(mask[2]) + r" =[^\n]*\n", first, re.M)
    assert splat
    definitions += [splat[0], mask[0]]
    text = text[:start] + text[end:]
    for line in definitions:
        text = text.replace(line, "", 1)
    loop = text.index(" = scf.for ")
    insert = text.index("\n", loop) + 1
    return text[:insert] + "".join(definitions) + segment + text[insert:]


def stream_store(text, _case):
    return re.sub(r"(tt.store [^\n]+?) :", r"\1 cacheModifier = cs :", text)


def two_groups(text, case):
    # A declared K-family specialization; odd group counts keep the parent.
    if case["shape"][2] % 256:
        return text
    loop = text.index(" = scf.for ")
    begin = text.rfind("\n", 0, loop) + 1
    body_begin = text.index("\n", loop) + 1
    body_end = text.index("      scf.yield", body_begin)
    yield_end = text.index("\n", body_end) + 1
    header = text[begin:body_begin]
    body = text[body_begin:body_end]
    outgoing = re.search(r"scf.yield (%\w+)", text[body_end:yield_end])[1]
    incoming = re.search(r"iter_args\((%\w+) = %acc\)", header)[1]
    definitions = set(re.findall(r"^\s*(%[A-Za-z_]\w*)\s*=", body, re.M))
    names = {n: n + "_g2" for n in definitions}
    names.update({"%k0": "%r5_second_group", incoming: outgoing})
    second = re.sub(r"%[A-Za-z_]\w*", lambda m: names.get(m[0], m[0]), body)
    second = "      %r5_second_group = arith.addi %k0, %r5_step128 : i32\n" + second
    header = re.sub(r"step %\w+ iter_args", "step %r5_step256 iter_args", header)
    new_yield = text[body_end:yield_end].replace(outgoing, names[outgoing])
    return (
        text[:begin]
        + "    %r5_step128 = arith.constant 128 : i32\n"
        + "    %r5_step256 = arith.constant 256 : i32\n"
        + header
        + body
        + second
        + new_yield
        + text[yield_end:]
    )


def main(stage):
    manifest = json.loads((OLD / "manifest.json").read_text())
    target = ROOT / stage
    source = OLD
    parent_name = "r2_group_pair"
    if stage != "r3":
        previous = ROOT / ("r3" if stage == "r4" else "r4")
        selection = json.loads((previous / "selection.json").read_text())
        source = previous
        parent_name = selection["selected"]
    if stage == "r3":
        transforms = {
            "group128": group128,
            "contiguous": contiguous,
            "contract": lambda s, c_: contiguous(group128(s, c_), c_),
        }
    elif stage == "r4":
        transforms = {
            f"pid_g{n}": (lambda s, c_, n=n: pid_group(s, c_, n)) for n in (2, 4, 8)
        }
    else:
        transforms = {
            "preload_pair": preload_pair,
            "early_scale": early_scale,
            "two_groups": two_groups,
        }
    records = []
    for case in manifest["cases"]:
        dest = target / "ir" / case["id"]
        baseline = (OLD / "ir" / case["id"] / "baseline.ttir").read_text()
        anchor = (OLD / "ir" / case["id"] / "r2_group_pair.ttir").read_text()
        parent = (source / "ir" / case["id"] / f"{parent_name}.ttir").read_text()
        for name, text in (
            ("baseline", baseline),
            ("anchor", anchor),
            ("parent", parent),
        ):
            c.write_once(dest / f"{name}.ttir", text)
        for name, fn in transforms.items():
            content = fn(parent, case)
            c.write_once(dest / f"{name}.ttir", content)
            records.append(
                {
                    "case": case["id"],
                    "name": name,
                    "parent_sha256": c.digest(parent),
                    "sha256": c.digest(content),
                }
            )
    c.write_once(target / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    c.write_once(
        target / "candidates.json",
        json.dumps(
            {
                "stage": stage,
                "names": list(transforms),
                "parent_from": str(source),
                "parent_name": parent_name,
                "records": records,
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["r3", "r4", "r5"])
    main(p.parse_args().stage)
