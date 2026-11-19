"""Deterministic direct TTIR transformations; never re-export a modified kernel."""

import re

from common import ROOT, ir_path, manifest, save_json, sha256


def without_locations(text):
    text = (
        "\n".join(line for line in text.splitlines() if not line.startswith("#loc"))
        + "\n"
    )
    while " loc(" in text:
        start = text.index(" loc(")
        end, depth = start + 5, 1
        while depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        text = text[:start] + text[end:]
    return text


def narrow_indices(text):
    # All frozen points have nonnegative offsets including masked lanes and
    # maximum (M*N*K + TILE_N*K + TILE_K) < 2^31. No signature type changes.
    aliases = {}
    pattern = re.compile(
        r"^\s*(%\w+) = arith.extsi (%\w+) : .*i32.* to .*i64.*\n", re.M
    )
    for match in pattern.finditer(text):
        aliases[match[1]] = match[2]
    text = pattern.sub("", text)
    for name, value in aliases.items():
        while value in aliases:
            value = aliases[value]
        text = re.sub(r"%" + re.escape(name[1:]) + r"\b", value, text)
    return text.replace("i64", "i32")


def trim_padding(text, config):
    meta = config["meta"]
    if "TILE_K" not in meta or not meta["ONE_TILE_PER_CTA"]:
        return text
    width = meta["TILE_N"]
    widths = [n for n in (128, 512, 2048, 4096) if n < width] + [width]
    if len(widths) == 1:
        return text
    lines = text.splitlines(keepends=True)
    header = "".join(lines[:2])
    body = "".join(lines[2:-2]).replace("    tt.return\n", "")

    def branch(index):
        size = widths[index]
        resized = body.replace(f"tensor<{width}x", f"tensor<{size}x").replace(
            f"end = {width} :", f"end = {size} :"
        )
        if index == len(widths) - 1:
            return resized
        return (
            f"    %limit_{size} = arith.constant {size} : i32\n"
            f"    %fits_{size} = arith.cmpi sle, %N, %limit_{size} : i32\n"
            f"    scf.if %fits_{size} {{\n"
            + resized
            + "    } else {\n"
            + branch(index + 1)
            + "    }\n"
        )

    return header + branch(0) + "    tt.return\n  }\n}\n"


def two_pass(text):
    if "scf.for" not in text:
        return text
    loop = re.search(
        r"^    (%\w+):2 = scf.for %start_n = (.*?) iter_args\(%m = (%\w+), (%\w+) = (%\w+)\) "
        r"-> \((tensor<[^>]+>), tensor<[^>]+>\)  : i32 \{\n(.*?)^    }\n",
        text,
        re.M | re.S,
    )
    assert loop, "Unexpected original online recurrence"
    result, bounds, initial_max, _, initial_sum, wide, body = loop.groups()
    update = re.search(r"^      %m_new = arith.maxnumf %m, (%\w+) :", body, re.M)
    assert update
    inp = update[1]
    loads = body[: update.start()]
    max_loop = (
        f"    %pass_max = scf.for %start_n = {bounds} iter_args(%m = {initial_max}) -> ({wide}) : i32 {{\n"
        + loads
        + f"      %next_max = arith.maxnumf %m, {inp} : {wide}\n"
        f"      scf.yield %next_max : {wide}\n    }}\n"
    )
    tail = text[loop.end() :]
    broadcast = re.search(
        r"^    (%z_\d+) = (tt.broadcast|tt.splat) (%m_reduced\w*) : (.*?) -> (tensor<[^>]+>)\n",
        tail,
        re.M,
    )
    assert broadcast
    max_result, max_type = broadcast[3], broadcast[4]
    reduction = tail[: broadcast.start()].replace(result + "#0", "%pass_max")
    z_reduce = re.search(r'^    (%z_\d+) = "tt.reduce"\((%z_\d+)\)', tail, re.M)
    assert z_reduce
    remaining = tail[z_reduce.start() :].replace(
        f'"tt.reduce"({z_reduce[2]})', '"tt.reduce"(%pass_sum)'
    )
    scalar = max_type == "f32"
    dense = "" if scalar else "dense<"
    close = "" if scalar else ">"
    safe = (
        f"    %pass_inf = arith.constant {dense}0xFF800000{close} : {max_type}\n"
        f"    %pass_zero = arith.constant {dense}0.000000e+00{close} : {max_type}\n"
        f"    %pass_all_inf = arith.cmpf oeq, {max_result}, %pass_inf : {max_type}\n"
    )
    bool_type = "i1" if scalar else max_type.replace("f32", "i1")
    select_type = max_type if scalar else f"{bool_type}, {max_type}"
    safe += (
        f"    %pass_safe = arith.select %pass_all_inf, %pass_zero, {max_result} : {select_type}\n"
        f"    %pass_broadcast = {broadcast[2]} %pass_safe : {max_type} -> {wide}\n"
    )
    sum_loop = (
        f"    %pass_sum = scf.for %start_n = {bounds} iter_args(%acc = {initial_sum}) -> ({wide}) : i32 {{\n"
        + loads
        + f"      %pass_delta = arith.subf {inp}, %pass_broadcast : {wide}\n"
        f"      %pass_exp = math.exp %pass_delta : {wide}\n"
        f"      %next_sum = arith.addf %acc, %pass_exp : {wide}\n"
        f"      scf.yield %next_sum : {wide}\n    }}\n"
    )
    return text[: loop.start()] + max_loop + reduction + safe + sum_loop + remaining


def main():
    variants = {
        "r1_i32": lambda text, cfg: narrow_indices(text),
        "r1_trim": trim_padding,
        "r1_two_pass": lambda text, cfg: two_pass(text),
        "r1_combined": lambda text, cfg: narrow_indices(
            two_pass(trim_padding(text, cfg))
        ),
    }
    records = {}
    for group, config in manifest()["groups"].items():
        original = ir_path(group, "baseline").read_text()
        clean = without_locations(original)
        records[group] = {}
        for name, transform in variants.items():
            output = transform(clean, config)
            path = ir_path(group, name)
            with path.open("x") as stream:
                stream.write(output)
            records[group][name] = dict(
                sha256=sha256(path),
                baseline_sha256=sha256(ir_path(group, "baseline")),
                calculation_changed=output != clean,
            )
    save_json(
        ROOT / "results/candidates_r1.json",
        dict(
            generator_sha256=sha256(__file__),
            groups=records,
            hypotheses=dict(
                r1_i32="Bounded integer address arithmetic narrowed to i32",
                r1_trim="Runtime N branches reduce non-inner padding; fixed TILE_K/grid/warps",
                r1_two_pass="Stable two-pass max/sum replaces online recurrence, same tile width",
                r1_combined="Composition of all three hypotheses",
            ),
        ),
    )


if __name__ == "__main__":
    main()
