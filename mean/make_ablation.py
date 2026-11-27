"""Write-once factorial TTIR: controlled flatten/index and round2 algorithm factors."""

import itertools
import re

from common import ALL_SHAPES, ROOT, ir_path, manifest, save_json, sha256
from make_round2 import branch, build

A_NAMES = ["baseline", "ab_f0i1", "ab_f1i0", "ab_f1i1", "optimized"]
B_NAMES = [f"ab_u{u}g{g}t{t}" for u, g, t in itertools.product((0, 1), repeat=3)]
NAMES = A_NAMES + B_NAMES


def narrow(text):
    # Only arithmetic widths change; remove now-identity sign extensions.
    aliases = {}
    lines = []
    for line in text.replace("i64", "i32").splitlines():
        match = re.search(r"(%[\w]+) = arith.extsi (%[\w]+) :", line)
        if match:
            aliases[match[1]] = match[2]
        else:
            lines.append(line)

    def resolve(match):
        token = match.group()
        while token in aliases:
            token = aliases[token]
        return token

    return re.sub(r"%[\w]+", resolve, "\n".join(lines) + "\n")


def flatten(text):
    # Retain all scalar address widths and float operations. Replace only the
    # original [128,8] Cartesian range and its K dimension by a flat range.
    start = text.index("    %5 = tt.make_range")
    end = text.index("    %17 =", start)
    replacement = (
        "\n".join(
            [
                "    %ab_range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>",
                "    %ab_range64 = arith.extsi %ab_range : tensor<1024xi32> to tensor<1024xi64>",
                "    %ab_kbase = tt.splat %4 : i64 -> tensor<1024xi64>",
                "    %16 = arith.addi %ab_kbase, %ab_range64 : tensor<1024xi64>",
            ]
        )
        + "\n"
    )
    text = text[:start] + replacement + text[end:]
    text = re.sub(
        r"^.*%cst_1 = arith.constant dense<8>.*\n", "", text, flags=re.MULTILINE
    )
    return text.replace("tensor<128x8x", "tensor<1024x")


def second(group, u, g, t):
    base = ir_path(group, "optimized").read_text()
    header = next(line for line in base.splitlines() if "tt.func public" in line)
    if "@mean_dim_kernel_non_inner_vec" not in base or "%N:" not in header:
        return base
    result = build(base, unroll=16 if u else 1)
    if g:
        result = branch(
            base,
            result,
            "    %min_n = arith.constant 8 : i32\n    %take = arith.cmpi sge, %N, %min_n : i32",
        )
        result = result.replace("%take", "%main_take")
    if t:
        tail = build(base, rows=256, width=1).replace(
            "scf.for %sub = %z to %full step %w", "scf.for %sub = %z to %one step %w"
        )
        predicate = "\n".join(
            [
                "    %t_width = arith.constant 1024 : i32",
                "    %t_one = arith.constant 1 : i32",
                "    %t_pk = tt.get_program_id y : i32",
                "    %t_base = arith.muli %t_pk, %t_width : i32",
                "    %t_remaining = arith.subi %K, %t_base : i32",
                "    %take = arith.cmpi eq, %t_remaining, %t_one : i32",
            ]
        )
        result = branch(result, tail, predicate)
    return result


def expected(group, name):
    base = ir_path(group, "baseline").read_text()
    if name in ("baseline", "optimized"):
        return ir_path(group, name).read_text()
    if name.startswith("ab_f"):
        if "@mean_dim_kernel_non_inner_vec" not in base:
            return base
        if "f1" in name:
            base = flatten(base)
        if "i1" in name:
            base = narrow(base)
        return base
    match = re.fullmatch(r"ab_u([01])g([01])t([01])", name)
    assert match, name
    return second(group, *map(int, match.groups()))


def main():
    records = []
    for m, n, k in ALL_SHAPES:
        assert (m - 1) * n * k + (n + 255) * k + ((k + 1023) // 1024) * 1024 < 2**31
    for group in manifest()["groups"]:
        for name in NAMES:
            text = expected(group, name)
            path = ir_path(group, name)
            if path.exists():
                assert path.read_text() == text, (group, name)
            else:
                with path.open("x") as stream:
                    stream.write(text)
            records.append(dict(group=group, name=name, sha256=sha256(path)))
        assert (
            ir_path(group, "ab_u1g1t1").read_bytes()
            == ir_path(group, "r2_final").read_bytes()
        )
    save_json(
        ROOT / "results/ablation_manifest.json",
        dict(
            plan_sha256=sha256(ROOT / "ablation_plan.md"),
            generator_sha256=sha256(__file__),
            variants=NAMES,
            records=records,
            references={
                str(p.relative_to(ROOT)): sha256(p)
                for p in (ROOT / "ir").glob("*/r2_final.ttir")
            },
        ),
    )


if __name__ == "__main__":
    main()
