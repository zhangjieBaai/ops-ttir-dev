"""Combine an unrolled main tile with a parallel reduction for one-column tails."""

from common import ROOT, ir_path, manifest, save_json, sha256
from make_round2 import branch, build


def combined(group, unroll):
    base = ir_path(group, "optimized").read_text()
    if "@mean_dim_kernel_non_inner_vec" not in base:
        return base
    header = next(line for line in base.splitlines() if "tt.func public" in line)
    if "%N:" not in header:
        return base
    main = build(base, unroll=unroll)
    guarded = branch(
        base,
        main,
        "    %min_n = arith.constant 8 : i32\n    %take = arith.cmpi sge, %N, %min_n : i32",
    )
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
    return branch(guarded.replace("%take", "%main_take"), tail, predicate)


def main():
    records = []
    for group in manifest()["groups"]:
        for factor in (8, 16):
            name = f"r2_combo{factor}_v2"
            path = ir_path(group, name)
            text = combined(group, factor)
            with path.open("x") as stream:
                stream.write(text)
            records.append(dict(group=group, name=name, sha256=sha256(path)))
    save_json(
        ROOT / "results/round2_combined_v2.json",
        dict(
            plan_sha256=sha256(ROOT / "optimization_plan_round2.md"),
            generator_sha256=sha256(__file__),
            records=records,
        ),
    )


if __name__ == "__main__":
    main()
