"""Direct TTIR libdevice/log algebra trials, original compiler options intact."""

import re

from common import ROOT, ir_path, manifest, save_json, sha256


def fast_log(text):
    return re.sub(
        r"math.log (%\w+) : (f32|tensor<[^>]+>)",
        lambda m: f'"tt.extern_elementwise"({m[1]}) '
        f'{{libname = "libdevice", libpath = "", symbol = "__nv_fast_logf", pure = true}} '
        f": ({m[2]}) -> {m[2]}",
        text,
    )


def base2(text):
    def replace(m):
        result, arg, typ = m.groups()
        constant = "0.693147182" if typ == "f32" else "dense<0.693147182>"
        return (
            f"{result}_ln2 = arith.constant {constant} : {typ}\n"
            f"    {result}_log2 = math.log2 {arg} : {typ}\n"
            f"    {result} = arith.mulf {result}_log2, {result}_ln2 : {typ}"
        )

    return re.sub(r"(%\w+) = math.log (%\w+) : (f32|tensor<[^>]+>)", replace, text)


def main():
    output = {}
    for g, c in manifest()["groups"].items():
        parent = ir_path(g, "r4_focus").read_text()
        output[g] = {}
        for n, t in [("r5_fastlog", fast_log(parent)), ("r5_log2", base2(parent))]:
            p = ir_path(g, n)
            with p.open("x") as f:
                f.write(t)
            output[g][n] = sha256(p)
    save_json(
        ROOT / "results/runs/r5_setup/candidates.json",
        dict(
            groups=output,
            generator_sha256=sha256(__file__),
            protocol_sha256=sha256(ROOT / "optimization_plan_round5.md"),
        ),
    )


if __name__ == "__main__":
    main()
