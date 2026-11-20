"""Static lowered-code/resource observations; no performance causality claim."""

import json
import re

from compare import ROOT, save_json


def main():
    pipeline = json.loads((ROOT / "results/main/pipeline.json").read_text())
    output = {}
    for group, variants in pipeline["groups"].items():
        output[group] = {}
        for name in ("source", "ttir"):
            k = variants[name]
            ptx = open(k["folder"] + "/kernel.ptx").read()
            llir = open(k["folder"] + "/kernel.llir").read()
            output[group][name] = dict(
                registers=k["registers"],
                spills=k["spills"],
                shared=k["metadata"]["shared"],
                ptx_instruction_counts={
                    op: len(re.findall(r"\b" + re.escape(op) + r"\b", ptx))
                    for op in (
                        "bar.sync",
                        "shfl.sync.bfly.b32",
                        "ex2.approx.f32",
                        "fma.rn.f32",
                        "fma.rn.ftz.f32",
                        "add.s64",
                        "mul.wide.s32",
                    )
                },
                llir_fma_intrinsics=llir.count("llvm.fma"),
                code_sections=k["text_sections"],
            )
    save_json(
        ROOT / "results/code_observations.json",
        dict(
            grade="C static observation; no isolated causal attribution", groups=output
        ),
    )


if __name__ == "__main__":
    main()
