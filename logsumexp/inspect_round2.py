"""Reproducible static evidence for directly modified TTIR and resulting code."""

import json

from audit_artifacts import text_sections
from common import ROOT, ir_path, manifest, save_json, sha256
from make_round2 import single_exp, unroll, widen
from make_round2_unroll import unroll_four


def main():
    transforms = {
        "r2_single_exp": lambda t, c: single_exp(t),
        "r2_select_after": lambda t, c: single_exp(t, "after"),
        "r2_wide256": lambda t, c: widen(single_exp(t), c, 256),
        "r2_wide512": lambda t, c: widen(single_exp(t), c, 512),
        "r2_unroll4": lambda t, c: unroll(single_exp(t)),
        "r2_explicit4": lambda t, c: unroll_four(single_exp(t)),
    }
    pipelines = [
        json.loads((ROOT / "results/runs/r2_setup" / p).read_text())
        for p in ("pipeline.json", "pipeline_explicit4.json")
    ]
    kernels = {}
    for pipeline in pipelines:
        for group, variants in pipeline["kernels"].items():
            kernels.setdefault(group, {}).update(variants)
    output = {
        "evidence_level": "C: static code facts, not performance causes",
        "groups": {},
    }
    for group, config in manifest()["groups"].items():
        parent = ir_path(group, "r1_trim").read_text()
        for name, transform in transforms.items():
            assert transform(parent, config) == ir_path(group, name).read_text()
        rows = {}
        for name, kernel in kernels[group].items():
            folder = ROOT / "artifacts" / group / name / kernel["hash"]
            ptx = (folder / "kernel.ptx").read_text()
            rows[name] = dict(
                ttir_sha256=sha256(ir_path(group, name)),
                ex2_instructions=ptx.count("ex2.approx"),
                registers=kernel["n_regs"],
                spills=kernel["n_spills"],
                shared=kernel["metadata"]["shared"],
                text_sections=text_sections(folder / "kernel.cubin"),
            )
        output["groups"][group] = rows
    save_json(ROOT / "results/runs/r2_setup/static_audit.json", output)


if __name__ == "__main__":
    main()
