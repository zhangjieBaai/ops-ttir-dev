"""Original baseline gate plus source-JIT/exported-TTIR exact recompile gate."""

import argparse
import json
import re
from pathlib import Path

import torch
import triton
from common_port import (
    MATRIX,
    OPTIONS,
    PARENT,
    ROOT,
    allocate,
    baseline_kernel,
    check_source,
    compile_routes,
    digest,
    group,
    launch,
    launch_source,
    preserve,
    save,
    snapshot,
)


def loop_dot_order(text):
    loops = []
    for match in re.finditer(r" = scf.for .*?\{\n(.*?)scf.yield", text, re.S):
        operations = []
        for line in match[1].splitlines():
            if "tt.dot" in line:
                operations.append(line.strip().split(" = ")[0])
            elif "%ppT = arith.truncf" in line:
                operations.append("ppT_cast")
        loops.append(operations)
    return loops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    result = dict(
        environment=snapshot(), source_manifest=check_source(), points=[], passed=True
    )
    for shape in MATRIX:
        b, h, n, d, c = shape
        t = allocate(*shape)
        kernels, metas = compile_routes(t)
        original_source = launch_source(baseline_kernel, t)
        frozen = (PARENT / "ir" / group(d, c) / "baseline.ttir").read_text()
        assert original_source.asm["ttir"] == frozen
        original_exact = {
            ext: original_source.asm[ext] == kernels["baseline"].asm[ext]
            for ext in ("llir", "ptx", "cubin")
        }
        assert all(original_exact.values())
        path = ROOT / "ir" / group(d, c) / "source_export.ttir"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_text() == kernels["source"].asm["ttir"]
        else:
            path.write_text(kernels["source"].asm["ttir"])
        recompiled = triton.compile(str(path), options=OPTIONS)
        recompile_meta = preserve("source_recompiled", d, c, recompiled)
        source_exact = {
            ext: kernels["source"].asm[ext] == recompiled.asm[ext]
            for ext in ("llir", "ptx", "cubin")
        }
        assert all(source_exact.values())
        launch(kernels["source"], t)
        torch.cuda.synchronize()
        expected = {k: t[k].clone() for k in ("dk", "dv")}
        launch(recompiled, t)
        torch.cuda.synchronize()
        assert all(torch.equal(t[k], v) for k, v in expected.items())
        options = metas["baseline"]["parsed_options"]
        for route in kernels:
            assert all(
                json.dumps(metas[route][key], sort_keys=True)
                == json.dumps(val, sort_keys=True)
                for key, val in options.items()
            ), route
        item = dict(
            shape=shape,
            group=group(d, c),
            original_baseline_gate=original_exact,
            source_recompile_gate=source_exact,
            source_recompile_outputs_bitwise=True,
            metadata=metas,
            recompile_metadata=recompile_meta,
            source_ir_text_equal=metas["source"]["elf_text_sections"]
            == metas["ir"]["elf_text_sections"],
            source_baseline_text_equal=metas["source"]["elf_text_sections"]
            == metas["baseline"]["elf_text_sections"],
            dot_order={r: loop_dot_order(k.asm["ttir"]) for r, k in kernels.items()},
            exported_ttir_sha256=digest(path),
        )
        result["points"].append(item)
        save(out / "pipeline.json", result)
        print(
            "pipeline",
            shape,
            "text_equal_source_ir",
            item["source_ir_text_equal"],
            item["dot_order"],
            flush=True,
        )


if __name__ == "__main__":
    main()
