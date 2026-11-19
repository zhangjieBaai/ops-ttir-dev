"""Export once, or verify the exact frozen upstream tutorial specialization."""

import argparse
import ast
import json

from common import (
    OPTIONS,
    PERF_SIZES,
    ROOT,
    allocate,
    gate,
    launch_source,
    save_json,
    sha256,
)


def verify_extraction():
    text = (ROOT / "upstream/01-vector-add.py.txt").read_text()
    upstream = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "add_kernel"
    )
    extracted = next(
        n
        for n in ast.parse((ROOT / "tutorial.py").read_text()).body
        if isinstance(n, ast.FunctionDef)
    )
    assert ast.dump(upstream) == ast.dump(extracted)
    provenance = json.loads((ROOT / "upstream/provenance.json").read_text())
    assert provenance["sha256"] == sha256(ROOT / "upstream/01-vector-add.py.txt")
    assert provenance["tutorial_sha256"] == sha256(ROOT / "tutorial.py")
    vals = [
        kw.value
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Benchmark"
        for kw in node.keywords
        if kw.arg == "x_vals"
    ]
    assert len(vals) == 1
    assert (
        eval(
            compile(ast.Expression(vals[0]), "<upstream x_vals>", "eval"),
            {"__builtins__": {}, "range": range},
        )
        == PERF_SIZES
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    gate()
    verify_extraction()
    x, y, out, _ = allocate(98432)
    ck = launch_source(x, y, out)
    path = ROOT / "ir/baseline.ttir"
    if args.verify:
        assert path.read_text() == ck.asm["ttir"]
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(ck.asm["ttir"])
        names = [
            "tutorial.py",
            "upstream/01-vector-add.py.txt",
            "upstream/provenance.json",
            "optimization_plan.md",
            "common.py",
            "ir/baseline.ttir",
        ]
        save_json(
            ROOT / "baseline.json",
            dict(
                experiment="vector_add_ttir_v1",
                options=OPTIONS,
                runtime_signature=["*fp32", "*fp32", "*fp32", "i32"],
                frozen_files={n: sha256(ROOT / n) for n in names},
                official_perf_sizes=PERF_SIZES,
            ),
        )
    print("PASS frozen tutorial TTIR")


if __name__ == "__main__":
    main()
