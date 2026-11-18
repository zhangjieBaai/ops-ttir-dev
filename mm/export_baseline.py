"""Verify six frozen per-shape autotune baselines against tutorial compilation."""

import argparse
import ast
import json

import torch
from common import (
    PERF_SHAPES,
    ROOT,
    allocate,
    config_for_shape,
    digest,
    frozen,
    gate,
    launch_source,
    sha256,
)


def verify_extraction():
    text = (ROOT / "upstream/03-matrix-multiplication.py.txt").read_text()
    upstream = {
        n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
    }
    extracted = ast.parse((ROOT / "tutorial.py").read_text())
    for node in extracted.body:
        if isinstance(node, ast.FunctionDef):
            original = upstream[node.name]
            assert ast.dump(ast.Module(body=node.body, type_ignores=[])) == ast.dump(
                ast.Module(body=original.body, type_ignores=[])
            )
    provenance = json.loads((ROOT / "upstream/provenance.json").read_text())
    assert provenance["sha256"] == sha256(
        ROOT / "upstream/03-matrix-multiplication.py.txt"
    )
    assert provenance["tutorial_sha256"] == sha256(ROOT / "tutorial.py")
    # Read the actual frozen tutorial's Benchmark x_vals, rather than testing
    # against a second hand-written list that could drift from the source.
    x_vals = [
        kw.value
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Benchmark"
        for kw in node.keywords
        if kw.arg == "x_vals"
    ]
    assert len(x_vals) == 1
    official_sizes = eval(
        compile(ast.Expression(x_vals[0]), "<frozen tutorial x_vals>", "eval"),
        {"__builtins__": {}, "range": range},
    )
    assert PERF_SHAPES == [(n, n, n) for n in official_sizes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="Verify immutable six-family export"
    )
    parser.parse_args()
    gate()
    verify_extraction()
    manifest = frozen()
    torch.manual_seed(42)
    for shape in PERF_SHAPES:
        family_id, config = config_for_shape(shape, manifest)
        tensors = allocate(*shape)
        ck = launch_source(tensors, config)
        assert ck.asm["ttir"] == (ROOT / config["ttir_path"]).read_text()
        assert digest(ck.asm["cubin"]) == config["autotune_cubin_sha256"]
        print(
            f"PASS: {shape} -> {family_id}; frozen tutorial TTIR/cubin unchanged",
            flush=True,
        )
    print("PASS: all 31 shapes match six frozen baseline TTIRs; no autotune rerun")


if __name__ == "__main__":
    main()
