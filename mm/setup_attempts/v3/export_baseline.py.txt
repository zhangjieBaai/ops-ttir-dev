"""One-time official autotune setup, then immutable source/TTIR/config freeze."""

import argparse
import ast
import json

import torch
import triton
from common import (
    PERF_SHAPES,
    ROOT,
    SEED_SHAPE,
    allocate,
    digest,
    gate,
    grid,
    launch_source,
    runtime_args,
    save_json,
    sha256,
)
from tutorial import get_cuda_autotune_config, matmul_kernel


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
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    gate()
    verify_extraction()
    torch.manual_seed(42)
    tensors = allocate(*SEED_SHAPE)
    if (ROOT / "baseline.json").exists():
        from common import frozen

        config = frozen()
        ck = launch_source(tensors, config)
        assert ck.asm["ttir"] == (ROOT / "ir/baseline.ttir").read_text()
        print("PASS: frozen source/config/TTIR unchanged; no autotune rerun")
        return
    assert not args.verify, "No baseline frozen yet"
    assert not (
        ROOT / "ir/baseline.ttir"
    ).exists(), "Partial freeze exists; investigate instead of overwriting"
    configs = get_cuda_autotune_config()
    tuner = triton.autotune(configs=configs, key=["M", "N", "K"], cache_results=False)(
        matmul_kernel
    )
    tuner[lambda meta: grid(tensors, meta)](*runtime_args(tensors), ACTIVATION="")
    selected = tuner.best_config
    config = dict(
        experiment="mm-fp16-sm90-fixed-v1",
        meta=dict(selected.kwargs, ACTIVATION=""),
        options=dict(
            num_warps=selected.num_warps,
            num_stages=selected.num_stages,
            num_ctas=1,
            enable_fp_fusion=True,
        ),
        # Source @jit specializes the three literal unit column strides.
        runtime_arg_indices=[0, 1, 2, 3, 4, 5, 6, 8, 10],
        representative_shape=SEED_SHAPE,
        selection_scope=(
            "One group: tutorial autotune at 1024^3; fixed across all tests; not per-shape best"
        ),
        autotune_candidates=[str(c) for c in configs],
        autotune_timings={str(c): value for c, value in tuner.configs_timings.items()},
        source_sha256=sha256(ROOT / "tutorial.py"),
        upstream_sha256=sha256(ROOT / "upstream/03-matrix-multiplication.py.txt"),
        contract_sha256=sha256(ROOT / "optimization_plan.md"),
        common_sha256=sha256(ROOT / "common.py"),
        protocol_sha256={
            name: sha256(ROOT / name)
            for name in ["validate.py", "benchmark.py", "verify_pipeline.py"]
        },
    )
    ck = launch_source(tensors, config)
    baseline = ROOT / "ir/baseline.ttir"
    with baseline.open("x") as f:
        f.write(ck.asm["ttir"])
    config["ttir_sha256"] = digest(ck.asm["ttir"])
    config["source_metadata"] = ck.metadata._asdict()
    save_json(ROOT / "baseline.json", config)
    print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
