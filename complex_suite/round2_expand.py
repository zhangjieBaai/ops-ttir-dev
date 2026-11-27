"""Freeze expanded baselines from original sources and extend direct TTIR parents."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from support import (
    compile_file,
    digest,
    environment,
    freeze,
    kernel_info,
    pipeline,
    save,
)

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(opname):
    folder = ROOT / opname
    sys.path.insert(0, str(folder))
    original = load("original_experiment", folder / "experiment.py")
    norm = opname == "norm_quant"
    if norm:
        shapes = [
            (m, n)
            for n in (257, 1024, 4096, 4097, 8192, 16384)
            for m in (1, 32, 128, 512, 2048)
        ]
        shapes[shapes.index((32, 257))] = (64, 257)
        shapes[shapes.index((512, 4097))] = (513, 4097)
        parent_builder = load(
            "norm_original_candidates", folder / "make_candidates.py"
        ).candidates
    else:
        shapes = [
            (m, n, k)
            for n, k in (
                (65536, 8),
                (65537, 8),
                (131072, 8),
                (262144, 8),
                (262144, 16),
                (1048576, 8),
            )
            for m in (1, 4, 16, 64, 128)
        ]
        parent_builder = load("topk_original_carry", folder / "make_carry.py").carry
    assert len(shapes) == 30 and len(set(shapes)) == 30
    old = json.loads((folder / "manifest.json").read_text())
    assert set(tuple(r["shape"]) for r in old["records"]).issubset(shapes)
    protocol = {
        "op": opname,
        "shapes": shapes,
        "count": 30,
        "config_policy": (
            "fixed original width-family; N4097=4warps, others=8"
            if norm
            else "fixed4warps/3stages, chunk1024"
        ),
        "original_source_sha256": digest(folder / "frozen_baseline.py"),
        "original_manifest_sha256": digest(folder / "manifest.json"),
    }
    target = folder / "round2/expanded"
    freeze(target / "protocol.json", json.dumps(protocol, indent=2) + "\n")
    assert not (target / "manifest.json").exists()
    env = environment()
    records = []
    for shape in shapes:
        torch.manual_seed(42)
        m, n = shape[:2]
        group = f"m{m}_n{n}" + (
            f"_w{4 if n == 4097 else 8}" if norm else f"_k{shape[2]}"
        )
        options = dict(
            num_warps=4 if not norm or n == 4097 else 8,
            num_stages=3,
            num_ctas=1,
            enable_fp_fusion=True,
        )
        ts, _ = original.allocate(*shape)
        stages = ["kernel"] if norm else ["local", "merge"]
        kernels, gates = {}, {}
        for stage in stages:
            ck = (
                original.source(ts, options["num_warps"])
                if norm
                else original.source(ts, shape[2], stage)
            )
            dest = target / "ir" / group
            if not norm:
                dest /= stage
            freeze(dest / "baseline.ttir", ck.asm["ttir"])
            parent = parent_builder(ck.asm["ttir"])
            if norm:
                parent = parent["stream_v1"]
            freeze(dest / "parent.ttir", parent)
            rebuilt = compile_file(dest / "baseline.ttir", options)
            gates[stage] = pipeline(ck, rebuilt)
            kernels[stage] = kernel_info(
                rebuilt,
                dest / "baseline.ttir",
                target / "artifacts" / group / stage / "baseline",
            )
        records.append(
            {
                "shape": shape,
                "group": group,
                "options": options,
                "pipeline": gates,
                "baseline": kernels,
            }
        )
        save(target / "setup_progress.json", {"environment": env, "records": records})
        print("PIPELINE", shape, "PASS", flush=True)
    save(
        target / "manifest.json",
        {"environment": env, "records": records, "protocol": protocol},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["norm_quant", "partial_topk"], required=True)
    main(p.parse_args().op)
