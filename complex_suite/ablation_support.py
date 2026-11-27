"""Validate and measure immutable, operator-local ablation manifests."""

import argparse
import importlib.util
import json
import math
import statistics
import sys

import torch

from .multiround_runner import measure_set
from .source_port_runner import info
from .support import compile_file, digest, environment, pipeline, save, tensor_digest


def run(op):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    folder = op / "ablation_v1"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    out = folder / "results" / args.run
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(op))
    spec = importlib.util.spec_from_file_location(
        "original_experiment", op / "experiment.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    norm = op.name == "norm_quant"
    names = manifest["names"]
    assert "baseline" in names and "final" in names
    result = dict(
        environment=environment(),
        seed=args.seed,
        rounds=args.rounds,
        rep_ms=args.rep,
        manifest_sha256=digest(manifest_path),
        names=names,
        records=[],
        pipeline=[],
        passed=False,
    )
    bundles = []
    for item in manifest["records"]:
        shape = tuple(item["shape"])
        torch.manual_seed(args.seed)
        ts, _ = old.allocate(*shape)
        stages = ["kernel"] if norm else ["local", "merge"]
        kernels, metadata, checks = {}, {}, {}
        for name in names:
            ks, km = {}, {}
            for stage in stages:
                path = op.parent / item["paths"][name][stage]
                assert digest(path) == item["sha256"][name][stage]
                ck = compile_file(path, item["options"])
                if name == "baseline":
                    src = (
                        old.source(ts, item["options"]["num_warps"])
                        if norm
                        else old.source(ts, shape[2], stage)
                    )
                    assert path.read_text() == src.asm["ttir"]
                    result["pipeline"].append(
                        dict(group=item["group"], stage=stage, gate=pipeline(src, ck))
                    )
                ks[stage] = ck
                km[stage] = info(
                    ck,
                    path,
                    op / "artifacts/ablation_v1" / item["group"] / stage / name,
                )
            kernel = ks["kernel"] if norm else ks
            checks[name] = old.validate(kernel, shape, args.seed, args.check_only)
            kernels[name], metadata[name] = kernel, km
        bundles.append(kernels)
        result["records"].append(
            dict(shape=shape, group=item["group"], kernels=metadata, validation=checks)
        )
        save(out / "validation.json", result)
        print("VALIDATED", shape, flush=True)
    # Every shape and every candidate passes before the first performance sample.
    if not args.check_only:
        for record, kernels in zip(result["records"], bundles):
            seed = args.seed + record["shape"][0] * 1000003 + sum(record["shape"][1:])
            torch.manual_seed(seed)
            ts, _ = old.allocate(*record["shape"])
            record["input_seed"] = seed
            record["input_sha256"] = [
                tensor_digest(t) for t in (ts[:3] if norm else ts[:1])
            ]
            functions = {n: (lambda ck=kernels[n]: old.launch(ck, ts)) for n in names}
            # Preserve a real parent route; otherwise use final for the A/A control.
            functions.setdefault("parent", functions["final"])
            functions.setdefault("anchor", functions["baseline"])
            record["timing"] = measure_set(functions, args.rounds, args.rep, seed)
            print(record["shape"], record["timing"]["baseline_over"], flush=True)
            save(out / "results.json", result)
        result["summary"] = {
            n: dict(
                geomean=math.exp(
                    statistics.mean(
                        math.log(r["timing"]["baseline_over"][n])
                        for r in result["records"]
                    )
                ),
                worst=min(r["timing"]["baseline_over"][n] for r in result["records"]),
            )
            for n in names
        }
        result["aa_passed"] = all(
            0.97 <= r["timing"]["aa_median"] <= 1.03 for r in result["records"]
        )
        assert result[
            "aa_passed"
        ], "A/A gate failed; preserve this run and use a new result ID"
    result["passed"] = True
    save(out / "results.json", result)
    print(result.get("summary", "validation passed"), flush=True)
