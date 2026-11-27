"""Sequential multi-round TTIR validation, parent/anchor/baseline timing."""

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
import traceback
from pathlib import Path

import torch
import triton.testing
from source_port_runner import info
from support import (
    compile_file,
    digest,
    environment,
    freeze,
    pipeline,
    save,
    tensor_digest,
)
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parents[1]


def measure_set(functions, rounds, rep, seed):
    rng = random.Random(seed)
    for fn in functions.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    for fn in functions.values():
        triton.testing.do_bench_cudagraph(fn, rep=100)

    def measure(name):
        return triton.testing.do_bench_cudagraph(functions[name], rep=rep) * 1000

    samples = []
    aa = []
    for turn in range(rounds):
        controls = [measure("parent"), measure("parent")]
        aa.append(
            {"round": turn, "times_us": controls, "ratio": controls[0] / controls[1]}
        )
        order = list(functions)
        rng.shuffle(order)
        times = {name: measure(name) for name in order}
        samples.append(
            {
                "round": turn,
                "order": order,
                "times_us": times,
                "parent_over": {n: times["parent"] / t for n, t in times.items()},
                "baseline_over": {n: times["baseline"] / t for n, t in times.items()},
                "anchor_over": {n: times["anchor"] / t for n, t in times.items()},
            }
        )
    return {
        "samples": samples,
        "aa": aa,
        "aa_median": statistics.median(x["ratio"] for x in aa),
        "median_us": {
            n: statistics.median(x["times_us"][n] for x in samples) for n in functions
        },
        "anchor_over": {
            n: statistics.median(x["anchor_over"][n] for x in samples)
            for n in functions
        },
        "parent_over": {
            n: statistics.median(x["parent_over"][n] for x in samples)
            for n in functions
        },
        "baseline_over": {
            n: statistics.median(x["baseline_over"][n] for x in samples)
            for n in functions
        },
    }


def summarize(records, names):
    return {
        name: {
            metric: {
                "geomean": math.exp(
                    statistics.mean(
                        math.log(r["timing"][metric][name]) for r in records
                    )
                ),
                "worst": min(r["timing"][metric][name] for r in records),
                "best": max(r["timing"][metric][name] for r in records),
            }
            for metric in ("parent_over", "anchor_over", "baseline_over")
        }
        for name in names
    }


def run(args):
    op = ROOT / args.op
    folder = op / "multiround" / f"r{args.round}"
    out = folder / "results" / args.run
    assert not out.exists(), "Result IDs are immutable"
    out.mkdir(parents=True)
    sys.path.insert(0, str(op))
    spec = importlib.util.spec_from_file_location(
        "original_experiment", op / "experiment.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    manifest_path = op / "round2/expanded/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    norm = args.op == "norm_quant"
    names = ["baseline", "anchor", "parent"] + args.names.split(",")
    assert len(names) == len(set(names))
    result = {
        "environment": environment(),
        "seed": args.seed,
        "round_id": args.round,
        "rounds": args.rounds,
        "rep_ms": args.rep,
        "manifest_sha256": digest(manifest_path),
        "failures": {},
        "records": [],
        "pipeline": [],
    }
    bundles = []
    # Finish all candidates' complete C0/C1 checks before any performance sampling.
    for item in manifest["records"]:
        shape = tuple(item["shape"])
        seed = args.seed + shape[0] * 1000003 + shape[1] + (shape[2] if not norm else 0)
        torch.manual_seed(seed)
        ts, _ = old.allocate(*shape)
        options = item["options"]
        stages = ["kernel"] if norm else ["local", "merge"]
        original = {
            s: (
                old.source(ts, options["num_warps"])
                if norm
                else old.source(ts, shape[2], s)
            )
            for s in stages
        }
        kernels, metas, validation = {}, {}, {}
        for name in names:
            if name in result["failures"]:
                continue
            try:
                paths = {}
                ks, km = {}, {}
                for stage in stages:
                    groupdir = (
                        (
                            op / "round2/expanded"
                            if name in ("baseline", "anchor")
                            else folder
                        )
                        / "ir"
                        / item["group"]
                    )
                    if not norm:
                        groupdir /= stage
                    filename = "optimized" if name == "anchor" else name
                    paths[stage] = path = groupdir / f"{filename}.ttir"
                    ck = compile_file(path, options)
                    fields = CUDABackend(ck.metadata.target).parse_options({}).__dict__
                    assert {k: ck.metadata._asdict()[k] for k in fields} == {
                        k: original[stage].metadata._asdict()[k] for k in fields
                    }
                    if name == "baseline":
                        assert path.read_text() == original[stage].asm["ttir"]
                        gate = pipeline(original[stage], ck)
                        result["pipeline"].append(
                            {"group": item["group"], "stage": stage, "gate": gate}
                        )
                    ks[stage] = ck
                    km[stage] = info(
                        ck, path, folder / "artifacts" / item["group"] / stage / name
                    )
                kernel = ks["kernel"] if norm else ks
                checks = old.validate(kernel, shape, args.seed, args.check_only)
                assert all(digest(paths[s]) == km[s]["ir_sha256"] for s in stages)
                kernels[name] = kernel
                metas[name] = km
                validation[name] = checks
            except Exception:
                if name in ("baseline", "anchor", "parent"):
                    raise
                result["failures"][name] = {
                    "shape": shape,
                    "traceback": traceback.format_exc(),
                }
                print("REJECT", name, shape, traceback.format_exc(), flush=True)
                save(out / "validation.json", result)
        bundles.append((item, kernels))
        result["records"].append(
            {
                "shape": shape,
                "group": item["group"],
                "kernels": metas,
                "validation": validation,
                "input_seed": seed,
            }
        )
        save(out / "validation.json", result)
    valid = [name for name in names if name not in result["failures"]]
    result["validated_names"] = valid
    freeze(
        out / "validated_ir.json",
        json.dumps(
            {
                r["group"]: {
                    name: {s: x["ir_sha256"] for s, x in r["kernels"][name].items()}
                    for name in valid
                }
                for r in result["records"]
            },
            indent=2,
        )
        + "\n",
    )
    if not args.check_only:
        for record, (item, kernels) in zip(result["records"], bundles):
            torch.manual_seed(record["input_seed"])
            ts, _ = old.allocate(*record["shape"])
            record["input_sha256"] = [
                tensor_digest(t) for t in (ts[:3] if norm else ts[:1])
            ]
            funcs = {
                name: (lambda ck=kernels[name]: old.launch(ck, ts)) for name in valid
            }
            record["timing"] = measure_set(
                funcs, args.rounds, args.rep, record["input_seed"]
            )
            print(
                record["shape"],
                {n: round(v, 4) for n, v in record["timing"]["parent_over"].items()},
                "AA",
                record["timing"]["aa_median"],
                flush=True,
            )
            save(out / "results.json", result)
        result["summary"] = summarize(result["records"], valid)
        result["aa_passed"] = all(
            0.97 <= r["timing"]["aa_median"] <= 1.03 for r in result["records"]
        )
        print(result["summary"], flush=True)
    result["passed"] = True
    save(out / "results.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True, choices=["norm_quant", "partial_topk"])
    p.add_argument("--run", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--round", type=int, choices=[3, 4, 5], required=True)
    run(p.parse_args())
