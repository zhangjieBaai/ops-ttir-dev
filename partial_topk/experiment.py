"""Frozen two-stage Top-K, precise tie checks and paired whole-sequence timings."""

import argparse
import json
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from frozen_baseline import partial_select  # noqa: E402

from complex_suite.support import (  # noqa: E402
    compile_file,
    digest,
    environment,
    freeze,
    kernel_info,
    paired,
    pipeline,
    save,
    summarize,
    tensor_digest,
)

SHAPES = [
    (1, 65536, 8),
    (16, 65536, 8),
    (128, 65536, 8),
    (16, 262144, 8),
    (16, 262144, 16),
    (4, 65537, 8),
]
OPTIONS = dict(num_warps=4, num_stages=3, num_ctas=1, enable_fp_fusion=True)
NAMES = ["baseline", "joint_v1", "local_joint_v1", "joint_cg_v1"]


def allocate(m, n, k, distribution="normal"):
    x = torch.randn((m, n), device="cuda")
    if distribution == "all_equal":
        x.fill_(2)
    elif distribution == "integer_ties":
        x.copy_(torch.randint(-4, 5, (m, n), device="cuda").float())
    elif distribution == "signed_zero":
        x.zero_()
        x[:, ::2] = -0.0
    elif distribution == "increasing":
        x.copy_(torch.arange(n, device="cuda").float()[None, :])
    elif distribution in ("positive_subnormal", "negative_subnormal"):
        bits = torch.arange(1, n + 1, device="cuda", dtype=torch.int32) % 8388607 + 1
        values = bits.view(torch.float32)
        x.copy_(
            values[None, :]
            if distribution == "positive_subnormal"
            else -values[None, :]
        )
    elif distribution == "lowest_finite":
        x.fill_(-torch.finfo(torch.float32).max)
    chunks = triton.cdiv(n, 1024)
    pv = torch.full((m * chunks * k + 32,), float("nan"), device="cuda")
    pi = torch.full((m * chunks * k + 32,), -123, device="cuda", dtype=torch.int32)
    v = torch.full((m * k + 32,), float("nan"), device="cuda")
    i = torch.full((m * k + 32,), -123, device="cuda", dtype=torch.int32)
    return (
        x,
        pv[: m * chunks * k],
        pi[: m * chunks * k],
        v[: m * k].view(m, k),
        i[: m * k].view(m, k),
    ), (pv, pi, v, i)


def args_for(ts, stage):
    x, pv, pi, v, i = ts
    return (x, pi, pv, pi) if stage == "local" else (pv, pi, v, i)


def grid(ts, stage):
    m, n = ts[0].shape
    return (m, triton.cdiv(n, 1024) if stage == "local" else 1, 1)


def source(ts, k, stage):
    n = ts[0].shape[1]
    chunks = triton.cdiv(n, 1024)
    length = n if stage == "local" else chunks * k
    block = 1024 if stage == "local" else triton.next_power_of_2(length)
    return partial_select[grid(ts, stage)](
        *args_for(ts, stage),
        length,
        k,
        block,
        chunks if stage == "local" else 1,
        stage == "merge",
        **OPTIONS,
    )


def launch(kernels, ts):
    for stage in ("local", "merge"):
        kernels[stage][grid(ts, stage)](*args_for(ts, stage))


def validate(kernels, shape, seed, small=False):
    m, n, k = shape
    torch.manual_seed(seed)
    records = []
    for dist in (
        "normal",
        "all_equal",
        "integer_ties",
        "signed_zero",
        "increasing",
        "positive_subnormal",
        "negative_subnormal",
        "lowest_finite",
    ):
        rows = min(m, 2) if small or dist != "normal" else m
        ts, stores = allocate(rows, n, k, dist)
        before = ts[0].clone()
        ids = torch.argsort(ts[0], dim=-1, descending=True, stable=True)[:, :k]
        values = torch.gather(ts[0], -1, ids)
        launch(kernels, ts)
        torch.cuda.synchronize()
        assert torch.equal(ts[4].long(), ids)
        assert torch.equal(ts[3], values)
        assert torch.equal(before, ts[0])
        assert all(
            (
                bool(torch.isnan(a[-32:]).all())
                if a.dtype == torch.float32
                else bool((a[-32:] == -123).all())
            )
            for a in stores
        )
        cpu = rows <= 2
        if cpu:
            cpu_ids = torch.argsort(
                before.cpu().double(), dim=-1, descending=True, stable=True
            )[:, :k]
            assert torch.equal(ts[4].cpu().long(), cpu_ids)
        records.append(
            {
                "shape": [rows, n, k],
                "distribution": dist,
                "indices_exact": True,
                "values_exact": True,
                "canary_input_unchanged": True,
                "cpu_fp64_stable_sort": cpu,
            }
        )
    return records


def setup():
    assert not (ROOT / "manifest.json").exists()
    env = environment()
    rows = []
    torch.manual_seed(42)
    for m, n, k in SHAPES:
        ts, _ = allocate(m, n, k)
        group = f"m{m}_n{n}_k{k}"
        info = {}
        kernels = {}
        for stage in ("local", "merge"):
            src = source(ts, k, stage)
            path = ROOT / "ir" / group / stage / "baseline.ttir"
            freeze(path, src.asm["ttir"])
            rebuilt = compile_file(path, OPTIONS)
            gate = pipeline(src, rebuilt)
            info[stage] = {
                "pipeline": gate,
                "baseline": kernel_info(
                    rebuilt, path, ROOT / "artifacts" / group / stage / "baseline"
                ),
            }
            kernels[stage] = rebuilt
        before = [t.clone() for t in ts[1:]]
        launch(kernels, ts)
        assert all(torch.equal(a, b) for a, b in zip(before, ts[1:]))
        validation = validate(kernels, (m, n, k), 42)
        rows.append(
            {
                "shape": [m, n, k],
                "group": group,
                "options": OPTIONS,
                "stages": info,
                "validation": validation,
            }
        )
        print(group, "pipeline/reference PASS", flush=True)
    save(
        ROOT / "manifest.json",
        {
            "environment": env,
            "source_sha256": digest(ROOT / "frozen_baseline.py"),
            "records": rows,
        },
    )


def run(args):
    env = environment()
    output = ROOT / "results" / args.run
    assert not output.exists()
    output.mkdir(parents=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["source_sha256"] == digest(ROOT / "frozen_baseline.py")
    names = args.names.split(",")
    assert names[0] == "baseline"
    result = {
        "environment": env,
        "seed": args.seed,
        "rounds": args.rounds,
        "rep_ms": args.rep,
        "manifest_sha256": digest(ROOT / "manifest.json"),
        "records": [],
    }
    for item in manifest["records"]:
        m, n, k = item["shape"]
        input_seed = args.seed + m * 1000003 + n + k
        torch.manual_seed(input_seed)
        ts, _ = allocate(m, n, k)
        input_hash = tensor_digest(ts[0])
        sources = {stage: source(ts, k, stage) for stage in ("local", "merge")}
        kernels, info, validation = {}, {}, {}
        for name in names:
            kernels[name], info[name] = {}, {}
            for stage in ("local", "merge"):
                use = (
                    "joint_v1"
                    if name == "local_joint_v1" and stage == "local"
                    else "baseline" if name == "local_joint_v1" else name
                )
                path = ROOT / "ir" / item["group"] / stage / f"{use}.ttir"
                ck = compile_file(path, OPTIONS)
                if name == "baseline":
                    assert sources[stage].asm["ttir"] == path.read_text()
                    pipeline(sources[stage], ck)
                info[name][stage] = kernel_info(
                    ck, path, ROOT / "artifacts" / item["group"] / stage / use
                )
                kernels[name][stage] = ck
            validation[name] = validate(
                kernels[name], (m, n, k), args.seed, args.check_only
            )
        record = {
            "shape": [m, n, k],
            "group": item["group"],
            "kernels": info,
            "validation": validation,
            "timing_input_seed": input_seed,
            "timing_input_sha256": input_hash,
        }
        if not args.check_only:
            funcs = {
                name: (lambda ks=ks: launch(ks, ts)) for name, ks in kernels.items()
            }
            record["timing"] = paired(
                funcs, args.rounds, args.rep, args.seed + m + n + k
            )
            print(
                item["shape"],
                {
                    n: round(v["speedup"], 4)
                    for n, v in record["timing"]["candidates"].items()
                },
                "AA",
                record["timing"]["aa_median"],
                flush=True,
            )
        result["records"].append(record)
        save(output / "results.json", result)
    result["passed"] = True
    if not args.check_only:
        result["summary"] = summarize(result["records"])
        result["aa_passed"] = all(
            0.97 <= r["timing"]["aa_median"] <= 1.03 for r in result["records"]
        )
        print(result["summary"], flush=True)
    save(output / "results.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setup", action="store_true")
    p.add_argument("--run", default="screen")
    p.add_argument("--names", default=",".join(NAMES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    setup() if a.setup else run(a)
