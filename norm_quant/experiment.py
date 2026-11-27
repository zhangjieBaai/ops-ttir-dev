"""Frozen TTIR setup, validation, paired benchmark and independent reruns."""

import argparse
import json
import sys
from pathlib import Path

import torch
import triton
import triton.testing

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from frozen_baseline import residual_rms_int8  # noqa: E402

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
    (1, 4096),
    (128, 1024),
    (128, 4096),
    (2048, 4096),
    (512, 8192),
    (513, 4097),
    (32, 16384),
    (64, 257),
]
NAMES = ["baseline", "factor_v1", "joint_v1", "joint_cancel_v1", "stream_v1"]


def allocate(m, n, distribution="normal"):
    x = torch.randn((m, n), device="cuda", dtype=torch.float16)
    r = torch.randn_like(x)
    w = torch.randn(n, device="cuda", dtype=x.dtype).clamp_(-8, 8)
    if distribution == "zero":
        x.zero_()
        r.zero_()
    elif distribution == "constant":
        x.fill_(2)
        r.fill_(-0.5)
    elif distribution == "offset":
        x.mul_(0.5).add_(1000)
        r.mul_(0.5)
    elif distribution == "cancellation":
        r.copy_(-x)
        r[:, 0] = 0
    elif distribution == "tiny":
        x.mul_(1e-4)
        r.mul_(1e-4)
    elif distribution == "signed_weight":
        w[::2] = -8
        w[1::2] = 8
    zstore = torch.full((m * n + 32,), float("nan"), device="cuda")
    qstore = torch.full((m * n + 32,), -128, device="cuda", dtype=torch.int8)
    sstore = torch.full((m + 32,), float("nan"), device="cuda")
    return (
        x,
        r,
        w,
        zstore[: m * n].view(m, n),
        qstore[: m * n].view(m, n),
        sstore[:m],
    ), (zstore, qstore, sstore)


def launch(ck, ts):
    ck[(ts[0].shape[0], 1, 1)](*ts)


def source(ts, warps):
    n = ts[0].shape[1]
    return residual_rms_int8[(ts[0].shape[0],)](
        *ts,
        n,
        triton.next_power_of_2(n),
        num_warps=warps,
        num_stages=3,
        num_ctas=1,
        enable_fp_fusion=True,
    )


def ref(ts, cpu=False):
    x, r, w = [v.cpu().double() if cpu else v.double() for v in ts[:3]]
    z = x + r
    y = (z * w) * torch.rsqrt((z * z).mean(-1, keepdim=True) + 1e-5)
    a = y.abs().amax(-1)
    s = torch.where(a > 0, a / 127, 1.0)
    q = torch.round(torch.clamp(y / s[:, None], -127, 127)).to(torch.int8)
    return z, y, s, q


def validate(ck, shape, seed, small=False):
    torch.manual_seed(seed)
    m, n = shape
    records = []
    for dist in (
        "normal",
        "zero",
        "constant",
        "offset",
        "cancellation",
        "tiny",
        "signed_weight",
    ):
        rows = min(m, 3) if small or dist != "normal" else m
        ts, stores = allocate(rows, n, dist)
        inputs = [t.clone() for t in ts[:3]]
        z, y, s, q = ref(ts)
        launch(ck, ts)
        torch.cuda.synchronize()
        torch.testing.assert_close(ts[3], z.float(), rtol=0, atol=0)
        torch.testing.assert_close(ts[5].double(), s, rtol=3e-5, atol=1e-9)
        qdiff = (ts[4].short() - q.short()).abs()
        assert bool(((ts[4] >= -127) & (ts[4] <= 127)).all())
        assert qdiff.max().item() <= 1
        err = (ts[4].double() * ts[5].double()[:, None] - y).abs()
        limit = 0.501 * s[:, None] + 3e-5 * y.abs().amax(-1, keepdim=True) + 1e-7
        assert bool(torch.all(err <= limit))
        assert bool(torch.isfinite(ts[3]).all() and torch.isfinite(ts[5]).all())
        assert bool(
            torch.isnan(stores[0][-32:]).all() and torch.isnan(stores[2][-32:]).all()
        )
        assert bool((stores[1][-32:] == -128).all())
        assert all(torch.equal(a, b) for a, b in zip(inputs, ts[:3]))
        cpu = rows <= 3
        if cpu:
            cz, cy, cs, cq = ref(ts, cpu=True)
            torch.testing.assert_close(ts[5].cpu().double(), cs, rtol=3e-5, atol=1e-9)
            assert (ts[4].cpu().short() - cq.short()).abs().max().item() <= 1
        records.append(
            {
                "shape": [rows, n],
                "distribution": dist,
                "max_code_error": qdiff.max().item(),
                "different_codes": (qdiff != 0).sum().item(),
                "max_scale_abs_error": (ts[5].double() - s).abs().max().item(),
                "max_reconstruction_error": err.max().item(),
                "cpu_fp64": cpu,
                "canary_input_unchanged": True,
            }
        )
    return records


def setup():
    env = environment()
    manifest_path = ROOT / "manifest.json"
    assert not manifest_path.exists(), "Already frozen; use verify"
    torch.manual_seed(42)
    rows = []
    for m, n in SHAPES:
        ts, _ = allocate(m, n)
        candidates = {}
        for warps in (4, 8):
            ck = source(ts, warps)
            # Correctness gate precedes setup timing as well.
            z, y, s, q = ref(ts)
            torch.testing.assert_close(ts[5].double(), s, rtol=3e-5, atol=1e-9)
            assert (ts[4].short() - q.short()).abs().max().item() <= 1
            values = [
                triton.testing.do_bench_cudagraph(lambda: source(ts, warps), rep=10)
                * 1000
                for _ in range(3)
            ]
            candidates[warps] = values
        import statistics

        warps = min(candidates, key=lambda w: statistics.median(candidates[w]))
        ck = source(ts, warps)
        group = f"m{m}_n{n}_w{warps}"
        path = ROOT / "ir" / group / "baseline.ttir"
        freeze(path, ck.asm["ttir"])
        opts = dict(num_warps=warps, num_stages=3, num_ctas=1, enable_fp_fusion=True)
        rebuilt = compile_file(path, opts)
        gate = pipeline(ck, rebuilt)
        before = [v.clone() for v in ts[3:]]
        launch(rebuilt, ts)
        assert all(torch.equal(a, b) for a, b in zip(before, ts[3:]))
        rows.append(
            {
                "shape": [m, n],
                "group": group,
                "options": opts,
                "tuning_us": candidates,
                "pipeline": gate,
                "source_file": str(ROOT / "frozen_baseline.py"),
                "source_sha256": digest(ROOT / "frozen_baseline.py"),
                "baseline": kernel_info(
                    rebuilt, path, ROOT / "artifacts" / group / "baseline"
                ),
            }
        )
        print(group, "pipeline PASS", flush=True)
    save(manifest_path, {"environment": env, "records": rows})


def run(args):
    env = environment()
    output = ROOT / "results" / args.run
    assert not output.exists()
    output.mkdir(parents=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
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
        m, n = item["shape"]
        input_seed = args.seed + m * 1000003 + n
        torch.manual_seed(input_seed)
        ts, _ = allocate(m, n)
        input_hashes = [tensor_digest(t) for t in ts[:3]]
        src = source(ts, item["options"]["num_warps"])
        assert digest(ROOT / "frozen_baseline.py") == item["source_sha256"]
        kernels = {}
        info = {}
        validation = {}
        for name in names:
            path = ROOT / "ir" / item["group"] / f"{name}.ttir"
            ck = compile_file(path, item["options"])
            if name == "baseline":
                assert path.read_text() == src.asm["ttir"]
                pipeline(src, ck)
            assert ck.metadata.num_warps == src.metadata.num_warps
            info[name] = kernel_info(
                ck, path, ROOT / "artifacts" / item["group"] / name
            )
            validation[name] = validate(ck, (m, n), args.seed, args.check_only)
            kernels[name] = ck
        record = {
            "shape": [m, n],
            "group": item["group"],
            "kernels": info,
            "validation": validation,
            "timing_input_seed": input_seed,
            "timing_input_sha256": input_hashes,
        }
        if not args.check_only:
            funcs = {
                name: (lambda ck=ck: launch(ck, ts)) for name, ck in kernels.items()
            }
            record["timing"] = paired(funcs, args.rounds, args.rep, args.seed + n + m)
            print(
                item["shape"],
                {
                    k: round(v["speedup"], 4)
                    for k, v in record["timing"]["candidates"].items()
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
