"""Numerically validate source witnesses, then measure against frozen TTIR."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics as st
import sys
from pathlib import Path

import torch
import triton
import triton.testing

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common import EPS, OPTIONS, allocate, compile_ir, launch  # noqa: E402
from environment import verify_distribution_files  # noqa: E402
from validate import cases  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob_sha(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else value.encode()
    ).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def measure(functions, rounds, rep, seed):
    rng = random.Random(seed)
    result, controls = {n: [] for n in functions if n != "frozen_baseline"}, []
    for fn in functions.values():
        triton.testing.do_bench_cudagraph(fn, rep=100)

    def time(name):
        return triton.testing.do_bench_cudagraph(functions[name], rep=rep) * 1000

    for turn in range(rounds):
        aa = [time("frozen_baseline"), time("frozen_baseline")]
        controls.append(dict(round=turn, us=aa, ratio=aa[0] / aa[1]))
        names = list(result)
        rng.shuffle(names)
        for name in names:
            order = ["frozen_baseline", name]
            rng.shuffle(order)
            times = {n: time(n) for n in order}
            result[name].append(
                dict(
                    round=turn,
                    order=order,
                    us=times,
                    speedup=times["frozen_baseline"] / times[name],
                )
            )
    return dict(
        samples=result, aa=controls, aa_median=st.median(c["ratio"] for c in controls)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Reserve an idle GPU first"
    environment = verify_distribution_files()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert sha(ROOT.parent / "tutorial.py") == manifest["tutorial_sha256"]
    torch.manual_seed(42)
    ts = allocate(1151, 8192)
    x, y, w, b, mean, rstd = ts[:6]
    compiled = {"frozen_baseline": compile_ir("baseline")}
    result = dict(
        environment=environment,
        visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        options=OPTIONS,
        source_manifest_sha256=sha(ROOT / "manifest.json"),
        pipeline={},
        accuracy=[],
        passed=False,
    )
    for name, entry in manifest["variants"].items():
        source_path = ROOT / entry["source"]
        assert sha(source_path) == entry["sha256"]
        spec = importlib.util.spec_from_file_location(f"step_{name}", source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = module.kernel[(1151,)](
            x,
            y,
            w,
            b,
            mean,
            rstd,
            x.stride(0),
            8192,
            EPS,
            BLOCK_SIZE=module.BLOCK_SIZE,
            **OPTIONS,
        )
        exported = args.output / f"{name}.ttir"
        exported.write_text(original.asm["ttir"])
        rebuilt = triton.compile(str(exported), options=OPTIONS)
        rebuilt._init_handles()
        assert all(original.asm[s] == rebuilt.asm[s] for s in ("llir", "ptx", "cubin"))
        assert all(getattr(rebuilt.metadata, k) == v for k, v in OPTIONS.items())
        compiled[name] = rebuilt
        result["pipeline"][name] = dict(
            source_sha256=sha(source_path),
            ttir_sha256=sha(exported),
            downstream={s: blob_sha(original.asm[s]) for s in ("llir", "ptx", "cubin")},
            source_file_identical=True,
            n_regs=rebuilt.n_regs,
            n_spills=rebuilt.n_spills,
        )
        print(
            "PIPELINE",
            name,
            "regs",
            rebuilt.n_regs,
            "spills",
            rebuilt.n_spills,
            flush=True,
        )
    save(args.output / "pipeline.json", result)
    suite = cases("full") + cases("round2-boundaries")
    for i, (m, n, pad, distribution, eps) in enumerate(suite):
        tensors = allocate(m, n, pad)
        x, y, w, b, mean, rstd, storage = tensors
        if distribution == "constant":
            x.fill_(2.0)
        elif distribution == "offset":
            x.add_(1000.0)
        elif distribution == "normal":
            x.mul_(0.5).add_(-2.3)
        reference = torch.nn.functional.layer_norm(x, (n,), w, b, eps).float()
        double = x.double()
        mean_ref = double.mean(-1).float()
        rstd_ref = torch.rsqrt(double.var(-1, unbiased=False) + eps).float()
        del double
        before = [t.clone() for t in (x, w, b)]
        for name, ck in compiled.items():
            storage.fill_(float("nan"))
            mean.fill_(float("nan"))
            rstd.fill_(float("nan"))
            launch(ck, tensors, eps)
            error = (y.float() - reference).abs().max().item()
            assert error < 1e-2, (name, m, n, distribution, error)
            torch.testing.assert_close(mean, mean_ref, rtol=2e-5, atol=1e-5)
            torch.testing.assert_close(rstd, rstd_ref, rtol=2e-4, atol=1e-5)
            assert bool(torch.isfinite(y).all()) and bool(storage[:, n:].isnan().all())
            assert all(torch.equal(a, t) for a, t in zip(before, (x, w, b)))
            result["accuracy"].append(
                dict(
                    name=name,
                    M=m,
                    N=n,
                    pad=pad,
                    distribution=distribution,
                    eps=eps,
                    y_error=error,
                    passed=True,
                )
            )
        if i % 20 == 0:
            print("ACCURACY", i + 1, "/", len(suite), flush=True)
    result["accuracy_passed"] = True
    save(args.output / "accuracy.json", result)
    if args.benchmark:
        result["benchmark"] = []
        for n in range(1024, 15873, 512):
            torch.manual_seed(42 + n)
            tensors = allocate(4096, n)
            tensors[0].mul_(0.5).add_(-2.3)
            functions = {
                name: (lambda ck=ck: launch(ck, tensors))
                for name, ck in compiled.items()
            }
            timing = measure(functions, rounds=5, rep=10, seed=42 + n)
            input_hashes = [
                blob_sha(t.cpu().numpy().tobytes())
                for t in (tensors[0], tensors[2], tensors[3])
            ]
            result["benchmark"].append(
                dict(
                    M=4096,
                    N=n,
                    input_seed=42 + n,
                    input_sha256=input_hashes,
                    timing=timing,
                )
            )
            print("BENCHMARK", n, "AA", timing["aa_median"], flush=True)
            save(args.output / "benchmark.json", result)
        result["aa_passed"] = all(
            0.97 <= r["timing"]["aa_median"] <= 1.03 for r in result["benchmark"]
        )
        assert result["aa_passed"], "Preserve failed A/A run and use a fresh result ID"
        result["benchmark_summary"] = {}
        for name in manifest["variants"]:
            values = [
                st.median(s["speedup"] for s in r["timing"]["samples"][name])
                for r in result["benchmark"]
            ]
            result["benchmark_summary"][name] = dict(
                geometric=math.exp(st.mean(map(math.log, values))),
                arithmetic=st.mean(values),
                minimum=min(values),
            )
        print(result["benchmark_summary"], flush=True)
    result["passed"] = True
    save(args.output / "results.json", result)


if __name__ == "__main__":
    main()
