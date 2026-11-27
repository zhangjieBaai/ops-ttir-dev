"""Three-way direct-TTIR validation and timing on a frozen 30-point matrix."""

import argparse
import json
import math
import random
import statistics
import struct
import sys
from pathlib import Path

import torch
import triton
import triton.testing
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parents[1]))
import common as c  # noqa: E402

from complex_suite.support import (  # noqa: E402
    compile_file,
    digest,
    environment,
    kernel_info,
    pipeline,
    save,
    sha,
    tensor_digest,
)

DISTRIBUTIONS = (
    "normal",
    "ties",
    "group_ties",
    "bias_dominant",
    "saturated",
    "negative",
)
DEFAULTS = [
    "r2_ordered",
    "r2_gather",
    "r2_rank",
    "r2_ordered_gather",
    "r2_rank_ordered_gather",
    "r2_unroll",
]


def text_hashes(blob):
    assert blob[:6] == b"\x7fELF\x02\x01"
    off = struct.unpack_from("<Q", blob, 40)[0]
    size, count, ni = struct.unpack_from("<HHH", blob, 58)
    hdrs = [
        struct.unpack_from("<IIQQQQIIQQ", blob, off + i * size) for i in range(count)
    ]
    n = hdrs[ni]
    names = blob[n[4] : n[4] + n[5]]
    return {
        names[h[0] :].split(b"\0", 1)[0].decode(): sha(blob[h[4] : h[4] + h[5]])
        for h in hdrs
        if names[h[0] :].startswith(b".text")
    }


def compile_pair(e, name):
    group = ROOT / "ir" / f"e{e}_w4"
    paths = [group / "baseline_s1.ttir", group / f"{name}_s2.ttir"]
    pairs, info = [], []
    for stage, path in enumerate(paths, 1):
        ck = compile_file(path, c.options(4))
        meta = kernel_info(
            ck, path, ROOT / "artifacts" / f"e{e}_w4" / name / f"s{stage}"
        )
        meta["text_sha256"] = text_hashes(ck.asm["cubin"])
        pairs.append(ck)
        info.append(meta)
    return tuple(pairs), info


def verify(points, kernels):
    checks = []
    for item in points:
        m, e = item["tokens"], item["experts"]
        t = c.allocate(m, e, seed=441)
        src = c.source_launch(t, 4)
        output = [t[k].clone() for k in ("g", "w", "i")]
        base = kernels[e, "baseline"]
        c.launch(base, t)
        assert all(torch.equal(a, t[k]) for a, k in zip(output, ("g", "w", "i")))
        stages = []
        for stage, (s, b) in enumerate(zip(src, base), 1):
            path = ROOT / "ir" / f"e{e}_w4" / f"baseline_s{stage}.ttir"
            assert s.asm["ttir"] == path.read_text()
            stages.append(pipeline(s, b))
        checks.append({"shape": [m, e], "stages": stages, "outputs_equal": True})
    return checks


def validate(points, names, kernels, seed):
    records = []
    failed = set()
    for item in points:
        m, e = item["tokens"], item["experts"]
        for dist in DISTRIBUTIONS:
            t = c.allocate(m, e, dist, seed + m + e)
            originals = (t["x"].clone(), t["b"].clone())
            rw, ri = c.reference(t)
            c.launch(kernels[e, "baseline"], t)
            reference = {k: t[k].clone() for k in ("g", "w", "i")}
            for name in names:
                if name in failed:
                    continue
                for k in ("g", "w"):
                    t[k].fill_(float("nan"))
                t["i"].fill_(-777)
                row = {"shape": [m, e], "distribution": dist, "variant": name}
                try:
                    c.launch(kernels[e, name], t)
                    torch.cuda.synchronize()
                    c.check_guards(t)
                    assert all(
                        torch.equal(a, t[k]) for a, k in zip(originals, ("x", "b"))
                    )
                    equal = {k: torch.equal(t[k], reference[k]) for k in reference}
                    assert all(equal.values()), str(equal)
                    assert torch.equal(t["i"].cpu().long(), ri)
                    torch.testing.assert_close(t["w"].cpu(), rw, rtol=2e-5, atol=2e-6)
                    row.update(
                        status="PASS",
                        outputs_equal=equal,
                        max_abs_cpu=(t["w"].cpu() - rw).abs().max().item(),
                    )
                except Exception as exc:
                    row.update(status="FAIL", error=str(exc))
                    failed.add(name)
                    print("FAIL", row, flush=True)
                records.append(row)
        print("validated", m, e, "failed", sorted(failed), flush=True)
    assert not {"baseline", "parent"}.intersection(failed)
    return {
        "checks": records,
        "failed": sorted(failed),
        "passed_variants": [n for n in names if n not in failed],
    }


def measure(t, pairs, rounds, rep, seed):
    rng = random.Random(seed)
    for pair in pairs.values():
        triton.testing.do_bench_cudagraph(lambda p=pair: c.launch(p, t), rep=100)
    samples, aa = [], []
    for turn in range(rounds):
        ats = [
            triton.testing.do_bench_cudagraph(
                lambda: c.launch(pairs["parent"], t), rep=rep
            )
            * 1000
            for _ in range(2)
        ]
        aa.append({"round": turn, "us": ats, "ratio": ats[0] / ats[1]})
        order = list(pairs)
        rng.shuffle(order)
        us = {
            name: triton.testing.do_bench_cudagraph(
                lambda p=pairs[name]: c.launch(p, t), rep=rep
            )
            * 1000
            for name in order
        }
        samples.append(
            {
                "round": turn,
                "order": order,
                "us": us,
                "parent_over_candidate": {
                    n: us["parent"] / v
                    for n, v in us.items()
                    if n not in ("baseline", "parent")
                },
                "baseline_over_candidate": {
                    n: us["baseline"] / v
                    for n, v in us.items()
                    if n not in ("baseline", "parent")
                },
                "baseline_over_parent": us["baseline"] / us["parent"],
            }
        )
    return {
        "samples": samples,
        "aa": aa,
        "aa_median": statistics.median(r["ratio"] for r in aa),
        "median_us": {n: statistics.median(r["us"][n] for r in samples) for n in pairs},
        "parent_over_candidate": {
            n: statistics.median(r["parent_over_candidate"][n] for r in samples)
            for n in pairs
            if n not in ("baseline", "parent")
        },
        "baseline_over_candidate": {
            n: statistics.median(r["baseline_over_candidate"][n] for r in samples)
            for n in pairs
            if n not in ("baseline", "parent")
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--variants", nargs="+", default=DEFAULTS)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    out = ROOT / "results" / args.run
    assert not out.exists()
    out.mkdir(parents=True)
    env = environment()
    manifest = json.loads((ROOT / "manifest30.json").read_text())
    names = list(dict.fromkeys(["baseline", "parent", *args.variants]))
    kernels, infos, errors = {}, {}, {}
    for e in (256, 512):
        for name in names:
            try:
                pair, meta = compile_pair(e, name)
                kernels[e, name], infos[f"e{e}/{name}"] = pair, meta
                if name != "baseline":
                    for s, b in zip(pair, kernels[e, "baseline"]):
                        fields = (
                            CUDABackend(b.metadata.target).parse_options({}).__dict__
                        )
                        assert {k: s.metadata._asdict()[k] for k in fields} == {
                            k: b.metadata._asdict()[k] for k in fields
                        }
                print(
                    "compiled",
                    e,
                    name,
                    pair[1].n_regs,
                    pair[1].metadata.shared,
                    flush=True,
                )
            except Exception as exc:
                errors[f"e{e}/{name}"] = repr(exc)
                print("COMPILE FAIL", e, name, repr(exc), flush=True)
    save(out / "compile.json", {"metadata": infos, "errors": errors})
    if args.compile_only:
        return
    names = [n for n in names if all((e, n) in kernels for e in (256, 512))]
    assert "baseline" in names and "parent" in names
    gate = verify(manifest["points"], kernels)
    save(out / "pipeline.json", gate)
    accuracy = validate(manifest["points"], names, kernels, args.seed)
    save(out / "accuracy.json", accuracy)
    if args.check_only:
        return
    names = accuracy["passed_variants"]
    result = {
        "environment": env,
        "manifest_sha256": digest(ROOT / "manifest30.json"),
        "seed": args.seed,
        "rounds": args.rounds,
        "rep_ms": args.rep,
        "names": names,
        "kernels": infos,
        "records": [],
    }
    for item in manifest["points"]:
        m, e = item["tokens"], item["experts"]
        seed = args.seed + m * 1000003 + e
        t = c.allocate(m, e, seed=seed)
        row = {
            "shape": [m, e],
            "input_seed": seed,
            "input_sha256": {k: tensor_digest(t[k]) for k in ("x", "b")},
            "timing": measure(
                t, {n: kernels[e, n] for n in names}, args.rounds, args.rep, seed
            ),
        }
        result["records"].append(row)
        save(out / "performance.json", result)
        print(
            "bench",
            m,
            e,
            row["timing"]["parent_over_candidate"],
            "AA",
            row["timing"]["aa_median"],
            flush=True,
        )
    result["summary"] = {}
    for name in names:
        if name in ("baseline", "parent"):
            continue
        result["summary"][name] = {}
        for factor in ("parent_over_candidate", "baseline_over_candidate"):
            values = [r["timing"][factor][name] for r in result["records"]]
            result["summary"][name][factor] = {
                "geomean": math.exp(statistics.mean(math.log(v) for v in values)),
                "worst": min(values),
                "best": max(values),
            }
    result["aa_passed"] = all(
        0.97 <= r["timing"]["aa_median"] <= 1.03 for r in result["records"]
    )
    save(out / "performance.json", result)
    print(result["summary"], "AA_PASS", result["aa_passed"], flush=True)


if __name__ == "__main__":
    main()
