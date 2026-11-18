"""Native paired six-family TTIR experiments, bound to the frozen baseline contract."""

import argparse
import json
import math
import os
import random
import re
import statistics
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
import triton.testing
from common import (
    ATOL,
    ROOT,
    RTOL,
    allocate,
    baseline_hashes,
    compile_ir,
    digest,
    dump_kernel,
    frozen,
    launch,
    save_json,
    sha256,
)
from validate import family_cases
from verify_pipeline import snapshot

TUNING = [256, 384, 640, 768, 1024, 1408, 1536, 2048, 2816, 4096, 2176, 2304]
NAMES = [
    "r1_canonical",
    "r1_unsigned",
    "r1_wrap_select",
    "r1_k_aligned",
    "r1_cache_cg",
    "r1_unroll2",
]


def idle():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    others = [
        line for line in raw.splitlines() if int(line.split(",", 1)[0]) != os.getpid()
    ]
    assert not others, f"External GPU workload: {others}"


def entry(ck):
    text = ck.src.src.split("tt.func public @", 1)[1].split(" attributes", 1)[0]
    text = re.sub(r' loc\("[^"\n]*"\(#loc\w*\)\)', "", text)
    return re.sub(r" loc\(#loc\w*\)", "", text)


def compatible(baseline, candidate, config):
    assert candidate.src.ext == baseline.src.ext == "ttir"
    assert candidate.metadata.target == baseline.metadata.target
    assert entry(candidate) == entry(baseline), "Runtime signature/attributes changed"
    # The baseline pipeline already checked all backend option fields.
    for key in baseline.metadata._fields:
        if key in [
            "hash",
            "shared",
            "tmem_size",
            "tensordesc_meta",
            "global_scratch_size",
            "global_scratch_align",
            "profile_scratch_size",
            "profile_scratch_align",
        ]:
            continue
        assert getattr(baseline.metadata, key) == getattr(candidate.metadata, key), key
    assert config["options"]["num_warps"] == candidate.metadata.num_warps


def executable_hashes(ck):
    data = ck.asm["cubin"]
    assert data[:6] == b"\x7fELF\x02\x01", "Expected little-endian ELF64 cubin"
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", data, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, section_offset + i * entry_size)
        for i in range(count)
    ]
    names = sections[names_index]
    strings = data[names[4] : names[4] + names[5]]
    hashes = {}
    for section in sections:
        if section[2] & 4:
            name = strings[section[0] :].split(b"\x00", 1)[0].decode()
            hashes[name] = digest(data[section[4] : section[4] + section[5]])
    assert hashes
    return hashes


def test_cases(config, suite):
    result = [(n, n, n, 0, "normal") for n in config["square_sizes"]]
    if suite == "sanitizer":
        result = []
    result += [(*shape, pad, dist) for shape, pad, dist in family_cases(suite, config)]
    # Aligned-K fast path boundaries and odd-number-of-tile-loop counts.
    bm, bn, bk = [
        config["meta"][k] for k in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]
    ]
    result += [
        (bm + 16, bn + 16, k, 16, "normal")
        for k in [bk - 16, bk, bk + 16, 2 * bk, 3 * bk]
        if k > 0
    ]
    return list(dict.fromkeys(result))


def validate(ck, config, seed, suite="full"):
    torch.manual_seed(seed)
    records = []
    passed = True
    for m, n, k, pad, dist in test_cases(config, suite):
        ts = allocate(m, n, k, pad=pad)
        a, b, c, storage = ts
        if dist == "zero":
            a.zero_()
        elif dist == "constant":
            a.fill_(0.25)
            b.fill_(0.25)
        elif dist == "alternating":
            b[::2].copy_(b[1::2])
            a[:, 1::2].copy_(-a[:, ::2])
        elif dist == "large":
            a.mul_(4)
            b.mul_(4)
        elif dist == "small":
            a.mul_(1e-3)
            b.mul_(1e-3)
        before = [a.clone(), b.clone()]
        refs = {
            "native_fp16": torch.mm(a, b),
            "native_fp32": torch.mm(a.float(), b.float()).half(),
        }
        if max(m, n, k) <= 272:
            refs["cpu_fp64"] = (
                torch.mm(a.cpu().double(), b.cpu().double()).half().cuda()
            )
        storage.fill_(float("nan"))
        launch(ck, ts, config)
        errors = {
            key: (c.float() - ref.float()).abs().max().item()
            for key, ref in refs.items()
        }
        ok = bool(torch.isfinite(c).all()) and bool(storage[:, n:].isnan().all())
        ok &= all(torch.equal(v, ref) for v, ref in zip([a, b], before))
        ok &= all(torch.allclose(c, ref, atol=ATOL, rtol=RTOL) for ref in refs.values())
        records.append(
            dict(
                shape=[m, n, k],
                padding=pad,
                distribution=dist,
                errors=errors,
                passed=ok,
            )
        )
        passed &= ok
    return dict(passed=passed, seed=seed, suite=suite, cases=records)


def check_pipeline(path, manifest):
    data = json.loads(Path(path).read_text())
    assert data["passed"] and data["config_sha256"] == sha256(ROOT / "baseline.json")
    assert data["baseline_hashes"] == baseline_hashes(manifest)
    assert data["torch_device_uuid"] == str(torch.cuda.get_device_properties(0).uuid)
    return data


def compile_validate(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest = frozen()
    pipeline = check_pipeline(args.pipeline, manifest)
    result = dict(
        passed=True,
        baseline_manifest_sha256=sha256(ROOT / "baseline.json"),
        plan_sha256=sha256(ROOT / "optimization_round1.md"),
        pipeline_sha256=sha256(args.pipeline),
        runner_sha256=sha256(__file__),
        records={},
    )
    for family, config in manifest["families"].items():
        idle()
        base = compile_ir(family)
        base_info = dump_kernel(base, f"{family}/baseline")
        assert (
            base_info["stages"]["cubin"]
            == pipeline["families"][family]["baseline"]["stages"]["cubin"]
        )
        result["records"][family] = {}
        for name in ["baseline", *args.names]:
            path = (ROOT / config["ttir_path"]).with_name(f"{name}.ttir")
            row = dict(ttir_path=str(path.relative_to(ROOT)), ttir_sha256=sha256(path))
            try:
                ck = base if name == "baseline" else compile_ir(family, name)
                compatible(base, ck, config)
                row["artifacts"] = dump_kernel(ck, f"{family}/{name}")
                row["executable_sections"] = executable_hashes(ck)
                if name == "r1_canonical":
                    assert row["executable_sections"] == executable_hashes(
                        base
                    ), "Canonical is not an A/A control"
                row["accuracy"] = validate(ck, config, args.seed)
                row["status"] = (
                    "PASS" if row["accuracy"]["passed"] else "CORRECTNESS_FAIL"
                )
            except Exception as exc:
                row.update(
                    status="COMPILE_OR_RUNTIME_FAIL",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            result["records"][family][name] = row
            save_json(out / f"{family}_{name}.json", row)
            print(f"{family} {name}: {row['status']}", flush=True)
            idle()
    save_json(out / "summary.json", result)


def confidence(rows):
    ratios = [row["speedup"] for row in rows]
    rng = np.random.default_rng(107)
    draws = []
    for row in rows:
        logs = np.log([pair["speedup"] for pair in row["pairs"]])
        draws.append(
            np.median(logs[rng.integers(0, len(logs), size=(2000, len(logs)))], axis=1)
        )
    aggregate = np.exp(np.mean(draws, axis=0))
    return dict(
        geometric_mean=math.exp(statistics.mean(math.log(r) for r in ratios)),
        arithmetic_mean=statistics.mean(ratios),
        min=min(ratios),
        max=max(ratios),
        ci95=[
            float(np.quantile(aggregate, 0.025)),
            float(np.quantile(aggregate, 0.975)),
        ],
    )


def measure(args):
    manifest = frozen()
    pipeline = check_pipeline(args.pipeline, manifest)
    validation = json.loads(args.validation.read_text())
    assert validation["baseline_manifest_sha256"] == sha256(ROOT / "baseline.json")
    assert validation["runner_sha256"] == sha256(__file__)
    choices = json.loads(args.selection.read_text()) if args.selection else None
    selected_families = args.families or list(manifest["families"])
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    result = dict(
        passed=True,
        seed=args.seed,
        rounds=args.rounds,
        rep_ms=20,
        metric="CUDA graph kernel time, preallocated output, paired random order",
        pipeline_sha256=sha256(args.pipeline),
        validation_sha256=sha256(args.validation),
        baseline_manifest_sha256=sha256(ROOT / "baseline.json"),
        runner_sha256=sha256(__file__),
        gpu_before=snapshot(),
        records=[],
        resources={},
        invalid_records=[],
        excluded_candidates=[],
    )
    for family in selected_families:
        idle()
        config = manifest["families"][family]
        names = list(
            dict.fromkeys(
                ["r1_canonical", choices[family] if choices else None]
                if choices
                else args.names
            )
        )
        names = [n for n in names if n and n != "baseline"]
        for name in list(names):
            status = validation["records"][family][name]["status"]
            if status != "PASS":
                assert choices is None, "A selected finalist must pass correctness"
                result["excluded_candidates"].append(
                    dict(family=family, name=name, status=status)
                )
                names.remove(name)
        kernels = {}
        for name in ["baseline", *names]:
            rec = validation["records"][family][name]
            assert rec["status"] == "PASS"
            assert sha256(ROOT / rec["ttir_path"]) == rec["ttir_sha256"]
            ck = compile_ir(family, name)
            compatible(ck if not kernels else kernels["baseline"], ck, config)
            info = dump_kernel(ck, f"{family}/{name}")
            assert info["stages"]["cubin"] == rec["artifacts"]["stages"]["cubin"]
            kernels[name] = ck
            result["resources"][f"{family}/{name}"] = dict(
                ttir_path=rec["ttir_path"], ttir_sha256=rec["ttir_sha256"], **info
            )
        assert (
            result["resources"][f"{family}/baseline"]["stages"]["cubin"]
            == pipeline["families"][family]["baseline"]["stages"]["cubin"]
        )
        sizes = (
            config["square_sizes"]
            if args.full
            else [n for n in config["square_sizes"] if n in TUNING]
        )
        for n in sizes:
            idle()
            ts = allocate(n, n, n)
            for ck in kernels.values():
                for _ in range(10):
                    launch(ck, ts, config)
            torch.cuda.synchronize()
            samples = {name: [] for name in names}
            for turn in range(args.rounds):
                order_candidates = list(names)
                rng.shuffle(order_candidates)
                for name in order_candidates:
                    order = ["baseline", name]
                    rng.shuffle(order)
                    times = {}
                    for label in order:
                        ck = kernels[label]
                        times[label] = (
                            triton.testing.do_bench_cudagraph(
                                lambda: launch(ck, ts, config), rep=20
                            )
                            * 1000
                        )
                    samples[name].append(
                        dict(
                            round=turn,
                            order=order,
                            baseline_us=times["baseline"],
                            candidate_us=times[name],
                            speedup=times["baseline"] / times[name],
                        )
                    )
            idle()
            control_pairs = samples.get("r1_canonical", [])
            control_ok = (
                bool(control_pairs)
                and abs(statistics.median(p["speedup"] for p in control_pairs) - 1)
                <= 0.02
            )
            for name, pairs in samples.items():
                med = statistics.median(p["speedup"] for p in pairs)
                drift = max(p["baseline_us"] for p in pairs) / min(
                    p["baseline_us"] for p in pairs
                )
                valid = drift <= 1.10 and control_ok
                row = dict(
                    family=family,
                    name=name,
                    shape=[n, n, n],
                    pairs=pairs,
                    speedup=med,
                    baseline_us=statistics.median(p["baseline_us"] for p in pairs),
                    candidate_us=statistics.median(p["candidate_us"] for p in pairs),
                    baseline_drift=drift,
                    valid=valid,
                )
                result["records"].append(row)
                if not valid:
                    result["invalid_records"].append([family, name, n])
                print(
                    f"{family} {name} N={n}: {med:.4f}x drift={drift:.3f} valid={valid}",
                    flush=True,
                )
    summaries = {}
    for family in selected_families:
        summaries[family] = {}
        for name in {r["name"] for r in result["records"] if r["family"] == family}:
            rows = [
                r
                for r in result["records"]
                if r["family"] == family and r["name"] == name
            ]
            summaries[family][name] = confidence(rows)
    result["summary"] = summaries
    result["gpu_after"] = snapshot()
    save_json(args.output, result)


def sanitize(args):
    m = frozen()
    choices = json.loads(args.selection.read_text())
    out = dict(
        passed=True, baseline_manifest_sha256=sha256(ROOT / "baseline.json"), records={}
    )
    for family, config in m["families"].items():
        idle()
        name = choices[family]
        ck = compile_ir(family, name)
        row = validate(ck, config, args.seed, "sanitizer")
        row.update(
            name=name,
            ttir_sha256=sha256((ROOT / config["ttir_path"]).with_name(f"{name}.ttir")),
        )
        out["records"][family] = row
        out["passed"] &= row["passed"]
        print(
            f"{family}/{name}: {len(row['cases'])} sanitizer cases pass={row['passed']}",
            flush=True,
        )
    save_json(args.output, out)
    assert out["passed"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["check", "bench", "sanitize"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pipeline", type=Path)
    p.add_argument("--validation", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--names", nargs="+", default=NAMES)
    p.add_argument("--families", nargs="+")
    p.add_argument("--full", action="store_true")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assert not args.output.exists(), "Use a new result path"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if args.mode == "check":
        compile_validate(args)
    elif args.mode == "bench":
        measure(args)
    else:
        sanitize(args)


if __name__ == "__main__":
    main()
