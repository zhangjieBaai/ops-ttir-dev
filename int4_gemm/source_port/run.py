"""Actual source-JIT / frozen TTIR comparison on the unchanged INT4 contract."""

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
import common as c  # noqa: E402
from source_v1 import _weight_int4pack_mm_kernel  # noqa: E402

NAMES = ("baseline", "ir", "source")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def elf_text(data):
    assert data[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, names_idx = struct.unpack_from("<HHH", data, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)
    ]
    names = data[headers[names_idx][4] : headers[names_idx][4] + headers[names_idx][5]]
    result = {}
    for h in headers:
        name = names[h[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text."):
            result[name] = {"sha256": c.digest(data[h[4] : h[4] + h[5]]), "bytes": h[5]}
    assert result
    return result


def info(ck, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for ext, data in ck.asm.items():
        (folder / f"kernel.{ext}").write_bytes(
            data if isinstance(data, bytes) else data.encode()
        )
    return dict(
        stages={k: c.digest(v) for k, v in ck.asm.items()},
        elf_text=elf_text(ck.asm["cubin"]),
        n_regs=ck.n_regs,
        n_spills=ck.n_spills,
        metadata=ck.metadata._asdict(),
        compiler_input_class=type(ck.src).__name__,
    )


def launch(ck, ts, config):
    shape = (ts[0].shape[0], ts[4].shape[1], ts[0].shape[1])
    if type(ck.src).__name__ == "ASTSource":
        ck[c.grid(shape, config)](*c.runtime_args(ts), *config[:3])
    else:
        c.launch(ck, ts, config)


def build(case, seed):
    config = case["config"]
    ts = c.allocate(case["shape"], seed)
    kernels = {}
    records = {}
    for name, filename in (("baseline", "baseline"), ("ir", "scalar_unsigned_v1")):
        path = c.ROOT / "ir" / case["id"] / f"{filename}.ttir"
        ck = triton.compile(str(path), options=c.options(config))
        ck._init_handles()
        kernels[name] = ck
        records[name] = info(ck, ROOT / "artifacts" / case["id"] / name)
        records[name]["input_path"] = str(path)
        records[name]["input_sha256"] = c.sha(path)
    original = c.source_launch(ts, config)
    assert (
        original.asm["ttir"]
        == (c.ROOT / "ir" / case["id"] / "baseline.ttir").read_text()
    )
    assert all(
        original.asm[s] == kernels["baseline"].asm[s] for s in ("llir", "ptx", "cubin")
    )
    source = _weight_int4pack_mm_kernel[c.grid(case["shape"], config)](
        *c.runtime_args(ts),
        BLOCK_M=config[0],
        BLOCK_N=config[1],
        BLOCK_K=config[2],
        **c.options(config),
    )
    source._init_handles()
    assert type(source.src).__name__ == "ASTSource"
    assert source.src.fn is _weight_int4pack_mm_kernel
    kernels["source"] = source
    path = ROOT / "ir" / case["id"] / "source_v1.ttir"
    c.write_once(path, source.asm["ttir"])
    rebuilt = triton.compile(str(path), options=c.options(config))
    rebuilt._init_handles()
    assert all(source.asm[s] == rebuilt.asm[s] for s in ("llir", "ptx", "cubin"))
    keys = CUDABackend(source.metadata.target).parse_options({}).__dict__
    opts = [
        {k: getattr(ck.metadata, k) for k in keys}
        for ck in [original, *kernels.values(), rebuilt]
    ]
    assert all(x == opts[0] for x in opts)
    assert all(ck.metadata.target == source.metadata.target for ck in kernels.values())
    records["source"] = info(source, ROOT / "artifacts" / case["id"] / "source")
    records["source"].update(
        input_path=str(ROOT / "source_v1.py"),
        input_sha256=c.sha(ROOT / "source_v1.py"),
        exported_ttir_sha256=c.sha(path),
    )
    expected = ts[4].clone()
    for ck in [rebuilt, *kernels.values()]:
        ts[4].fill_(float("nan"))
        launch(ck, ts, config)
        assert torch.equal(expected, ts[4])
    return (
        kernels,
        records,
        dict(
            baseline_source_file_identical=True,
            source_export_file_identical=True,
            full_options_identical=True,
            outputs_bitwise=True,
            options=opts[0],
        ),
    )


def accuracy(case, kernels, seed):
    rows = []
    for mode in ("random", "extreme_nibble", "zero_scale", "zero", "constant"):
        ts = c.allocate(case["shape"], seed, mode)
        inputs = [t.clone() for t in ts[:4]]
        ref = c.reference(ts)
        cpu = c.reference(ts, cpu=True) if case["shape"][1] <= 256 else None
        launch(kernels["baseline"], ts, case["config"])
        base = ts[4].clone()
        for name, ck in kernels.items():
            ts[4].fill_(float("nan"))
            launch(ck, ts, case["config"])
            error = c.assert_output(ts, ref)
            assert torch.equal(ts[4], base)
            if cpu is not None:
                c.assert_output(ts, cpu)
            assert all(torch.equal(a, b) for a, b in zip(ts[:4], inputs))
            rows.append(
                dict(
                    mode=mode,
                    path=name,
                    max_abs_error=error,
                    baseline_bitwise=True,
                    cpu_fp64=cpu is not None,
                    canary=True,
                    inputs_unchanged=True,
                )
            )
    return rows


def measure(case, kernels, seed, rounds, rep):
    ts = c.allocate(case["shape"], seed)
    hashes = [c.digest(t.detach().cpu().contiguous().numpy().tobytes()) for t in ts[:4]]
    funcs = {
        name: (lambda ck=ck: launch(ck, ts, case["config"]))
        for name, ck in kernels.items()
    }
    for fn in funcs.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    for fn in funcs.values():
        triton.testing.do_bench_cudagraph(fn, rep=50)
    rng = random.Random(seed)
    samples = []
    aa = []
    for ri in range(rounds):
        order = list(NAMES)
        rng.shuffle(order)
        us = {
            n: triton.testing.do_bench_cudagraph(funcs[n], rep=rep) * 1000
            for n in order
        }
        aa_us = [
            triton.testing.do_bench_cudagraph(funcs["baseline"], rep=rep) * 1000
            for _ in range(2)
        ]
        aa.append(dict(round=ri, us=aa_us, ratio=aa_us[0] / aa_us[1]))
        samples.append(
            dict(
                round=ri,
                order=order,
                us=us,
                baseline_over_ir=us["baseline"] / us["ir"],
                baseline_over_source=us["baseline"] / us["source"],
                ir_over_source=us["ir"] / us["source"],
            )
        )
    aa_median = statistics.median(x["ratio"] for x in aa)
    return dict(
        input_seed=seed,
        input_sha256=hashes,
        samples=samples,
        aa=aa,
        aa_median=aa_median,
        aa_passed=0.97 <= aa_median <= 1.03,
        median_us={n: statistics.median(s["us"][n] for s in samples) for n in NAMES},
        ratios={
            key: statistics.median(s[key] for s in samples)
            for key in ("baseline_over_ir", "baseline_over_source", "ir_over_source")
        },
    )


def summary(rows):
    return {
        key: dict(
            geomean=math.exp(
                statistics.mean(math.log(r["timing"]["ratios"][key]) for r in rows)
            ),
            minimum=min(r["timing"]["ratios"][key] for r in rows),
            maximum=max(r["timing"]["ratios"][key] for r in rows),
        )
        for key in ("baseline_over_ir", "baseline_over_source", "ir_over_source")
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    out = ROOT / "results" / args.run
    assert not out.exists()
    out.mkdir(parents=True)
    env = c.environment()
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = json.loads((ROOT / "source_manifest.json").read_text())
    assert frozen["sha256"] == c.sha(ROOT / "source_v1.py")
    result = dict(
        environment=env,
        seed=args.seed,
        rounds=args.rounds,
        rep_ms=args.rep,
        source_sha256=frozen["sha256"],
        old_manifest_sha256=c.sha(c.ROOT / "baseline.json"),
        records=[],
    )
    for case in c.manifest()["cases"]:
        seed = args.seed + sum(case["shape"])
        ks, infos, gate = build(case, seed)
        checks = accuracy(case, ks, seed)
        row = dict(
            case=case["id"],
            shape=case["shape"],
            config=case["config"],
            pipeline=gate,
            kernels=infos,
            accuracy=checks,
        )
        if not args.check_only:
            row["timing"] = measure(case, ks, seed, args.rounds, args.rep)
            print(
                case["id"],
                row["timing"]["ratios"],
                "AA",
                row["timing"]["aa_median"],
                flush=True,
            )
        else:
            print(case["id"], "accuracy PASS", flush=True)
        result["records"].append(row)
        save(out / "results.json", result)
    result["passed"] = True
    if not args.check_only:
        result["summary"] = summary(result["records"])
        result["aa_passed"] = all(r["timing"]["aa_passed"] for r in result["records"])
        print(result["summary"], flush=True)
    save(out / "results.json", result)


if __name__ == "__main__":
    main()
