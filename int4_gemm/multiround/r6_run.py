"""Round6 TTIR-only runner, preserving prior timing and full-matrix validation."""

import argparse
import json
import math
import random
import re
import statistics
import struct
import sys
import traceback
from pathlib import Path

import torch
import triton
import triton.testing
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "r3"
sys.path.insert(0, str(ROOT.parent))
import common as c  # noqa: E402


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def launch(ck, ts, config):
    shape = (ts[0].shape[0], ts[4].shape[1], ts[0].shape[1])
    names = [
        "A_ptr",
        "mat2_ptr",
        "qScale_ptr",
        "qZeros_ptr",
        "C_ptr",
        "M",
        "N",
        "K",
        "K2",
        "qGroupSize",
        "G",
        "stride_am",
        "stride_ak",
        "stride_mn",
        "stride_mk",
        "stride_sg",
        "stride_sn",
        "stride_zg",
        "stride_zn",
        "stride_cm",
        "stride_cn",
    ]
    values = dict(zip(names, c.runtime_args(ts)))
    header = next(
        line for line in ck.asm["ttir"].splitlines() if "tt.func public" in line
    )
    actual = re.findall(r"%([A-Za-z_]\w*):", header)
    ck[c.grid(shape, config)](*(values[name] for name in actual))


def elf_text(blob):
    assert blob[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, ni = struct.unpack_from("<HHH", blob, 58)
    heads = [
        struct.unpack_from("<IIQQQQIIQQ", blob, offset + i * size) for i in range(count)
    ]
    names = blob[heads[ni][4] : heads[ni][4] + heads[ni][5]]
    result = {}
    for h in heads:
        name = names[h[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text"):
            result[name] = {"sha256": c.digest(blob[h[4] : h[4] + h[5]]), "bytes": h[5]}
    assert result
    return result


def compile_case(case, name):
    path = BASE / "ir" / case["id"] / f"{name}.ttir"
    ck = triton.compile(str(path), options=c.options(case["config"]))
    ck._init_handles()
    dest = BASE / "artifacts" / case["id"] / name
    dest.mkdir(parents=True, exist_ok=True)
    for stage, value in ck.asm.items():
        (dest / f"kernel.{stage}").write_bytes(
            value if isinstance(value, bytes) else value.encode()
        )
    info = c.record(ck, path)
    info["path"] = str(path)
    info["elf_text"] = elf_text(ck.asm["cubin"])
    return ck, info


def pipeline(case, baseline, seed):
    ts = c.allocate(case["shape"], seed)
    source = c.source_launch(ts, case["config"])
    assert (
        source.asm["ttir"] == (BASE / "ir" / case["id"] / "baseline.ttir").read_text()
    )
    assert all(source.asm[s] == baseline.asm[s] for s in ("llir", "ptx", "cubin"))
    keys = CUDABackend(source.metadata.target).parse_options({}).__dict__
    options = {k: getattr(source.metadata, k) for k in keys}
    assert options == {k: getattr(baseline.metadata, k) for k in keys}
    expected = ts[4].clone()
    ts[4].fill_(float("nan"))
    launch(baseline, ts, case["config"])
    assert torch.equal(ts[4], expected)
    return {
        "passed": True,
        "llir_ptx_cubin_identical": True,
        "outputs_bitwise": True,
        "options": options,
    }


def accuracy(case, kernels, seed):
    rows = []
    for mode in ("random", "extreme_nibble", "zero_scale", "zero", "constant"):
        ts = c.allocate(case["shape"], seed, mode)
        inputs = [x.clone() for x in ts[:4]]
        ref = c.reference(ts)
        cpu = c.reference(ts, cpu=True) if case["shape"][1] <= 256 else None
        launch(kernels["baseline"], ts, case["config"])
        base = ts[4].clone()
        for name, ck in kernels.items():
            ts[4].fill_(float("nan"))
            launch(ck, ts, case["config"])
            error = c.assert_output(ts, ref)
            assert torch.equal(ts[4], base), f'not bitwise {name} {case["id"]} {mode}'
            assert torch.equal(ts[4].view(torch.int16), base.view(torch.int16)), (
                name,
                case["id"],
                mode,
                "raw FP16 bits differ",
            )
            if cpu is not None:
                c.assert_output(ts, cpu)
            assert all(torch.equal(x, y) for x, y in zip(inputs, ts[:4]))
            rows.append(
                {
                    "case": case["id"],
                    "name": name,
                    "mode": mode,
                    "max_abs_error": error,
                    "baseline_bitwise": True,
                    "baseline_raw_int16_bits": True,
                    "cpu_fp64": cpu is not None,
                    "canary_inputs_unchanged": True,
                }
            )
    return rows


def measure(case, kernels, seed, rounds, rep):
    ts = c.allocate(case["shape"], seed)
    hashes = [c.digest(t.cpu().contiguous().numpy().tobytes()) for t in ts[:4]]
    funcs = {
        name: (lambda ck=ck: launch(ck, ts, case["config"]))
        for name, ck in kernels.items()
    }
    for fn in funcs.values():
        triton.testing.do_bench_cudagraph(fn, rep=100)
    rng = random.Random(seed)
    samples = []
    aa = []
    for turn in range(rounds):
        order = list(funcs)
        rng.shuffle(order)
        us = {
            name: triton.testing.do_bench_cudagraph(funcs[name], rep=rep) * 1000
            for name in order
        }
        a = [
            triton.testing.do_bench_cudagraph(funcs["parent"], rep=rep) * 1000
            for _ in range(2)
        ]
        aa.append({"round": turn, "us": a, "ratio": a[0] / a[1]})
        samples.append(
            {
                "round": turn,
                "order": order,
                "us": us,
                "incremental": {
                    name: us["parent"] / value for name, value in us.items()
                },
                "cumulative": {
                    name: us["baseline"] / value for name, value in us.items()
                },
                "anchor_relative": {
                    name: us["anchor"] / value for name, value in us.items()
                },
            }
        )
    return {
        "input_seed": seed,
        "input_sha256": hashes,
        "samples": samples,
        "aa": aa,
        "aa_median": statistics.median(x["ratio"] for x in aa),
        "median_us": {n: statistics.median(x["us"][n] for x in samples) for n in funcs},
        "incremental": {
            n: statistics.median(x["incremental"][n] for x in samples) for n in funcs
        },
        "cumulative": {
            n: statistics.median(x["cumulative"][n] for x in samples) for n in funcs
        },
        "anchor_relative": {
            n: statistics.median(x["anchor_relative"][n] for x in samples)
            for n in funcs
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
                "minimum": min(r["timing"][metric][name] for r in records),
                "maximum": max(r["timing"][metric][name] for r in records),
            }
            for metric in ("incremental", "anchor_relative", "cumulative")
        }
        for name in names
    }


def main():
    global BASE
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["r6"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--names")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--rep", type=int, default=10)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    BASE = ROOT / a.stage
    out = BASE / "results" / a.run
    assert not out.exists()
    out.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = {
        "environment": c.environment(),
        "seed": a.seed,
        "rounds": a.rounds,
        "rep_ms": a.rep,
        "manifest_sha256": c.sha(BASE / "manifest.json"),
        "records": [],
        "failures": [],
    }
    manifest = json.loads((BASE / "manifest.json").read_text())
    assert manifest["source_sha256"] == c.sha(ROOT.parent / "tutorial.py")
    names = (
        a.names.split(",")
        if a.names
        else ["baseline", "anchor", "parent"]
        + json.loads((BASE / "candidates.json").read_text())["names"]
    )
    assert names[:3] == ["baseline", "anchor", "parent"]
    kernels = {}
    metadata = {}
    gates = {}
    failed = set()
    for case in manifest["cases"]:
        group = case["id"]
        kernels[group] = {}
        metadata[group] = {}
        for name in names:
            try:
                ck, record = compile_case(case, name)
                kernels[group][name] = ck
                metadata[group][name] = record
            except Exception as exc:
                failed.add(name)
                data["failures"].append(
                    {
                        "case": group,
                        "name": name,
                        "stage": "compile",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                save(out / "failures.json", data["failures"])
        assert not {"baseline", "anchor", "parent"} & failed
        gates[group] = pipeline(case, kernels[group]["baseline"], a.seed)
        for ck in kernels[group].values():
            assert {
                k: getattr(ck.metadata, k) for k in gates[group]["options"]
            } == gates[group]["options"]
        print("compiled/pipeline", group, "failed=", sorted(failed), flush=True)
    save(out / "pipeline.json", gates)
    valid = [name for name in names if name not in failed]
    # Entire expanded correctness matrix gates every candidate before any timing.
    validation = {}
    for case in manifest["cases"]:
        group = case["id"]
        validation[group] = []
        for name in valid.copy():
            try:
                ks = {"baseline": kernels[group]["baseline"]}
                if name != "baseline":
                    ks[name] = kernels[group][name]
                checks = accuracy(case, ks, a.seed)
                validation[group].extend(x for x in checks if x["name"] == name)
            except Exception as exc:
                failed.add(name)
                valid.remove(name)
                data["failures"].append(
                    {
                        "case": group,
                        "name": name,
                        "stage": "accuracy",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                save(out / "failures.json", data["failures"])
                assert name not in ("baseline", "anchor", "parent")
        print("accuracy", group, valid, flush=True)
    save(
        out / "accuracy.json",
        {
            "passed": True,
            "valid_names": valid,
            "checks": validation,
            "kernels": metadata,
        },
    )
    for case in manifest["cases"]:
        group = case["id"]
        record = {
            "case": group,
            "shape": case["shape"],
            "config": case["config"],
            "pipeline": gates[group],
            "kernels": {n: metadata[group][n] for n in valid},
            "validation": validation[group],
        }
        if not a.check_only:
            record["timing"] = measure(
                case,
                {n: kernels[group][n] for n in valid},
                a.seed
                + case["shape"][0] * 100003
                + case["shape"][1]
                + case["shape"][2],
                a.rounds,
                a.rep,
            )
            print(
                "bench",
                group,
                "parent/new",
                record["timing"]["incremental"],
                "AA",
                record["timing"]["aa_median"],
                flush=True,
            )
        data["records"].append(record)
        save(out / "results.json", data)
    data["passed"] = True
    data["valid_names"] = valid
    if not a.check_only:
        data["summary"] = summarize(data["records"], valid)
        data["aa_passed"] = all(
            0.97 <= r["timing"]["aa_median"] <= 1.03 for r in data["records"]
        )
        print(data["summary"], flush=True)
    save(out / "results.json", data)


if __name__ == "__main__":
    main()
