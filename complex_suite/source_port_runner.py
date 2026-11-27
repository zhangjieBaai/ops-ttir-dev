"""Time actual source-JIT kernels against frozen baseline and optimized TTIR."""

import argparse
import importlib.util
import json
import random
import re
import statistics
import struct
import sys
from pathlib import Path

import torch
import triton
import triton.testing
from support import (
    compile_file,
    digest,
    environment,
    freeze,
    kernel_info,
    pipeline,
    save,
    sha,
    tensor_digest,
)
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceAdapter:
    """Append source constexpr Python slots; the CUDA kernel ABI is unchanged."""

    def __init__(self, ck, constants):
        self.ck = ck
        self.constants = constants

    def __getitem__(self, grid):
        return lambda *runtime_args: self.ck[grid](*runtime_args, *self.constants)


def elf_text(blob):
    assert blob[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", blob, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", blob, offset + i * size) for i in range(count)
    ]
    names = headers[names_index]
    names = blob[names[4] : names[4] + names[5]]
    sections = {}
    for h in headers:
        name = names[h[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text"):
            sections[name] = sha(blob[h[4] : h[4] + h[5]])
    return sections


def three_way(functions, rounds, rep, seed):
    rng = random.Random(seed)
    for fn in functions.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    for fn in functions.values():
        triton.testing.do_bench_cudagraph(fn, rep=50)

    def measure(name):
        return triton.testing.do_bench_cudagraph(functions[name], rep=rep) * 1000

    samples = []
    aa = []
    for turn in range(rounds):
        at = [measure("baseline"), measure("baseline")]
        aa.append({"round": turn, "us": at, "ratio": at[0] / at[1]})
        order = ["baseline", "ir", "source"]
        rng.shuffle(order)
        us = {name: measure(name) for name in order}
        samples.append(
            {
                "round": turn,
                "order": order,
                "us": us,
                "baseline_over_ir": us["baseline"] / us["ir"],
                "baseline_over_source": us["baseline"] / us["source"],
                "ir_over_source": us["ir"] / us["source"],
            }
        )
    return {
        "samples": samples,
        "aa": aa,
        "aa_median": statistics.median(x["ratio"] for x in aa),
        "median_us": {
            n: statistics.median(s["us"][n] for s in samples) for n in functions
        },
        "ratios": {
            n: statistics.median(s[n] for s in samples)
            for n in ("baseline_over_ir", "baseline_over_source", "ir_over_source")
        },
    }


def info(ck, path, artifacts):
    result = kernel_info(ck, path, artifacts)
    result["elf_text_sha256"] = elf_text(ck.asm["cubin"])
    ttir = ck.asm["ttir"]
    result["structural_counts"] = {
        "reductions": ttir.count('"tt.reduce"'),
        "maxui": ttir.count("arith.maxui"),
        "cache_cs": len(re.findall(r"tt.store[^\n]+cacheModifier = cs", ttir)),
        "loop_result_types": re.findall(r"scf.for[^\n]+-> ([^\n]+)", ttir),
        "signed_shift_before_loop": "arith.shrsi" in ttir.split("scf.for")[0],
        "redux_max_u32": ck.asm["ptx"].count("redux.sync.max.u32"),
    }
    return result


def run(args):
    folder = ROOT / args.op
    port = folder / "source_port"
    out = port / "results" / args.run
    assert not out.exists(), "Use a new result directory"
    out.mkdir(parents=True)
    sys.path.insert(0, str(folder))
    old = load_module("original_experiment", folder / "experiment.py")
    new = load_module("source_port_kernel", port / "source_v1.py")
    manifest = json.loads((folder / "manifest.json").read_text())
    source_record = {
        "source_sha256": digest(port / "source_v1.py"),
        "original_manifest_sha256": digest(folder / "manifest.json"),
        "contract": "complex_suite/source_port_plan.md",
        "source_compiler_path": "public @triton.jit frontend; actual source CompiledKernel timed",
    }
    freeze(port / "source_manifest.json", json.dumps(source_record, indent=2) + "\n")
    result = {
        "environment": environment(),
        "source": source_record,
        "seed": args.seed,
        "rounds": args.rounds,
        "rep_ms": args.rep,
        "source_consts_adapter": "Original CUDA runtime ABI unchanged; source constexpr Python slots appended",
        "records": [],
    }
    norm = args.op == "norm_quant"
    candidate = "stream_v1" if norm else "ordered_carry_v3"
    for item in manifest["records"]:
        shape = tuple(item["shape"])
        m, n = shape[:2]
        seed = args.seed + m * 1000003 + n + (0 if norm else shape[2])
        torch.manual_seed(seed)
        ts, _ = old.allocate(*shape)
        input_hashes = [tensor_digest(t) for t in (ts[:3] if norm else ts[:1])]
        stages = ["kernel"] if norm else ["local", "merge"]
        kernels = {name: {} for name in ("baseline", "ir", "source")}
        sources = {}
        source_exports = {}
        metadata = {name: {} for name in kernels}
        gates = {}
        options = item["options"]
        for stage in stages:
            srcbase = (
                old.source(ts, options["num_warps"])
                if norm
                else old.source(ts, shape[2], stage)
            )
            groupdir = folder / "ir" / item["group"]
            if not norm:
                groupdir /= stage
            for name, filename in (("baseline", "baseline"), ("ir", candidate)):
                path = groupdir / f"{filename}.ttir"
                ck = compile_file(path, options)
                kernels[name][stage] = ck
                metadata[name][stage] = info(
                    ck, path, port / "artifacts" / item["group"] / stage / name
                )
            assert srcbase.asm["ttir"] == (groupdir / "baseline.ttir").read_text()
            gates[stage] = {"original": pipeline(srcbase, kernels["baseline"][stage])}
            if norm:
                grid = (m, 1, 1)
                constants = (n, triton.next_power_of_2(n))
                source_ck = new.residual_rms_int8[grid](*ts, *constants, **options)
            else:
                k = shape[2]
                chunks = triton.cdiv(n, 1024)
                length = n if stage == "local" else chunks * k
                block = 1024 if stage == "local" else triton.next_power_of_2(length)
                constants = (
                    length,
                    k,
                    block,
                    chunks if stage == "local" else 1,
                    stage == "merge",
                )
                source_ck = new.partial_select[old.grid(ts, stage)](
                    *old.args_for(ts, stage), *constants, **options
                )
            source_ck._init_handles()
            sources[stage] = source_ck
            kernels["source"][stage] = SourceAdapter(source_ck, constants)
            path = port / "ir" / item["group"] / stage / "source_v1.ttir"
            freeze(path, source_ck.asm["ttir"])
            source_exports[stage] = compile_file(path, options)
            gates[stage]["source_export"] = pipeline(source_ck, source_exports[stage])
            fields = CUDABackend(source_ck.metadata.target).parse_options({}).__dict__
            actual_options = [
                {k: ck.metadata._asdict()[k] for k in fields}
                for ck in [kernels["baseline"][stage], kernels["ir"][stage], source_ck]
            ]
            assert actual_options[0] == actual_options[1] == actual_options[2]
            metadata["source"][stage] = info(
                source_ck, path, port / "artifacts" / item["group"] / stage / "source"
            )
        if norm:
            launches = {name: ks["kernel"] for name, ks in kernels.items()}
            exported = source_exports["kernel"]
        else:
            launches = kernels
            exported = source_exports
        old.launch(launches["source"], ts)
        before = [x.clone() for x in (ts[3:] if norm else ts[1:])]
        old.launch(exported, ts)
        assert all(
            torch.equal(a, b) for a, b in zip(before, ts[3:] if norm else ts[1:])
        )
        validation = {
            name: old.validate(ck, shape, args.seed, args.check_only)
            for name, ck in launches.items()
        }
        # Same normal input, output-by-output equality is an observation beyond the contract.
        outputs = {}
        for name, ck in launches.items():
            old.launch(ck, ts)
            outputs[name] = [x.clone() for x in (ts[3:] if norm else ts[3:5])]
        equality = {
            "source_ir_outputs_equal": [
                torch.equal(a, b) for a, b in zip(outputs["source"], outputs["ir"])
            ],
            "source_baseline_outputs_equal": [
                torch.equal(a, b)
                for a, b in zip(outputs["source"], outputs["baseline"])
            ],
        }
        record = {
            "shape": shape,
            "group": item["group"],
            "options": options,
            "input_seed": seed,
            "input_sha256": input_hashes,
            "pipeline": gates,
            "kernels": metadata,
            "validation": validation,
            "normal_output_equality": equality,
            "source_ir_text_equal": {
                stage: metadata["source"][stage]["elf_text_sha256"]
                == metadata["ir"][stage]["elf_text_sha256"]
                for stage in stages
            },
        }
        if not args.check_only:
            funcs = {
                name: (lambda ck=ck: old.launch(ck, ts))
                for name, ck in launches.items()
            }
            record["timing"] = three_way(funcs, args.rounds, args.rep, seed)
            print(
                shape,
                record["timing"]["ratios"],
                "AA",
                record["timing"]["aa_median"],
                "text",
                record["source_ir_text_equal"],
                flush=True,
            )
        result["records"].append(record)
        save(out / "results.json", result)
    result["passed"] = True
    if not args.check_only:
        result["aa_passed"] = all(
            0.97 <= r["timing"]["aa_median"] <= 1.03 for r in result["records"]
        )
    save(out / "results.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["norm_quant", "partial_topk"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--check-only", action="store_true")
    run(p.parse_args())
