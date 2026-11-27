"""Evidence helpers for the new norm and selection experiments."""

import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path

import torch
import triton
import triton.testing
from triton._C import libtriton
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "suite_env", REPO / "layer_norm/environment.py"
)
_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env)


def sha(data):
    return hashlib.sha256(
        data if isinstance(data, bytes) else data.encode()
    ).hexdigest()


def digest(path):
    return sha(Path(path).read_bytes())


def tensor_digest(tensor):
    return sha(tensor.detach().cpu().contiguous().numpy().tobytes())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def freeze(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, f"Refusing to overwrite {path}"
    else:
        path.write_text(text)


def environment():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    result = _env.verify_distribution_files()
    result.update(
        torch=torch.__version__,
        device=os.environ["CUDA_VISIBLE_DEVICES"],
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,clocks.sm,power.draw,temperature.gpu",
                "--format=csv",
            ],
            text=True,
        ),
        toolchain={
            "libtriton": digest(libtriton.__file__),
            "backend": digest(inspect.getfile(CUDABackend)),
            "ptxas": digest(get_ptxas(90).path),
        },
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("TRITON", "CUDA", "LD_", "PYTHON"))
        },
        clocks_locked=False,
        cache_flushed=False,
    )
    return result


def compile_file(path, options):
    _env.check_environment()
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    ck = triton.compile(str(path), options=options)
    ck._init_handles()
    return ck


def kernel_info(ck, path, artifacts):
    artifacts.mkdir(parents=True, exist_ok=True)
    for stage, data in ck.asm.items():
        p = artifacts / f"kernel.{stage}"
        p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return {
        "ir_path": str(path),
        "ir_sha256": digest(path),
        "stage_sha256": {k: sha(v) for k, v in ck.asm.items()},
        "n_regs": ck.n_regs,
        "n_spills": ck.n_spills,
        "metadata": ck.metadata._asdict(),
    }


def pipeline(source, rebuilt):
    for stage in ("llir", "ptx", "cubin"):
        assert source.asm[stage] == rebuilt.asm[stage], stage
    backend = CUDABackend(source.metadata.target)
    fields = backend.parse_options({}).__dict__
    source_options = {k: source.metadata._asdict()[k] for k in fields}
    rebuilt_options = {k: rebuilt.metadata._asdict()[k] for k in fields}
    assert source_options == rebuilt_options
    assert source.metadata.target == rebuilt.metadata.target
    return {
        "llir_ptx_cubin_identical": True,
        "full_options_identical": True,
        "options": source_options,
        "source_stages": {k: sha(v) for k, v in source.asm.items()},
    }


def paired(callables, rounds, rep, seed):
    rng = random.Random(seed)
    for fn in callables.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    triton.testing.do_bench_cudagraph(callables["baseline"], rep=50)
    rows = {n: [] for n in callables if n != "baseline"}
    aa = []

    def measure(n):
        return triton.testing.do_bench_cudagraph(callables[n], rep=rep) * 1000

    for turn in range(rounds):
        at = [measure("baseline"), measure("baseline")]
        aa.append({"round": turn, "us": at, "ratio": at[0] / at[1]})
        names = list(rows)
        rng.shuffle(names)
        for name in names:
            order = ["baseline", name]
            rng.shuffle(order)
            values = {n: measure(n) for n in order}
            rows[name].append(
                {
                    "round": turn,
                    "order": order,
                    "baseline_us": values["baseline"],
                    "candidate_us": values[name],
                    "speedup": values["baseline"] / values[name],
                }
            )
    return {
        "aa": aa,
        "aa_median": statistics.median(x["ratio"] for x in aa),
        "candidates": {
            n: {
                "samples": v,
                "speedup": statistics.median(x["speedup"] for x in v),
                "baseline_us": statistics.median(x["baseline_us"] for x in v),
                "candidate_us": statistics.median(x["candidate_us"] for x in v),
            }
            for n, v in rows.items()
        },
    }


def summarize(records):
    names = records[0]["timing"]["candidates"]
    return {
        n: {
            "geomean": math.exp(
                statistics.mean(
                    math.log(r["timing"]["candidates"][n]["speedup"]) for r in records
                )
            ),
            "worst": min(r["timing"]["candidates"][n]["speedup"] for r in records),
            "best": max(r["timing"]["candidates"][n]["speedup"] for r in records),
        }
        for n in names
    }
