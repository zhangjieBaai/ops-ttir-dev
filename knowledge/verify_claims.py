"""Reproduce bounded TTIR capability probes and audit historical paired samples.

Run in the frozen venv, on an idle isolated GPU, with a new output directory.
These independent FP32 probes are not LayerNorm optimization candidates.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "layer_norm"))

import torch  # noqa: E402
import triton  # noqa: E402
from common import OPTIONS  # noqa: E402
from environment import verify_distribution_files  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(path):
    data = json.loads(path.read_text())
    groups = {}
    for row in data["records"]:
        samples = row["samples"]
        ratios = []
        for sample in samples:
            b, c = sample["baseline_us"], sample["candidate_us"]
            assert math.isfinite(b) and math.isfinite(c) and min(b, c) > 0
            assert math.isclose(b / c, sample["speedup"], rel_tol=1e-10)
            assert sorted(sample["order"]) == sorted(["baseline", row["kernel"]])
            ratios.append(b / c)
        assert len(samples) == data["rounds"]
        assert math.isclose(statistics.median(ratios), row["speedup"], rel_tol=1e-10)
        groups.setdefault(row["kernel"], []).append((row, ratios))
    out = {}
    for name, rows in groups.items():
        medians = [statistics.median(ratios) for _, ratios in rows]
        rng = random.Random(42)
        # Resample paired rounds within each fixed shape; not a population-of-shapes CI.
        boots = sorted(
            math.exp(
                statistics.mean(
                    math.log(statistics.median(rng.choices(rs, k=len(rs))))
                    for _, rs in rows
                )
            )
            for _ in range(2000)
        )
        worst = min(rows, key=lambda pair: statistics.median(pair[1]))
        out[name] = dict(
            shapes=len(rows),
            paired_samples=sum(len(rs) for _, rs in rows),
            geometric_mean=math.exp(statistics.mean(map(math.log, medians))),
            arithmetic_mean=statistics.mean(medians),
            worst_shape=[worst[0]["M"], worst[0]["N"]],
            worst_speedup=statistics.median(worst[1]),
            conditional_paired_bootstrap_95=[boots[49], boots[1949]],
            observed_pair_range=[
                min(min(rs) for _, rs in rows),
                max(max(rs) for _, rs in rows),
            ],
        )
    return dict(
        sha256=digest(path),
        rounds=data["rounds"],
        rep_ms=data["rep_ms"],
        summary=out,
        limitation="Bootstrap conditions on fixed shapes and recorded paired rounds; "
        "does not cover device contention, clock drift, selection or cross-process uncertainty.",
    )


def module(body):
    return """module {
  tt.func public @probe(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>) {
    %i = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %xp = tt.splat %X : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %p = tt.addptr %xp, %i : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %x = tt.load %p : tensor<256x!tt.ptr<f32>>
BODY
    %yp = tt.splat %Y : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %q = tt.addptr %yp, %i : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %q, %y : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
""".replace(
        "BODY", body
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Isolate an idle device first"
    args.output.mkdir(parents=True, exist_ok=False)
    assert not triton.knobs.compilation.override
    assert triton.knobs.runtime.add_stages_inspection_hook is None
    result = dict(
        environment=verify_distribution_files(),
        argv=sys.argv,
        script_sha256=digest(__file__),
        visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        torch_version=torch.__version__,
    )
    result["gpu_snapshot_before"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
            "--format=csv",
        ],
        text=True,
    )
    result["target"] = str(triton.runtime.driver.active.get_current_target())
    bodies = {
        "tensor_negf": "    %y = arith.negf %x : tensor<256xf32>",
        "negative_zero_subf": "    %z = arith.constant dense<-0.0> : tensor<256xf32>\n"
        "    %y = arith.subf %z, %x : tensor<256xf32>",
        "positive_zero_subf": "    %z = arith.constant dense<0.0> : tensor<256xf32>\n"
        "    %y = arith.subf %z, %x : tensor<256xf32>",
        "inline_asm": '    %y = "tt.elementwise_inline_asm"(%x) '
        '<{asm_string = "neg.f32 $0, $1;", constraints = "=f,f", '
        "pure = true, packed_element = 1 : i32}> : (tensor<256xf32>) -> tensor<256xf32>",
    }
    for name, attr in [
        ("same_load", ""),
        ("ca_load", " cacheModifier = ca"),
        ("volatile_load", " {isVolatile = true}"),
    ]:
        bodies[name] = (
            f"    %v = tt.load %p{attr} : tensor<256x!tt.ptr<f32>>\n"
            "    %y = arith.addf %x, %v : tensor<256xf32>"
        )
    result["probes"] = {}
    for name, body in bodies.items():
        path = args.output / f"{name}.ttir"
        path.write_text(module(body))
        entry = dict(ttir_sha256=digest(path))
        result["probes"][name] = entry
        try:
            ck = triton.compile(str(path), options=OPTIONS)
        except Exception as exc:
            entry.update(status="COMPILE_FAILED", error=str(exc))
            continue
        ck._init_handles()
        entry.update(
            status="COMPILED",
            n_regs=ck.n_regs,
            n_spills=ck.n_spills,
            metadata=ck.metadata._asdict(),
        )
        entry["artifacts"] = {}
        for ext, value in ck.asm.items():
            # IRSource's compiled TTIR is a parsed/printed module with locations.
            # Preserve the exact input file separately from every stage artifact.
            artifact = args.output / f"{name}.compiled.{ext}"
            artifact.write_bytes(value if isinstance(value, bytes) else value.encode())
            entry["artifacts"][ext] = digest(artifact)
        assert entry["ttir_sha256"] == digest(path), "Input TTIR was overwritten"
        x = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"), 1e-30]
            * 32,
            device="cuda",
        )
        y = torch.full_like(x, float("nan"))
        ck[(1, 1, 1)](x, y)
        torch.cuda.synchronize()
        expected = x + x if name.endswith("load") else -x
        torch.testing.assert_close(y, expected, rtol=0, atol=0, equal_nan=True)
        zeros = expected == 0
        entry["values_match"] = True
        entry["zero_signs_match"] = bool(
            torch.equal(torch.signbit(y[zeros]), torch.signbit(expected[zeros]))
        )
        entry["ptx_load_instructions"] = [
            line.strip()
            for line in ck.asm["ptx"].splitlines()
            if re.search(r"\bld\.(?:volatile\.)?global\.", line)
        ]
    result["historical_samples"] = {}
    for name in [
        "screening",
        "full_benchmark",
        "independent_repeat",
        "round2/screen_batch1",
        "round2/screen_batch2",
        "round2/screen_ca",
        "round2/full_stable",
        "round2/independent_repeat",
    ]:
        path = ROOT / "layer_norm/results" / f"{name}.json"
        result["historical_samples"][name] = audit(path)
    result["limitations"] = (
        "Capability probes only: no new LayerNorm speed, sanitizer or final-candidate qualification."
    )
    result["gpu_snapshot_after"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
            "--format=csv",
        ],
        text=True,
    )
    (args.output / "audit.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                k: {a: b for a, b in v.items() if a not in ["artifacts", "metadata"]}
                for k, v in result["probes"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
