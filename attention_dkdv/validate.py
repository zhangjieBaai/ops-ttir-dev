"""Numerical and coverage validation, both original math and baseline comparisons."""

import argparse
import json
from pathlib import Path

import torch
from common import MATRIX, ROOT, allocate, compile_ir, launch, save, snapshot


def cpu_autograd(t):
    q = t["q"].cpu().double().requires_grad_(True)
    k = t["original_k"].cpu().double().requires_grad_(True)
    v = t["v"].cpu().double().requires_grad_(True)
    scores = q @ k.transpose(-1, -2) * t["scale"]
    n = q.shape[-2]
    if t["causal"]:
        scores = scores.masked_fill(
            torch.arange(n)[:, None] < torch.arange(n)[None, :], float("-inf")
        )
    output = scores.softmax(-1) @ v
    dk, dv = torch.autograd.grad(output, (k, v), t["do"].cpu().double())
    for actual, expected in zip(t["reference"], (dk, dv)):
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-10, rtol=1e-10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--names", default="all")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--small", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    names = (
        ["baseline"]
        + list(json.loads((ROOT / "candidates.json").read_text())["d64_c0"])
        if a.names == "all"
        else a.names.split(",")
    )
    result = dict(
        environment=snapshot(),
        seed=a.seed,
        rtol=0.03,
        atol=0.008,
        compile={},
        checks=[],
        passed=True,
    )
    kernels = {}
    for d in (64, 128):
        for c in (False, True):
            for name in names:
                key = (name, d, c)
                try:
                    kernels[key], meta = compile_ir(name, d, c)
                    result["compile"][str(key)] = dict(status="PASS", metadata=meta)
                except Exception as exc:
                    result["compile"][str(key)] = dict(status="FAIL", error=repr(exc))
                    result["passed"] = False
                save(out / "accuracy.json", result)
    cases = [] if a.small else [(shape, "normal") for shape in MATRIX]
    for d in (64, 128):
        for c in (False, True):
            for dist in ("normal", "zero_do", "constant", "large"):
                cases.append(((1, 4, 128, d, c), dist))
    for shape, dist in cases:
        b, h, n, d, c = shape
        t = allocate(*shape, distribution=dist, seed=a.seed)
        if n == 128 and dist == "normal":
            cpu_autograd(t)
        before = {key: t[key].clone() for key in ("q", "k", "v", "do", "m", "delta")}
        base = None
        for name in names:
            key = (name, d, c)
            if key not in kernels:
                continue
            t["dk"].fill_(float("nan"))
            t["dv"].fill_(float("nan"))
            item = dict(
                shape=shape,
                distribution=dist,
                name=name,
                status="PASS",
                errors=[],
                max_abs={},
                bitwise_baseline={},
            )
            try:
                launch(kernels[key], t)
                torch.cuda.synchronize()
                for k, ref in zip(("dk", "dv"), t["reference"]):
                    assert torch.isfinite(t[k]).all(), k + " nonfinite/unwritten"
                    err = (t[k].double() - ref).abs()
                    item["max_abs"][k] = err.max().item()
                    torch.testing.assert_close(
                        t[k].double(), ref, atol=0.008, rtol=0.03
                    )
                    if base is not None:
                        item["bitwise_baseline"][k] = torch.equal(t[k], base[k])
                assert all(
                    torch.equal(t[k], v) for k, v in before.items()
                ), "input mutation"
            except Exception as exc:
                item["status"] = "FAIL"
                item["errors"].append(str(exc))
                result["passed"] = False
            if name == "baseline":
                base = {key: t[key].clone() for key in ("dk", "dv")}
            result["checks"].append(item)
            save(out / "accuracy.json", result)
            print(
                "accuracy",
                shape,
                dist,
                name,
                item["status"],
                item["max_abs"],
                flush=True,
            )
    assert result["passed"], "Some checks failed; see immutable result."


if __name__ == "__main__":
    main()
