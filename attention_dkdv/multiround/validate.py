"""Expanded contract and exact-parent/baseline checks for direct TTIR candidates."""

import argparse
import json
from pathlib import Path

import torch
from common_mr import (
    MATRIX,
    allocate,
    candidates,
    compile_ir,
    cpu_autograd,
    group,
    input_hashes,
    launch,
    save,
    snapshot,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--selection")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    names = (
        candidates()
        if not a.selection
        else {
            g: [v]
            for g, v in json.loads(Path(a.selection).read_text())[
                "tested_candidate"
            ].items()
        }
    )
    result = {
        "environment": snapshot(),
        "seed": a.seed,
        "compile": {},
        "checks": [],
        "passed": True,
    }
    kernels = {}
    for d in (64, 128):
        for c in (False, True):
            g = group(d, c)
            for name in ["baseline", "anchor", "parent"] + names[g]:
                key = (name, d, c)
                try:
                    ck, meta = compile_ir(name, d, c)
                    kernels[key] = ck
                    result["compile"][str(key)] = {"status": "PASS", "metadata": meta}
                except Exception as exc:
                    result["compile"][str(key)] = {"status": "FAIL", "error": str(exc)}
                save(out / "accuracy.json", result)
    matrix = (
        [(1, 4, 128, d, c) for d in (64, 128) for c in (False, True)]
        if a.pilot
        else MATRIX
    )
    cases = [(s, "normal") for s in matrix]
    for d in (64, 128):
        for c in (False, True):
            for dist in ("zero_do", "constant", "large"):
                cases.append(((1, 4, 128, d, c), dist))
    for shape, dist in cases:
        b, h, n, d, c = shape
        g = group(d, c)
        t = allocate(*shape, distribution=dist, seed=a.seed)
        if n == 128 and dist == "normal":
            cpu_autograd(t)
        hashes = input_hashes(t)
        before = {k: t[k].clone() for k in ("q", "k", "v", "do", "m", "delta")}
        baseline = None
        for name in ["baseline", "anchor", "parent"] + names[g]:
            key = (name, d, c)
            if key not in kernels:
                continue
            t["dk"].fill_(float("nan"))
            t["dv"].fill_(float("nan"))
            row = {
                "shape": shape,
                "distribution": dist,
                "name": name,
                "group": g,
                "input_sha256": hashes,
                "status": "PASS",
                "max_abs": {},
            }
            try:
                launch(kernels[key], t)
                torch.cuda.synchronize()
                for k, ref in zip(("dk", "dv"), t["reference"]):
                    assert bool(torch.isfinite(t[k]).all()), k + " nonfinite/unwritten"
                    row["max_abs"][k] = (t[k].double() - ref).abs().max().item()
                    torch.testing.assert_close(
                        t[k].double(), ref, atol=0.008, rtol=0.03
                    )
                    if baseline is not None:
                        assert torch.equal(
                            t[k].view(torch.int16), baseline[k].view(torch.int16)
                        ), (k + " not bitwise baseline")
                assert all(
                    torch.equal(t[k], v) for k, v in before.items()
                ), "input changed"
                row["bitwise_baseline"] = True
            except Exception as exc:
                row["status"] = "FAIL"
                row["error"] = str(exc)
            if name == "baseline":
                baseline = {k: t[k].clone() for k in ("dk", "dv")}
            result["checks"].append(row)
            save(out / "accuracy.json", result)
            print(
                "ACCURACY",
                shape,
                dist,
                name,
                row["status"],
                row.get("error", ""),
                flush=True,
            )
    statuses = {}
    for g, gs in names.items():
        for name in ["baseline", "anchor", "parent"] + gs:
            records = [
                r for r in result["checks"] if r["group"] == g and r["name"] == name
            ]
            statuses[g + "/" + name] = bool(records) and all(
                r["status"] == "PASS" for r in records
            )
    result["valid"] = statuses
    result["passed"] = all(
        statuses[g + "/" + n] for g in names for n in ["baseline", "anchor", "parent"]
    )
    save(out / "accuracy.json", result)


if __name__ == "__main__":
    main()
