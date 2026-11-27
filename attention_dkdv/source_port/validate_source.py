"""Keep the original 28-case numerical contract; add the real source-JIT route."""

import argparse
from pathlib import Path

import torch
from common_port import (
    INPUT_KEYS,
    MATRIX,
    ROUTES,
    allocate,
    check_source,
    compile_routes,
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
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--source-only", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    result = dict(
        environment=snapshot(),
        source_manifest=check_source(),
        seed=a.seed,
        rtol=0.03,
        atol=0.008,
        compile={},
        checks=[],
        cpu_autograd=[],
        passed=True,
    )
    kernels = {}
    routes = ("source",) if a.source_only else ROUTES
    cases = [(shape, "normal") for shape in MATRIX] + [
        ((1, 4, 128, d, c), dist)
        for d in (64, 128)
        for c in (False, True)
        for dist in ("normal", "zero_do", "constant", "large")
    ]
    for shape, dist in cases:
        d, c = shape[-2:]
        g = group(d, c)
        t = allocate(*shape, distribution=dist, seed=a.seed)
        if g not in kernels:
            kernels[g], result["compile"][g] = compile_routes(t)
        if shape[2] == 128 and dist == "normal":
            cpu_autograd(t)
            result["cpu_autograd"].append(dict(shape=shape, status="PASS"))
        hashes = input_hashes(t)
        before = {k: t[k].clone() for k in INPUT_KEYS}
        base = None
        for route in routes:
            t["dk"].fill_(float("nan"))
            t["dv"].fill_(float("nan"))
            item = dict(
                shape=shape,
                distribution=dist,
                route=route,
                input_sha256=hashes,
                status="PASS",
                max_abs={},
                bitwise_baseline={},
                errors=[],
            )
            try:
                launch(kernels[g][route], t)
                torch.cuda.synchronize()
                for key, ref in zip(("dk", "dv"), t["reference"]):
                    assert torch.isfinite(t[key]).all(), key + " nonfinite/unwritten"
                    item["max_abs"][key] = (t[key].double() - ref).abs().max().item()
                    torch.testing.assert_close(
                        t[key].double(), ref, atol=0.008, rtol=0.03
                    )
                    if base is not None:
                        item["bitwise_baseline"][key] = torch.equal(t[key], base[key])
                assert all(
                    torch.equal(t[k], v) for k, v in before.items()
                ), "input mutation"
            except Exception as exc:
                item["status"] = "FAIL"
                item["errors"].append(str(exc))
                result["passed"] = False
            if route == "baseline":
                base = {k: t[k].clone() for k in ("dk", "dv")}
            result["checks"].append(item)
            save(out / "accuracy.json", result)
            print(
                "accuracy",
                shape,
                dist,
                route,
                item["status"],
                item["max_abs"],
                flush=True,
            )
    assert result["passed"]


if __name__ == "__main__":
    main()
