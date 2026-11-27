"""Semantic, coverage and guard checks before timings."""

import argparse
import json

import common as c
import torch


def validate(output, variants, seed=0, quick=False):
    result = dict(environment=c.environment(), seed=seed, checks=[], pass_all=True)
    manifest = json.loads((c.ROOT / "baseline.json").read_text())
    shapes = c.PERF if not quick else [(16, 256), (128, 512)]
    for m, e in shapes:
        w = manifest["mapping"][f"{m}x{e}"]["warps"]
        kernels = {name: c.compile_pair(e, w, name) for name in variants}
        baseline = c.compile_pair(e, w)
        distributions = (
            "normal",
            "ties",
            "group_ties",
            "bias_dominant",
            "saturated",
            "negative",
        )
        for dist in distributions:
            t = c.allocate(m, e, dist, seed + m + e)
            xb = (t["x"].clone(), t["b"].clone())
            ref_w, ref_i = c.reference(t)
            c.launch(baseline, t)
            base_w, base_i, base_g = [t[k].clone() for k in ("w", "i", "g")]
            for name, pair in kernels.items():
                t["w"].fill_(float("nan"))
                t["g"].fill_(float("nan"))
                t["i"].fill_(-777)
                c.launch(pair, t)
                torch.cuda.synchronize()
                row = dict(shape=[m, e], distribution=dist, variant=name)
                try:
                    c.check_guards(t)
                    assert torch.equal(xb[0], t["x"]) and torch.equal(xb[1], t["b"])
                    row["indices_vs_baseline"] = torch.equal(t["i"], base_i)
                    row["weights_bitwise_vs_baseline"] = torch.equal(t["w"], base_w)
                    row["groups_bitwise_vs_baseline"] = torch.equal(t["g"], base_g)
                    row["indices_vs_cpu"] = torch.equal(t["i"].cpu().long(), ref_i)
                    row["max_abs_weight_error_cpu"] = (
                        (t["w"].cpu() - ref_w).abs().max().item()
                    )
                    assert row["indices_vs_baseline"]
                    assert row["weights_bitwise_vs_baseline"]
                    assert row["groups_bitwise_vs_baseline"]
                    assert row[
                        "indices_vs_cpu"
                    ], f"CPU index mismatches: {(t['i'].cpu() != ref_i).sum().item()}"
                    torch.testing.assert_close(
                        t["w"].cpu(), ref_w, rtol=2e-5, atol=2e-6
                    )
                    row["status"] = "PASS"
                except Exception as exc:
                    row.update(status="FAIL", error=str(exc))
                    result["pass_all"] = False
                    print("FAIL", row, flush=True)
                result["checks"].append(row)
        print("validated", m, e, flush=True)
    result["variants"] = {
        f"{e}_{w}_{name}": c.artifacts(c.compile_pair(e, w, name), e, w, name)
        for e, w in {(x["experts"], x["warps"]) for x in manifest["mapping"].values()}
        for name in variants
    }
    c.save(output, result)
    print(
        f"Validation {'PASS' if result['pass_all'] else 'FAIL'}: {len(result['checks'])} checks",
        flush=True,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "membership", "argmax", "combined"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    assert not c.Path(args.output).exists()
    out = validate(args.output, args.variants, args.seed, args.quick)
    raise SystemExit(0 if out["pass_all"] else 1)
