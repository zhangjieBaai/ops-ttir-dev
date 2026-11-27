"""Check original frozen-source baseline for every expanded round2 point."""

import argparse
from pathlib import Path

import torch
from common_mr import (
    MATRIX,
    ROOT,
    _attn_bwd_dkdv_only,
    allocate,
    compile_ir,
    digest,
    group,
    launch,
    launch_source,
    save,
    snapshot,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    assert not out.exists()
    out.mkdir(parents=True)
    result = {"environment": snapshot(), "points": []}
    kernels = {}
    for b, h, n, d, c in MATRIX:
        t = allocate(b, h, n, d, c)
        source = launch_source(_attn_bwd_dkdv_only, t)
        expected = [t[k].clone() for k in ("dk", "dv")]
        g = group(d, c)
        path = ROOT / "ir" / g / "baseline.ttir"
        assert path.read_text() == source.asm["ttir"], (
            b,
            h,
            n,
            d,
            c,
            "specialization changed",
        )
        if g not in kernels:
            kernels[g] = compile_ir("baseline", d, c)
        ck, meta = kernels[g]
        assert all(source.asm[k] == ck.asm[k] for k in ("llir", "ptx", "cubin"))
        assert all(
            source.metadata._asdict()[k] == v for k, v in meta["parsed_options"].items()
        )
        launch(ck, t)
        torch.cuda.synchronize()
        assert all(torch.equal(t[k], v) for k, v in zip(("dk", "dv"), expected))
        result["points"].append(
            {
                "shape": [b, h, n, d, c],
                "group": g,
                "ttir_sha256": digest(path),
                "llvm_ptx_cubin_equal": True,
                "outputs_bitwise_equal": True,
                "metadata": meta,
            }
        )
        save(out / "pipeline.json", result)
        print("PIPELINE", b, h, n, d, c, flush=True)
    result["passed"] = True
    save(out / "pipeline.json", result)


if __name__ == "__main__":
    main()
