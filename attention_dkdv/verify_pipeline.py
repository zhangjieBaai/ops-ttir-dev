"""Export once or recheck all shape/source/file pipelines before measurement."""

import argparse
from pathlib import Path

import torch
from common import (
    MATRIX,
    ROOT,
    allocate,
    compile_ir,
    digest,
    group,
    launch,
    launch_source,
    save,
    snapshot,
)
from frozen_source import _attn_bwd, _attn_bwd_dkdv_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--export", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    result = dict(environment=snapshot(), points=[])
    for b, h, n, d, c in MATRIX:
        t = allocate(b, h, n, d, c)
        source = launch_source(_attn_bwd_dkdv_only, t)
        expected = [t[k].clone() for k in ("dk", "dv")]
        path = ROOT / "ir" / group(d, c) / "baseline.ttir"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            assert a.export
            path.write_text(source.asm["ttir"])
        assert path.read_text() == source.asm["ttir"], (b, h, n, d, c, "ttir differs")
        ck, meta = compile_ir("baseline", d, c)
        exact = {
            ext: source.asm[ext] == ck.asm[ext] for ext in ("llir", "ptx", "cubin")
        }
        assert all(exact.values()), exact
        launch(ck, t)
        torch.cuda.synchronize()
        assert all(torch.equal(t[k], v) for k, v in zip(("dk", "dv"), expected))
        launch_source(_attn_bwd, t)
        torch.cuda.synchronize()
        assert all(torch.equal(t[k], v) for k, v in zip(("dk", "dv"), expected))
        result["points"].append(
            dict(
                shape=(b, h, n, d, c),
                group=group(d, c),
                ttir_sha256=digest(path),
                downstream_byte_exact=exact,
                extraction_dkdv_bitwise=True,
                metadata=meta,
            )
        )
        save(out / "pipeline.json", result)
        print("pipeline", b, h, n, d, c, exact, flush=True)
    save(
        ROOT / "baseline_manifest.json",
        dict(
            matrix=MATRIX,
            groups={
                group(d, c): digest(ROOT / "ir" / group(d, c) / "baseline.ttir")
                for d in (64, 128)
                for c in (False, True)
            },
        ),
    )


if __name__ == "__main__":
    main()
