"""Verify all frozen shape mappings before a measurement process."""

import argparse
import json

import common as c
import torch


def verify(output):
    result = dict(environment=c.environment(), points=[])
    manifest = json.loads((c.ROOT / "baseline.json").read_text())
    for item in manifest["mapping"].values():
        m, e, w = (item[k] for k in ("tokens", "experts", "warps"))
        t = c.allocate(m, e, seed=333)
        src = c.source_launch(t, w)
        source_outputs = [t[k].clone() for k in ("g", "w", "i")]
        native = c.compile_pair(e, w)
        for k in ("g", "w", "i"):
            t[k].fill_(-777)
        c.launch(native, t)
        torch.cuda.synchronize()
        assert all(
            torch.equal(t[k], ref) for k, ref in zip(("g", "w", "i"), source_outputs)
        )
        checks = []
        for stage, (s, n) in enumerate(zip(src, native), 1):
            path = c.ROOT / "ir" / c.group(e, w) / f"baseline_s{stage}.ttir"
            assert s.asm["ttir"] == path.read_text()
            same = {ext: s.asm[ext] == n.asm[ext] for ext in ("llir", "ptx", "cubin")}
            assert all(same.values())
            sm, nm = s.metadata._asdict(), n.metadata._asdict()
            assert {k: v for k, v in sm.items() if k != "hash"} == {
                k: v for k, v in nm.items() if k != "hash"
            }
            checks.append(same)
        result["points"].append(
            dict(
                item,
                checks=checks,
                outputs_bitwise=True,
                artifacts=c.artifacts(native, e, w, "baseline"),
            )
        )
    c.save(output, result)
    print(f"Pipeline PASS: {len(result['points'])} mappings", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert not c.Path(args.output).exists()
    verify(args.output)
