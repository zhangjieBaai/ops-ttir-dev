"""Prove all candidate files can be reproduced directly from frozen TTIR."""

from common import ROOT, ir_path, manifest, save_json, sha256
from make_candidates import tiled

records = []
for group in manifest()["groups"]:
    original = ir_path(group, "baseline").read_text()
    for rows in [1, 4, 16, 32]:
        path = ir_path(group, f"r1_tile{rows}")
        expected = (
            tiled(original, rows)
            if "@mean_dim_kernel_non_inner_vec" in original
            else original
        )
        assert expected == path.read_text()
        records.append(dict(group=group, rows=rows, sha256=sha256(path)))
    assert (
        ir_path(group, "optimized").read_bytes()
        == ir_path(group, "r1_tile1").read_bytes()
    )
save_json(
    ROOT / "results/generator_reproduction.json", dict(passed=True, records=records)
)
