"""Freeze one uniform vector-path candidate after tuning-only screening."""

import json

from common import ROOT, ir_path, manifest, save_json, sha256

screen = ROOT / "results/runs/screen.json"
data = json.loads(screen.read_text())
assert data["aa_passed"] and data["suite"] == "tuning"
selected = max(data["summaries"], key=lambda n: data["summaries"][n]["geometric"])
assert selected == "r1_tile1"
records = {}
for group in manifest()["groups"]:
    source = ir_path(group, selected)
    path = ir_path(group, "optimized")
    with path.open("x") as f:
        f.write(source.read_text())
    records[group] = dict(candidate=selected, sha256=sha256(path))
save_json(
    ROOT / "results/selection.json",
    dict(
        selected=selected,
        screen_sha256=sha256(screen),
        rule="Same candidate in every group; inner/non-inner groups equal baseline bytes",
        groups=records,
    ),
)
