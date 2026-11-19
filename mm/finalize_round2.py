"""Apply the preregistered full-group gates without changing round1 selection."""

import json

from common import ROOT, frozen, save_json, sha256
from experiment_round1 import confidence
from make_round2 import FAMILY

root = ROOT / "results/runs/mm_opt_r2_20261119"
m = frozen()
config = m["families"][FAMILY]
main, repeat = [
    json.loads((root / n).read_text()) for n in ["full.json", "repeat.json"]
]
name = json.loads((root / "finalist.json").read_text())["name"]
criteria = []
metrics = {}
for label, data in [("main", main), ("repeat", repeat)]:
    assert data["baseline_manifest_sha256"] == sha256(ROOT / "baseline.json")
    assert data["runner_sha256"] == sha256(ROOT / "experiment_round2.py")
    assert data["pipeline_sha256"] == sha256(root / "pipeline.json")
    validation_path = root / (
        "correction_check/summary.json"
        if label == "main"
        else "repeat_check/summary.json"
    )
    assert data["validation_sha256"] == sha256(validation_path)
    validation = json.loads(validation_path.read_text())
    assert validation["records"][name]["status"] == "PASS"
    candidate_path = (ROOT / config["ttir_path"]).with_name(name + ".ttir")
    assert data["resources"][name]["ttir_sha256"] == sha256(candidate_path)
    assert validation["records"][name]["ttir_sha256"] == sha256(candidate_path)
    rows = [r for r in data["records"] if r["name"] == name]
    controls = [r for r in data["records"] if r["name"] == "r1_canonical"]
    assert sorted(r["shape"][0] for r in rows) == config["square_sizes"]
    assert len(controls) == len(rows) == 6
    assert sorted(r["shape"][0] for r in controls) == config["square_sizes"]
    summary = confidence(rows)
    assert summary == data["summary"][name]
    valid = all(r["valid"] for r in rows + controls)
    criteria.append(
        valid
        and summary["geometric_mean"] >= 1.02
        and summary["ci95"][0] > 1
        and summary["min"] >= 0.95
    )
    metrics[label] = dict(valid=valid, **summary)
accepted = all(criteria)
source = (
    (ROOT / config["ttir_path"]).with_name(name + ".ttir")
    if accepted
    else ROOT / config["ttir_path"]
)
selection = dict(
    family=FAMILY,
    status="ACCEPTED_GAIN" if accepted else "BASELINE_FALLBACK",
    name=name if accepted else "baseline",
    tested_finalist=name,
    source_path=str(source.relative_to(ROOT)),
    source_sha256=sha256(source),
    baseline_manifest_sha256=sha256(ROOT / "baseline.json"),
    metrics=metrics,
    reason=(
        "Accepted only if both full-group runs meet all preregistered gates"
        if accepted
        else (
            "No candidate reached>=1.02 full-group geomean in both runs; "
            "do not promote a small local or screening effect."
        )
    ),
    old_round1_selection_sha256=sha256(ROOT / "results/round1_selection.json"),
    unchanged_round1_public_entry=True,
)
if (root / "selection.json").exists():
    assert json.loads((root / "selection.json").read_text()) == selection
else:
    save_json(root / "selection.json", selection)
print(json.dumps(selection, indent=2))
