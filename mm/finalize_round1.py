"""Select each family from identity-matched complete primary and repeat evidence."""

import hashlib
import json
from pathlib import Path

from common import frozen, sha256
from experiment_round1 import confidence

root = Path("/root/ops-ttir-dev/mm")
r = root / "results/runs/mm_opt_r1_20261118"
m = frozen()
a, b = [json.loads((r / name).read_text()) for name in ["full.json", "repeat.json"]]
for run, validation_path in [
    (a, r / "followup_check/summary.json"),
    (b, r / "repeat_check/summary.json"),
]:
    assert run["baseline_manifest_sha256"] == sha256(root / "baseline.json")
    assert run["runner_sha256"] == sha256(root / "experiment_round1.py")
    assert run["validation_sha256"] == sha256(validation_path)
    assert run["pipeline_sha256"] == sha256(r / "pipeline.json")
choices = json.loads((r / "finalists.json").read_text())
result = dict(
    baseline_manifest_sha256=hashlib.sha256(
        (root / "baseline.json").read_bytes()
    ).hexdigest(),
    families={},
    plan_path="optimization_round1.md",
    plan_sha256=hashlib.sha256(
        (root / "optimization_round1.md").read_bytes()
    ).hexdigest(),
    full_results=str((r / "full.json").relative_to(root)),
    repeat_results=str((r / "repeat.json").relative_to(root)),
    full_sha256=hashlib.sha256((r / "full.json").read_bytes()).hexdigest(),
    repeat_sha256=hashlib.sha256((r / "repeat.json").read_bytes()).hexdigest(),
)
for family, c in m["families"].items():
    proposed = choices[family]
    accepted = False
    details = {}
    if proposed != "baseline":
        valid = True
        for label, run in [("main", a), ("repeat", b)]:
            rows = [
                x
                for x in run["records"]
                if x["family"] == family and x["name"] == proposed
            ]
            controls = [
                x
                for x in run["records"]
                if x["family"] == family and x["name"] == "r1_canonical"
            ]
            assert sorted(x["shape"][0] for x in rows) == c["square_sizes"]
            assert sorted(x["shape"][0] for x in controls) == c["square_sizes"]
            s = run["summary"][family][proposed]
            assert s == confidence(rows), "Summary differs from raw paired samples"
            valid &= all(x["valid"] for x in rows + controls)
            details[label] = s
            valid &= (
                s["geometric_mean"] >= 1.02 and s["ci95"][0] > 1 and s["min"] >= 0.95
            )
        accepted = bool(valid)
    selected = proposed if accepted else "baseline"
    src = (root / c["ttir_path"]).with_name(selected + ".ttir")
    dest = src.with_name("optimized.ttir")
    if dest.exists():
        assert dest.read_bytes() == src.read_bytes()
    else:
        with dest.open("xb") as f:
            f.write(src.read_bytes())
    reason = (
        "Meets preregistered per-family primary/repeat geomean, CI, worst-shape and validity gates"
        if accepted
        else (
            "Baseline retained: bounded attempts did not establish an acceptable reproducible gain; "
            "invalid/negative evidence retained"
        )
    )
    result["families"][family] = dict(
        name=selected,
        optimized_path=str(dest.relative_to(root)),
        optimized_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        source_path=str(src.relative_to(root)),
        source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        status="ACCEPTED_GAIN" if accepted else "BASELINE_FALLBACK",
        reason=reason,
        measurements=details,
        square_sizes=c["square_sizes"],
    )
with (root / "results/round1_selection.json").open("x") as f:
    f.write(json.dumps(result, indent=2) + "\n")
with (r / "selected_names.json").open("x") as f:
    f.write(json.dumps({f: "optimized" for f in m["families"]}, indent=2) + "\n")
for f, c in result["families"].items():
    print(f, c["status"], c["name"], c["measurements"])
