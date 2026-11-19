"""Select under the user's warning exception; no synchronization qualification implied."""

# flake8: noqa: E402 -- standalone MM helpers.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import frozen, save_json, sha256
from experiment_round1 import confidence
from performance_v2 import BASE, NAMES, WAIVER, fingerprints
from probe import CONFIG, ROOT

R = ROOT / "results/performance_v2"


def save_or_verify(path, data):
    if path.exists():
        assert json.loads(path.read_text()) == data, "Do not rewrite a frozen selection"
    else:
        save_json(path, data)


def main():
    frozen()
    a, b = [json.loads((R / name).read_text()) for name in ["main.json", "repeat.json"]]
    decisions = {}
    declared = json.loads((ROOT / "results/prototype_inputs_v4.json").read_text())[
        "inputs"
    ]
    for name in [BASE, *NAMES]:
        assert sha256(ROOT / "ir" / f"{name}.ttir") == declared[f"{name}.ttir"]
    for data, validation in [
        (a, R / "check/summary.json"),
        (b, R / "repeat_check/summary.json"),
    ]:
        assert data["fingerprints"] == fingerprints()
        assert data["validation_sha256"] == sha256(validation)
        assert data["baseline_manifest_sha256"] == sha256(ROOT.parent / "baseline.json")
        assert data["sync_qualified"] is False and data["warning_exception"] == WAIVER
    for name in NAMES:
        acceptable = True
        metrics = {}
        for label, data in [("main", a), ("repeat", b)]:
            rows = [r for r in data["records"] if r["name"] == name]
            controls = [r for r in data["records"] if r["name"] == "control"]
            assert [r["shape"][0] for r in rows] == CONFIG["square_sizes"]
            assert [r["shape"][0] for r in controls] == CONFIG["square_sizes"]
            stats = confidence(rows)
            assert stats == data["summary"][name]
            valid = all(r["valid"] for r in rows + controls)
            acceptable &= (
                valid
                and stats["geometric_mean"] >= 1.02
                and stats["ci95"][0] > 1
                and stats["min"] >= 0.95
            )
            metrics[label] = dict(valid=valid, **stats)
            assert (
                data["resources"][name]["stages"]["cubin"]
                == json.loads(
                    (
                        R
                        / ("check" if label == "main" else "repeat_check")
                        / "summary.json"
                    ).read_text()
                )["records"][name]["artifacts"]["stages"]["cubin"]
            )
        decisions[name] = dict(
            accepted_performance_with_user_exception=bool(acceptable), metrics=metrics
        )
    eligible = [
        name
        for name, d in decisions.items()
        if d["accepted_performance_with_user_exception"]
    ]
    selected = (
        max(eligible, key=lambda n: decisions[n]["metrics"]["main"]["geometric_mean"])
        if eligible
        else BASE
    )
    source = ROOT / "ir" / f"{selected}.ttir"
    target = ROOT / "ir/selected_host.ttir"
    if target.exists():
        assert target.read_bytes() == source.read_bytes()
    else:
        with target.open("xb") as f:
            f.write(source.read_bytes())
    registry = json.loads((ROOT / "results/prototype_inputs_v4.json").read_text())
    registry["inputs"][target.name] = sha256(target)
    save_or_verify(ROOT / "results/prototype_inputs_v5.json", registry)
    result = dict(
        selected=selected,
        baseline=BASE,
        source_path=str(source.relative_to(ROOT)),
        source_sha256=sha256(source),
        selected_alias="ir/selected_host.ttir",
        selected_sha256=sha256(target),
        decisions=decisions,
        status=(
            "PERFORMANCE_SELECTED_UNDER_EXPLICIT_USER_RACE_EXCEPTION"
            if eligible
            else "BASELINE_FALLBACK"
        ),
        sync_qualified=False,
        warning_exception=WAIVER,
        old_mm_optimized_unchanged=True,
        plan_sha256=sha256(ROOT / "optimization_plan_v2.md"),
        fingerprints=fingerprints(),
        main_sha256=sha256(R / "main.json"),
        repeat_sha256=sha256(R / "repeat.json"),
        registry_sha256=sha256(ROOT / "results/prototype_inputs_v5.json"),
        scope="Only six target-family shapes, new host ABI/local reader, official backend/runtime; kernel graph metric not old interface end-to-end gain",
    )
    save_or_verify(R / "selection.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
