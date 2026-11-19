"""Apply preregistered promotion rules and materialize hash-bound optimized aliases."""

import json

from common import ROOT, frozen, hashes, save_json, sha256, write_once


def main():
    main_path = ROOT / "results/runs/round1/full.json"
    repeat_path = ROOT / "results/runs/round1/repeat.json"
    primary, repeat = [
        json.loads(path.read_text()) for path in (main_path, repeat_path)
    ]
    assert primary["aa_passed"] and repeat["aa_passed"]
    assert len(primary["controls"]) == len(repeat["controls"]) == 30
    assert primary["gpu_uuid"] == repeat["gpu_uuid"]
    eligible = []
    for name, metrics in primary["summaries"].items():
        second = repeat["summaries"][name]
        if (
            min(metrics["geometric"], second["geometric"]) > 1.01
            and min(metrics["worst"], second["worst"]) >= 0.95
        ):
            eligible.append(name)
    selected = (
        max(eligible, key=lambda name: primary["summaries"][name]["geometric"])
        if eligible
        else "baseline"
    )
    if selected != "baseline":
        for tool in ["memcheck", "racecheck"]:
            assert json.loads((ROOT / f"results/runs/final/{tool}.json").read_text())[
                "passed"
            ]
    for config in frozen()["groups"].values():
        source = (ROOT / config["path"]).with_name(f"{selected}.ttir")
        write_once(source.with_name("optimized.ttir"), source.read_text())
    result = dict(
        selected=selected,
        reason=(
            "No candidate exceeds 1.01x geometric speedup in both independent runs; preserve baseline"
            if selected == "baseline"
            else "Passed frozen numerical, timing and sanitizer promotion gates"
        ),
        primary_sha256=sha256(main_path),
        repeat_sha256=sha256(repeat_path),
        selected_hashes=hashes(selected),
        optimized_hashes=hashes("optimized"),
        main=primary["summaries"],
        repeat=repeat["summaries"],
    )
    assert result["selected_hashes"] == result["optimized_hashes"]
    save_json(ROOT / "results/selection.json", result)
    print(result["selected"], result["reason"])


if __name__ == "__main__":
    main()
