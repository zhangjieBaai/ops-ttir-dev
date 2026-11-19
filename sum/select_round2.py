"""Promote a hash-bound launch_v2 result after independent timing and sanitizers."""

import json

from common import PERF_SIZES, ROOT, save_json, sha256, write_once
from round2 import compile_path, frozen, identities, load, manifest, require_evidence

RUN = ROOT / "results/runs/r2_launch"


def main():
    names = ["baseline", "r2_single16384"]
    kernels, routes = load(names, frozen(), manifest())
    evidence = identities(kernels, routes)
    paths = {
        label: RUN / file
        for label, file in {
            "pipeline": "pipeline_v2.json",
            "accuracy": "accuracy_v2.json",
            "main": "full.json",
            "repeat": "repeat.json",
            "memcheck": "memcheck.json",
            "racecheck": "racecheck.json",
        }.items()
    }
    records = {}
    for label, path in paths.items():
        data = json.loads(path.read_text())
        if label in ["main", "repeat"]:
            assert data["aa_passed"]
            assert len(data["controls"]) == 30
            rows = [r for r in data["records"] if r["name"] == names[1]]
            assert [r["n"] for r in rows] == PERF_SIZES
            assert all(len(r["pairs"]) == 7 for r in rows)
            metrics = data["summaries"][names[1]]
            assert metrics["geometric"] > 1.10 and metrics["worst"] >= 0.95
            assert data["pipeline_sha256"] == sha256(paths["pipeline"])
            assert data["validation_sha256"] == sha256(paths["accuracy"])
            for name, shapes in json.loads(json.dumps(evidence, default=str)).items():
                assert data["artifacts"][name] == shapes
        else:
            require_evidence(path, evidence)
        assert data["uncontended"]
        assert data["driver_sha256"] == sha256(ROOT / "round2.py")
        assert data["round2_protocol_sha256"] == sha256(
            ROOT / "optimization_plan_round2.md"
        )
        assert data["round2_mapping_sha256"] == sha256(ROOT / "round2_manifest.json")
        records[label] = data
    assert records["main"]["gpu_uuid"] == records["repeat"]["gpu_uuid"]
    assert records["main"]["seed"] != records["repeat"]["seed"]
    aliases = {}
    for n, stages in routes[names[1]].items():
        aliases[n] = []
        for index, stage in enumerate(stages):
            source = ROOT / stage["path"]
            alias = ROOT / f"ir/launch_v2/selected/n{n}/stage{index + 1}.ttir"
            write_once(alias, source.read_text())
            updated = dict(stage, path=str(alias.relative_to(ROOT)))
            compiled = compile_path(updated)
            original = kernels[stage["path"]]
            assert compiled.asm["cubin"] == original.asm["cubin"]
            aliases[n].append(updated)
    save_json(
        ROOT / "results/selection_round2.json",
        dict(
            experiment="sum_launch_v2",
            selected=names[1],
            threshold=16384,
            statement="New launch contract; small inputs use one kernel. Not fixed-launch TTIR-only gain.",
            evidence_sha256={label: sha256(path) for label, path in paths.items()},
            main=records["main"]["summaries"][names[1]],
            repeat=records["repeat"]["summaries"][names[1]],
            artifacts=evidence[names[1]],
            routes=routes[names[1]],
            selected_aliases=aliases,
        ),
    )
    print(
        "PASS >1.10x main/repeat; exact hashes/options/artifacts/sanitizers; selected aliases cubin identical"
    )


if __name__ == "__main__":
    main()
