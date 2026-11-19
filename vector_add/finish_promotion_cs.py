"""Bind formal alias results and exact-identity sanitizer reuse to the promotion."""

import json

from common import PERF_SIZES, ROOT, save_json, sha256

RUN = ROOT / "results/runs/promote_cs_20261119"


def main():
    primary = json.loads((RUN / "full.json").read_text())
    repeat = json.loads((RUN / "repeat.json").read_text())
    expected = sha256(ROOT / "ir/r1_store_cs.ttir")
    assert expected == sha256(ROOT / "ir/optimized.ttir")
    checks = {}
    for label, result in (("main", primary), ("repeat", repeat)):
        assert result["aa_passed"] and result["suite"] == "full"
        assert sorted(r["n"] for r in result["records"]) == PERF_SIZES
        assert all(r["name"] == "optimized" for r in result["records"])
        assert result["input_ttir"]["optimized"]["sha256"] == expected
        assert (
            result["artifacts"]["optimized"]["stages"]["cubin"]
            == primary["artifacts"]["optimized"]["stages"]["cubin"]
        )
        accuracy = RUN / f"accuracy_{label}.json"
        validated = json.loads(accuracy.read_text())
        assert validated["passed"] and len(validated["cases"]) == 36
        assert validated["kernels"]["optimized"] == expected
        checks[label] = dict(
            accuracy_sha256=sha256(accuracy),
            performance_sha256=sha256(
                RUN / ("full.json" if label == "main" else "repeat.json")
            ),
        )
    reused = {}
    for tool in ("memcheck", "racecheck"):
        path = ROOT / f"results/runs/r1_20261119/{tool}_final.json"
        prior = json.loads(path.read_text())
        assert prior["passed"] and prior["kernels"]["r1_store_cs"] == expected
        assert (
            prior["artifacts"]["r1_store_cs"]["stages"]["cubin"]
            == primary["artifacts"]["optimized"]["stages"]["cubin"]
        )
        reused[tool] = dict(
            status="REUSED_EXACT_TTIR_AND_CUBIN_IDENTITY",
            receipt=str(path.relative_to(ROOT)),
            receipt_sha256=sha256(path),
            log_sha256=sha256(path.with_suffix(".log")),
            cases=len(prior["cases"]),
            original_visible_devices=prior["visible_devices"],
            note="Existing GPU 7 H100 SM90 sanitizer evidence; not a new GPU 0 sanitizer run",
        )
    save_json(
        RUN / "sanitizer_reuse.json",
        dict(
            ttir_sha256=expected,
            cubin_sha256=primary["artifacts"]["optimized"]["stages"]["cubin"],
            reused=reused,
        ),
    )
    selection = json.loads((ROOT / "results/selection.json").read_text())
    selection.update(
        status="COMPLETE",
        acceptance_basis=(
            "Explicit user acceptance of local gain and modest overall gain; "
            "original 1.02x threshold waived"
        ),
        performance_evidence="results/runs/promote_cs_20261119/full.json",
        independent_repeat="results/runs/promote_cs_20261119/repeat.json",
        main_summary=primary["summaries"]["optimized"],
        repeat_summary=repeat["summaries"]["optimized"],
        both_aa_passed=True,
        checks=checks,
        sanitizer_evidence="results/runs/promote_cs_20261119/sanitizer_reuse.json",
        actual_cubin_sha256=primary["artifacts"]["optimized"]["stages"]["cubin"],
    )
    save_json(RUN / "completed_selection.json", selection)
    (ROOT / "results/selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Formal optimized/baseline comparison completed")


if __name__ == "__main__":
    main()
