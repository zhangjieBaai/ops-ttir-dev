"""Freeze six families from saved official autotune evidence, without re-tuning."""

import argparse
import json

from common import PERF_SHAPES, ROOT, digest, save_json, sha256

SELECTION = ROOT / "results/runs/autotune_count_20261118T135029Z/summary.json"
PROTOCOL = [
    "common.py",
    "export_baseline.py",
    "verify_pipeline.py",
    "validate.py",
    "benchmark.py",
    "freeze_baselines.py",
    "environment.py",
    "run_baseline.sh",
    "test_accuracy.sh",
    "device_guard.sh",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initialize-six",
        action="store_true",
        help="Explicit migration from archived v3 manifest",
    )
    args = parser.parse_args()
    manifest_path = ROOT / "baseline.json"
    current = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if current and "families" in current:
        from common import frozen

        frozen()
        print("PASS: six-family manifest and immutable TTIR hashes verified")
        return
    assert args.initialize_six, "Use --initialize-six for explicit migration"
    archive = ROOT / "setup_attempts/v3/baseline.json"
    if current:
        assert sha256(archive) == sha256(
            manifest_path
        ), "Old manifest must be archived byte for byte"
    selection = json.loads(SELECTION.read_text())
    assert selection["passed"] and selection["distinct_selected_ttir"] == 6
    assert [row["shape"] for row in selection["records"]] == [
        list(s) for s in PERF_SHAPES
    ]
    families, mapping = {}, {}
    for group in selection["selected_groups"]:
        h = group["ttir_sha256"]
        records = [
            row for row in selection["records"] if row["selected_ttir_sha256"] == h
        ]
        assert [row["shape"][0] for row in records] == group["square_sizes"]
        c = records[0]["selected_config"]
        assert all(
            row["selected_config"] == c and row["correctness_passed"] for row in records
        )
        name = (
            f"bm{c['BLOCK_SIZE_M']}_bn{c['BLOCK_SIZE_N']}_bk{c['BLOCK_SIZE_K']}"
            f"_g8_w{c['num_warps']}_s{c['num_stages']}"
        )
        src = SELECTION.parent / "ttir" / f"{h}.ttir"
        assert sha256(src) == h
        path = ROOT / "ir" / name / "baseline.ttir"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == src.read_bytes()
        else:
            with path.open("xb") as f:
                f.write(src.read_bytes())
        families[name] = dict(
            ttir_path=str(path.relative_to(ROOT)),
            ttir_sha256=h,
            meta={
                key: c[key]
                for key in [
                    "BLOCK_SIZE_M",
                    "BLOCK_SIZE_N",
                    "BLOCK_SIZE_K",
                    "GROUP_SIZE_M",
                ]
            },
            options={key: c[key] for key in ["num_warps", "num_stages", "num_ctas"]},
            runtime_arg_indices=[0, 1, 2, 3, 4, 5, 6, 8, 10],
            square_sizes=group["square_sizes"],
            autotune_cubin_sha256=records[0]["selected_cubin_sha256"],
        )
        families[name]["meta"]["ACTIVATION"] = ""
        families[name]["options"]["enable_fp_fusion"] = True
        for row in records:
            assert (
                row["selected_cubin_sha256"] == families[name]["autotune_cubin_sha256"]
            )
            mapping[str(row["shape"][0])] = name
    data = dict(
        experiment="mm-fp16-sm90-autotuned-six-v4",
        schema_version=4,
        source_sha256=sha256(ROOT / "tutorial.py"),
        upstream_sha256=sha256(ROOT / "upstream/03-matrix-multiplication.py.txt"),
        contract_sha256=sha256(ROOT / "optimization_plan.md"),
        common_sha256=sha256(ROOT / "common.py"),
        protocol_sha256={name: sha256(ROOT / name) for name in PROTOCOL},
        selection_source=str(SELECTION.relative_to(ROOT)),
        selection_sha256=sha256(SELECTION),
        shape_to_family=mapping,
        families=families,
        performance_shapes=PERF_SHAPES,
        historical_baseline=dict(
            manifest="setup_attempts/v3/baseline.json",
            sha256=sha256(archive),
            ttir_path="ir/baseline.ttir",
            ttir_sha256=json.loads(archive.read_text())["ttir_sha256"],
        ),
    )
    # Only the explicitly archived legacy active manifest can be replaced.
    if current:
        manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    else:
        save_json(manifest_path, data)
    assert (
        len(families)
        == len(
            {digest((ROOT / f["ttir_path"]).read_bytes()) for f in families.values()}
        )
        == 6
    )
    print(
        "Frozen 6 baseline.ttir files and exact 31-shape mapping from saved autotune evidence"
    )


if __name__ == "__main__":
    main()
