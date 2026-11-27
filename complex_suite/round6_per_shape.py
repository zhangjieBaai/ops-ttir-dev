"""Export baseline/final speedups using only qualified independent repeats."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "complex_suite"
OPS = {
    "int4_gemm": ("INT4反量化＋GEMM", "results.json", "M, N, K"),
    "moe_routing": ("Grouped／biased MoE", "performance.json", "tokens, experts"),
    "norm_quant": ("Norm＋动态INT8", "results.json", "M, N"),
    "attention_dkdv": ("Attention dK／dV", "benchmark.json", "B, H, N, D, causal"),
    "partial_topk": ("大N partial Top-K", "results.json", "M, N, K"),
}
VERSIONS = {
    "int4_gemm": "保留r3 `group128`，r6未晋升。",
    "moe_routing": "r6 `batch4`；G=8、top_group=4、topK=8固定。",
    "norm_quant": "保留r2 `reciprocal_early`，r6未晋升。",
    "attention_dkdv": "D128采用r6 `dv_late`；D64保留原版本。causal=False为非因果，True为因果。",
    "partial_topk": "r6 `merge_heads_packed_v2`，计时包含local＋merge两个kernel。",
}


def load(path, sources):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def main():
    sources = {}
    aggregate = load(OUT / "round6_summary.json", sources)
    all_records = []
    run_metadata = {}
    sections = []
    for op, (label, filename, dimensions) in OPS.items():
        folder = ROOT / op / "multiround/r6"
        selection = load(folder / "selection.json", sources)
        runs = {
            phase: load(folder / "results" / phase / filename, sources)
            for phase in ("repeat",)
        }
        run_metadata[op] = {
            phase: {key: run[key] for key in ("seed", "rounds", "rep_ms")}
            for phase, run in runs.items()
        }
        by_shape = {}
        for phase, run in runs.items():
            assert run["aa_passed"] and run.get("passed", True)
            assert run["rounds"] == 5
            assert run["seed"] == 123
            assert run["rep_ms"] == 20
            records = run.get("records", run.get("points"))
            assert len(records) == (32 if op == "attention_dkdv" else 30)
            seen = set()
            for rec in records:
                shape = tuple(rec["shape"])
                assert shape not in seen
                seen.add(shape)
                route = (
                    selection["groups"][rec["group"]]["selected"]
                    if op == "attention_dkdv"
                    else selection["selected"]
                )
                timing = rec.get("timing", rec)
                samples = timing["samples"]
                assert len(samples) == run["rounds"]
                times = [sample.get("times_us", sample.get("us")) for sample in samples]
                assert all(
                    all(math.isfinite(v) and v > 0 for v in t.values()) for t in times
                )
                ratios = {
                    f"{base}_over_final": statistics.median(
                        t[base] / t[route] for t in times
                    )
                    for base in ("baseline",)
                }
                close(
                    statistics.median(t[route] for t in times),
                    timing["median_us"][route],
                )
                close(
                    statistics.median(t["baseline"] for t in times),
                    timing["median_us"]["baseline"],
                )
                aa = timing["aa"]
                aa_median = statistics.median(
                    (a.get("times_us", a.get("us")))[0]
                    / (a.get("times_us", a.get("us")))[1]
                    for a in aa
                )
                assert 0.97 <= aa_median <= 1.03
                item = by_shape.setdefault(
                    shape,
                    {"operator": op, "shape": list(shape), "route": route},
                )
                assert item["route"] == route
                item["speedup"] = ratios["baseline_over_final"]
                item["aa_median"] = aa_median
        ordered = [by_shape[s] for s in sorted(by_shape)]
        geo = math.exp(statistics.mean(math.log(r["speedup"]) for r in ordered))
        close(geo, aggregate["operators"][op]["final"]["repeat"]["baseline_over_final"])
        all_records.extend(ordered)
        sections += [
            f"## {label}（{len(ordered)}点）",
            "",
            f"Shape顺序：`({dimensions})`。{VERSIONS[op]}",
            "",
            "| Shape | 相对原始基线加速比 |",
            "|---|---:|",
        ]
        for rec in ordered:
            shape_text = "(" + ", ".join(map(str, rec["shape"])) + ")"
            sections.append(f"| {shape_text} | {rec['speedup']:.4f}× |")
        sections += ["", f"[该算子完整报告](../{op}/multiround/r6/report.md)", ""]
    assert len(all_records) == 152
    payload = {
        "round": 6,
        "unique_shape_count": 152,
        "measurement": "independent_repeat",
        "metric": "Median of paired original-baseline/final time ratios per shape",
        "sources": sources,
        "run_metadata": run_metadata,
        "records": all_records,
    }
    (OUT / "round6_per_shape.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    headers = ["operator", "shape", "speedup"]
    with (OUT / "round6_per_shape.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for rec in all_records:
            row = {
                "operator": rec["operator"],
                "shape": json.dumps(rec["shape"]),
                "speedup": rec["speedup"],
            }
            writer.writerow(row)
    intro = [
        "# 第六轮：全部152个shape的加速比",
        "",
        "统一使用2026-11-24完成的第六轮独立复测结果，按shape升序排列。",
        "",
        "**加速比 = 原始基线耗时 ÷ 最终优化版耗时。** 每个shape对5轮复测的配对比值取中位数，"
        "rep=20ms。大于1表示提速，小于1表示回退。",
        "",
        "Attention固定SM1200MHz，其余四类未主动锁频；结果仅对应各自冻结配置、数值合同和硬件。"
        "表内INT4、Norm及Attention D64保留旧版本，基线累计收益仍由本轮直接计时得到。",
        "",
        "[CSV表格](round6_per_shape.csv) · [完整精度JSON与来源哈希](round6_per_shape.json) · "
        "[第六轮方法与验证报告](round6_report.md)",
        "",
    ]
    (OUT / "round6_per_shape.md").write_text("\n".join(intro + sections))
    print(
        "PASS:152 unique shapes; raw ratios reproduce all five final aggregate summaries."
    )


if __name__ == "__main__":
    main()
