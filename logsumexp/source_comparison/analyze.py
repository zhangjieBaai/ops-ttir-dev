"""Recompute paired samples and write the source-vs-TTIR report and CSV."""

import csv
import json
import math
import statistics
from pathlib import Path

from compare import PERF_SHAPES, ROOT, SELECTED, save_json, sha256


def audit_run(folder):
    data = json.loads((folder / "benchmark.json").read_text())
    pipeline = json.loads((folder / "pipeline.json").read_text())
    accuracy = json.loads((folder / "accuracy.json").read_text())
    assert pipeline["passed"] and accuracy["passed"] and data["aa_passed"]
    assert len(data["controls"]) == 30 and all(c["passed"] for c in data["controls"])
    assert len(accuracy["records"]) == 51 * 8 * 3
    for field, file in [
        ("source_sha256", "optimized_source.py"),
        ("runner_sha256", "compare.py"),
        ("comparison_protocol_sha256", "protocol.md"),
    ]:
        assert data[field] == pipeline[field] == accuracy[field] == sha256(ROOT / file)
    for row in [*data["records"], *data["controls"]]:
        assert len(row["pairs"]) == data["rounds"]
        for pair in row["pairs"]:
            raw = pair["samples_us"]
            a = statistics.mean(raw[row["reference"]])
            b = statistics.mean(raw[row["candidate"]])
            assert pair["reference_us"] == a and pair["candidate_us"] == b
            assert math.isclose(pair["speedup"], a / b, rel_tol=1e-12)
        assert row["median_speedup"] == statistics.median(
            p["speedup"] for p in row["pairs"]
        )
    assert sha256(folder / "telemetry.csv") == data["telemetry_sha256"]
    for group, variants in pipeline["groups"].items():
        for name in ("source", "ttir", "baseline", "roundtrip"):
            info = variants[name]
            for stage, expected in info["stage_hashes"].items():
                assert sha256(Path(info["folder"]) / ("kernel." + stage)) == expected
        assert variants["source_file_downstream_equal"]
    for metric in ("baseline/source", "baseline/ttir", "ttir/source"):
        ref, candidate = metric.split("/")
        rows = [
            r
            for r in data["records"]
            if r["reference"] == ref and r["candidate"] == candidate
        ]
        assert {tuple(r["shape"]) for r in rows} == set(PERF_SHAPES)
        geo = math.exp(statistics.mean(math.log(r["median_speedup"]) for r in rows))
        assert math.isclose(geo, data["summaries"][metric]["geometric"], rel_tol=1e-12)
    return data, pipeline, accuracy


def main():
    first, pipeline, accuracy = audit_run(ROOT / "results/main")
    second, repeat_pipeline, repeat_accuracy = audit_run(ROOT / "results/repeat")
    assert first["physical_uuid"] == second["physical_uuid"]
    assert first["source_sha256"] == second["source_sha256"]
    for tool in ("memcheck", "racecheck"):
        data = json.loads((ROOT / "results" / tool / "accuracy.json").read_text())
        assert data["passed"] and len(data["records"]) == 306
        log = (ROOT / "results" / (tool + ".log")).read_text()
        assert (
            "ERROR SUMMARY: 0 errors" if tool == "memcheck" else "0 errors, 0 warnings"
        ) in log
    result = dict(
        passed=True,
        selected=SELECTED,
        source_sha256=first["source_sha256"],
        metric="CUDA graph replay device latency; TTIR/source>1 means source faster",
        main_summaries=first["summaries"],
        repeat_summaries=second["summaries"],
        code_identical_groups=sum(
            g["source_ttir_text_code_equal"] for g in pipeline["groups"].values()
        ),
        total_groups=len(pipeline["groups"]),
        code_identical_perf_shapes=sum(
            pipeline["groups"][p["group"]]["source_ttir_text_code_equal"]
            for p in pipeline["points"]
            if tuple(p["shape"]) in PERF_SHAPES
        ),
        total_performance_shapes=30,
        bitwise_equal_distributions=sum(
            r.get("source_ttir_output_bitwise_equal", False)
            for r in accuracy["records"]
        ),
        total_compared_distributions=51 * 8,
        accuracy_counts=[len(accuracy["records"]), len(repeat_accuracy["records"])],
        sanitizers="PASS all51shapes2distributions3versions",
        evidence_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ROOT / "results").rglob("*.json")
        },
    )
    save_json(ROOT / "results/acceptance.json", result)
    lookups = []
    for data in (first, second):
        lookups.append(
            {
                (tuple(r["shape"]), r["reference"], r["candidate"]): r
                for r in data["records"]
            }
        )
    table = []
    for shape in PERF_SHAPES:
        row = dict(M=shape[0], N=shape[1], K=shape[2])
        for label, lookup in zip(("main", "repeat"), lookups):
            paired = lookup[shape, "ttir", "source"]
            row.update(
                {
                    label + "_ttir_us": paired["reference_us"],
                    label + "_source_us": paired["candidate_us"],
                    label + "_ttir_over_source": paired["median_speedup"],
                    label
                    + "_baseline_over_source": lookup[shape, "baseline", "source"][
                        "median_speedup"
                    ],
                    label
                    + "_baseline_over_ttir": lookup[shape, "baseline", "ttir"][
                        "median_speedup"
                    ],
                }
            )
        row["same_machine_code"] = pipeline["groups"][
            lookups[0][shape, "ttir", "source"]["group"]
        ]["source_ttir_text_code_equal"]
        table.append(row)
    with (ROOT / "results/all_shapes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "# Logsumexp：Triton源码与直接TTIR优化的对照实验",
        "",
        "本实验把已验收的 `r4_focus` 优化方法回写为公开 Triton Python，和该不可变 TTIR 文件直接配对。"
        "下表中的 TTIR/源码 大于1表示源码版更快；不使用两个历史平均值相除推算差异。",
        "",
        "| 对照 | 主测几何平均（95%配对轮次bootstrap） | 独立复测 |",
        "|---|---|---|",
    ]
    for name in ("ttir/source", "baseline/source", "baseline/ttir"):
        a = first["summaries"][name]
        b = second["summaries"][name]
        lines.append(
            f'| {name} | {a["geometric"]:.5f}× {a["bootstrap_95"]} | {b["geometric"]:.5f}× {b["bootstrap_95"]} |'
        )
    lines += [
        "",
        "baseline（原始对照实现）来自用户指定 FlagGems logsumexp.py。"
        "性能矩阵、原启发式分组、完整编译选项、设备参数签名和主机grid保持不变。"
        "同一H100物理GPU、官方Triton3.7.0、FP16连续输入输出；原始数值容差不变。",
        "",
        f'29个组中有{result["code_identical_groups"]}组ELF代码段逐字节一致，覆盖{result["code_identical_perf_shapes"]}/30个性能点。'
        "完整cubin包含函数名和调试信息，不能把代码段一致说成整个cubin一致。",
        "",
        "## 逐shape直接比较",
        "",
        "| M,N,K | TTIR主测μs | 源码主测μs | TTIR/源码 主测 | TTIR/源码 复测 | 机器指令相同 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in table:
        lines.append(
            f'| ({r["M"]},{r["N"]},{r["K"]}) | {r["main_ttir_us"]:.3f} | {r["main_source_us"]:.3f} | '
            f'{r["main_ttir_over_source"]:.5f}× | {r["repeat_ttir_over_source"]:.5f}× | '
            f'{"是" if r["same_machine_code"] else "否"} |'
        )
    lines += [
        "",
        "## 如何表达这些优化",
        "",
        "按N分支缩小tile；在线统计量用q=exp(-abs(x-m))和tl.where；大行FP16最大值用自定义tl.reduce后转FP32，"
        "因为tl.max会自动提升；长行用65536宽tile与运行时回退；小输入用8行合并、32lane分组；"
        "非内层索引保持i32。全部用公开tl API，没有内联汇编、内部IR builder或二进制替换。",
        "",
        "常规源码多经过make_ttir前端优化，直接TTIR文件从make_ttgir开始。"
        "源码自己的TTIR文件重编译与原生源码编译的LLVM/PTX/cubin字节一致，"
        "TTGIR经相同parser归一化一致。"
        "原始源码/基线同样逐点核对；没有跨编译路线混比。",
        "",
        "## 验证与计时边界",
        "",
        f"两个种子分别{51*8*3}项精度检查通过，覆盖51个shape、8种分布和3个版本；"
        f'源码/TTIR输出逐位一致为{result["bitwise_equal_distributions"]}/{51*8}个输入组合。'
        "逐位不同不等于数值失败：两者均按FP64舍入参考及nativeFP32参考检查rtol=1e-3/atol=2e-3。"
        "每次重置输出sentinel（用于发现漏写的预填哨兵值），检查边界与输入未变；memcheck/racecheck各306项通过。",
        "",
        "主测7轮、独立进程不同种子复测5轮，每次rep=20ms，随机配对顺序，全部30个A/A对照通过。"
        "计时是预分配输出的CUDA graph设备时间，不含CPU提交、分配、编译、模块加载；无清缓存，未主动锁频、锁频配置未知。"
        "全部原始replay样本、执行顺序、运行中遥测、编译产物和内容哈希（用于确认文件是否改变的校验值）保留。",
        "",
        "ASTSource原生启动器保留constexpr占位参数，IRSource启动器只接受运行时参数。"
        "公共适配包装按各自正确接口传参，实际设备ABI/grid/stream一致。"
        "早期适配错误在精度/编译准备阶段被发现并修复，未进入计时；原日志与失败分类保留。",
        "",
        "结论仅覆盖此固定版本和输入矩阵；不外推其他布局/dtype/NaN/+inf、未映射shape或完整FlagGems API。"
        "两组机器指令不同也不自动证明显著性能差异，需结合直接配对、A/A波动和复测。"
        "已经观察过的shape属于回归测试集，不是独立新shape测试集。",
        "",
        "## 复现",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python \\",
        "  logsumexp/source_comparison/compare.py --output <新结果目录> --rounds 7 --seed 42",
        "```",
        "",
        "完整逐点耗时、相对原始实现的加速比及资源证据见 results/all_shapes.csv 和 results/acceptance.json。"
        "该对照不修改当前selected/optimized别名，不将源码导出的IR重新称为直接TTIR优化。",
    ]
    # Wrap prose while keeping the comparison table unchanged.
    wrapped = []
    for line in lines:
        while len(line) > 125 and not line.startswith(
            ("|", "#", "CUDA_VISIBLE_DEVICES")
        ):
            quoted = False
            cuts = []
            for i, char in enumerate(line[:126]):
                if char == "`":
                    quoted = not quoted
                if not quoted and char in "，；。 ":
                    cuts.append(i + 1)
            end = next((v for v in reversed(cuts) if v >= 45), 125)
            wrapped.append(line[:end].rstrip())
            line = line[end:].lstrip()
        wrapped.append(line)
    with (ROOT / "report.md").open("x") as stream:
        stream.write("\n".join(wrapped) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "main_summaries",
                    "repeat_summaries",
                    "code_identical_groups",
                    "code_identical_perf_shapes",
                    "bitwise_equal_distributions",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
