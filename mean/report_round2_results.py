"""Recompute round2 acceptance and Markdown tables from immutable raw receipts."""

import json
import math
import statistics

from benchmark import bootstrap, summary
from common import ALL_SHAPES, PERF_SHAPES, ROOT, ir_path, manifest, save_json, sha256

RUN = ROOT / "results/runs/round2/final"


def read(name):
    return json.loads((RUN / f"{name}.json").read_text())


def parent_rows(rows):
    return [
        dict(
            row,
            median_speedup=row["median_parent_speedup"],
            pairs=[dict(pair, speedup=pair["parent_speedup"]) for pair in row["pairs"]],
        )
        for row in rows
    ]


def main():
    data = {
        name: read(name)
        for name in (
            "pipeline",
            "accuracy",
            "accuracy_repeat",
            "main",
            "repeat",
            "memcheck",
            "racecheck",
        )
    }
    a, b = data["main"], data["repeat"]
    finalist = json.loads((ROOT / "results/round2_finalist.json").read_text())
    checks = {}
    for name in ("pipeline", "accuracy", "accuracy_repeat", "memcheck", "racecheck"):
        checks[name] = data[name]["passed"]
    for name, marker in [
        ("memcheck", "ERROR SUMMARY: 0 errors"),
        ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"),
    ]:
        checks[name] &= marker in (RUN / f"{name}.log").read_text()
    plan_hash = sha256(ROOT / "optimization_plan_round2.md")
    checks["plan"] = (
        a["round2_plan_sha256"]
        == b["round2_plan_sha256"]
        == finalist["plan_sha256"]
        == plan_hash
    )
    checks["same_gpu"] = len({item["physical_uuid"] for item in data.values()}) == 1
    checks["main_aa"] = a["aa_passed"]
    checks["repeat_aa"] = b["aa_passed"]
    checks["matrix"] = all(
        [tuple(row["shape"]) for row in bench["records"]] == PERF_SHAPES
        for bench in (a, b)
    )
    checks["protocol"] = all(
        bench["reference"] == "baseline"
        and bench["parent"] == "optimized"
        and bench["rep_ms"] == 20
        for bench in (a, b)
    )
    for bench in (a, b):
        for row in bench["records"]:
            pairs = row["pairs"]
            assert len(pairs) == bench["rounds"]
            assert row["median_speedup"] == statistics.median(
                pair["speedup"] for pair in pairs
            )
            assert row["median_parent_speedup"] == statistics.median(
                pair["parent_speedup"] for pair in pairs
            )
            for pair in pairs:
                assert set(pair["order"]) == {"baseline", "r2_final", "parent"}
                samples = pair["graph_replay_samples_us"]
                for label, time_key in [
                    ("baseline", "baseline_us"),
                    ("r2_final", "candidate_us"),
                    ("parent", "parent_us"),
                ]:
                    assert samples[label] and all(
                        math.isfinite(t) and t > 0 for t in samples[label]
                    )
                    assert math.isclose(statistics.mean(samples[label]), pair[time_key])
                assert math.isclose(
                    pair["speedup"], pair["baseline_us"] / pair["candidate_us"]
                )
                assert math.isclose(
                    pair["parent_speedup"], pair["parent_us"] / pair["candidate_us"]
                )
    checks["identities"] = True
    for group in manifest()["groups"]:
        for name in ("baseline", "optimized", "r2_final"):
            expected = data["pipeline"]["kernels"][group][name]
            assert expected["input_sha256"] == sha256(ir_path(group, name))
            for record in (data["accuracy"], data["accuracy_repeat"]):
                actual = record["kernels"][group][name]
                assert actual["input_sha256"] == expected["input_sha256"]
                assert actual["stages"]["cubin"] == expected["stages"]["cubin"]
            for bench in (a, b):
                if group in bench["artifacts"]:
                    actual = bench["artifacts"][group][name]
                    assert actual["input_sha256"] == expected["input_sha256"]
                    assert actual["stages"]["cubin"] == expected["stages"]["cubin"]
        assert sha256(ir_path(group, "r2_final")) == finalist["groups"][group]["sha256"]
    checks["round_counts"] = a["rounds"] == 7 and b["rounds"] == 5
    checks["accuracy_coverage"] = all(
        len(data[name]["records"]) == len(ALL_SHAPES) * 8 * 3
        for name in ("accuracy", "accuracy_repeat")
    )
    checks["sanitizer_coverage"] = all(
        len(data[name]["records"]) == len(ALL_SHAPES) * 2 * 2
        for name in ("memcheck", "racecheck")
    )
    assert all(checks.values()), checks
    results = {}
    for label, bench in [("main", a), ("repeat", b)]:
        rows = bench["records"]
        results[label] = dict(
            baseline=dict(**summary(rows), ci95=bootstrap(rows)),
            parent=dict(
                **summary(parent_rows(rows)), ci95=bootstrap(parent_rows(rows))
            ),
        )
        results[label]["by_path"] = {}
        for path, indices in [
            ("inner", range(6)),
            ("noninner", range(6, 12)),
            ("vector", range(12, 30)),
        ]:
            subset = [rows[i] for i in indices]
            results[label]["by_path"][path] = dict(
                baseline=summary(subset), parent=summary(parent_rows(subset))
            )
    ar, br = results["main"], results["repeat"]
    accuracy_max = max(
        row["max_abs_error_fp64_rounded"]
        for row in data["accuracy"]["records"]
        if row["name"] == "r2_final"
    )
    repeat_max = max(
        row["max_abs_error_fp64_rounded"]
        for row in data["accuracy_repeat"]["records"]
        if row["name"] == "r2_final"
    )
    lines = [
        "# mean_dim TTIR 第二轮结果",
        "",
        f"30 个 FP16 shape：相对原始 baseline 的几何平均加速为 **{ar['baseline']['geometric']:.4f}×**，"
        f"独立进程复测 **{br['baseline']['geometric']:.4f}×**。相对第一轮 optimized 分别为 "
        f"**{ar['parent']['geometric']:.4f}× / {br['parent']['geometric']:.4f}×**。",
        "",
        f"原始 baseline 对照的95%配对轮次 bootstrap 区间：{ar['baseline']['ci95']} / {br['baseline']['ci95']}。"
        f"第一轮对照的区间：{ar['parent']['ci95']} / {br['parent']['ci95']}。"
        "区间仅表示固定测试点已保存轮次的波动，不代表所有输入或设备。",
        "",
        f"主测最快 {ar['baseline']['best']:.4f}×，最慢 {ar['baseline']['worst']:.4f}×；"
        f"相对第一轮最慢 {ar['parent']['worst']:.4f}×。全部验收项：`{checks}`。",
        "",
        "## 改动与契约",
        "",
        "最终候选为 `r2_combo16_v2`，每组写一次的 `r2_final.ttir` 与它字节相同。"
        "第一轮 `optimized.ttir`、原始 baseline、源码快照、baseline.json 和第一轮报告均未改写。"
        "26 个原始 TTIR/配置组、42 个输入映射、签名、grid、warps、完整 backend options、容差均沿用原契约。",
        "",
        "K≥1024 的主路径将每次循环展开为16行加载，然后按原顺序累加。"
        "N<8 保留第一轮路径；当该 CTA 的剩余输出恰好为一列时，使用256×1的 N 方向分块归约。"
        "后者保持相同 CTA 和输出归属，不增加 kernel 或工作区，但改变尾列的浮点归约顺序。"
        "K<1024 的两条路径与原始 baseline 保持字节相同。分支均根据运行时 N 和 K/CTA位置决定，没有按测试 shape 查表。",
        "",
        "所有候选直接由 TTIR 文本构建；未生成候选 Python kernel、修改 TTGIR/LLVM/PTX/cubin，"
        "也未替换 binary 或 launch metadata。编译与启动继续使用 common.compile_ir/common.launch。"
        "这些循环展开与归约策略也可用 Triton 语法表达，不声称为 TTIR 独有。",
        "",
        "## 搜索与负结果",
        "",
        "第一阶段比较顺序展开4/8、独立累加器4/8，以及 N×K 为16×256、64×64的本地子块。"
        "只使用既定偶数编号15点，每点3轮；全部 A/A 控制通过。",
        "",
        "| 候选 | 相对第一轮初筛几何平均 | 最差点 |",
        "|---|---:|---:|",
    ]
    for screen_name in ("screen", "screen_combined"):
        screen = json.loads(
            (ROOT / f"results/runs/round2/{screen_name}.json").read_text()
        )
        assert screen["aa_passed"]
        for name, value in screen["summaries"].items():
            lines.append(
                f"| {name} | {value['geometric']:.4f}× | {value['worst']:.4f}× |"
            )
    lines += [
        "",
        "这些是 B 级初筛结果。窄 K 子块方案的额外循环和归约伴随明显回退，停止该方向；"
        "16次展开组合优于8次组合，按整体初筛几何平均选定，再冻结最终候选。"
        "没有从各点挑选最优候选。主路径展开、短 N 分支、尾列归约作为组合验收，未完成所有单项消融。",
        "",
        "初筛有两次静态构造错误：无返回值 scf.for 被命名、嵌套分支重复定义 %take。"
        "首次候选文件及诊断保留，修复使用新 _v2 名称；错误版本从未执行。"
        "没有将失败编译当作性能退步。记录见 results/round2_compile_failure.json 和 "
        "results/round2_combined_parse_failure.json。",
        "",
        "30个性能点均已在第一轮暴露；本轮奇数编号15点不参与候选选择，但不能称作全新独立holdout。"
        "baseline.json.partition 是第一轮遗留且未使用的标签，实际划分以计时结果 actual_indices 为准。"
        "所有42点均参与数值验证。首阶段反思和预算处置见 results/runs/round2/reflection.json。",
        "",
        "## 完整30点结果",
        "",
        "| M,N,K | 原始 µs | 第一轮 µs | 第二轮 µs | 对原始主测 | 对第一轮主测 | 对原始复测 | 对第一轮复测 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    repeats = {tuple(row["shape"]): row for row in b["records"]}
    for row in a["records"]:
        repeat = repeats[tuple(row["shape"])]
        shape = ",".join(map(str, row["shape"]))
        lines.append(
            f"| {shape} | {row['baseline_us']:.3f} | {row['parent_us']:.3f} | {row['candidate_us']:.3f} | "
            f"{row['median_speedup']:.4f}× | {row['median_parent_speedup']:.4f}× | "
            f"{repeat['median_speedup']:.4f}× | {repeat['median_parent_speedup']:.4f}× |"
        )
    lines += [
        "",
        "## 分路径与退步",
        "",
        "| 路径 | 点数 | 对原始主测/复测几何 | 对第一轮主测/复测几何 |",
        "|---|---:|---:|---:|",
    ]
    for path, count in [("inner", 6), ("noninner", 6), ("vector", 18)]:
        sa, sb = ar["by_path"][path], br["by_path"][path]
        lines.append(
            f"| {path} | {count} | {sa['baseline']['geometric']:.4f}× / {sb['baseline']['geometric']:.4f}× | "
            f"{sa['parent']['geometric']:.4f}× / {sb['parent']['geometric']:.4f}× |"
        )
    for shape in [(1, 63, 1025), (1, 1024, 1024), (4, 1024, 8192)]:
        row = next(row for row in a["records"] if tuple(row["shape"]) == shape)
        repeat = repeats[shape]
        lines += [
            "",
            f"`{shape}`：本轮主测/复测对原始为 {row['median_speedup']:.4f}× / "
            f"{repeat['median_speedup']:.4f}×，对第一轮为 {row['median_parent_speedup']:.4f}× / "
            f"{repeat['median_parent_speedup']:.4f}×。",
        ]
    regressions = [row for row in a["records"] if row["median_parent_speedup"] < 0.97]
    lines += [
        "",
        "相对第一轮主测退步超过3%的点："
        + str([(row["shape"], row["median_parent_speedup"]) for row in regressions])
        + "。"
        "接近1×的点有测量波动；前两路径未发生算法优化。",
        "",
        "## 正确性、环境和计时",
        "",
        f"最终42点 pipeline 全部通过；seed42和seed99分别 {len(data['accuracy']['records'])} 项检查通过，"
        f"对新候选的 CPU FP64 均值转FP16最大绝对误差分别为 {accuracy_max:.8g} / {repeat_max:.8g}。"
        "参考、分布、atol=rtol=0.002、NaN/Inf、输出sentinel、canary和输入不变检查沿用原协议。"
        "memcheck与racecheck各168项，分别报告0 errors和0 hazards/0 errors/0 warnings。",
        "",
        f"GPU UUID `{a['physical_uuid']}`；本轮主测、复测与验证均在同一物理卡。"
        "环境为官方Triton3.7.0的指定venv，工具链、RECORD与实际模块路径见pipeline.json。"
        "源码/file每点TTIR一致，规范化TTGIR一致，LLIR/PTX/cubin字节相同，baseline输出逐位相同。",
        "",
        "主测7轮、独立进程不同数据种子复测5轮，每次rep20ms CUDA graph replay，预分配输出。"
        "每轮将原始baseline、第一轮optimized、第二轮候选随机排序，直接保存两种配对比值；"
        "A/A另行交错，全部原始graph样本、顺序、资源及遥测保留。计时前验证TTIR/cubin与准确性证据一致。"
        "计时不含分配、编译、加载和Python提交。未锁频、未清缓存，不称作冷缓存。",
        "",
        "## 资源、解释与限制",
        "",
        "| TTIR组 | 原始 regs | 第一轮 regs | 第二轮 regs | 第二轮 spill/shared bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, kernels in data["pipeline"]["kernels"].items():
        if "@mean_dim_kernel_non_inner_vec" in ir_path(group, "baseline").read_text():
            old, parent, new = (
                kernels[name] for name in ("baseline", "optimized", "r2_final")
            )
            lines.append(
                f"| {group} | {old['n_regs']} | {parent['n_regs']} | {new['n_regs']} | "
                f"{new['n_spills']} / {new['metadata']['shared']} |"
            )
    lines += [
        "",
        "C级静态证据：展开候选改变了循环体加载与算术结构；对齐组资源达到48 regs，未对齐组117 regs，均无spill。"
        "资源增大没有阻止该组合的实测收益；没有profiler采集或完整消融，不能把全部收益归因为某类stall或寄存器数。"
        "尾列并行归约的单项贡献尚未隔离。全部优化仍受原始grid并行度约束。",
        "",
        "32位索引仅承诺冻结42点；verify_round2.py对含masked lane的保守最大偏移检查小于2^31。"
        "未验证其他dtype、非连续/负stride、空归约、任意新shape、改变grid或多kernel拆分归约。"
        "本轮收益相对已固定原始启发式配置，不声称胜过任意重新autotune实现。",
        "",
        "复现：先确认设备空闲，再从仓库根目录运行 "
        "`CUDA_VISIBLE_DEVICES=5 bash mean/test_round2.sh`；"
        "`bash mean/test_round2.sh accuracy`只运行pipeline和准确性。每次创建新结果目录。"
        "最终文件为各组r2_final.ttir，第一轮入口仍使用optimized.ttir。",
        "",
        "证据：results/runs/round2/final/{pipeline,accuracy,accuracy_repeat,main,repeat,memcheck,racecheck}.json；"
        "候选选择results/round2_finalist.json；直接TTIR重建及索引范围results/runs/round2/generator_reproduction_bounds.json。"
        "结论等级A只覆盖本轮验收矩阵；B级初筛和D级机制假设不外推为最终收益。",
    ]
    (ROOT / "report_round2.md").write_text("\n".join(lines) + "\n")
    save_json(
        ROOT / "results/round2_acceptance.json",
        dict(
            accepted=all(checks.values()),
            checks=checks,
            results=results,
            plan_sha256=plan_hash,
            report_sha256=sha256(ROOT / "report_round2.md"),
            evidence={path.name: sha256(path) for path in RUN.glob("*.json")},
        ),
    )
    print(json.dumps(dict(checks=checks, results=results), indent=2))


if __name__ == "__main__":
    main()
