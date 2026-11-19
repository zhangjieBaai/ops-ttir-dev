"""Recompute the final report from immutable paired samples and sanitizer receipts."""

import argparse
import json
import math
import statistics
from pathlib import Path

from common import CONTRACT, ROOT, digest, save

RUN = ROOT / "results/runs/r2"
FINAL = RUN / "final_gpu5"


def read(path):
    return json.loads(Path(path).read_text())


def verify_timing(data):
    assert data["valid"]
    assert len(data["records"]) == 30
    assert [r["shape"] for r in data["records"]] == CONTRACT["performance"]["shapes"]
    ratios = []
    for row in data["records"]:
        paired = [p["baseline_us"] / p["candidate_us"] for p in row["pairs"]]
        assert paired == row["paired_ratios"]
        assert statistics.median(paired) == row["speedup"]
        ratios.append(row["speedup"])
    geo = math.exp(statistics.mean(map(math.log, ratios)))
    assert abs(geo - data["summary"]["optimized"]["geomean"]) < 1e-12
    return data["summary"]["optimized"]


def record_existing(path, value):
    if path.exists():
        assert read(path) == value, f"Conflicting immutable evidence: {path}"
    else:
        save(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "report_round2.md")
    args = parser.parse_args()
    candidate = read(RUN / "provisional_selection.json")["candidate"]
    expected = digest(ROOT / f"ir/{candidate}.ttir")
    assert expected == digest(ROOT / "ir/optimized.ttir")
    main_data, repeat = read(FINAL / "full.json"), read(FINAL / "repeat.json")
    main_summary, repeat_summary = verify_timing(main_data), verify_timing(repeat)
    assert main_data["gpu_uuid"] == repeat["gpu_uuid"]
    for data in [main_data, repeat]:
        assert data["resources"]["optimized"]["ttir_sha256"] == expected
    for filename in [
        "accuracy.json",
        "repeat_accuracy.json",
        "memcheck_accuracy.json",
        "racecheck_accuracy.json",
    ]:
        accuracy = read(FINAL / filename)
        assert (
            accuracy["passed"]
            and accuracy["identities"]["optimized"]["ttir_sha256"] == expected
        )
    for tool in ["memcheck", "racecheck"]:
        log = (FINAL / f"{tool}.log").read_text()
        if tool == "memcheck":
            assert "ERROR SUMMARY: 0 errors" in log
        else:
            assert (
                "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in log
            )
    controls, repeat_controls = read(FINAL / "controls.json"), read(
        FINAL / "repeat_controls.json"
    )
    assert controls["valid"] and repeat_controls["valid"]
    main_aa = {tuple(r["shape"]): r for r in controls["records"] if r["name"] == "aa"}
    repeat_aa = {
        tuple(r["shape"]): r for r in repeat_controls["records"] if r["name"] == "aa"
    }
    spread = {
        str(shape): repeat_aa[shape]["baseline_us"] / row["baseline_us"]
        for shape, row in main_aa.items()
    }
    pipeline = read(FINAL / "pipeline.json")
    assert pipeline["passed"] and len(pipeline["records"]) == 30
    screen = read(RUN / "screen_gpu5.json")
    assert screen["valid"]
    confirmation = read(RUN / "confirmation_gpu5.json")
    assert confirmation["valid"]
    resources = main_data["resources"]
    accuracy = read(FINAL / "accuracy.json")
    errors = {
        name: dict(
            max_abs_var=max(
                r["max_abs_var"] for r in accuracy["records"] if r["name"] == name
            ),
            max_abs_mean=max(
                r["max_abs_mean"] for r in accuracy["records"] if r["name"] == name
            ),
        )
        for name in ["baseline", "optimized"]
    }
    low = min(main_data["records"], key=lambda r: r["speedup"])
    high = max(main_data["records"], key=lambda r: r["speedup"])
    selected = dict(
        candidate=candidate,
        ttir_sha256=expected,
        baseline_sha256=digest(ROOT / "ir/baseline.ttir"),
        contract_sha256=digest(ROOT / "contract.json"),
        main=main_summary,
        repeat=repeat_summary,
        evidence_grade="A: complete 30-point paired matrix plus independent process repeat",
        metric=CONTRACT["performance"]["metric"],
        gpu_uuid=main_data["gpu_uuid"],
        full_sha256=digest(FINAL / "full.json"),
        repeat_sha256=digest(FINAL / "repeat.json"),
        correctness_errors=errors,
        crossprocess_control_baseline_ratios=spread,
        limitations=CONTRACT["excluded"],
        qualification=(
            "Scoped TTIR experiment; no KernelForge operator execution adapter "
            "or ATen integration qualification"
        ),
    )
    record_existing(RUN / "selection.json", selected)
    record_existing(
        RUN / "outcome.json",
        dict(
            passed=True,
            selection_sha256=digest(RUN / "selection.json"),
            sanitizer={
                t: digest(FINAL / f"{t}.log") for t in ["memcheck", "racecheck"]
            },
            checks={
                "source_pipeline": True,
                "accuracy_two_seeds": True,
                "paired30_main": True,
                "paired30_independent_repeat": True,
                "memcheck_subset": True,
                "racecheck_subset": True,
            },
            unmeasured=[
                "API latency",
                "host issue latency",
                "cold-cache performance",
                "profiler achieved occupancy",
            ],
            reflection=(
                "Eight corrected TTIR candidates, one screen batch; reject wide single tiles "
                "for severe regressions. Freeze bounded hybrid after confirmation. "
                "No further research needed to close scoped experiment."
            ),
        ),
    )
    ci = main_summary["ci95_conditional_bootstrap"]
    rci = repeat_summary["ci95_conditional_bootstrap"]
    text = f"""# var_mean TTIR 优化实验（修正后的第二轮）

30 个 shape 的主测几何平均加速
**{main_summary['geomean']:.4f}×**，配对轮次条件 bootstrap 95% 区间 [{ci[0]:.4f}, {ci[1]:.4f}]；
独立进程复测 **{repeat_summary['geomean']:.4f}×**，区间 [{rci[0]:.4f}, {rci[1]:.4f}]。
这是预分配输出的 CUDA graph kernel 时间，不含 Python、分配、编译或加载。

本报告仅使用恢复 int64 行索引后的 r2 数据，契约为
`{CONTRACT['version']}`，设备为物理 GPU5（H100，UUID `{main_data['gpu_uuid']}`）。
30 个 shape 曾在已废弃的 int32 提取版本中暴露，属于确认和回归矩阵，不能称为全新 holdout。
旧轮数据完整保存在 [setup_attempts/i32_pid](setup_attempts/i32_pid/README.md)，不计入下列加速比。

## 冻结范围与基线

输入是连续 FP32 二维张量，对最后一维按行计算 mean 和中心化方差；correction 为 0 或 1。性能点为 M∈{{1,17,128,1024,4096}} 与
N∈{{128,512,1040,4096,8192,16384}} 的笛卡尔积，共30点，等权；性能用 correction=1。支持契约要求 N 是16的倍数且16≤N≤16384、1≤M≤4096，无别名。
全张量两阶段路径、FP16/BF16、非连续输入及 API 端到端性能不在本轮范围。

原始实现快照来自 `/root/FlagGems/src/flag_gems/ops/var_mean.py`。只去除主机缓存装饰器，将 `ext.program_id(0)` 展开为原 helper 的
`tl.program_id(0).to(tl.int64)`；[AST检查](results/runs/r2/extraction.json)证明其余 kernel 计算体一致。源码 SHA256 为
`{digest(ROOT/'upstream/var_mean.py.snapshot')}`。

原实现此路径没有 autotune。baseline 固定 BLOCK_N=1024，4 warps、3 stages、1 CTA，开启 FP fusion；两版使用完全相同的后端 options、签名、每行一个 CTA
的 grid、CompiledKernel launcher 和原生 Triton 文件入口。环境为官方 Triton3.7.0、独立 Python3.12.3 venv，PyTorch版本见
[pipeline](results/runs/r2/final_gpu5/pipeline.json)。统一设置 TRITON_DISABLE_LINE_INFO=1，避免文件解析 unknown loc 的调试行号差异；
最终30点源码/文件 baseline 的 LLVM、PTX、完整 cubin 逐字节一致，归一化 TTGIR 一致，两项输出逐位一致。

## 选中的 TTIR 改写

候选 `{candidate}` 已按相同 SHA256 复制为 [optimized.ttir](ir/optimized.ttir)，哈希 `{expected}`。baseline 哈希
`{digest(ROOT/'ir/baseline.ttir')}`。

- N≤128/512/1024 时，分别采用128/512/1024宽的单 tile：一次加载，先归约 mean，再计算中心化平方和。
- N>1024 时保留两遍中心化算法，将动态循环 tile 从1024改为256。
- 在 TTIR 为 N 增加已由契约和 launcher 检查的16整除属性，使 int64 行地址计算可以利用该前提。

以上改动只发生在 TTIR；没有改 kernel Python 计算后重导出，没有编辑 TTGIR/LLVM/PTX/cubin，也没有改变两版的启动配置。baseline 与最终版均为
{resources['baseline']['n_regs']}/{resources['optimized']['n_regs']} registers per thread、
{resources['baseline']['n_spills']}/{resources['optimized']['n_spills']} spill、
{resources['baseline']['shared']}/{resources['optimized']['shared']} bytes shared。

PTX 中可见最终版的向量化 load，而基线有多条标量 load。这是静态产物证据（C级），不能把全部性能收益归因于单项指令变化；没有 profiler 或完整最终矩阵单项消融。短行展开、循环宽度、整数前提共同构成最终组合。

## 候选与负结果

下表为同进程12点、3轮、rep=10ms初筛（B级），不是最终完整矩阵加速比。所有候选先通过 CPU FP64 数值检查。

| 候选 | 几何平均 | 最差点 | registers/thread | 处理 |
|---|---:|---:|---:|---|
"""
    candidate_identities = read(RUN / "accuracy_candidates.json")["identities"]
    for name, summary in screen["summary"].items():
        status = (
            "进入18点确认"
            if name == candidate
            else ("严重退步，弃用" if summary["worst"] < 0.9 else "保留对照")
        )
        text += (
            f"| {name} | {summary['geomean']:.4f}× | {summary['worst']:.4f}× | "
            f"{candidate_identities[name]['n_regs']} | {status} |\n"
        )
    csummary = confirmation["summary"][candidate]
    text += f"""
其余18点确认几何平均
{csummary['geomean']:.4f}×、最差 {csummary['worst']:.4f}×；确认后没有再修改候选。大 tile 单次加载版本的最坏退步说明，减少加载或提高局部吞吐并不足以保证所有行数上的
收益；高寄存器资源是观测到的静态线索，未做因果隔离。

## 最终配对计时

主测30点×7轮，seed42；独立进程复测30点×5轮，seed123；两者 rep=20ms，每轮随机 baseline/候选先后顺序，预热后 graph replay，同一物理 GPU5。
每点取配对加速比中位数再等权汇总，保存全部轮次、顺序、IR/cubin哈希及负载时钟/功耗/进程采样。两轮没有监测到其他 GPU5 进程。协作锁只协调使用同一锁的任务，采样不能排除所有瞬时外部干扰；没有主动锁频或主动清缓存，
不能称为冷缓存或端到端收益。区间仅是保存配对轮次下的条件不确定性，不覆盖选择偏差和所有系统误差。

主测算术平均
{main_summary['arithmetic']:.4f}×；最小中位数 {low['speedup']:.4f}×，shape={tuple(low['shape'])}；
最大 {high['speedup']:.4f}×，shape={tuple(high['shape'])}。独立复测最差 {repeat_summary['worst']:.4f}×。

| M | N | baseline µs | optimized µs | 主测配对加速 | 独立复测 |
|---:|---:|---:|---:|---:|---:|
"""
    for row, other in zip(main_data["records"], repeat["records"]):
        m, n = row["shape"]
        text += (
            f"| {m} | {n} | {row['baseline_us']:.4f} | {row['candidate_us']:.4f} | "
            f"{row['speedup']:.4f}× | {other['speedup']:.4f}× |\n"
        )
    text += f"""
A/A控制（3个代表shape）几何比
{controls['summary']['aa']['geomean']:.6f}，重复进程 {repeat_controls['summary']['aa']['geomean']:.6f}；
已知双次启动的 baseline/双次序列几何比
{controls['summary']['double']['geomean']:.6f}，复测 {repeat_controls['summary']['double']['geomean']:.6f}，接近预期0.5。
两次控制进程对应 baseline 耗时比（repeat/main）为 `{spread}`，仅描述会话内这3点的跨进程漂移，不是全域噪声界。

## 数值、内存与同步检查

最终 baseline/optimized 每个 seed 各564项数值比较，两进程各1128条记录；包含30性能shape、18边界N、M=1/17、correction0/1，
以及normal、constant、offset10000、lowvar、zero、alternating、first-element outlier。每次 launch 前重置输出NaN，检查两端各16元素 canary。
CPU FP64 `torch.var_mean` 是主参考，variance rtol=2e-4/atol=2e-6，mean rtol=2e-5/atol=2e-5。以下是测试集绝对误差，判断使用上述相对+绝对容差，
不能把它们解释为全部输入误差上界。

| 输出误差对CPU FP64 | baseline | optimized |
|---|---:|---:|
| variance最大绝对误差 | {errors['baseline']['max_abs_var']:.9g} | {errors['optimized']['max_abs_var']:.9g} |
| mean最大绝对误差 | {errors['baseline']['max_abs_mean']:.9g} | {errors['optimized']['max_abs_mean']:.9g} |

native GPU FP32 ATen误差也逐项保存，但大偏移输入本身可能偏离FP64；不通过放松FP64门槛适配它。最终 memcheck/racecheck 各覆盖12个边界/压力用例×2
correction×2版本，共48项检查；均为0 errors，racecheck为0 hazards。该子集不能证明所有支持输入都经过sanitizer。

## 复现与证据

从 `/root/ops-ttir-dev` 执行：

```bash
CUDA_VISIBLE_DEVICES=5 bash var_mean/test_accuracy.sh
CUDA_VISIBLE_DEVICES=5 bash var_mean/test_performance.sh
```

脚本生成新结果目录，拒绝覆盖已有JSON。候选可由 make_candidates.py → make_dispatch.py → make_hybrid.py 重建；重建时核对已有记录而不覆盖旧证据。
所有结果和重建脚本已归档到Git；ignored artifacts可由同一原生编译入口再生，哈希在原始JSON中。

- [源文件/编译链证据](results/runs/r2/final_gpu5/pipeline.json)
- [主测全部配对样本](results/runs/r2/final_gpu5/full.json)、[独立复测](results/runs/r2/final_gpu5/repeat.json)
- [正确性](results/runs/r2/final_gpu5/accuracy.json)、[第二seed](results/runs/r2/final_gpu5/repeat_accuracy.json)
- [memcheck](results/runs/r2/final_gpu5/memcheck.log)、[racecheck](results/runs/r2/final_gpu5/racecheck.log)
- [候选选择与最终哈希](results/runs/r2/selection.json)、[验收记录](results/runs/r2/outcome.json)

KernelForge当前没有var_mean执行适配器：本实验使用其流程、证据和只读独立审查机制，不声明KernelForge正式算子集成资格，本实验未修改FlagGems算子源码。
已废弃的提取、正则生成失败和GPU7干扰记录均保留；不得拿这些数据替代本报告的修正基线结果。
"""
    path = args.output
    if path.exists():
        assert path.read_text() == text, f"Use a new report output: {path}"
    else:
        path.write_text(text)
    print(main_summary, repeat_summary)


if __name__ == "__main__":
    main()
