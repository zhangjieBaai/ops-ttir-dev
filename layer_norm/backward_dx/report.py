"""Generate a dX-only report from finalized evidence; no new measurements."""

import json
import math
import statistics

from common import BLOCKS, ROOT, save


def main():
    r = ROOT / "results"

    def load(name):
        return json.load(open(r / name))

    selection = load("selection.json")
    main_run = load("full.json")
    repeat = load("repeat.json")
    assert selection["passed"]
    assert (
        load("pipeline_selected_final.json")["passed"]
        and load("selected_accuracy.json")["passed"]
    )
    assert load("selected_benchmark.json")["aa_passed"]
    selected_bench = load("selected_benchmark.json")
    observed = selected_bench["summary"]["optimized"]
    selected_accuracy = load("selected_accuracy.json")
    max_fp64 = max(v["fp64_max_abs"] for v in selected_accuracy["records"])
    max_native = max(v["native_max_abs"] for v in selected_accuracy["records"])
    max_rms = max(v["normalized_rms"] for v in selected_accuracy["records"])
    resources = {}
    for block in BLOCKS:
        for name in ["baseline", "optimized", "joint_coefficients", "grouped2048"]:
            p = ROOT / f"artifacts/b{block}/{name}/metadata.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            resources[f"b{block}/{name}"] = {
                k: d[k]
                for k in [
                    "n_regs",
                    "n_spills",
                    "shared",
                    "num_warps",
                    "num_ctas",
                    "num_stages",
                ]
            }
    save(r / "resources_final.json", resources)
    # Frozen family policy applied to historical main/repeat, without relabeling raw records.
    composed = {}
    for label, data in [("main", main_run), ("repeat", repeat)]:
        ratios = []
        for n in range(1024, 15873, 512):
            b = max(1024, 1 << (n - 1).bit_length())
            name = selection["families"][str(b)]["candidate"]
            ratio = (
                1.0
                if name == "baseline"
                else next(
                    v["speedup"]
                    for v in data["records"]
                    if v["N"] == n and v["kernel"] == name
                )
            )
            ratios.append(ratio)
        composed[label] = dict(
            geometric_mean=math.exp(statistics.mean(math.log(v) for v in ratios)),
            arithmetic_mean=statistics.mean(ratios),
            worst=min(ratios),
            best=max(ratios),
        )
    save(
        r / "selected_summary.json",
        dict(
            composed=composed,
            direct_selected_benchmark=observed,
            limitation=(
                "Composed policy uses ratio=1 for retained baseline families; "
                "direct alias benchmark independently measured"
            ),
        ),
    )
    lines = [
        "# LayerNorm backward dX TTIR 首轮报告",
        "",
        "baseline是官方Triton3.7.0教程`_layer_norm_bwd_dx_fused`中dX计算段的提取版本，**不是完整融合dX+dW+dB kernel**。"
        "两版均不执行dW/dB或锁；删去这些工作不计为优化收益。保存源码提取来源，5组检查确认dX与原完整融合kernel逐位一致。",
        "",
        "## 结果",
        "",
        f"同一组baseline对比、原生Triton3.7.0编译链、30点CUDA graph：最终分组选择直接测量几何平均 **{observed['geometric_mean']:.4f}×**，"
        f"算术平均 **{observed['arithmetic_mean']:.4f}×**，逐点范围{observed['worst']:.4f}×–{observed['best']:.4f}×。",
        "",
        f"按预先冻结组级策略从主测/独立复测组合的几何均值分别为 **{composed['main']['geometric_mean']:.4f}× / "
        f"{composed['repeat']['geometric_mean']:.4f}×**；保留baseline的组按1计，不伪装成独立新样本。"
        "最终别名另跑完整矩阵，原始记录见selected_benchmark.json。",
        "",
        "本轮改善较小，未复现前向的1.44×；中等行宽没有达到2%的组级门槛，明确保留baseline。"
        "大tile重载显著降低寄存器，但其额外加载并非所有尺寸都值得。",
        "",
        "## 分组、契约和选择",
        "",
        "FP16 X/DY/W/DX、FP32保存Mean/Rstd、内部FP32计算；N/stride正16倍数，N≤16384、列连续、"
        "X/DY/DX同行stride、无别名。M=4096，N=1024..15872 step512。BLOCK=next_power_of_two(N)，最小1024；"
        "warps=min(max(BLOCK/256,1),8)，1CTA、3stages、fp_fusion=True。每组保持原配置/ABI/grid，不进行额外调参。",
        "",
        "| BLOCK | warps | 最终候选 | 组主测几何 | 组独立复测几何 | baseline/new registers |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for b in BLOCKS:
        sel = selection["families"][str(b)]
        name = sel["candidate"]
        ms = 1 if name == "baseline" else sel["main"][name]["geometric_mean"]
        rs = 1 if name == "baseline" else sel["repeat"]["geometric_mean"]
        base = resources[f"b{b}/baseline"]
        opt = resources[f"b{b}/optimized"]
        lines.append(
            f"| {b} | {base['num_warps']} | {name} | {ms:.4f}× | {rs:.4f}× | {base['n_regs']} / {opt['n_regs']} |"
        )
    lines += [
        "",
        "## 采用的TTIR改动",
        "",
        "- 把`sum(xhat*wdy)`和`sum(wdy)`表达为二输入/二输出`tt.reduce`，允许后端共同lowering两个独立和。",
        "- 将除以N改为共享1/N再乘法，预计算行级alpha=c1*rstd、beta=c2*rstd，"
        "输出改写为wdy*rstd-beta-xhat*alpha，允许后端FMA融合。浮点舍入改变，按冻结容差检验。",
        "- 本轮不使用inline PTX，不修改TTGIR/LLVM/PTX/cubin，不手改barrier，不宣称联合规约一定只需某个固定barrier数量。",
        "",
        "## 逐shape直接计时",
        "",
        "| N | BLOCK | baseline µs | selected µs | 配对加速 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in selected_bench["records"]:
        lines.append(
            f"| {row['N']} | {row['block']} | {row['baseline_us']:.3f} | "
            f"{row['candidate_us']:.3f} | {row['speedup']:.4f}× |"
        )
    lines += [
        "",
        "## 验证和测量",
        "",
        "- 5组源码与TTIR文件路径的LLVM/PTX/cubin字节一致；TTGIR统一parser后相同，输出逐位相同。"
        "最终各组optimized与选定候选hash/binary一致，完整后端options与baseline相同。",
        "- 全部候选先C1后计时。完整135例含30个性能点、组边界±16、小M、padding16、normal/offset/constant/lowvar/zero_dy，"
        "参考native ATen autograd和基于实际FP32保存统计量的FP64公式；小用例额外CPU FP64 autograd。"
        "rtol=0.01、atol=0.01、normalized RMS≤0.003，检查有限性、输出sentinel/padding及输入不修改。"
        "高rstd梯度可能有较大绝对误差，不能把相对容差验证写成全部绝对误差<1e-2。",
        "- 最终候选和baseline经30个边界/padding例的memcheck和racecheck，工具均0 errors、racecheck无warnings。",
        "- 主测30点7轮rep20ms，独立进程seed123/5轮rep10ms；保存所有配对样本、随机顺序和逐点A/A控制，"
        "控制中位数要求0.97..1.03。相同graph稳定阶段、预分配输入输出；计时不含前向或计算保存统计量、dW/dB、"
        "分配/编译/加载/CPU launch。未锁时钟、未清缓存，不称端到端或冷缓存收益。",
        "",
        f"最终别名复验中最大绝对误差：FP64保存统计量参考{max_fp64:.6g}，native参考{max_native:.6g}；"
        f"最大normalized RMS为{max_rms:.6g}。RMS定义为误差RMS除以max(参考RMS,1)。"
        "较大绝对误差发生于低方差/常量等高rstd梯度场景，使用冻结的逐元素绝对+相对容差验证。",
        "",
        "## 负结果与NCU机制线索",
        "",
        "- 初筛canonical约1.0000×；单独倒数复用约1.0021×、联合规约约1.0107×，不能预设标量优化有显著整体收益。",
        "- grouped2048将16384组222寄存器降至118、无spill，NCU实际occupancy约12%→23%；"
        "但完整30点几何0.9782×，8192点约0.8194×。额外重载/分组降低活跃状态的收益有尺寸边界，未采用。",
        "- NCU使用base clock/cache-control=all，采集occupancy、WarpStateStats、MemoryWorkloadAnalysis等；"
        "C级诊断，不替换常规配对计时，不把occupancy翻倍等同于性能翻倍。",
        "",
        "| N | baseline NCU µs | joint_coefficients NCU µs | grouped NCU µs |",
        "|---:|---:|---:|---:|",
    ]
    ncu = load("ncu_summary.json")
    for n in [1024, 4096, 8192, 8704, 15872]:
        vals = {
            v["kernel"]: float(v["metrics"]["gpu__time_duration.sum"]["value"])
            for v in ncu
            if v["N"] == n
        }
        lines.append(
            f"| {n} | {vals['baseline']:.3f} | {vals['joint_coefficients']:.3f} | {vals['grouped2048']:.3f} |"
        )
    lines += [
        "",
        "主实验初次候选生成因Python共享模块common名称冲突而在编译前失败，"
        "日志accuracy_round1.log保留；改为局部文本变换后成功，未将脚本失败算作kernel失败。"
        "本轮仅尝试冻结的8个候选，停止继续扩展。",
        "",
        "## 文件入口",
        "",
        "- `ir/b*/baseline.ttir` / `optimized.ttir`：各组冻结基线和最终别名。",
        "- `results/selection.json`：实际组级选择、输入hash、主/复测来源。",
        "- `results/full.json` / `repeat.json` / `selected_benchmark.json`：原始性能/A/A数据。",
        "- `results/ncu_finalists.ncu-rep` / `ncu.csv` / `ncu_summary.json`：NCU证据。",
        "- `test_accuracy.sh` / `test_performance.sh`：实际运行入口，见README。",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines))
    print(composed, observed)


if __name__ == "__main__":
    main()
