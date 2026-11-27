"""Recompute norm/selection summaries and freeze only qualified selections."""

import json
import math
import random
import statistics
from pathlib import Path

from support import digest, freeze, save, summarize

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(records, name, seed=71):
    rng = random.Random(seed)
    draws = []
    for _ in range(3000):
        ratios = []
        for row in records:
            s = row["timing"]["candidates"][name]["samples"]
            ratios.append(statistics.median(rng.choice(s)["speedup"] for _ in s))
        draws.append(math.exp(statistics.mean(math.log(v) for v in ratios)))
    draws.sort()
    return [draws[75], draws[2925]]


def finalize(op, candidate, title, explanation, limits):
    folder = ROOT / op
    main = json.loads((folder / "results/main_v2/results.json").read_text())
    repeat = json.loads((folder / "results/repeat_v2/results.json").read_text())
    assert main["passed"] and repeat["passed"]
    summary = summarize(main["records"])[candidate]
    other = summarize(repeat["records"])[candidate]
    assert main["aa_passed"] and repeat["aa_passed"]
    assert min(summary["geomean"], other["geomean"]) >= 1.02
    assert min(summary["worst"], other["worst"]) >= 0.95
    assert (
        "ERROR SUMMARY: 0 errors"
        in (folder / "results/logs/memcheck_final.log").read_text()
    )
    assert (
        "0 errors, 0 warnings"
        in (folder / "results/logs/racecheck_final.log").read_text()
    )
    pairs = []
    for row, rep in zip(main["records"], repeat["records"]):
        assert row["shape"] == rep["shape"]
        stage_names = ("local", "merge") if op == "partial_topk" else (None,)
        for stage in stage_names:
            info = (
                row["kernels"][candidate][stage] if stage else row["kernels"][candidate]
            )
            again = (
                rep["kernels"][candidate][stage] if stage else rep["kernels"][candidate]
            )
            assert info["ir_sha256"] == again["ir_sha256"] == digest(info["ir_path"])
            base = (
                row["kernels"]["baseline"][stage]
                if stage
                else row["kernels"]["baseline"]
            )
            # All parsed backend options are saved; cache hash/resources may differ.
            from triton.backends.nvidia.compiler import CUDABackend
            from triton.runtime.driver import driver

            fields = (
                CUDABackend(driver.active.get_current_target())
                .parse_options({})
                .__dict__
            )
            assert {k: info["metadata"][k] for k in fields} == {
                k: base["metadata"][k] for k in fields
            }
            src = Path(info["ir_path"])
            freeze(src.parent / "optimized.ttir", src.read_text())
            pairs.append(
                {
                    "group": row["group"],
                    "stage": stage,
                    "candidate": candidate,
                    "ir_sha256": info["ir_sha256"],
                }
            )
    selection = {
        "candidate": candidate,
        "accepted": True,
        "evidence_level": "A, frozen matrix only",
        "main": summary,
        "repeat": other,
        "main_bootstrap95": bootstrap(main["records"], candidate),
        "repeat_bootstrap95": bootstrap(repeat["records"], candidate),
        "groups": pairs,
        "evidence": {
            name: digest(folder / f"results/{name}/results.json")
            for name in ("main_v2", "repeat_v2")
        },
    }
    save(folder / "selection.json", selection)
    lines = [
        f"# {title}",
        "",
        f"最终采用 `{candidate}`。冻结{len(main['records'])}点几何平均：主测 "
        f"**{summary['geomean']:.4f}×**，独立进程复测 **{other['geomean']:.4f}×**；"
        f"最差点分别 **{summary['worst']:.4f}× / {other['worst']:.4f}×**。"
        f"性能证据为A，仅限本矩阵和本研究baseline。",
        "",
        explanation,
        "",
        "| shape | baseline μs | candidate μs | 主测 | 复测 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row, rep in zip(main["records"], repeat["records"]):
        t = row["timing"]["candidates"][candidate]
        rt = rep["timing"]["candidates"][candidate]
        lines.append(
            f"| {tuple(row['shape'])} | {t['baseline_us']:.3f} "
            f"| {t['candidate_us']:.3f} | {t['speedup']:.4f}× "
            f"| {rt['speedup']:.4f}× |"
        )
    lines += [
        "",
        "时间为样本中位数，加速比为配对比值中位数，两者不必恰好相除。",
        "",
        "## 协议与证据",
        "",
        "固定官方 Triton3.7.0 / PyTorch2.12.0+cu130 / H100。"
        "基线先从冻结源码导出，再与文件入口核对LLVM/PTX/cubin字节一致；每次主/复测在计时前重复核验，"
        "实际完整backend options一致。所有候选从TTIR文件编译，未改TTGIR/LLVM/PTX/cubin。"
        "",
        "",
        "主测7轮、独立进程复测5轮、每轮20ms；配对内随机次序、每轮A/A。预分配完整设备序列CUDA "
        "graph时间，不含编译、分配和Python launch；cache未清理、时钟未锁定。每shape保存计时输入seed与SHA256。"
        "初筛后冻结单一候选，不按独立复测择优。",
        "",
        f"固定shape、重采样配对轮次的95% bootstrap区间：主测 `{selection['main_bootstrap95']}`，"
        f"复测 `{selection['repeat_bootstrap95']}`。它只反映这些样本的波动，"
        f"不表示跨模型/设备的不确定性。",
        "",
        "主证据：[main_v2](results/main_v2/results.json)、"
        "[repeat_v2](results/repeat_v2/results.json)、"
        "[manifest](manifest.json)、[选择和哈希](selection.json)。"
        "memcheck **0 errors**，racecheck **0 errors/0 warnings**，"
        "只过滤检查本实验kernel；[memcheck日志](results/logs/memcheck_final.log)、[racecheck日志](results/logs/racecheck_final.log)。"
        "",
        "",
        "初版main/repeat也保留；审阅发现它们的timing tensor在seed初始化前分配，"
        "配对公平但无法按所记seed完整重建输入。本报告改用已修复seed并保存输入哈希的main_v2/repeat_v2，"
        "没有把旧样本混入平均。",
        "",
        "## 数值、资源与负结果",
        "",
    ]
    validation_count = sum(
        len(v)
        for dataset in (main, repeat)
        for r in dataset["records"]
        for v in r["validation"].values()
    )
    lines.append(
        f"正式主/复测共{validation_count}项版本/分布检查通过；另保留各次初筛与sanitizer内检查。所有输出有canary、sentinel及输入不修改检查。"
    )
    lines += [
        "",
        "| shape | regs baseline→candidate | spill baseline→candidate | shared B baseline→candidate |",
        "|---|---|---|---|",
    ]
    for row in main["records"]:
        bs = row["kernels"]["baseline"]
        cs = row["kernels"][candidate]
        if op == "partial_topk":
            bs, cs = bs["local"], cs["local"]
        lines.append(
            f"| {tuple(row['shape'])} | {bs['n_regs']}→{cs['n_regs']} "
            f"| {bs['n_spills']}→{cs['n_spills']} | {bs['metadata']['shared']}→{cs['metadata']['shared']} "
            f"|"
        )
    lines += [
        "",
        "资源是C级静态证据；Top-K表列local阶段，merge资源在原始JSON。不把寄存器或同步减少直接解释为全部性能收益。",
        "",
        limits,
        "",
        "## 复现",
        "",
        "先查看 `nvidia-smi` 并设置独占 `CUDA_VISIBLE_DEVICES`，"
        "Python固定 `/root/venvs/ir-ops-dev/bin/python`。在仓库根执行下列命令，"
        "输出目录必须不存在：",
        "",
        "```bash",
        f"CUDA_VISIBLE_DEVICES={6 if op == 'norm_quant' else 7} "
        f"/root/venvs/ir-ops-dev/bin/python {op}/experiment.py "
        f"\\",
        f"  --run my_recheck --names baseline,{candidate} --rounds 5 --rep 20 --seed 234",
        "```",
        "",
        "脚本先自动执行pipeline/数值门槛，再计时；`--check-only`只验证。候选生成器对已有IR做字节一致性断言，拒绝覆盖不同内容。已冻结manifest时不要再次`--setup`。",
        "",
    ]
    (folder / "report_round1.md").write_text("\n".join(lines))
    return selection


if __name__ == "__main__":
    result = {}
    result["norm_quant"] = finalize(
        "norm_quant",
        "stream_v1",
        "Residual RMSNorm + 动态 INT8：直接 TTIR 实验",
        "融合研究baseline一次完成残差加法、RMSNorm、amax、INT8量化，并写出FP32 "
        "residual、scale和INT8。它参考FlagGems结构，但不是生产kernel原样提取；"
        "每个点只在4/8 warps中预调优，未穷尽配置。候选把max(abs(rho*u))改为rho*max(abs(u))，"
        "将sum(z²)/max(abs(u))合为tuple reduction，再对输出使用cs "
        "cache hint；单kernel、ABI/grid/options不变。",
        "量化合同预先允许Q相对FP64至多差1 code、scale rtol=3e-5/atol=1e-9，"
        "另验证重构误差；**不声称INT8逐位等价**。Z精确、Q显式限制在[-127,127]，因此-128 "
        "sentinel不能蒙混过关。覆盖normal/zero/constant/offset/cancellation/tiny/signed-weight，"
        "小rows另有CPU FP64。\n\n4种候选的完整初筛保留在 `results/screen/results.json`（B）：因式分解、联合统计、消去rho、输出cache "
        "hint；stream收益是组合结果。N=16384、rows32约1.4%回退仍保留，未用逐点挑选掩盖。"
        "未涵盖FP8、BF16、任意非连续layout、NaN/Inf。初次setup曾错误比较metadata的cache "
        "key，修正为完整backend options比较；失败日志保留，没有放过LLVM/PTX/cubin一致性门槛。"
        "",
    )
    result["partial_topk"] = finalize(
        "partial_topk",
        "ordered_carry_v3",
        "大 N partial Top-K：直接 TTIR 的循环状态重写",
        "固定两阶段研究baseline：每1024输入保留K候选，再合并成全局Top-K；沿用FlagGems分块重复选择思路，"
        "未全排序。输出value/原始index，值降序，同分按较小index优先。候选把FP32 "
        "score映射为单调uint32 key，以unsigned max归约，并将编码移出K循环、key作为loop-carried "
        "SSA值保留；min-index tie规则保留。kernel数、workspace、ABI/grid和4warps/3stages不变，"
        "计时覆盖两kernel。此baseline配置固定，未逐点autotune。",
        "有限FP32的value数值精确、indices严格精确；**±0等价，不要求保留零符号位**，"
        "baseline的max结果本来就不保证选中index处的零位型。v3先归一化±0以保持同分index规则。"
        "已选值/padding使用-inf的有序key，小于所有finite key；尾块不足K也不能胜过真实输入。"
        "主测扩展到正/负subnormal及最低有限值，另含normal/全tie/整数tie/±0/递增；"
        "小rows对CPU FP64稳定排序。\n\n负结果：联合value/index argmax导致全6点约0.810×；"
        "仅改有序max但每轮重算key约0.980×。两者在旧main完整7轮有B级证据，见 `results/main/results.json`；"
        "joint/local_joint/cg消融在screen。v3的改变是循环状态表示，不能把全部收益简单归因于少一次reduce。"
        "未实施packed64候选；未测试NaN/Inf、不同K、不同chunk配置或动态workspace协议。"
        "",
    )
    save(ROOT / "complex_suite/norm_topk_summary.json", result)
