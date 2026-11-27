"""Audit paired final-version ablations and render independent operator reports."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OP = ROOT.parent
NAMES = json.loads((ROOT / "frozen/candidates.json").read_text())["names"]
LABELS = {
    "baseline": "原始融合基线",
    "parent": "首轮组合：共享+标量化+unsigned",
    "narrow": "再缩窄 scale/zero 加载",
    "full_k": "首轮组合仅去 K mask",
    "narrow_full_k": "窄加载+去 K mask",
    "group_pair": "再加 BK64 组内复用（最终去右移）",
    "final": "最终：再用 group128 右移",
    "no_full_k": "最终恢复 K mask",
    "no_pair": "最终去组内复用",
    "no_narrow_pair": "最终联合去窄加载和组内复用",
}
EDGES = [
    ("原始基线 → 首轮组合", "baseline", "parent"),
    ("首轮组合 → 窄加载", "parent", "narrow"),
    ("首轮组合 → 去 K mask", "parent", "full_k"),
    ("窄加载 → 再去 K mask", "narrow", "narrow_full_k"),
    ("去 K mask → 再窄加载", "full_k", "narrow_full_k"),
    ("两项均启用 → 组内复用", "narrow_full_k", "group_pair"),
    ("组内复用 → group128 右移", "group_pair", "final"),
]
REMOVALS = [
    ("group128 右移", "group_pair", "final"),
    ("K mask 简化", "no_full_k", "final"),
    ("BK64 组内复用", "no_pair", "final"),
    ("窄加载和组内复用（联合移除）", "no_narrow_pair", "final"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(phase, rounds):
    path = ROOT / "results" / f"v2_{phase}" / "results.json"
    data = json.loads(path.read_text())
    assert data["passed"] and data["aa_passed"] and not data["failures"]
    assert data["rounds"] == rounds and data["rep_ms"] == 20
    assert data["valid_names"] == NAMES and len(data["records"]) == 30
    pipeline = json.loads((ROOT / "results" / f"v2_{phase}_pipeline.json").read_text())
    assert pipeline["passed"] and len(pipeline["records"]) == 30
    for row in data["records"]:
        assert row["pipeline"]["llir_ptx_cubin_identical"]
        assert len(row["timing"]["samples"]) == rounds
        assert len(row["validation"]) == len(NAMES) * 5
        assert 0.97 <= row["timing"]["aa_median"] <= 1.03
        final = OP / "multiround/r6/ir" / row["case"] / "optimized.ttir"
        for name, kernel in row["kernels"].items():
            path = ROOT / "frozen/ir" / row["case"] / f"{name}.ttir"
            assert kernel["ttir_sha256"] == sha(path)
        assert sha(final) == sha(ROOT / "frozen/ir" / row["case"] / "final.ttir")
    return data


def ratios(data, before, after):
    return {
        row["case"]: statistics.median(
            sample["us"][before] / sample["us"][after]
            for sample in row["timing"]["samples"]
        )
        for row in data["records"]
    }


def geo(data, before, after):
    return statistics.geometric_mean(ratios(data, before, after).values())


def main():
    runs = [read("main", 7), read("repeat", 5)]
    cumulative = {name: [geo(run, "baseline", name) for run in runs] for name in NAMES}
    edges = {
        label: [geo(run, before, after) for run in runs]
        for label, before, after in EDGES
    }
    removals = {
        label: [geo(run, before, after) for run in runs]
        for label, before, after in REMOVALS
    }
    representative = next(
        row for row in runs[0]["records"] if row["shape"] == [128, 11008, 4096]
    )
    report = [
        "# INT4 GEMM 最终版 TTIR 消融实验",
        "",
        "研究第六轮保留的 `r3/group128`；不修改冻结基线、候选或优化别名。每个 shape 的 `final.ttir` "
        "与第六轮 `optimized.ttir` SHA256 相同。固定 H100、官方 Triton3.7.0、30点、FP16/uint8、"
        "group128及逐点原配置。主测 GPU0 seed42 7轮，独立进程 GPU1 seed123 5轮，均 rep20ms。"
        "两次在各自同卡测量全部版本与共同 baseline，跨卡不直接相除绝对时延。[完整协议](ablation_plan.md)",
        "",
        "## 各版本相对原始 baseline",
        "",
        "| 版本 | 主测几何加速 | 独立复测 | 代表点寄存器 | spill |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        resource = representative["kernels"][name]
        report.append(
            f"| {LABELS[name]} (`{name}`) | {cumulative[name][0]:.4f}× | "
            f"{cumulative[name][1]:.4f}× | {resource['n_regs']} | {resource['n_spills']} |"
        )
    report += [
        "",
        "寄存器/spill只列 `(128,11008,4096)` 的静态值（C级），完整各点资源和IR/PTX/cubin哈希见原始JSON。"
        "相对基线性能为本30点主测/独立复测的A级证据，不能从资源变化单独推导性能原因。",
        "",
        "## 逐项加入的条件收益",
        "",
        "每个 shape 先取同轮 `before_us/after_us` 的中位数，再跨30点求几何平均。"
        "这些是直接同会话的条件对照；百分比不能相加。首轮组合仍把共享、标量化和unsigned合并，"
        "本报告不把其收益拆给未单独复测的首轮成分。",
        "",
        "| 对照 | 主测 | 独立复测 |",
        "|---|---:|---:|",
    ]
    report += [f"| {label} | {v[0]:.4f}× | {v[1]:.4f}× |" for label, v in edges.items()]
    mask_worst = [min(ratios(run, "parent", "full_k").values()) for run in runs]
    report += [
        "",
        f"去K mask单独启用的均值仅{edges['首轮组合 → 去 K mask'][0]:.4f}×/"
        f"{edges['首轮组合 → 去 K mask'][1]:.4f}×，最差shape为{mask_worst[0]:.4f}×/"
        f"{mask_worst[1]:.4f}×（均为M256,N4096,K4096）。在窄加载之后加入则约1.039×。"
        "因此不能把K mask删除称为无条件提升；当前收益与其余编译结构有关。",
        "",
        "## 从最终版移除改动",
        "",
        "数值为缺少改动的版本耗时/最终版耗时。组内复用生成器依赖窄加载结构，"
        "最后一行联合移除两项，不能解释为窄加载的纯独立贡献；窄加载的干净对照见上一节2×2。"
        "BK128上的组内复用步骤本来就是恒等变换，不把它当作该子集的一项收益。",
        "",
        "| 去掉的改动 | 主测条件加速 | 独立复测 |",
        "|---|---:|---:|",
    ]
    report += [
        f"| {label} | {v[0]:.4f}× | {v[1]:.4f}× |" for label, v in removals.items()
    ]
    report += [
        "",
        "## 正确性、计时与限制",
        "",
        "每次显式pipeline均通过30点源码/TTIR文件入口LLVM/PTX/cubin及输出一致性。"
        "各会话30×10×5=1500项版本/分布检查全部通过，先完成完整正确性矩阵才计时。"
        "random/extreme_nibble/zero_scale/zero/constant五种分布，对照FP32 ATen，N≤256另对照CPU FP64，"
        "候选与baseline逐元素完全一致（torch.equal）、输出canary和输入不变。全部A/A中位数在[0.97,1.03]。",
        "",
        "每版每点100ms稳定阶段，随机次序、预分配CUDA graph单fused kernel计时，保留所有样本。"
        "未锁时钟、未给置信区间；低于约1%的差异不据此证明可靠提升。"
        "只覆盖既定有限shape/分布，不外推其他packing/dtype/GPU，也不代表冷cache或端到端。"
        "第六轮正式路径已有[memcheck](multiround/r6/memcheck.log)/[racecheck](multiround/r6/racecheck.log)"
        "零错误记录；本轮新增中间消融没有另跑sanitizer。",
        "",
        "## 普通 Triton 源码表达能力",
        "",
        "共享scale/zero、标量group索引、unsigned索引、窄tl.load后broadcast、删除契约内恒真mask、"
        "两次dot共用组参数、以`k0 >> 7`代替除128，都能用公开Triton表达。"
        "已有[首轮源码回写实测](source_port/report.md)只覆盖首轮组合，不能当作本最终组合的源码性能对照。"
        "本轮没有运行最终组合的源码回写；可表达同一算法不意味着相同机器码或相同性能。",
        "",
        "| 改动 | 公开Triton对应写法 | 本轮结论 |",
        "|---|---|---|",
        "| 共享low/high组参数 | 同一scale/zero值被两支引用 | 可表达；首轮已有组合源码实测 |",
        "| group标量化/unsigned | 标量`k0 // group`后广播；`tl.cast(..., tl.uint32)` | 可表达 |",
        "| 窄加载 | `tl.load`读取1×N参数，再broadcast到K维 | 可表达；最终源码性能未测 |",
        "| 去K mask | 在契约保证整除时省去K维mask | 可表达，不能推广到K尾部 |",
        "| BK64组内复用 | 两次顺序`tl.dot`共用组参数 | 可表达，dot顺序须保持 |",
        "| group128右移 | 非负`k0 >> 7` | 可表达，group必须为128 |",
        "",
        "## 全部 shape",
        "",
        "以下是当前最终组合与原始baseline的同场配对结果。[各版本完整CSV](ablation/results/v2_all_shapes.csv)。",
        "",
        "| M,N,K | 主测最终加速比 | 独立复测 |",
        "|---|---:|---:|",
    ]
    per_shape = [ratios(run, "baseline", "final") for run in runs]
    for row in sorted(runs[0]["records"], key=lambda row: row["shape"]):
        key = row["case"]
        report.append(
            f"| {tuple(row['shape'])} | {per_shape[0][key]:.4f}× | {per_shape[1][key]:.4f}× |"
        )
    report += [
        "",
        "证据：[主测](ablation/results/v2_main/results.json)、[独立复测](ablation/results/v2_repeat/results.json)、"
        "[主测正确性](ablation/results/v2_main/accuracy.json)、[复测正确性](ablation/results/v2_repeat/accuracy.json)、"
        "[主pipeline](ablation/results/v2_main_pipeline.json)、[复pipeline](ablation/results/v2_repeat_pipeline.json)、"
        "[汇总及证据哈希](ablation/results/v2_summary.json)。",
        "",
        "复现：`CUDA_VISIBLE_DEVICES=<空闲卡> bash int4_gemm/run_experiment.sh <新会话名> [seed] [rounds]`。"
        "`ablation/generate.py`生成冻结IR，`ablation/verify_pipeline.py`核对编译链，`ablation/run.py`先数值后计时，"
        "`ablation/render.py`从全部原始样本重算本报告。旧的[第二轮消融](ablation/plan.md)和历史结果保留。",
    ]
    (OP / "report_ablation.md").write_text("\n".join(report) + "\n")
    final_geo = cumulative["final"]
    main_report = [
        "# INT4 反量化融合 GEMM 的 TTIR 优化",
        "",
        f"当前最终版本为第三轮 `group128`，第六轮继续保留；本次完整30点消融会话相对原始融合baseline的"
        f"几何加速比为 **{final_geo[0]:.4f}× / {final_geo[1]:.4f}×**（主测/独立复测，A级）。"
        "各版本、逐项加入、最终移除及完整矩阵见[独立消融报告](report_ablation.md)。",
        "",
        "## 实验契约与基线",
        "",
        "源自 FlagGems commit `ef8aea78fed8321c3e02a15bf31e847cd0579e4f` 的"
        " `_weight_int4pack_mm_with_scales_and_zeros.py`。[上游快照](upstream/source.py.snapshot)、"
        "[冻结提取源码](tutorial.py)、[首轮协议](optimization_plan.md)保留来源及提取说明。"
        "基线本来已经在同一设备kernel融合nibble解包、groupwise反量化及Tensor Core dot，"
        "没有用全量反量化中间张量作为唯一基线。",
        "",
        "A/scale/zero为连续FP16，packed为`[N,K/2]` uint8，低nibble对应偶K；group固定128，"
        "K为128倍数，N为16倍数，M允许尾部，无别名，有限输入且输出无溢出。"
        "权重按`FP16((FP32(q)-FP32(zero))*FP32(scale))`舍入，dot用FP32累积、输出FP16。"
        "对照FP32 ATen matmul使用`atol=0.035, rtol=0.003`，N≤256另对照CPU FP64。",
        "",
        "H100/SM90、官方Triton3.7.0；[30点冻结shape/config映射](round2/expanded/manifest.json)。"
        "首轮从预定四配置选择基线，新增点沿用配置族；属于有限配置调优，不是全面autotune或最强库比较。"
        "每点baseline/全部候选的ABI、grid、tile和完整后端options一致。"
        "所有被测路径只launch从TTIR原生编译的kernel，参考计算仅用于测试。",
        "",
        "## 优化过程",
        "",
        "| 步骤 | 直接修改TTIR | 合法性与边界 |",
        "|---|---|---|",
        "| 首轮组合 | 合并low/high量化组参数，group索引标量化，非负索引除余用unsigned | group128、BK64/128对齐；保留原FP16舍入和dot顺序 |",
        "| 窄加载 | scale/zero从宽K×N张量改成1×N加载再广播 | 同一K tile属于同一量化组 |",
        "| 去K mask | 契约内恒真的A/packed K mask改常真 | K为128倍数，N/M尾部mask保留 |",
        "| 组内复用 | BK64每128个K执行两次dot，共用scale/zero | dot顺序不变；BK128路径保持恒等 |",
        "| group128 | 运行时除group替换为逻辑右移7位 | 仅group=128、非负K索引契约 |",
        "",
        "前四项的独立/组合效果及最终移除右移、K mask简化和组内复用均由本次同场消融实测，"
        "不是用历史搜索轮平均相乘。共享/标量化/unsigned仍合并为首轮起点；其内部未独立复测的收益不作分摊。",
        "",
        "| 历史轮次 | 结论与证据入口 |",
        "|---|---|",
        "| 首轮7点 | scalar_unsigned_v1晋升；[历史完整报告](report_round1.md) |",
        "| 第二轮30点 | r2_group_pair晋升；[报告](round2/report.md) |",
        "| 第三轮 | group128晋升；[报告](multiround/r3/report.md) |",
        "| 第四、五轮 | 未产生新晋升；[多轮报告](multiround/report.md) |",
        "| 第六轮 | magic_fp32_v1对parent主/复测仅0.9536×/0.9537×，保留group128；[负结果](multiround/r6/report.md) |",
        "",
        "## 验证与完整性能结果",
        "",
        "本次两会话各30点×10路径×5分布共1500项检查通过；所有版本先完成正确性再计时。"
        "检查原生源码/文件baseline LLVM/PTX/cubin字节一致、候选输出与baseline一致、"
        "FP32/CPU参考、输入不变及输出canary。正式最终路径另有第六轮memcheck/racecheck零错误记录。",
        "",
        "预分配CUDA graph单fused kernel；主测7轮、复测5轮，rep20ms，各版先稳定100ms。"
        "主测GPU0与独立复测GPU1各自同场测baseline和所有候选，全部A/A中位数在[0.97,1.03]。"
        "每shape取同轮配对比中位数，再跨shape求几何平均；不跨GPU比较绝对时延。",
        "",
        "| M,N,K | 主测最终加速比 | 独立复测 |",
        "|---|---:|---:|",
    ]
    for row in sorted(runs[0]["records"], key=lambda row: row["shape"]):
        key = row["case"]
        main_report.append(
            f"| {tuple(row['shape'])} | {per_shape[0][key]:.4f}× | {per_shape[1][key]:.4f}× |"
        )
    main_report += [
        "",
        "## 归因与适用范围",
        "",
        "性能数字为该30点/该H100/该编译链的A级证据；资源及机器码哈希为C级。"
        "窄加载、K mask与组内复用存在交互，最终移除窄加载必须联合去掉当前复用结构，"
        "不能把条件收益相加或全归因于寄存器下降。未锁时钟，无置信区间，不外推新量化格式、dtype或其他GPU。"
        "不含分配/编译/CPU launch，不代表冷cache或生产端到端。",
        "",
        "公开Triton可表达上述算法；[已有源码回写性能](source_port/report.md)只覆盖首轮组合，"
        "本最终组合尚无源码回写对照，不声称TTIR独占收益或A/B/C信息优势。",
        "",
        "复现见[README](README.md)、[消融协议](ablation_plan.md)；"
        "本次[主测](ablation/results/v2_main/results.json)、[独立复测](ablation/results/v2_repeat/results.json)、"
        "[逐项消融及资源](report_ablation.md)、[全部CSV](ablation/results/v2_all_shapes.csv)。"
        "[第六轮最终选择](multiround/r6/selection.json)及所有冻结alias保持原样。",
    ]
    (OP / "report.md").write_text("\n".join(main_report) + "\n")
    summary = {
        "operator": "int4_gemm",
        "shape_count": 30,
        "final": "r3/group128 (r6 retained)",
        "cumulative": cumulative,
        "additions": edges,
        "removals": removals,
        "evidence": "A: 30-shape main/repeat; C: compiled resources and identity",
        "source_port_measured": False,
        "sanitizer_new_ablations_measured": False,
        "sessions": {
            phase: {
                "seed": run["seed"],
                "rounds": run["rounds"],
                "rep_ms": run["rep_ms"],
                "cuda_visible_devices": run["environment"]["cuda_visible_devices"],
            }
            for phase, run in zip(("main", "repeat"), runs)
        },
        "script_sha256": {
            str(path.relative_to(OP)): sha(path)
            for path in (
                ROOT / "generate.py",
                ROOT / "run.py",
                ROOT / "verify_pipeline.py",
                ROOT / "render.py",
                OP / "round2/run.py",
                OP / "round2/make_candidates.py",
                OP / "multiround/generate.py",
            )
        },
        "sources": {
            str(path.relative_to(OP)): sha(path)
            for path in sorted((ROOT / "results").glob("v2_*/results.json"))
            + sorted((ROOT / "results").glob("v2_*_pipeline.json"))
        },
    }
    (ROOT / "results/v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "results/v2_all_shapes.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "M", "N", "K", "version", "baseline_over_version"])
        for phase, run in zip(("main", "repeat"), runs):
            for name in NAMES:
                values = ratios(run, "baseline", name)
                for row in run["records"]:
                    writer.writerow([phase, *row["shape"], name, values[row["case"]]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
