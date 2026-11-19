"""Second-round dX performance diagnosis and bounded negative/positive experiment report."""

import json
import math
import statistics

from common import ROOT

R = ROOT / "results/round2"


def main():
    def load(n):
        return json.load(open(R / n))

    full = load("full.json")
    repeat = load("repeat.json")
    decision = load("decision.json")
    screen = load("screen_batch1.json")
    assert full["aa_passed"] and repeat["aa_passed"] and decision["passed"]
    rows = {
        name: {r["N"]: r for r in full["records"] if r["kernel"] == name}
        for name in full["summary"]
    }
    parent = rows["r1_parent"]

    def geo(values):
        return math.exp(statistics.mean(math.log(v) for v in values))

    main_summary = {
        n: dict(
            vs_baseline=full["summary"][n]["geometric_mean"],
            vs_parent=geo(
                parent[k]["candidate_us"] / v["candidate_us"] for k, v in data.items()
            ),
        )
        for n, data in rows.items()
    }
    lines = [
        "# backward dX 第二轮：瓶颈诊断与TTIR尝试",
        "",
        (
            "本轮新增8个TTIR候选，**未晋升新版本**。"
            if not decision["promoted_blocks"]
            else f"本轮新增8个TTIR候选，晋升组：{decision['promoted_blocks']}。"
        )
        + "没有达到接受门槛的组继续使用第一轮parent，原始baseline与前向成果保持不变。",
        "",
        "## 当前结果",
        "",
        "| 版本 | 主测30点几何 / baseline | 主测几何 / parent | 独立复测几何 / baseline |",
        "|---|---:|---:|---:|",
    ]
    for name, v in main_summary.items():
        lines.append(
            f"| {name} | {v['vs_baseline']:.4f}× | {v['vs_parent']:.4f}× | "
            f"{repeat['summary'][name]['geometric_mean']:.4f}× |"
        )
    lines += [
        "",
        "主测7轮rep20ms，独立seed123/5轮rep10ms，全部30点保留A/A控制。"
        "相对parent是同会话各kernel耗时中位数之比，原始各kernel仍与同一baseline配对；"
        "未伪造直接parent/candidate的配对样本。阈值为组级对parent几何≥1.02、最差≥0.95，"
        "同时对baseline几何≥1.02，复测不达标保留parent。",
        "",
        "## 瓶颈：为什么寄存器降了却更慢",
        "",
        "1. **大行宽有资源限制，但规约同步已经简化。** b16384的parent为220 registers、0 spill，"
        "NCU实际occupancy约12%；静态SASS已有2处CTA barrier，原baseline为5处。"
        "继续降低寄存器能提高驻留量，不代表足以抵消重载和控制指令。",
        "2. **8192组已具较高内存吞吐。** N8192的parent在NCU条件下约2.92TB/s，"
        "增加重载后即使occupancy从约34.5%升至61.1%，耗时仍从62.9升至74.5us。"
        "这是供给/依赖成本与资源的权衡，不是只看barrier或occupancy就能定位的单因素问题。",
        "3. **大N循环版可能增加实际DRAM流量。** 补充指标直接采集dram bytes读写总量，"
        "见下表；同一编译器、同输入/缓存/时钟条件下比较。读取是否命中L1/L2不能仅由TTIR load条数推断。",
        "4. **范围组合的寄存器额度由整个kernel决定。** 单独loop2048为40寄存器，"
        "加入静态fallback后range2048为123、range4096为126，约落在128分配档；"
        "N8704的动态路径不能继续享受独立40-register kernel的驻留能力。"
        "组合是否改变其他调度仍需产物检查，不能精确断言所有差距都来自寄存器。",
        "",
        "## 第一批候选（B级6点初筛；资源为C级）",
        "",
        "| 候选 | 初筛几何 / baseline | 最差点 | b16384 registers |",
        "|---|---:|---:|---:|",
    ]
    analysis = load("binary_analysis.json")
    for name, s in screen["summary"].items():
        if name == "r1_parent":
            continue
        resource = analysis[f"{name}/b16384"]
        lines.append(
            f"| {name} | {s['geometric_mean']:.4f}× | {s['worst']:.4f}× | {resource['n_regs']} |"
        )
    lines += [
        "",
        "只重载X、只重载DY/W、全部重载均有退步；两遍窄循环仅刚超过8192的行宽受益，"
        "N15872的loop2048降到约0.627×。reshape/fold联合规约候选也未改善整体性能。"
        "这说明前向的窄分组/重载经验不能直接迁移到反向dX：此处保存的xhat和wdy是后续输出需要的值，"
        "重算/重读具有实际成本。",
        "",
        "## 最后两个范围候选：逐shape复核",
        "",
        "b16384且N≤10240走动态循环，其他N走第一轮静态grouped2048；其他四组与parent字节相同。"
        "并未改grid、warps、stages或ABI。",
        "",
        "| N | parent µs | range2048 µs | range4096 µs | range4096相对parent |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in range(1024, 15873, 512):
        t = parent[n]["candidate_us"]
        a = rows["r2_range2048"][n]["candidate_us"]
        b = rows["r2_range4096"][n]["candidate_us"]
        lines.append(f"| {n} | {t:.3f} | {a:.3f} | {b:.3f} | {t/b:.4f}× |")
    lines += [
        "",
        "## NCU机制线索（C级）",
        "",
        "固定base clock/cache-control=all；LaunchStats、Occupancy、WarpStateStats、MemoryWorkloadAnalysis。"
        "计时与常规CUDA graph口径不同，不计作最终加速比。stall/issue是比值，不是耗时百分比。",
        "",
        "| N | 版本 | NCU µs | 寄存器 | actual occupancy | DRAM TB/s |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in load("ncu_summary.json"):
        if row["N"] not in [8192, 8704, 15872]:
            continue
        m = row["metrics"]
        lines.append(
            f"| {row['N']} | {row['kernel']} | {float(m['gpu__time_duration.sum']['value']):.3f} | "
            f"{m['launch__registers_per_thread']['value']} | "
            f"{float(m['sm__warps_active.avg.pct_of_peak_sustained_active']['value']):.1f}% | "
            f"{float(m['dram__bytes.sum.per_second']['value']):.3f} |"
        )
    traffic = load("traffic_summary.json")
    lines += [
        "",
        "### DRAM字节数专项采集",
        "",
        "| N | 版本 | read MB | write MB | NCU µs |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in traffic:
        if row["N"] not in [8192, 8704, 15872]:
            continue
        lines.append(
            f"| {row['N']} | {row['kernel']} | {row['read_bytes']/1e6:.2f} | "
            f"{row['write_bytes']/1e6:.2f} | {row['us']:.3f} |"
        )
    lines += [
        "",
        "N15872的动态循环版观察到实际DRAM读取增加；具体倍数按表中采集值计算。"
        "该指标记录采集窗口里的实际DRAM计数，写回时机和kernel replay状态仍会影响数值；"
        "不从单次计数建立绝对带宽下界。重载的缓存行为与源级访存次数应分开讨论。",
        "",
        "## 验证与边界",
        "",
        "- 官方Triton3.7.0独立venv，原生TTIR→TTGIR→LLVM→PTX→cubin，双方同options/加载/launcher；"
        "未使用inline PTX或手改下游产物。",
        "- 第一批1080次、范围候选540次完整C1；独立seed123复验、48个新增分支/旧边界用例的memcheck/racecheck结果见各JSON和日志。"
        "原135用例包含保存统计量FP64/native梯度参考、小用例CPU FP64、压力分布和padding。"
        "沿用rtol/atol=0.01、归一化RMS≤0.003，不调整容差接受候选。",
        "- 所有原始样本、IR哈希和A/A结果保留；没有将首次初筛的局部最佳点当作最终优化。"
        "未锁常规计时时钟，未清缓存，不称端到端/冷缓存或完整dX+dW+dB性能。",
        "",
        "## 下一步",
        "",
        "在相同配置和普通TTIR约束下，优先寻找能够减少中间数据而不增加大规模DRAM重读的表达。"
        "若进一步研究每CTA行数、num_warps或多kernel算法，需要建立新契约与匹配baseline；"
        "本轮未偷偷改变这些条件。保留已经验证的第一轮成果及本轮负结果，停止8候选预算内的搜索。",
        "",
        "入口与原始证据：`results/round2/decision.json`、`full.json`、`repeat.json`、"
        "`binary_analysis.json`、`ncu_batch1.ncu-rep`、`traffic.ncu-rep`和`manifest.json`。",
        "",
    ]
    (ROOT / "report_round2.md").write_text("\n".join(lines))
    print(main_summary)


if __name__ == "__main__":
    main()
