"""Render round-three results from measured, hash-gated evidence."""

import json
import math
import statistics

from common import ROOT

R = ROOT / "results/round3"


def geo(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def main():
    selection = json.load(open(R / "selection_final.json"))
    sets = {
        name: json.load(open(R / name))
        for name in [
            "full_stream.json",
            "repeat_stream.json",
            "full.json",
            "repeat.json",
        ]
    }
    out = [
        "# backward dX 第三轮：中心化代数与联合规约",
        "",
        f"本轮6个新候选，晋升family：{selection['promoted_blocks']}。"
        "`optimized.ttir`仅更新通过主测、独立复测和数值/sanitizer检查的组。前向实验与所有原始baseline未修改。",
        "",
        "baseline是官方Triton3.7.0教程的dX段提取，双方不做dW/dB或锁；结果不能称为完整fused backward的加速。FP16输入输出、FP32统计量"
        "与累加，N/stride是16倍数且N≤16384，无别名。",
        "",
        "## 最终分组结果（A级）",
        "",
        "| family | 最终候选 | 主测 / parent | 独立复测 / parent | 主测 / baseline |",
        "|---|---|---:|---:|---:|",
    ]
    for block, family in selection["families"].items():
        passed = int(block) in selection["promoted_blocks"]
        main_stats = selection["main"][block]
        rep = selection["repeat"][block]
        rows = [
            r
            for r in sets["full_stream.json"]["records"]
            if r["block"] == int(block)
            and r["kernel"] == ("r3_centered_stream" if passed else "r3_parent")
        ]
        baseline = geo([r["speedup"] for r in rows])
        main_ratio = main_stats["vs_parent"]["geometric_mean"] if passed else 1
        repeat_ratio = rep["vs_parent"]["geometric_mean"] if passed else 1
        out.append(
            f"| b{block} | {family['candidate']} | {main_ratio:.4f}× | {repeat_ratio:.4f}× | {baseline:.4f}× |"
        )
    out += [
        "",
        "表中未晋升组的parent比值按保留版本写1；baseline列为该组最终选择在同会话的实测值。主测7轮rep20ms，独立复测seed123/5轮rep1"
        "0ms，M4096、N1024..15872步长512，共30点。组级接受门槛：对parent几何≥1.02、最差≥0.95，并且对原baseline几何≥1.02，主测"
        "和复测均须通过。",
        "",
        "| 测量 | 最终组合 / baseline | 最终组合 / parent | parent / baseline | A/A |",
        "|---|---:|---:|---:|---|",
    ]
    composite = {}
    for label, filename in [
        ("主测", "full_stream.json"),
        ("独立复测", "repeat_stream.json"),
    ]:
        data = sets[filename]
        assert data["aa_passed"]
        parents = {r["N"]: r for r in data["records"] if r["kernel"] == "r3_parent"}
        selected = [
            r
            for r in data["records"]
            if r["kernel"]
            == (
                "r3_centered_stream"
                if r["block"] in selection["promoted_blocks"]
                else "r3_parent"
            )
        ]
        assert len(selected) == 30
        ratios = [parents[r["N"]]["candidate_us"] / r["candidate_us"] for r in selected]
        composite[filename] = dict(
            vs_baseline=geo([r["speedup"] for r in selected]), vs_parent=geo(ratios)
        )
        out.append(
            f"| {label} | {composite[filename]['vs_baseline']:.4f}× | {geo(ratios):.4f}× | "
            f"{data['summary']['r3_parent']['geometric_mean']:.4f}× | 30/30通过 |"
        )
    out += [
        "",
        "最终组合由同一会话内各shape已冻结选择的候选记录汇总；对parent为各kernel耗时中位数之比，每条原始计时仍与共同baseline配对，不伪称直接parent/"
        "candidate交错配对。别名的IR/cubin身份由最终pipeline检查验证。",
        "",
        "## 有效方向与限制",
        "",
        "将规约从sum(xhat*wdy)改为sum((x-mean)*wdy)，把rstd移到行级alpha系数；仍显式中心化和FP32运算，输出使用rstd³*c1与rstd"
        "*c2。另加DX的cs store。b1024/b2048的寄存器仍为32、零spill、2处静态CTA barrier；静态SASS的FMUL从25减至19，store"
        "变为STG.E.EF.128。这是编译产物线索，未把全部收益归因于某一条指令或cache属性。大N不能继承小N收益。",
        "",
        "## 未采用的方向",
        "",
        "- r3_rawmoment规约sum(x*wdy)，再减mean*sum(wdy)：常规135例通过，但新增72例中18例失败；其他候选和baseline在同一save"
        "d stats参考下均通过。存在抵消误差，不放宽容差，不做性能筛选。",
        "- r3_joined把两个向量join为[B,2]再规约：b16384升到237寄存器；未表现出优于parent的初筛整体收益。",
        "- r3_group_center1024/2048：b16384降到128/126寄存器，零spill；2048版大组主测对parent仅1.0019×、复测0.999"
        "6×，未通过晋升。整30点相对baseline主测0.9850×、复测0.9857×。单点N8704收益不能代表整个family。",
        "- 第一份screen_batch1在N8704的A/A=1.0413，整份不用于晋升；保留原件，补跑screen_batch1_repeat后全部A/A通过。",
        "",
        "## NCU瓶颈证据（C级）",
        "",
        "固定base clock、cache-control=all，NCU仅用于诊断；其耗时不混入CUDA graph性能结果。下面是parent→group_center2048：",
        "",
        "| N | registers | achieved occupancy | DRAM读取MB | long-scoreboard / issue | NCU µs |",
        "|---:|---|---|---|---|---|",
    ]
    ncu = json.load(open(R / "ncu_summary.json"))
    for n in [8192, 8704, 15872]:
        a, b = [
            next(r for r in ncu if r["N"] == n and r["kernel"] == name)
            for name in ["r3_parent", "r3_group_center2048"]
        ]

        def pair(key, scale=1, precision=2):
            return f"{a[key]['value']*scale:.{precision}f}→{b[key]['value']*scale:.{precision}f}"

        out.append(
            f"| {n} | {pair('launch__registers_per_thread', precision=0)} | "
            f"{pair('sm__warps_active.avg.pct_of_peak_sustained_active')}% | "
            f"{pair('dram__bytes_read.sum', 1e-6)} | "
            f"{pair('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio')} | "
            f"{pair('gpu__time_duration.sum', 1e-3)} |"
        )
    out += [
        "",
        "分组中心化避免了第二轮纯重载的DRAM读取近翻倍，但N15872占用率提升后仍变慢，long-scoreboard指标增加；指令、加载依赖和调度仍可能抵消资源收益，尚未完成单因素因果拆分。",
        "",
        "小N的NCU未复现graph收益：N1024 parent→centered_stream为9.728→9.984µs，N2048为15.936→16.384µs。"
        "NCU为base clock且每次清cache，性能协议是反复CUDA graph执行；两者条件不同。不能宣称冷cache或实际单次调用获得相同加速，"
        "静态FMUL减少仍只是机制线索。原始数据见results/round3/ncu_stream_summary.json。",
        "",
        "## 正确性与编译路径",
        "",
        "首批810次、第二批540次常规检查通过；额外48边界/版本，额外压力72例/版本；独立seed123再验证。压力测试原先用autograd重算stats，导致basel"
        "ine和parent也有17例不一致；原件保留，新增显式aten.native_layer_norm_backward传入同一mean/rstd，同时比较FP64 sav"
        "ed-stats公式，未改容差。常规135例继续要求原native autograd参考。",
        "",
        "memcheck/racecheck分别覆盖30个边界与padding用例/版本，均零错误；候选使用不可变名称和TTIR哈希。原始数值指标全部保留，不声称高rstd梯度的绝对误差统一小于0.01。",
        "",
        "```text",
        "baseline.ttir  ─┐",
        "                ├─ triton.compile(.ttir, 同组相同options)",
        "optimized.ttir ─┘     → make_ttgir → make_llir → PTX → Triton自带ptxas → cubin",
        "                      → CompiledKernel._init_handles → 相同CUDA launcher/相同grid与参数",
        "```",
        "",
        "官方upstream Triton3.7.0固定venv，无FlagTree；TTIR文件入口从make_ttgir继续。五组各自固定warps/stages等，不改gr"
        "id/ABI/options，不使用独立llc、手改低层IR或内联PTX。最终pipeline要求每组baseline LLVM/PTX/cubin与冻结教程提取源码一致"
        "，并验证最终候选IR与cubin身份。",
        "",
        "证据：[主测](results/round3/full_stream.json)、[独立复测](results/round3/repeat_stream.json)、[最"
        "终选择](results/round3/selection_final.json)、[分组失败决策](results/round3/decision.json)、[NCU"
        "](results/round3/ncu_summary.json)、[计划](optimization_plan_v3.md)。",
    ]
    direct = json.load(open(R / "final_selected_benchmark.json"))
    assert direct["aa_passed"] and len(direct["records"]) == 30
    direct_geo = direct["summary"]["optimized"]["geometric_mean"]
    out += [
        "",
        "## 最终别名直接复核",
        "",
        f"更新后的optimized入口直接跑完整30点、3轮rep10ms：几何{direct_geo:.4f}×原baseline，30/30 A/A通过。"
        "这是最终入口的补充复核；组级晋升依据仍是前述7轮主测与5轮独立复测。",
        "[直接复核记录](results/round3/final_selected_benchmark.json)、"
        "[最终编译链](results/round3/pipeline_final.json)、[最终别名正确性](results/round3/accuracy_selected.json)。",
    ]
    (ROOT / "report_round3.md").write_text("\n".join(out) + "\n")
    (R / "composite_summary.json").write_text(json.dumps(composite, indent=2) + "\n")


if __name__ == "__main__":
    main()
