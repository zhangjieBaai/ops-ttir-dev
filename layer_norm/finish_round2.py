"""Freeze a round-2 winner only after paired timing, independent checks and sanitizers."""

import argparse
import json
import math
import shutil
import statistics

from common import ROOT, save_json, sha256

RESULTS = ROOT / "results/round2"


def read(name):
    return json.loads((RESULTS / name).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--select",
        action="store_true",
        help="Record provisional selection without changing optimized",
    )
    args = p.parse_args()
    full = read("full_stable.json")
    assert len(full["records"]) == 90 and full["rounds"] == 7
    candidates = {n: s for n, s in full["summary"].items() if n != "optimized"}
    winner = max(candidates, key=lambda n: candidates[n]["geometric_mean"])
    selection = dict(
        kernel=winner,
        sha256=sha256(ROOT / f"ir/{winner}.ttir"),
        parent="tile_dispatch_math_v2",
        criterion="Highest geometric mean on full 30-shape, 7-round paired native-Triton benchmark",
        benchmark_sha256=sha256(RESULTS / "full_stable.json"),
    )
    path = RESULTS / "selection.json"
    if args.select:
        if path.exists():
            assert read("selection.json") == selection
        save_json(path, selection)
        print(winner)
        return
    assert read("selection.json") == selection
    accuracy = read("accuracy_all.json")
    extra = read("boundary_accuracy.json")
    repeat = read("independent_repeat.json")
    assert accuracy["passed"] and extra["passed"]
    assert (
        accuracy["kernels"][winner] == extra["kernels"][winner] == selection["sha256"]
    )
    assert repeat["rounds"] == 5 and len(repeat["records"]) == 60
    assert (
        repeat["summary"][winner]["geometric_mean"]
        > repeat["summary"]["tile_dispatch_math_v2"]["geometric_mean"]
    )
    for tool in ["memcheck", "racecheck"]:
        check = read(f"{tool}_accuracy.json")
        assert check["passed"] and check["kernels"][winner] == selection["sha256"]
        log = (RESULTS / f"{tool}.log").read_text()
        if tool == "memcheck":
            assert "ERROR SUMMARY: 0 errors" in log
        else:
            assert (
                "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in log
            )
    comparison = read("llir_comparison.json")
    assert comparison["ttir_name"] == winner
    assert len(comparison["records"]) == 30 and all(
        r["passed"] for r in comparison["accuracy"]
    )
    assert full["kernels"][winner]["ttir_sha256"] == selection["sha256"]
    # Historical raw results continue to refer to the original parent hash.
    previous = ROOT / "results/selection.json"
    if not (RESULTS / "previous_selection.json").exists():
        shutil.copyfile(previous, RESULTS / "previous_selection.json")
    shutil.copyfile(ROOT / f"ir/{winner}.ttir", ROOT / "ir/optimized.ttir")
    save_json(previous, selection)
    s = candidates[winner]
    old = full["summary"]["optimized"]
    rep = repeat["summary"][winner]
    comp_rows = comparison["records"]
    llir_ratios = [
        statistics.median(
            v["times_us"]["ttir_baseline"] / v["times_us"]["llir_optimized"]
            for v in r["samples"]
        )
        for r in comp_rows
    ]
    llir_geo = math.exp(statistics.mean(math.log(v) for v in llir_ratios))
    llir_arith = statistics.mean(llir_ratios)
    ttir_comp = comparison["summary"]["ttir_speedup"]
    relative = comparison["summary"]["optimized_ratio"]["geometric_mean"]
    old_rows = {r["N"]: r for r in full["records"] if r["kernel"] == "optimized"}
    new_rows = [r for r in full["records"] if r["kernel"] == winner]
    rr = {r["N"]: r for r in repeat["records"] if r["kernel"] == winner}
    llir_times = {r["N"]: r["times_us"]["llir_optimized"] for r in comp_rows}
    ncu = read("ncu_finalists_summary.json")
    resource = json.loads((ROOT / f"artifacts/{winner}/metadata.json").read_text())
    versus_parent = [
        old_rows[r["N"]]["candidate_us"] / r["candidate_us"] for r in new_rows
    ]
    lines = [
        "# LayerNorm TTIR 第二轮优化报告",
        "",
        f"采用 `{winner}.ttir`，已更新 `ir/optimized.ttir`，与该候选字节一致。",
        "",
        f"证据等级 A（完整矩阵与独立复测）：30 个 shape、7轮：算术平均 **{s['arithmetic_mean']:.3f}×**，几何平均 **{s['geometric_mean']:.3f}×**；"
        f"同轮上一版为 {old['arithmetic_mean']:.3f}× / {old['geometric_mean']:.3f}×。独立进程 seed=123、5轮复测："
        f"**{rep['arithmetic_mean']:.3f}× / {rep['geometric_mean']:.3f}×**。",
        "",
        f"相对parent有 {sum(v > 1 for v in versus_parent)}/30 个shape更快；最大回退为 "
        f"{(1/min(versus_parent)-1)*100:.2f}%（按同轮各版耗时中位数计算），不宣称逐点全部改善。",
        "",
        "## 与 LLIR 的实际距离",
        "",
        f"B级补充观测（单进程3轮）：四版本同 GPU/同 CUDA graph 计时，用**当前原生 baseline 作为共同参考**："
        f"LLIR 成熟版算术/几何平均为 **{llir_arith:.3f}× / {llir_geo:.3f}×**，"
        f"本轮 TTIR 为 **{ttir_comp['arithmetic_mean']:.3f}× / {ttir_comp['geometric_mean']:.3f}×**。"
        f"新 TTIR/LLIR 优化版耗时的几何平均比值为 **{relative:.3f}**（<1表示TTIR更快）。",
        "",
        "LLIR 保留 cubin 来自旧离线后端，仅作为性能参考；主实验两份 TTIR 全部经过同一官方 Triton 3.7.0 原生链。"
        "该比较没有把旧 LLIR 的较慢 baseline 作为 TTIR 参考，也不把结果外推为层级本身的优劣。",
        "",
        "## 保持不变的契约",
        "",
        "H100 GPU7 / SM90；官方 Triton 3.7.0、独立 venv、无FlagTree。FP16 X/Y/W/B，FP32累加与Mean/Rstd，"
        "中心化方差。8 warps、1 CTA、3 stages、相同签名/grid=(M,1,1)/launcher。"
        "N/stride为16倍数、列连续、X/Y同行stride、无别名；主矩阵M=4096，N=1024..15872 step512。"
        "只编辑TTIR，无内联PTX、无下游手工修改，无maxnreg调参。",
        "",
        "## 实际采用的变换",
        "",
        "- 静态范围分支：1024/2048/3072/4096/6144/8192/10240/12288/14336/16384；其他正N保留动态分组循环。",
        "- 将8192宽累加器改为2048宽（部分小N使用1024宽），在分组之间累加后再规约。输出也按分组处理，缩短中间值存活。",
        "- mean加载显式使用TTIR `cacheModifier=ca`，与variance默认load区分，避免公共表达式合并把整行输入长时间保留。"
        "它是普通缓存load，不是volatile；实际PTX/SASS由原生后端生成。",
        "- Y store使用TTIR `cacheModifier=cs`，减少流式输出对缓存的干扰；沿用倒数复用与rsqrt。",
        "",
        f"资源：从parent的80 registers/thread、n_spills=2，变为 **{resource['n_regs']} registers/thread、"
        f"n_spills={resource['n_spills']}**，dynamic shared={resource['shared']} B。仍使用后端默认规约，不宣称达成LLIR的两次barrier。",
        "",
        "## 逐shape计时",
        "",
        "下表baseline/parent/new是同轮全量配对测量；LLIR是补充比较中的耗时，单列参考，不能跨测量直接作单点精确归因。",
        "",
        "| N | baseline µs | parent µs | new µs | new/base加速 | 独立复测加速 | LLIR参考 µs |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in new_rows:
        lines.append(
            f"| {r['N']} | {r['baseline_us']:.3f} | {old_rows[r['N']]['candidate_us']:.3f} | "
            f"{r['candidate_us']:.3f} | {r['speedup']:.3f}× | {rr[r['N']]['speedup']:.3f}× | {llir_times[r['N']]:.3f} |"
        )
    lines += [
        "",
        "## NCU证据（C级机制线索）",
        "",
        "`ncu_batch1` 与 `ncu_finalists` 均为 base clock、cache-control=all，采集 LaunchStats/Occupancy/SpeedOfLight/"
        "WarpStateStats/MemoryWorkloadAnalysis。每次被采集launch同时通过Y精度检查。NCU时长不替换graph benchmark。",
        "",
        "| N | 版本 | NCU µs | 实际occupancy | DRAM Tbyte/s |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in ncu:
        if row["kernel"] not in ["optimized", winner]:
            continue
        met = row["metrics"]
        lines.append(
            f"| {row['N']} | {'parent' if row['kernel'] == 'optimized' else 'new'} | "
            f"{float(met['gpu__time_duration.sum']['value']):.3f} | "
            f"{float(met['sm__warps_active.avg.pct_of_peak_sustained_active']['value']):.1f}% | "
            f"{float(met['dram__bytes.sum.per_second']['value']):.3f} |"
        )
    lines += [
        "",
        "低寄存器路径的实际occupancy从约33%~35%升至约55%~57%。中等N时加载依赖隐藏和吞吐改善，"
        "但最大N的吞吐/时长仍有回退，不能仅凭occupancy判定更快。第一批volatile重载在大N的long-scoreboard/issue比值显著增加，"
        "指导后续改用普通ca加载。该比值是平均stall/issue统计，不是可直接消除的耗时百分比。",
        "",
        "## 验证、负结果与测量有效性",
        "",
        "- baseline和14个可编译候选/parent在完整142用例上验证；累计记录包含重复baseline，详见accuracy_all.json。"
        "额外覆盖新增范围边界±16、padding=0/16、normal/offset分布，共120例/版本，见boundary_accuracy.json。",
        "- 最终版与baseline在120个新增分支边界/压力/padding用例上分别通过memcheck/racecheck；均0 errors，racecheck无warnings。",
        "- affine初版的arith.negf在当前TTIR文件入口lowering失败；以新候选名修复，未覆盖失败IR。动态小分组循环、"
        "全输入复用和fold候选存在性能退步，全部保留。16候选预算用完后停止扩展，非无限搜索。",
        "- 首次完整测量的N=1024出现baseline 31~68us大幅漂移，整轮排除并中止，原始partial数据保留。"
        "重跑采用所有版本相同的额外untimed graph预热(rep=100ms/版本)，独立复测同样预热。"
        "时钟未锁定，缓存未强制清理；报告为预分配重复CUDA graph kernel时间，不含编译、加载、CPU launch或分配。",
        "",
        "## 复现与证据",
        "",
        "候选生成：`make_round2.py`。原始结果位于`results/round2/`：candidates.json、accuracy_*.json、"
        "screen_*.json、full_stable.json、independent_repeat.json、llir_comparison.json、"
        "ncu_*.ncu-rep/csv/summary.json、memcheck.log、racecheck.log、selection.json。",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=7 bash layer_norm/test_performance.sh",
        "# 直接复现本轮候选，结果写到新的输出路径：",
        "/root/venvs/ir-ops-dev/bin/python layer_norm/make_round2.py",
        "CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python layer_norm/benchmark.py \\",
        f"  tile_dispatch_math_v2 {winner} --full --rounds 7 --rep 20 --stabilize-ms 100 \\",
        "  --validation layer_norm/results/round2/accuracy_finalists.json --output /root/tmp/new_round2_benchmark.json",
        "```",
        "",
    ]
    (ROOT / "report_round2.md").write_text("\n".join(lines))
    print(winner, s)


if __name__ == "__main__":
    main()
