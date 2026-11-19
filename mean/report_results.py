"""Derive the report and acceptance record solely from saved experiment receipts."""

import json

from benchmark import summary
from common import ROOT, save_json, sha256

run = ROOT / "results/runs/final_gpu6"
main = json.loads((run / "main.json").read_text())
repeat = json.loads((run / "repeat.json").read_text())
accuracy = json.loads((run / "accuracy.json").read_text())
pipeline = json.loads((run / "pipeline.json").read_text())
checks = {
    name: json.loads((run / f"{name}.json").read_text())["passed"]
    for name in ["memcheck", "racecheck"]
}
for tool, marker in [
    ("memcheck", "ERROR SUMMARY: 0 errors"),
    ("racecheck", "RACECHECK SUMMARY: 0 hazards"),
]:
    checks[tool] = checks[tool] and marker in (run / f"{tool}.log").read_text()
checks.update(
    accuracy=accuracy["passed"],
    pipeline=pipeline["passed"],
    main_aa=main["aa_passed"],
    repeat_aa=repeat["aa_passed"],
)
assert main["physical_uuid"] == repeat["physical_uuid"] == pipeline["physical_uuid"]
rows = main["records"]
rep = {tuple(r["shape"]): r for r in repeat["records"]}

paths = {}
for label, pred in [
    ("K=1", lambda k: k == 1),
    ("1<K<1024", lambda k: 1 < k < 1024),
    ("K>=1024", lambda k: k >= 1024),
]:
    paths[label] = dict(
        main=summary([r for r in rows if pred(r["shape"][2])]),
        repeat=summary([r for r in repeat["records"] if pred(r["shape"][2])]),
    )
s = main["summaries"]["optimized"]
t = repeat["summaries"]["optimized"]
lines = [
    f"""# mean_dim TTIR 首轮结果

30 shape 主测几何平均 **{s['geometric']:.4f}×**，独立进程复测 **{t['geometric']:.4f}×**。主测95%配对轮次 bootstrap 区间
{s['bootstrap_95']}；复测 {t['bootstrap_95']}。这些区间仅描述固定 shape 已保存轮次的波动，不代表其他负载。A/A 控制主测 {main['aa_passed']}、
复测 {repeat['aa_passed']}。最差点 {s['worst']:.4f}×，最快点 {s['best']:.4f}×。证据等级：若全部验收项通过，则为 A，范围仅本轮 FP16
冻结矩阵。

本轮在 `/root/ops-ttir-dev/mean` 完成，未修改 FlagGems 的 mean.py。基线采用用户指定本地实现，源码 commit 及哈希见 upstream/provenance.json；
不声称它与用户链接的历史 commit 完全相同。覆盖连续轴、Non-inner Heuristic 和 K≥1024 的串行向量累积三条路径。性能集30点，边界正确性12点；26个不可变 TTIR/完整配置组。
输入为连续 FP16 [M,N,K]，沿 N 取均值，FP32累积，FP16输出。

## 优化与负结果

最终选择 `r1_tile1`，每组 optimized.ttir 与该候选哈希一致。K≥1024 将逻辑128×8向量状态改写为1×1024、32位索引 TTIR；保留逐 N 循环累积和归约后除法。
K tile=1024、grid、8 warps、signature及完整编译 options均保持基线契约。前两条路径直接保留 baseline 字节。

四个候选完全由 make_candidates.py 直接生成 TTIR，没有修改 Python kernel 后导出。候选为1/4/16/32行的二维 tile：4/16/32使用分行独立累积，
循环后沿N维归约，改变浮点加法顺序。`verify_candidates.py`逐组验证可重现。tile1是扁平化与索引窄化的组合；没有独立消融，不能把全部收益归因于32位地址运算。PTX和资源可用于继续诊断，
不能代替性能实测。

初筛仅使用偶数编号的15点，每点3轮，全部A/A通过。tile1几何1.1249×、tile4 0.9933×、tile16 1.0269×、tile32 1.0553×，均为B级初筛。N=1时多行tile约0.50×，
因此未选。没有按单点结果拼装最优映射。奇数编号15点直到finalist冻结后才测候选性能；所有42点均在初筛前暴露于正确性验证，不称作完全独立的holdout。预算内完成四个候选后已有稳定候选，
顺序展开/cache提示方向未运行。

## 验证和计时

源码与TTIR文件入口在全部42点逐一核查：冻结TTIR相同，正规化TTGIR相同，LLVM/PTX/cubin字节相同，输出逐位相同。最初含行号导出在K=1025出现LLVM调试位置不同、PTX/cubin相同；
已保留原件，统一设置TRITON_DISABLE_LINE_INFO=1重新冻结，再开始优化。

所有被测kernel从TTIR文件经同一triton.compile原生后端得到CompiledKernel，同一launcher启动，没有独立llc、替换binary、改变grid或工作区。
每次验证、计时按冻结shape映射选组，未映射输入拒绝。环境为指定venv官方Triton3.7.0，RECORD/路径/模块来源及libtriton、backend、ptxas哈希见pipeline.json。

最终主测与复测GPU UUID：`{main['physical_uuid']}`。主测7轮、复测5轮、rep=20ms，预分配输出、CUDA graph replay，随机配对顺序，保存全部样本、
逐点产物及遥测。指标不包含分配、编译、加载或Python提交。未锁定时钟、未清缓存，不称为冷缓存。GPU7两次正式计时因其他进程中止，见final/和final_clean/；这些数据全部排除。
初筛在GPU7，最终比较和独立复测在GPU6，绝不混合两卡样本计算加速比。

最终基线和候选各42 shape×8分布，共672项检查；CPU FP64均值转FP16及native ATen对照，atol=rtol=0.002。
涵盖normal/zero/constant/offset/alternating/outlier/-Inf/NaN，
输入不变、输出sentinel与两侧canary检查。主测前seed42通过，GPU6 seed99也通过。所有候选初始验证共1680项。最终memcheck/racecheck各42 shape×2分布×2版本=168项，
状态见下。

验收项：{checks}

## 分路径结果

| 路径 | 点数 | 主测几何 | 复测几何 |
|---|---:|---:|---:|"""
]
for label, value in paths.items():
    count = sum(
        (
            r["shape"][2] == 1
            if label == "K=1"
            else (
                1 < r["shape"][2] < 1024
                if label == "1<K<1024"
                else r["shape"][2] >= 1024
            )
        )
        for r in rows
    )
    lines.append(
        f"| {label} | {count} | {value['main']['geometric']:.4f}× | {value['repeat']['geometric']:.4f}× |"
    )
lines += [
    "\n## 完整30点\n",
    "| M,N,K | baseline µs | optimized µs | 主测加速 | 复测加速 |",
    "|---|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {','.join(map(str, r['shape']))} | {r['baseline_us']:.3f} | {r['candidate_us']:.3f} | "
        f"{r['median_speedup']:.4f}× | {rep[tuple(r['shape'])]['median_speedup']:.4f}× |"
    )
lines += [
    """
## 资源与范围

向量路径7组的baseline→optimized寄存器为12→10、32→28、30→28、30→28、32→32、30→28、32→32；全部0 spill/0 shared。不把资源下降视为加速原因已被证明。
完整分组资源、TTIR和cubin哈希在pipeline.json和每次benchmark artifacts中。

32位索引合法性只针对冻结42点：输入元素数及尾tile最大被计算偏移均远小于2^31；正尺寸、非负stride、连续且对齐、无别名。不能外推到大于2^31的张量、非连续输入、FP32/BF16、
空归约或多个归约轴。此轮主要研究K≥1024；前两条路径未优化，接近1×的微小波动不算算法收益。未穷举配置，也未证明胜过任意新autotune实现。

原始证据：results/runs/final_gpu6/{pipeline,accuracy,main,repeat,memcheck,racecheck}.json；负结果results/runs/screen.json。
复现入口见README.md。
"""
]
(ROOT / "report_round1.md").write_text("\n".join(lines))
save_json(
    ROOT / "results/acceptance.json",
    dict(
        checks=checks,
        accepted=all(checks.values()),
        main=s,
        repeat=t,
        by_path=paths,
        evidence={p.name: sha256(p) for p in run.glob("*.json")},
        report_sha256=sha256(ROOT / "report_round1.md"),
    ),
)
