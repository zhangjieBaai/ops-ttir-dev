"""Render measured outcomes from the bounded matched-configuration IR borrowing round."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/runs/mm_opt_r2_20261119"


def read(name):
    return json.loads((RUN / name).read_text())


def main():
    audit = read("reference_ir_audit.json")
    screen = read("screen.json")
    correction = read("correction_screen.json")
    primary = read("full.json")
    repeat = read("repeat.json")
    selection = read("selection.json")
    assert selection["status"] == "BASELINE_FALLBACK"
    assert all(
        x["valid"] for d in [screen, correction, primary, repeat] for x in d["records"]
    )
    stats = selection["metrics"]
    references = [
        "| 实现/编译器 | 入口与访存 | registers | shared bytes | PTX bulk-copy出现次数 | PTX普通cp.async出现次数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in audit["records"]:
        if row["shape"] != 1152:
            continue
        label = row["implementation"]
        entry = (
            "host descriptor / TMA"
            if label == "flaggems_flagtree"
            else "三指针 / 普通load+mask"
        )
        references.append(
            f"| {label} | {entry} | {row['n_regs']} | {row['metadata']['shared']} | "
            f"{row['ptx_static_counts']['cp.async.bulk']} | {row['ptx_static_counts']['cp.async.cg']} |"
        )
    checks = read("check/summary.json")["records"]
    correction_check = read("correction_check/summary.json")["records"]
    attempts = [
        "| 候选 | 改动 | 初筛几何均值 | registers | spill |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "r2_pid_direct": "借用pid%group_size行排列",
        "r2_zero_pad": "原始连续索引+M/N/K边界zero mask",
        "r2_absolute": "k计算本轮地址，仅携带accumulator",
        "r2_zero_absolute": "zero mask+absolute",
        "r2_zero_pid": "zero mask+pid",
        "r2_absolute_pid": "absolute+pid",
        "r2_zero_absolute_pid": "三项组合",
        "r2_k_offset": "保留指针递增，只将K循环改为实际offset步进64",
    }
    for name, label in labels.items():
        data = correction if name == "r2_k_offset" else screen
        check = correction_check if name == "r2_k_offset" else checks
        a = check[name]["artifacts"]
        attempts.append(
            f"| {name} | {label} | {data['summary'][name]['geometric_mean']:.5f}× | {a['n_regs']} | {a['n_spills']} |"
        )
    full = [
        "| M=N=K | baseline主测µs | candidate主测µs | 主测配对比值 | 独立复测 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in primary["records"]:
        if row["name"] != "r2_k_offset":
            continue
        other = next(
            r
            for r in repeat["records"]
            if r["name"] == "r2_k_offset" and r["shape"] == row["shape"]
        )
        full.append(
            f"| {row['shape'][0]} | {row['baseline_us']:.3f} | {row['candidate_us']:.3f} | "
            f"{row['speedup']:.5f}× | {other['speedup']:.5f}× |"
        )
    count = sum(
        len(row["accuracy"]["cases"])
        for name in [
            "check/summary.json",
            "correction_check/summary.json",
            "repeat_check/summary.json",
        ]
        for row in read(name)["records"].values()
    )
    mem, race = read("memcheck_accuracy.json"), read("racecheck_accuracy.json")
    assert mem["passed"] and race["passed"] and read("dispatch.json")["passed"]
    toolcount = sum(len(row["cases"]) for row in mem["records"].values())
    probe = read("descriptor_probe.json")
    acc = read("acc_buffer_probe.json")
    assert (
        probe["compiled"]
        and probe["launched"] is False
        and probe["requires_global_scratch"] == 384
    )
    assert (
        acc["compiled"] and not acc["launched"] and acc["same_executable_as_baseline"]
    )
    geo_main = stats["main"]["geometric_mean"]
    geo_repeat = stats["repeat"]["geometric_mean"]
    text = f"""# MM 第二轮：借鉴FlagGems IR，未获得可接受的TTIR收益

已对比1152³、1280³匹配配置下的实际FlagGems/FlagTree与教程IR，
并在官方Triton3.7.0环境直接修改本组TTIR尝试了8个功能候选。
**本轮不晋升新版本，继续使用原baseline。**
最好候选r2_k_offset在本组全部6点主测/复测几何均值为 **{geo_main:.5f}× / {geo_repeat:.5f}×**，
没有达到冻结的>=1.02条件。
原round1其他组收益与公共入口保留不变。


## 相同配置下，实际IR差在哪

配置均为BM64、BN128、BK64、GROUP8、warps4、stages4、CTAs1。
但相同参数不意味着相同访存算法或ABI。
1152和1280在每个实现内确实共用相同cubin；
输入产物hash均重新核对，
见 [reference_ir_audit.json](results/runs/mm_opt_r2_20261119/reference_ir_audit.json)。


{chr(10).join(references)}

FlagGems TTIR使用`tt.descriptor_load/store`，
TTGIR为`ttng.async_tma_copy_global_to_local`；
教程为原始指针的`tt.load/store`和`ttg.async_copy_global_to_local`，
并带逐元素K边界mask和行列取模。
两者均使用WGMMA，
不是“一个用Tensor Core另一个不用”。
FlagTree教程是126 registers，
upstream教程124，
FG90；
FG与upstream shared仅相差32 bytes，
却有不同地址生成、load、同步和入口结构。


静态PTX普通cp.async的48处与bulk-copy的11处是**文本出现次数**，
含流水线不同位置，
不是动态执行次数，
也不是访存字节或性能因果证明。
`ttgir_comparison.json`记录对应op；
upstream教程另有2处warp_group_dot_wait，
FG/FlagTree1处。
不能直接编辑TTGIR/LLVM/PTX来复制这些差异，
本轮只允许从TTIR原生lowering产生变化。


## 能迁移并验证的改动

1. **program ID映射**：FG使用`group_base + pid%group_size`，
教程先`pid%group_width`再`%group_size`。
每个列chunk有group_size个连续pid，
两式可产生同组行的不同排列，
但仍完整覆盖同一tile集合；
grid规则不变。
额外测试M704/N256/K80、padding0/16，
其非初始三行partial group从pid16开始，
覆盖非零行旋转，
避免只测size1尾组的假覆盖。

2. **descriptor式边界零填充的普通load模拟**：取消A行/B列取模，
使用原始连续索引并将M/N有效性与原K load mask组合，
无效元素填0，
原store mask保留。
合法输出的dot不变；
实际mask合法性通过精度、输出canary和memcheck核查。
这是软件模拟，
并没有变成TMA。

3. **accumulator-only循环**：用k计算本轮A/B指针，
只携带accumulator，
借鉴FG循环结构。
普通指针路径中该改写增加了地址工作/活跃状态，
实际结果回退；
这只是结构与计时观测，
不把每一项成本当作已消融因果。

4. **有依据的一项修正**：保留原指针递增，
将`k=0..ceil(K/64), offset=k*64`改为`offset=0..K, step64`，
相同dot次数/顺序及K尾mask。
它仅有很小局部变化，
没有整组有效收益。


{chr(10).join(attempts)}

以上初筛均为4点768/1024/1152/1280、3轮、rep20ms配对几何均值（B级）。
候选raw hash、完整options/metadata、正确性和可执行section hash均保存在check/。
zero_pad虽然从124降到120 registers，
却约0.895×；
再次说明资源下降不能替代计时。
其余组合没有挽回这个回退，
未继续重复同类变体。


## 为什么没有直接使用TMA

FG的host descriptor作为入口参数，
和冻结的三指针+六标量ABI不同，
不能直接拷贝它的TTIR作为候选。
本轮另手写了一个**保持9参数入口、在设备内创建descriptor**的编译诊断：[device_descriptor_probe.ttir](results/runs/mm_opt_r2_20261119/device_descriptor_probe.ttir)。
官方3.7.0确实能编译为TMA路径，
90 registers，
但metadata要求 **每CTA384 bytes global_scratch**，
用于3个descriptor；
原baseline为0。


该诊断**没有启动、没有精度验证或性能测量，
不属于功能候选或排名**。
原生driver会按该metadata向runtime allocator申请scratch，
默认NullAllocator提示必须设置allocator。
本轮没有新增allocator或工作区分配。
实验要求第6节规定改变工作区约定必须另建实验，
因此没有将其混入当前固定workspace的“只改TTIR”收益。
这不证明TTIR无法表达TMA，
也不证明device descriptor能复现host descriptor的收益；
它还会引入每CTA描述符构造成本，
需在另组验证。


另检查了accumulator buffering提示：`tt.disallow_acc_multi_buffer`编译探针与原baseline的**可执行section完全相同**，
仍124 registers、98304 shared。
只作静态诊断，
未启动或计时；
不能凭TTGIR中wait数量不同就宣布这个提示能减少等待。


## 完整本组复核和验收

冻结finalist r2_k_offset后测本组6点（768/896/1024/1152/1280/1408），
主测7轮、独立进程seed123复测5轮，
rep20ms，
随机对内顺序，
预分配输出、同公共launcher和stream路径。
两轮所有要求点及A/A控制均有效，
没有删除回退点。
条件95%bootstrap区间分别为 **[{stats['main']['ci95'][0]:.6f},{stats['main']['ci95'][1]:.6f}] /
[{stats['repeat']['ci95'][0]:.6f},{stats['repeat']['ci95'][1]:.6f}]**；
只覆盖保存轮次波动，
不覆盖选择偏差或所有系统误差。


{chr(10).join(full)}

1152/1280最终只观测约0.5%的局部变化，
768/896/1024/1408有回退；
不按shape挑候选或降低验收门槛。
896/1408本轮未参与初筛，
但历史已暴露，
不称全新独立holdout。
选择结果为 [selection.json](results/runs/mm_opt_r2_20261119/selection.json) 的BASELINE_FALLBACK。


本轮累计 **{count}次功能kernel精度检查通过**（包括baseline和A/A重复，
不等于独立shape数），
8个功能候选均通过初验，
finalist另用seed123验证。
覆盖组全部性能点、partial-group、tile/K边界、padding与数值压力；
native FP16/禁TF32 FP32，
小尺寸CPU FP64，
atol=.01、rtol=.001，
输出NaN sentinel、padding canary、有限性和输入不变检查。


memcheck/racecheck分别覆盖baseline、pid、zero_pad、absolute和k_offset **5版本×17用例={toolcount}项/工具**，
均0 errors，
racecheck0 hazards/0 warnings。
组合候选仅做精度初验，
未单独跑sanitizer；
它们未被接受。
现有最终公共入口另外执行全部31调用、实际TTIR/cubin对应和未映射输入拒绝检查通过，
其他5组无新性能结论。


## 环境、证据和复现

主实验仅指定venv、官方Triton3.7.0、H100 SM90、物理GPU1；
FlagTree只提供旧产物参考，
本轮未加载其编译器。
每组baseline与候选通过同一个未修改的`common.compile_ir`与`common.launch`，
完整backend options、target、9参数签名及scratch约定匹配。
fresh pipeline对全部31点源码/文件的TTIR、LLVM/PTX/cubin、TTGIR归一化和输出逐位一致核查通过。
无锁频/清cache或连续独占声称，
同卡阶段不并行运行其他GPU工作。
未修改基线、配置映射、旧r1选择或别名。


证据目录：[mm_opt_r2_20261119](results/runs/mm_opt_r2_20261119/)。
计划见[optimization_round2.md](optimization_round2.md)，
直接TTIR生成器为[make_round2.py](make_round2.py)和[make_round2_correction.py](make_round2_correction.py)，
验证/计时为[experiment_round2.py](experiment_round2.py)，
选择复核为[finalize_round2.py](finalize_round2.py)。
正式报告复算脚本为[report_round2.py](report_round2.py)。
全部负结果保留。
独立KernelForge计划审查已批准，
具体partial-group建议已落实；
最终close审查以 `/root/KernelForge-workspace/runs/mm-ttir-borrow-r2/advisors/` 为状态来源，
不宣称框架CLI qualification。


```bash
# 只复算已有证据与选择，不启动GPU kernel
/root/venvs/ir-ops-dev/bin/python mm/finalize_round2.py
/root/venvs/ir-ops-dev/bin/python mm/report_round2.py
# 先确认GPU空闲；完整复验写新结果目录，不重写旧候选
CUDA_VISIBLE_DEVICES=1 bash mm/test_round2.sh
```
"""
    (ROOT / "report_round2.md").write_text(text)
    print("Rendered scoped negative result; baseline and round1 choice remain active")


if __name__ == "__main__":
    main()
