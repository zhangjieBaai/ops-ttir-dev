# 五类算子第六轮TTIR优化

[全部152个shape的加速比](round6_per_shape.md) · [CSV表格](round6_per_shape.csv)

更新：2026-11-24。在r3–r5完成后继续进行新一轮结构搜索，仍覆盖152个唯一shape（四类各30、Attention32）。本轮从r5最终验收版本出发，只改变TTIR，不改变dtype、编译配置、ABI、grid、kernel数或workspace。

## 最终有效路线的直接实测

parent指r5最终有效选择，anchor指第二轮验收选择，baseline指原始冻结实现。下面的主测／独立复测均来自第六轮同场随机配对，未相乘历史加速比。

| 算子 | 点数 | 相对r5 parent：主测／复测 | 相对原始baseline：主测／复测 |
|---|---:|---:|---:|
| [INT4反量化＋GEMM](../int4_gemm/multiround/r6/report.md) | 30 | 保留parent | 2.519096×／2.518154× |
| [Grouped／biased MoE](../moe_routing/multiround/r6/report.md) | 30 | 1.023423×／1.023135× | 1.542143×／1.543441× |
| [Norm＋动态INT8](../norm_quant/multiround/r6/report.md) | 30 | 保留parent | 1.434825×／1.437892× |
| [Attention dK／dV](../attention_dkdv/multiround/r6/report.md) | 32 | 1.023446×／1.023624× | 1.799777×／1.799615× |
| [大N partial Top-K](../partial_topk/multiround/r6/report.md) | 30 | 1.034961×／1.034752× | 1.275513×／1.275095× |

没有晋升时使用本轮实测parent时间计算累计收益；相同代码在不同测量轮次的微小波动不算新增收益。Attention仅在原D/causal分组上决定晋升，全32点平均包含D64的未改路径，具体如下。

| Attention组 | 正式候选 | 主测增量 | 复测增量 | 决策 |
|---|---|---:|---:|---|
| d64_c0 | r6_fission_v_then_k_v2 | 0.999938× | 0.999036× | 保留parent |
| d64_c1 | r6_fission_v_then_k_v2 | 1.000145× | 1.000039× | 保留parent |
| d128_c0 | r6_dv_late | 1.051183× | 1.051661× | 接受 |
| d128_c1 | r6_dv_late | 1.043713× | 1.043967× | 接受 |

D64表中的候选名称对应与parent字节相同的控制文件，没有对D64执行拆分遍历。D128合计16点的复测增量为1.047807×，累计为2.874342×；D64的16点累计为1.126732×。

Attention本轮仍临时固定SM1200MHz，结束后恢复自动频率；其余四类未主动锁频。不能将自动频率历史绝对耗时与本轮锁频绝对耗时混用。

## 本轮改动与发现

| 算子 | 新方向 | 结果解释与表达方式 |
|---|---|---|
| INT4 | uint8先拼接再FP32反量化；以整数位构造精确0..15浮点值；组合 | 精度通过但整体回退，保留parent。公开Triton join/trans/reshape、cast/bitcast和整数运算可表达 |
| MoE | 同一活动CTA批量处理2/4/8个连续组的stage1 top2，stage2保持parent | 保持原M×8 grid，非组首CTA返回；公开Triton分支、二维张量和axis1 tuple reduce可表达。工作分配变化不是实际occupancy的测量证据 |
| Norm | 直接最近偶数舍入至i32，整数clamp；联合统计分4组 | 近似持平或回退，保留parent。libdevice float2int_rn有公开Triton CUDA接口；分组reduce和整数clamp有公开语法 |
| Attention | 在已接受64-key结构上后移dV；或将dV和dK分两遍查询计算 | 两遍计算重复QK、概率及加载，减少部分spill却整体回退；后移dV在新parent上重新检验。公开表达式顺序与循环可表达 |
| Top-K | merge仅保存每个局部有序列表当前头部和cursor；比较uint32两次归约与packed uint64稳定选择 | local保持原样，推进胜出列表后masked load下一项；公开Triton可表达，以更小的merge活跃状态换取额外依赖加载 |

这些是新的结构实验：Attention第一轮曾在128-key基线上测试dV后移，本轮是在第二轮形成的64-key parent上重新验证；Top-K利用已有局部排序约束缩小merge状态，不是改变kernel数或用完整排序替代partial Top-K。

## 资格、负结果与复现

完整矩阵先做原始源码／文件baseline LLVM、PTX、cubin和完整options核验，再做数值检查。初筛3轮rep10，冻结候选后主测7轮rep20、独立进程seed123复测5轮rep20，A/A中位要求[0.97,1.03]。晋升要求两次增量几何均值≥1.02、最差点≥0.95，并通过sanitizer。

所有新增晋升均有memcheck零错误和racecheck零错误／零警告证据。Norm／Top-K正常分布使用完整M验证，其他压力分布及sanitizer使用min(M,3)／min(M,2)行；覆盖30映射，不把缩减行数当作每个完整M均在sanitizer下执行。Top-K保持稳定id与数值exact，±0等价；Norm沿用code误差≤1及scale／重构误差合同。

解析失败版本与初筛负结果均保留。Top-K packed初版SSA命名冲突在v2修复，选拔使用完整screen_v2。Attention fission的解析修正与无收益记录见分报告。仅主/复测确认结果属于固定矩阵A级性能证据，静态寄存器、spill、shared及.text身份属于C级，不能单独证明动态性能原因。

**r2–r6新增结构尚未做Triton源码回写实测。** 可表达性不等于已验证源码性能或机器指令相同；第一轮41点源码回写的结论仅适用于当时版本。

[预声明计划](round6_plan.md) · [机器可读汇总](round6_summary.json) · [r3–r5历史报告](multiround_report.md) · [第一轮源码回写](source_port_comparison.md)

复现汇总：`/root/venvs/ir-ops-dev/bin/python complex_suite/round6_report.py`。各算子报告和workflow记录包含完整测量命令、原始样本、每shape选择及哈希。
