# 五类算子连续三轮 TTIR 优化（r3–r5）

> 后续已完成[第六轮TTIR优化](round6_report.md)，新增MoE、Attention D128与Top-K收益。本文保留r3–r5结束时的历史结果。

报告更新：2026-11-24。第二轮之后继续完成三个顺序搜索轮，每轮均覆盖原有152个唯一性能点：INT4、MoE、Norm、Top-K各30点，Attention32点。三轮共456次shape评估，不是456个不同shape。

本次新增晋升为INT4的GROUP128常量特化、MoE的top2归约与整数常量组合、Top-K的合同特化。Norm与Attention的三轮新候选未达门槛，最终保留第二轮版本。

## 最终有效版本

所有数字均来自最后一轮同场计时：baseline为原始实现，anchor为第二轮验收实现。下表是各自固定矩阵内A级性能证据；原始baseline／最终版为累计收益，anchor／最终版为本次三轮之后的净增量。没有相乘历史加速比。

| 算子 | 点数 | 最终版本 | 相对第二轮：主测／复测 | 相对原始baseline：主测／复测 |
|---|---:|---|---:|---:|
| [INT4 反量化＋GEMM](../int4_gemm/multiround/report.md) | 30 | r3/group128 | 1.030287×／1.030268× | 2.518282×／2.517380× |
| [Grouped／biased MoE](../moe_routing/multiround/report.md) | 30 | r5/pair_integer_constants | 1.072442×／1.071690× | 1.507532×／1.507741× |
| [Norm＋动态 INT8](../norm_quant/multiround/report.md) | 30 | 第二轮版本 | 1.000044×／1.000204× | 1.433824×／1.435725× |
| [Attention dK／dV](../attention_dkdv/multiround/report.md) | 32 | 保留第二轮分组选择 | 1.000197×／1.000010× | 1.759292×／1.759590× |
| [大 N partial Top-K](../partial_topk/multiround/report.md) | 30 | r5/contract | 1.022419×／1.022347× | 1.232721×／1.232306× |

保留parent时使用本轮实际测得的parent耗时，接近1的anchor比值是重复计时波动，不是新收益。Attention完整32点包含D64和D128；独立复测相对原baseline分别为D64的1.126879×（16点）和D128的2.747550×（16点）。这些结果均在SM1200MHz下测得，不能与旧自动频率实验的绝对时间混用。

## 每轮实际尝试与决策

以下为冻结进入正式确认的候选相对当轮parent的直接增量，负结果也保留。只有主测和独立复测均满足几何均值≥1.02、最差点≥0.95，并通过数值、A/A和sanitizer，才替换parent。

| 算子／组 | 轮次 | 正式候选 | 主测增量 | 复测增量 | 决策 |
|---|---|---|---:|---:|---|
| INT4 反量化＋GEMM | r3 | `group128` | 1.030644× | 1.030233× | 接受 |
| INT4 反量化＋GEMM | r4 | `pid_g4` | 0.994102× | 0.994036× | 保留parent |
| INT4 反量化＋GEMM | r5 | `preload_pair` | 0.988765× | 0.988884× | 保留parent |
| Grouped／biased MoE | r3 | `unroll_experts` | 1.001101× | 1.000858× | 保留parent |
| Grouped／biased MoE | r4 | `pair_top2_v2` | 1.018584× | 1.019415× | 保留parent |
| Grouped／biased MoE | r5 | `pair_integer_constants` | 1.072454× | 1.071772× | 接受 |
| Norm＋动态 INT8 | r3 | `gain_cancel` | 1.004073× | 1.004587× | 保留parent |
| Norm＋动态 INT8 | r4 | `joint_uint` | 0.980665× | 0.981627× | 保留parent |
| Norm＋动态 INT8 | r5 | `stream1024_large` | 0.951529× | 0.951581× | 保留parent |
| Attention dK／dV / d64_c0 | r3 | `r3_structure` | 0.931192× | 0.932382× | 保留parent |
| Attention dK／dV / d64_c1 | r3 | `r3_structure` | 0.956264× | 0.955100× | 保留parent |
| Attention dK／dV / d128_c0 | r3 | `r3_structure` | 0.991912× | 0.992097× | 保留parent |
| Attention dK／dV / d128_c1 | r3 | `r3_structure` | 1.007681× | 1.007960× | 保留parent |
| Attention dK／dV / d64_c0 | r4 | `r4_qdo_cg` | 0.999867× | 1.000224× | 保留parent |
| Attention dK／dV / d64_c1 | r4 | `r4_delta_early` | 1.001891× | 1.002060× | 保留parent |
| Attention dK／dV / d128_c0 | r4 | `r4_qdo_cg` | 0.999923× | 1.000965× | 保留parent |
| Attention dK／dV / d128_c1 | r4 | `r4_delta_early` | 1.000954× | 1.001738× | 保留parent |
| Attention dK／dV / d64_c0 | r5 | `r5_store_cs` | 0.999846× | 1.000402× | 保留parent |
| Attention dK／dV / d64_c1 | r5 | `r5_store_cs` | 1.000391× | 1.000146× | 保留parent |
| Attention dK／dV / d128_c0 | r5 | `r5_qdo_keep` | 0.999923× | 1.000189× | 保留parent |
| Attention dK／dV / d128_c1 | r5 | `r5_store_cs` | 1.000301× | 1.000388× | 保留parent |
| 大 N partial Top-K | r3 | `hier8` | 0.804095× | 0.803950× | 保留parent |
| 大 N partial Top-K | r4 | `store2` | 0.967804× | 0.967631× | 保留parent |
| 大 N partial Top-K | r5 | `contract` | 1.022228× | 1.022150× | 接受 |

## 优化了什么，Triton能否表达

| 算子 | 三轮探索与结果 | 公开Triton表达方式 |
|---|---|---|
| INT4 | r3把既定GROUP128的非负除法改为右移7位并晋升；r4的PID分组、r5的预加载／两group展开均未进一步获益 | 常量或`tl.constexpr`、整数右移、`tl.program_id`重排、循环与加载顺序 |
| MoE | r3的gather属性／选择展开不获益；r4单独tuple top2复测1.0194×未达门槛；r5将其与固定G、topK、stride等整数常量组合后晋升 | `tl.reduce`自定义双值组合器、`tl.constexpr`或已知常量。未晋升的`efficient_layout`属性在3.7的`tl.gather`无直接公开参数 |
| Norm | r3合并或消去量化gain；r4混合sum／uint max归约；r5重载或1024分块输出，均未达门槛 | 标量算术、`tl.reduce`、bitcast、循环和`tl.load` |
| Attention | r3组合展开／指针重算或查询迭代；r4调整Q/dO缓存和delta读取；r5调整存储／驱逐提示，均未晋升 | 循环、表达式次序、`tl.load`／`tl.store`缓存与驱逐参数 |
| Top-K | r3分层uint32／uint64选择、r4批量store退步；r5在有限值、固定grid及完整块条件下删除恒真mask／耗尽判断后晋升，尾块保护保留 | `tl.constexpr`分支、逻辑索引、`tl.load` mask、归约与reshape |

上述可表达性是结构与API对应关系，不是新的源码性能实验。第一轮41点源码回写曾测得与TTIR相同的可执行ELF `.text`；第二轮及r3–r5的新改动尚未做源码回写计时，不能直接继承其机器指令或性能相等结论。

## 有价值的负结果与限制

MoE的单独top2归约没有达到晋升门槛，第三轮将它与常量特化重新组合后才获得稳定收益；这是新的组合实测。INT4消除变量除法时寄存器与spill也改变，不能把收益全部归因于一条指令。

Attention查询展开增加活跃状态，部分组spill和shared显著上升并回退；Q/dO的`.cg`以及`.ca+evict_last`候选虽然改变上层属性，四组最终可执行`.text`仍与parent相同。这些是本编译器与配置下的静态证据，不代表缓存提示普遍无效，也没有证明实际occupancy或动态瓶颈。

Norm没有因更少的算术或更窄的输出循环而获得整体提升；Top-K分层归约与批量存储也慢于parent。因此只保留通过完整矩阵门槛的版本，不按复测结果事后挑选单shape赢家。

## 验证与复现范围

官方Triton3.7.0／H100，保持第二轮dtype、固定编译配置、ABI、grid、kernel数与workspace，未新增autotune或pipeline-depth调参。每轮先做正确性和原始源码／文件baseline的LLVM、PTX、cubin一致性检查，再初筛、冻结选择、主测7轮、独立进程复测5轮。

正式计时为预分配CUDA graph设备时间，随机交错同输入配对，`rep=20ms`为计时库参数，保留外层全部样本；每点A/A中位要求在[0.97,1.03]。Attention等条件预热并锁SM1200MHz，结束后已恢复自动频率；其他四类不主动锁频。

最终新增晋升均有memcheck／racecheck零错误证据。INT4、Norm、Top-K、Attention三轮正式候选均有新sanitizer记录；MoE仅晋升的r5新增sanitizer，r3/r4未晋升候选没有新sanitizer，保留的parent沿用已有资格。Norm／Top-K的sanitizer覆盖30个映射，为控制开销使用缩减行数；不是每个压力分布都验证原始完整M。具体数值合同、覆盖范围与哈希见分报告。

数值合同各不相同：Norm沿用量化code误差、scale／重构误差及输出范围检查；Top-K要求稳定索引和值数值相等，±0等价。未宣称二者所有输出均保持原始位型。

[共同协议](multiround_plan.md) · [机器可读汇总](multiround_summary.json) · [第二轮152点报告](round2_report.md) · [第一轮源码回写实测](source_port_comparison.md)

复现汇总：`/root/venvs/ir-ops-dev/bin/python complex_suite/multiround_report.py`。各算子报告链接给出原始配对样本、选择与IR哈希、候选生成器及GPU复现命令。
