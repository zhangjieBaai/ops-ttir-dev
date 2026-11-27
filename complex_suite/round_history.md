# 复杂融合算子 TTIR 扩展实验（2026-11-23）

按用户文章的五类建议，新增 INT4 groupwise反量化GEMM、完整grouped/biased MoE routing、Residual RMSNorm+动态INT8、Attention backward dK/dV、大N partial Top-K。使用官方Triton3.7.0，在5张独占H100上并行实验；所有候选直接修改冻结TTIR，baseline/候选共用文件入口编译、配置和launcher。未修改FlagGems生产算子。

下表是各自冻结矩阵的**A级性能证据**：主测7轮、独立进程复测5轮，每轮20ms，预分配CUDA graph设备时间。每个算子的矩阵、baseline强度和测量边界不同，不计算跨算子总平均。

| 任务 | 最终候选/范围 | 主测几何均值 | 独立复测 | 报告 |
|---|---|---:|---:|---|
| INT4反量化+GEMM | scalar_unsigned_v1，7点 | 1.0948× | 1.0958× | [INT4](../int4_gemm/report_round1.md) |
| Grouped/biased MoE routing | membership，8点 | 1.1502× | 1.1502× | [MoE](../moe_routing/report_round1.md) |
| Residual RMSNorm+动态INT8 | stream_v1，8点 | 1.0548× | 1.0521× | [Norm+量化](../norm_quant/report_round1.md) |
| Attention backward dK/dV | r1_dv_late，D64的6点 | 1.1241× | 1.1214× | [Attention](../attention_dkdv/report_round1.md) |
| 大N partial Top-K | ordered_carry_v3，6点 | 1.0827× | 1.0816× | [Top-K](../partial_topk/report_round1.md) |

Attention共测12点，D128未获得稳定收益，保留baseline；表中D64数值不能冒充全12点收益。Norm最差点复测0.9859×，保留该小幅回退，未用逐shape择优抹去。MoE的4096×512点约1.379×；INT4文章代表点M16/K4096/N4096约1.149×/1.147×，M128/K4096/N11008约1.095×/1.095×。完整逐点时间在分报告。

## 与文章假设对应的结果

- **INT4**：同byte low/high的group/scale/zero共享、对齐K tile内group标量化及有值域证明的unsigned索引组合获益。cast挪到join/reshape之后的候选与baseline机器指令相同，没有收益；没有据源码注释虚构编译缺陷。
- **MoE**：把[E,G] membership比较+归约换成连续分组broadcast/reshape，显著缩小中间状态。联合argmax虽然减少静态barrier，却在大batch回退；保留负结果。
- **Norm+INT8**：利用正rho将amax(y)分解，组合sum/max统计，再测试输出cache hint。合同在实验前允许量化值相对FP64至多差1 code，并检查scale和重构误差，**不是逐位INT8等价**；FP8未测。
- **Attention**：将dV消费概率的cast/dot移到dK之后，在D64有效，D128因果反而变慢。结果仅包括已有LSE/delta、预缩放K后的dK/dV，不能当作完整backward收益。
- **Top-K**：联合value/index归约初筛显著回退；有序整数max逐轮编码仍回退。把有序key作为循环SSA状态、仅在循环外编码后获得收益。PTX出现redux.sync.max.u32是C级静态事实；不是单靠reduce数量减少。两kernel/工作区/配置保持一致。

## 验证与解释边界

五条线均完成源码→TTIR文件入口LLVM/PTX/cubin一致性、参考数值检查、配对性能、独立进程复测和最终memcheck/racecheck（0 errors，racecheck 0 warnings）。Top-K indices严格一致、values数值一致，±0等价；覆盖tie、尾块、正负subnormal与最低有限值。输入契约、IR/编译产物哈希、资源、原始配对样本和失败候选在各目录。

INT4/MoE使用本地FlagGems融合源码基线并有限调参；Attention使用官方固定配置提取；Norm与Top-K是明确声明的新融合/分块研究基线，其中Top-K固定4warps未逐shape调参。没有声称这些基线是H100最优库实现。时钟未锁、cache未主动清理。Norm/Top-K审阅后补齐计时输入seed/哈希并重新主测复测，最终采用main_v2/repeat_v2，旧证据保留且未混算。

本轮遵守现有仓库的“只改TTIR”实验合同，**没有开展文章中的源码A/看IR改源码B/直接改IR C盲化对照**。这些变换大多也能回写Triton源码；结果证明具体TTIR候选在固定任务里有效，不能证明IR信息独占优势。复现总入口：[complex_suite/README.md](README.md)。

对应的方法分析见 [五类优化如何用 Triton 源码表达](optimization_methods_and_triton.md)。该方法报告记录源码回写前的静态判断；后续实际源码编译与性能验证见下节。


# 五类优化的 Triton 源码回写实测（2026-11-23）

已将上一轮五类最终TTIR改动忠实写回公开Triton源码；没有重新autotune或增加算法优化。41个原性能点中，源码与对应优化TTIR的可执行ELF `.text`均相同，实际源码JIT对象的设备时间也与TTIR优化版基本一致。完整cubin可能因调试信息不同，不能混称字节一致。

| 算子/范围 | 独立复测baseline/源码 | 独立复测baseline/TTIR | 独立复测TTIR时间/源码时间 |
|---|---:|---:|---:|
| INT4反量化＋GEMM（7点） | 1.0955× | 1.0947× | 1.000816× |
| Grouped/biased MoE（8点） | 1.1507× | 1.1509× | 1.000107× |
| Norm＋动态INT8（8点） | 1.0558× | 1.0550× | 1.000569× |
| Attention dK/dV，D64（6点） | 1.1239× | 1.1218× | 1.001609× |
| 大N partial Top-K（6点） | 1.0823× | 1.0830× | 0.999474× |

这些比值都来自本轮三路同输入、同配置、随机交错配对，主测7轮/独立复测5轮、每轮20ms，不能由旧报告比值推算。最后一列接近1表示两优化路径近似同速。Attention首轮A/A超门槛，旧轮保留并排除，最终采用统一500ms预热后的合格完整重测；D128仍保留负结果，不混入D64收益。

新源码均通过原数值/压力合同与memcheck/racecheck（0 errors、0 warnings）；源码计时使用正常前端产生的CompiledKernel，导出TTIR重编译只用于pipeline核验。

结论由“源码可以表达但尚未测量”推进为“本轮已知优化可忠实回写，源码复现机器指令与收益”。这仍不是盲化A/B/C搜索效率对照，不能据此判断看TTIR是否提高发现优化的概率。

[完整三路性能与源码入口](source_port_comparison.md) · [机器可读汇总](source_port_comparison.json)


# 五类算子第二轮 TTIR 优化：30／32 点扩展矩阵（2026-11-23）

本轮在第一轮已验收版本上继续直接修改 TTIR，正式矩阵为 INT4、MoE、Norm、Top-K 各30点，Attention32点，共152点，保留全部原测试点。以下均是独立复测的同轮配对几何平均；增量相对上一轮 parent，累计相对原始 baseline。

| 算子 | 性能点数 | 收益覆盖 | 增量加速 | 累计加速 |
|---|---:|---|---:|---:|
| [INT4反量化＋GEMM](../int4_gemm/round2/report.md) | 30 | 全部30点 | 2.2571× | 2.4458× |
| [Grouped/biased MoE](../moe_routing/round2/report.md) | 30 | 全部30点 | 1.1860× | 1.4066× |
| [Norm＋动态INT8](../norm_quant/round2/report.md) | 30 | 全部30点 | 1.3808× | 1.4367× |
| [Attention dK/dV](../attention_dkdv/round2/report.md) | 32 | D128的16点；D64保留parent | 2.7420× | 2.7422× |
| [大N partial Top-K](../partial_topk/round2/report.md) | 30 | 全部30点 | 1.1274× | 1.2051× |

INT4缩窄scale/zero加载并在两次64-K迭代间复用；MoE结合稳定组排名、uint32选择状态和一次gather；Norm复用标量倒数并提前写Z；Attention将原CTA内128-key输出拆成64-key子块并按组重算指针或展开；Top-K固定原始ids并展开K次选择。ABI、grid、kernel数、工作区和同组编译选项不变。

Attention因未锁频的同一parent存在明显A/A波动，正式32点统一固定GPU4 SM为1200MHz后重测；全部路线同频，旧无效计时保留但未混入。结束后已恢复自动频率。其余四类未主动锁频；不能把Attention子集的2.742×冒充全32点收益。

新方案均通过原数值／压力合同及memcheck/racecheck，0 errors、racecheck 0 warnings；主测7轮、独立复测5轮，do_bench_cudagraph的rep参数为20ms。实际全部外层配对样本、A/A、输入／IR／产物哈希、资源及负结果保留。Norm额外90项CPU FP64极端检查通过；Top-K重叠的首轮main/repeat已全部排除，只采用隔离重跑。

新增shape使用预先声明的固定配置族，未声称重新全面autotune或胜过最强库实现。累计收益与上一轮小矩阵平均不可混算。新结构可以用公开Triton表达，但本轮没有再次进行源码回写性能验证；上一轮源码与TTIR指令一致的结论不自动适用于本轮新候选。

[完整第二轮报告](round2_report.md) · [机器可读汇总](round2_summary.json) · [共同协议](round2_plan.md)

# 五类算子连续三轮TTIR优化：r3–r5（2026-11-24）

每轮覆盖原152个唯一shape：四类各30点、Attention32点。逐轮从已验收parent继续搜索；筛选后冻结候选，主测7轮、独立复测5轮，未达晋升门槛则保留parent。三轮重复评估没有扩大唯一shape数。

| 算子 | 最终版本 | 独立复测：相对第二轮 | 独立复测：相对原始baseline |
|---|---|---:|---:|
| INT4反量化＋GEMM，30点 | r3 `group128` | 1.030268× | 2.517380× |
| Grouped／biased MoE，30点 | r5 `pair_integer_constants` | 1.071690× | 1.507741× |
| Norm＋动态INT8，30点 | 保留第二轮 | 1.000204× | 1.435725× |
| Attention dK／dV，32点 | 保留第二轮分组选择 | 1.000010× | 1.759590× |
| Partial Top-K，30点 | r5 `contract` | 1.022347× | 1.232306× |

比值全部来自最后一轮同场直接配对，未相乘历史收益。INT4接受GROUP128除法转右移；MoE接受top2双值归约与整数常量特化组合；Top-K接受有限值／固定grid／完整块合同下的冗余判断消除。其余尝试的回退和近似恒等结果完整保留。

Attention正式实验固定SM1200MHz，结束后已恢复；D64／D128各16点复测累计分别为1.126879×／2.747550×。新增晋升均有独立复测及sanitizer资格，MoE负结果r3/r4没有新增sanitizer；Norm／Top-K的sanitizer覆盖30个映射但使用缩减行数。最终晋升的新结构可用公开Triton表达，r2–r5尚未进行源码回写性能对照。

[完整多轮报告](multiround_report.md) · [机器可读汇总](multiround_summary.json) · [共同协议](multiround_plan.md)

# 五类算子第六轮TTIR优化（2026-11-24）

本轮继续覆盖152个唯一shape。以下为独立复测的直接配对几何均值，累计收益来自本轮实测，不与旧数字相乘。

| 算子 | 最终选择 | 相对r5增量 | 相对原始baseline |
|---|---|---:|---:|
| INT4反量化＋GEMM（30点） | 保留r3 `group128` | 保留parent | 2.518154× |
| Grouped／biased MoE（30点） | r6 `batch4` | 1.023135× | 1.543441× |
| Norm＋动态INT8（30点） | 保留r2 `reciprocal_early` | 保留parent | 1.437892× |
| Attention dK／dV（32点） | D128采用r6 `dv_late`；D64保留 | 1.023624× | 1.799615× |
| 大N partial Top-K（30点） | r6 `merge_heads_packed_v2` | 1.034752× | 1.275095× |

MoE以单个活动CTA处理4个组；Attention在已接受64-key结构上后移dV；Top-K只维护局部有序列表的当前头部进行归并。INT4与Norm的新候选未晋升，负结果保留。Attention D128的16点新增1.047807×、累计2.874342×；完整32点包含16个未改D64点，测量在SM1200MHz下完成并已恢复自动频率。

最终选择均通过数值、A/A及sanitizer验收；Norm／Top-K的sanitizer使用缩减行数覆盖30个映射。r2–r6的新改动尚未做Triton源码回写性能对照。

[完整第六轮报告](round6_report.md) · [机器可读汇总](round6_summary.json)
