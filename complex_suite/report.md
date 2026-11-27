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
