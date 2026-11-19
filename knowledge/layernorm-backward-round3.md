# LayerNorm backward dX：中心化代数与资源取舍

范围：官方 upstream Triton 3.7.0、H100、FP16 X/DY/W/DX、FP32保存统计量与运算；dX-only教程提取，不包含dW/dB或锁。固定五组编译options/grid/ABI，仅改TTIR。结论不能直接外推到完整fused backward或其他dtype。

## 可复用经验

- **规约常数可以移到标量路径，但要保留中心化。** 把`sum(xhat*wdy)`改为`sum((x-mean)*wdy)`，将rstd并入行级输出系数，减少逐元素乘法。`r3_centered_stream`还使用cs store。这是组合候选，不能把收益全部归因于某一个变化；按family验收，不能将小N收益推广到大N。A级最终选择和逐组主/复测见[报告](../layer_norm/backward_dx/report_round3.md)与[冻结选择](../layer_norm/backward_dx/results/round3/selection_final.json)；不要引用6点初筛替代完整矩阵。
- **资源相同仍可能有算术收益。** b1024/b2048的parent与centered_stream都是32 registers、零spill、2处静态CTA barrier，但静态FMUL为25→19、store为STG.E.128→STG.E.EF.128。这是C级编译产物事实，不是单因素归因；指令数是静态计数，不等于所有线程实际执行总量。[产物统计](../layer_norm/backward_dx/results/round3/binary_analysis.json)。
- **带saved stats的backward要对齐参考输入。** 大偏移、低方差压力测试中，重新做forward的autograd可能使用不同的mean/rstd，使baseline也失败。先记录原失败，再用`aten.native_layer_norm_backward`显式传入相同stats，与FP64 saved-stats公式交叉验证。保持原数值阈值，不删原记录；常规135用例仍要求独立native autograd。原测试baseline有17/72例不一致，参考对齐后baseline/parent为0/72。[原记录](../layer_norm/backward_dx/results/round3/stress_batch1.json)、[对齐记录](../layer_norm/backward_dx/results/round3/stress_saved_stats.json)。

## 无效经验与限制

- **先规约原始矩再减均值会有抵消误差。** `sum(x*wdy)-mean*sum(wdy)`虽然数学等价、b16384从220降到167 registers，且原135例通过，但新增72例中18例失败。低normalized RMS不能替代逐元素allclose；禁止放宽容差或只保留随机分布来接受它。未为这个失败候选测性能。[数值记录](../layer_norm/backward_dx/results/round3/stress_saved_stats.json)。
- **join不能保证生成更好的联合规约。** 将两个向量变成[B,2]后规约，b16384寄存器升至237、仍零spill；初筛没有整体优于parent的证据。B级初筛和C级资源只能支持当前候选的结论。[有效初筛](../layer_norm/backward_dx/results/round3/screen_batch1_repeat.json)。
- **避免额外DRAM读取仍不足以保证收益。** 分组中心化2048版b16384为126 registers，N15872的DRAM读取260.15→260.75MB，避免了第二轮重载候选近翻倍的流量；achieved occupancy 12.23%→22.36%，但NCU 132.90→136.99µs，long-scoreboard/issue 2.80→5.43。C级机制线索表明依赖/调度代价仍值得查，不支持把所有差距归因于单一指标。[NCU](../layer_norm/backward_dx/results/round3/ncu_summary.json)。
- **单点收益不能接受整组替换。** group_center2048在N8704主测比parent快约7.9%，b16384整组却仅1.0019×，独立复测0.9996×；未达到1.02×门槛。完整30点相对原baseline主测0.9850×、复测0.9857×。这是A级负结果，保留IR和原始记录，不把局部点写成最终收益。[分组试验决策](../layer_norm/backward_dx/results/round3/decision.json)。
- **A/A超阈值就保留并排除整份初筛。** screen_batch1的N8704 A/A=1.0413，超过0.97..1.03阈值；没有挑选其中有利点，补跑整份screen_batch1_repeat。正式主/复测都保留全部30点A/A。NCU的base-clock/cache-control=all耗时单独记账，不混入CUDA graph计时。

- **graph收益不等于清cache后的单次收益。** 小N最终候选在NCU条件下并未更快：N1024 parent→候选9.728→9.984µs，N2048为15.936→16.384µs；graph完整主/复测才显示局部收益。不同clock/cache/replay条件不能混合解释，不把FMUL减少直接当作全部加速来源。[小N NCU](../layer_norm/backward_dx/results/round3/ncu_stream_summary.json)。

下一步若继续大N实验，应同时控制规约前状态、规约后load依赖和整个kernel资源额度；不要只追求最低registers。对cache策略、算术重排和调度分别做消融，才可拆分收益来源。
