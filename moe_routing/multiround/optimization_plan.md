# MoE 连续三轮直接 TTIR 优化：r3 / r4 / r5

沿用round2已冻结30点、FP32、G8/top4组/top8专家、两kernel、4warps/3stages、ABI/grid/workspace及严格输出合同。初始anchor为r2_rank_ordered_gather，source_port和历史文件不改。每轮只从已接受parent的TTIR生成新候选，未接受则下一轮继续原parent。

预算：r3检查实际gather的efficient_layout属性、展开8次expert选择、展开后以标量indices拼接替代循环tensor状态；r4优化此前未改的stage1 top2统计，比较有序uint32 max和一次tuple top2归约；r5对固定合同中的整数参数做明示常量特化，比较仅stage1、两stage整数、再加routed_scale=1的特化。每轮2-3候选，失败与无机器码变化均保留；不使用优化Python kernel导出、不修改后端IR、不内联PTX、不改变轮次配置。

普通tl.gather在本地Triton3.7.0没有公开efficient_layout参数；libtriton包含该属性名。属性试验只作候选，不预设合法布局或收益；编译、数值和产物检查决定是否采用。其他变换可用公开Triton表达，源码性能未测。

每轮先原baseline源码/文件LLVM/PTX/cubin一致性，全部30点x6分布验证，再3轮rep10ms初筛。冻结最佳全矩阵候选后7轮rep20ms主测、独立seed123的5轮复测。记录每轮baseline、round2 anchor、当前parent、candidate四类实际耗时，直接计算parent/candidate增量、anchor/candidate链收益、baseline/candidate累计；不相乘历史数字。parent A/A各点中位须[.97,1.03]。增量两轮geo>=1.02且最差>=.95，且memcheck/racecheck无错误才接受。无收益同样保留正式负结果。若候选在所有组可执行.text与parent相同，可按C级同码证据结束而不虚构性能。

GPU2独占，固定/root/venvs/ir-ops-dev/bin/python、官方Triton3.7.0、H100；不锁频、不主动清cache，计时为预分配两kernel CUDA graph设备序列，排除编译/分配/Python。rep参数不是整轮实际持续时间；保存的是do_bench_cudagraph默认内部重试均值组成的全部外层配对样本。

执行修订：r3初版gather布尔属性被解析拒绝，改为unit属性新名gather_unit_v2；scalar_indices中的tt.cat允许重排，违反精确输出，改为join/trans/reshape新名scalar_indices_v2。失败原件保留。r3正式主测后，为r4负数选择增加negative_bias/wide_bias两分布，从r3复测起验证8分布，不改变性能矩阵与误差合同。r4初版tuple区域形参%b与外围常量冲突，v2改独立SSA名；先前仅ordered有效的完整筛选及其已启动主测全部列为pilot/superseded，正式使用screen_v2/main_v2/repeat_v2。

r4的完整筛选发现tuple top2约1.9%增量，低于预设2%门槛。r5在生成前调整第三候选：如果r4保留parent，则从该parent再次施加已验证tuple top2并与整数常量特化组合；如果r4已接受tuple，则第三候选仍测试float routed_scale=1特化。仍是三个候选，不把未通过晋升的r4直接当新parent，也不改变精度/性能阈值。
