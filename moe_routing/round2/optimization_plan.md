MoE 第二轮只修改当前 membership 父版本的 TTIR，原8点、FP32、G8/top4 groups/top8 experts、两kernel、workspace、ABI/grid/options与严格indices/weights合同不变。GPU2独占，官方Triton3.7.0。原文件不改；本目录保存不可变baseline/parent副本与SHA256。

预先限定6个结构候选：r2_ordered 将biased expert分数编码为有序uint32并作为循环状态，保留min-index ties；r2_gather 去掉逐个选中logit重载/tanh与weights循环状态，选完indices后一次gather已经算好的原始sigmoid分数；r2_rank 用8×8稳定排名替代group的4轮threshold选择和scan；r2_ordered_gather与r2_rank_ordered_gather为组合；r2_unroll直接展开8轮expert选择。biased分数可能为负，ordered编码处理符号并将±0归一化；不从biased分数减回bias。gather保留原sigmoid值而非近似重算，必须严格通过基线逐位weights门槛。

每次计时先逐8点验证原baseline源码/TTIR的LLIR/PTX/cubin一致，再完整6分布数值/CPU/输入不改/canary门槛。候选C0失败保留；任何数值失败不进计时。3轮10ms完整初筛后按增量geo选单一候选冻结，再7轮20ms主测与独立seed123五轮20ms复测。每轮随机baseline/parent/candidate顺序，记录全部样本和parent A/A。A/A中位数须在[0.97,1.03]，最终parent/candidate两轮geo均>=1.02且最差>=.95并通过memcheck/racecheck才晋升。未达到则保留parent，不用相对原baseline的旧收益冒充本轮收益。

主指标是预分配完整两kernel CUDA graph设备时间，非Python/分配/编译端到端。所有候选同等100ms预热，未锁频、未主动清cache。报告同轮baseline/candidate累计与parent/candidate增量、资源/代码差异及公开Triton能否表达；不声称TTIR独有。

用户在首轮候选生成后、任何GPU验证/计时前要求扩展到约30点：本轮正式矩阵冻结为M={16,32,64,128,256,384,512,768,1024,1536,2048,3072,4096,6144,8192} × E={256,512}，共30点。原8点仅作历史/可能pilot。新M明确固定同一4warps/3stages映射，不声称已逐新点autotune。原完整6压力分布及正式主/复测、sanitizer覆盖30点。详见manifest30.json。
