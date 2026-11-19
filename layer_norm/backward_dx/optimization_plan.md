# LayerNorm backward dX：第一轮契约

## 来源与边界

使用本项目已保存的官方Triton v3.7.0 tutorial05快照。原教程 `_layer_norm_bwd_dx_fused` 同时计算dX和带锁的dW/dB部分和。本实验只提取加载X/DY/W/Mean/Rstd、两次规约、计算及写出dX的原始语句，删除仅用于dW/dB的参数/地址/累加/锁代码，明确称为“教程dX计算段提取baseline”，不称完整原版fused backward。双方均dX-only，不把删去dW/dB算作优化收益。提取脚本、原始语句hash保存。前向目录及IR不修改。

官方Triton3.7.0独立venv，无FlagTree/内联汇编/下游手工修改/独立llc/cubin替换。候选与baseline共用编译及launcher。

## 冻结分组

按教程 `BLOCK_SIZE_N=next_power_of_2(N)`，`num_warps=min(max(BLOCK/256,1),8)`，1CTA/3stages/fp_fusion=True。组1024(w4)、2048/4096/8192/16384(w8)，各自baseline不可变；每个N用对应组，不给小N人为套大tile。FP16 X/DY/W/DX，FP32 Mean/Rstd和内部累加。N/stride正16倍数，N<=16384、列连续、X/DY/DX共享行stride且无别名、指针16字节对齐、元素偏移<2^31。

主性能矩阵 M=4096，N=1024..15872 step512，30点；主测7轮rep20ms，独立进程seed123/5轮rep10ms，CUDA graph、预分配同输入/输出、随机配对顺序，A/A控制。稳定阶段所有版本相同。计算Saved Mean/Rstd不计入dX-only时间。

正确性对native ATen autograd梯度及基于实际FP32 saved stats的FP64公式参考，小用例额外CPU FP64 autograd；finite、sentinel、padding、输入不修改。容差对native/FP64分别为rtol=0.01,atol=0.01，归一化RMS误差<=0.003；large offset/near-constant高rstd压力按相同相对容差检查，不将大绝对值梯度强行限制为绝对1e-2。测试包括normal/offset/constant/lowvar、zero_dy、signed权重、小M、每组边界和padding。两进程不同seed。sanitizer最终候选覆盖全部组边界。

## 假设、预算与接受

最多8个直接TTIR候选：canonical控制、联合规约两个独立sum、N倒数复用、输出行级系数/FMA重写、分组统计+重载输出、流式store及有限组合。原有中心化/保存统计量语义保留。不更改grid/options，不跨组复制不匹配的metadata。

先pipeline与baseline正确性、基线Git提交，再候选C0/C1。初筛6点1024/3072/4096/8192/8704/15872，3轮rep10ms；入围两版全矩阵复核。NCU固定base clock/cache-control=all，定位occupancy、shared/barrier和加载依赖，只作机制证据。最终完整矩阵和独立复测几何加速>=1.02、最差shape>=0.95、A/A中位数0.97..1.03且数值/内存检查通过才晋升，否则optimized保持baseline。保留失败与所有原始样本；无显著收益如实报告。

## 初筛与入围

8个候选（含canonical对照）完成。joint_coefficients初筛几何1.0257x；grouped2048为0.9651x，但N=8704局部1.1085x。后者将16384组寄存器222降为118，仍需核查全组其他尺寸是否退步，不能凭单点提升晋升。两个候选进入全量30点/7轮及独立5轮，按预先冻结的配置组统计几何平均，允许组内保留baseline；每组选择只根据主测冻结，复测仅验收，不用复测重选有利点。各组候选主/复测均>=1.02且组内最差>=0.95才晋升，该规则是原总验收向组级展开。NCU作为机制分析，不计入性能验收。


## 第一轮结果记录

- 全部8候选先做135例/版本C1，两个入围候选全30点主测7轮、独立进程5轮。joint_coefficients全量几何1.0231x/1.0221x；grouped2048为0.9782x/0.9797x，不晋升。
- 分组主测选择joint_coefficients用于1024/2048/16384；4096/8192保留baseline。独立复测这三组分别约1.0287x、1.0232x、1.0364x，均通过冻结的2%门槛。原始组级主测数据保留，不用复测重选其他候选。
- NCU：16384组grouped候选把寄存器222降至118、无spill，实际occupancy约12%升至23%，但全量性能退步；说明重载与分组的代价不能只用occupancy判断。联合规约仍由原生lowering决定同步，不声称手工复现LLIR规约策略。
- memcheck/racecheck各baseline与joint_coefficients两版×30边界例通过，0 errors，racecheck无warnings。最终别名再做5组pipeline一致性、135例正确性及完整30点7轮性能验证。前向源码/IR不修改。
- 本轮停止于8候选预算，收益小于前向，如实报告；未测BF16/FP32、非16对齐N、非有限值或完整dW/dB backward。
