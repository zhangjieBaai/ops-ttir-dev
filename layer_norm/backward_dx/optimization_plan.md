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
