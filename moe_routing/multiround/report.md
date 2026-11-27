# MoE 连续三轮 TTIR 优化（r3–r5）

本任务实际完成三轮不同结构的搜索，性能矩阵始终为原30点；不是将主测与复测算成不同轮次。
每轮从上一轮已接受结果出发；不满足2%全矩阵增量门槛时保留parent。原baseline、round2 anchor和全部历史文件未改。

| 轮次 | 冻结测试候选 | 主测增量 | 独立复测增量 | 最差复测增量 | 决策 |
|---|---|---:|---:|---:|---|
| [r3](r3/report.md) | unroll_experts | 1.001101× | 1.000858× | 0.983112× | 保留parent |
| [r4](r4/report.md) | pair_top2_v2 | 1.018584× | 1.019415× | 1.002169× | 保留parent |
| [r5](r5/report.md) | pair_integer_constants | 1.072454× | 1.071772× | 1.010373× | 接受 |

最后一轮实际执行路径相对round2 anchor：主测 **1.072442×**，独立复测 **1.071690×**。
相对原baseline累计：**1.507532× / 1.507741×**。
这些是最后一轮baseline/anchor/parent/candidate直接同场计时，未相乘历史轮次的加速比。重复三轮共90个shape评估，唯一shape数仍为30。

## r3：gather属性、展开与indices状态

实际tt.gather的efficient_layout是unit属性；初版BoolAttr=true被解析器拒绝，修正版gather_unit_v2编译且精度通过，但没有2%增量。
普通3.7公开tl.gather(src,index,axis)没有直接设置这个属性的参数；这不证明等价布局不可能通过源码重构得到。
展开8次expert选择通过精确检查，但整体约1×；以标量索引构造有序向量也未获益。
初版scalar_indices直接tt.cat允许后端重排，导致严格weights位型检查失败；v2改为join/trans/reshape后正确，仍慢于parent。
因此r3正式确认后保留round2版本。解析/精度失败文件、screen和screen_v2均保留。

## r4：此前未改的stage1 top2统计

ordered_top2把两次浮点max换成有序uint32 max，保留最大值计数与第二大值规则；pair_top2用一次双值归约直接合并两个局部top2。
组合函数为hi=max(a,c)，lo=max(min(a,c),max(b,d))，最终仍按max1+max2写出，有限输入和ties下值相同。
tuple初版region形参%b与外层变量冲突，被解析拒绝；v2使用独立SSA名。原完整筛选与当时已启动的部分主测标为superseded，
正式选拔/确认使用screen_v2、main_v2、repeat_v2，旧selection_pre_main.json保留，最终冻结选择在selection_pre_main_v2.json。
r4获得接近2%的稳定收益，但未过预先门槛；没有事后降低阈值。

## r5：固定合同参数的常量特化与组合

合同中G=8、top_group=4、topK=8、连续strides以及E=256/512族已固定。新IR把这些整数参数的使用替换成对应常量，仍保留原函数签名和运行时参数槽。
这使stage1的除余和两stage的边界/地址计算有机会折叠；不把未声明的任意stride/任意G当作支持域。M仍为运行时参数，全部15种合法M保留。
r4未达到晋升门槛后，r5第三候选从仍被接受的parent重新施加tuple top2并与整数特化组合；不是把未接受r4偷偷当parent。
三项为stage1整数特化、两stage整数特化、tuple+两stage整数特化。最终组合经过新的完整验证/主测/复测，而非预设单项收益可相乘。

## 数值、时钟与验证

30点沿用M={16,32,64,128,256,384,512,768,1024,1536,2048,3072,4096,6144,8192}乘E256/512。
G8/top4/top8、FP32、连续等长分组、原tanh sigmoid/归一化不变；仍是两个kernel、4warps/3stages、相同ABI/grid/workspace，无autotune。
原六分布全部保留；r3复测起增加全负bias和宽幅有限bias，针对signed-key分支与大幅度ties。除了torch.equal，本轮用int32 view比较FP32原始位型。
原baseline源码与文件入口LLVM/PTX/cubin每次主实验逐30映射检查；所有候选先数值后性能。每个被测文件/输入/产物均有哈希。
主测7轮、独立seed123复测5轮、rep20参数，全部随机顺序与外层配对样本保留；A/A每点中位要求[0.97,1.03]。
GPU2独占，官方Triton3.7.0/H100；预分配完整两kernel CUDA graph设备时间，不含Python/分配/编译；未锁频、不主动清cache。
接受新版本需要主/复测增量geo>=1.02、最差>=.95和memcheck/racecheck通过；负结果保留parent，未把旧收益冒充本轮增量。
三轮均有完整主/独立复测性能证据，静态资源/属性判断为C级。未晋升的新kernel未额外运行sanitizer，沿用未改parent的已有资格；晋升轮次有新sanitizer结果。

## Triton源码表达能力

稳定top2 tuple可用公开tl.reduce自定义组合函数表达，常量特化可用tl.constexpr或已知常量表达，循环展开和有序concat也有公开语法。
未接受的efficient_layout属性没有直接公开参数；本轮没有发现依靠该属性获得的独占性能优势，也未做新源码回写计时。

[共同协议](optimization_plan.md) · [机器可读汇总](summary.json) · [静态分析](static_analysis.json)

复现命令见README，结果目录必须不存在。候选生成器对已存在TTIR做字节一致断言。
