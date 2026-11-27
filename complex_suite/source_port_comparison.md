# 直接改 Triton 源码的性能：五类 TTIR 优化的忠实回写对照

**实测结论：这五类现有优化直接写成公开 Triton 源码，可以复现直接改 TTIR 的收益。**
所有受测配置中，优化源码与对应TTIR优化版生成的可执行 ELF `.text` 完全相同，资源也一致；
同轮配对计时的两优化路径差异很小，没有测出必须直接改TTIR才能获得的独占收益。
这次已经从上一份报告的‘源码可表达，性能未验证’，推进到实际编译和测量。

| 算子 | 源码相对baseline：主测 | 源码相对baseline：独立复测 | 独立复测TTIR相对baseline | 独立复测IR时间/source时间 |
|---|---:|---:|---:|---:|
| INT4反量化＋GEMM（7点） | 1.0949× | 1.0955× | 1.0947× | 1.000816× |
| Grouped/biased MoE（8点） | 1.1502× | 1.1507× | 1.1509× | 1.000107× |
| Norm＋动态INT8（8点） | 1.0583× | 1.0558× | 1.0550× | 1.000569× |
| Attention dK/dV，D64（6点） | 1.1211× | 1.1239× | 1.1218× | 1.001609× |
| 大N partial Top-K（6点） | 1.0826× | 1.0823× | 1.0830× | 0.999474× |

最后一列接近1表示两种优化路径时间接近，>1表示源码略快。
三种比值均使用本轮同shape、同输入、同轮测量；没有拿旧报告的加速比推算源码性能。
每点先取配对比值的中位数，再做几何汇总，因此几何汇总之间不必满足简单相除关系。
微小差异可能受计时、缓存状态或模块位置等影响；指令相同不要求每次耗时逐位一致。

## 忠实回写了什么

| 算子 | 源码改动 | 编译后观察 | 实现与逐点报告 |
|---|---|---|---|
| INT4反量化＋GEMM（7点） | 共享low/high scale/zero；group标量化；公开uint32除余 | 7点可执行.text一致 | [源码](../int4_gemm/source_port/source_v1.py) / [报告](../int4_gemm/source_port/report.md) |
| Grouped/biased MoE（8点） | 连续组mask用broadcast/reshape扩展到expert mask | 前端保留目标形状，消除dense membership；.text一致 | [源码](../moe_routing/source_port/membership_source_v1.py) / [报告](../moe_routing/source_port/report.md) |
| Norm＋动态INT8（8点） | sum/max tuple reduce、amax因式分解、store .cs | 前端保留联合统计与cache hint；8点.text一致 | [源码](../norm_quant/source_port/source_v1.py) / [报告](../norm_quant/source_port/report.md) |
| Attention dK/dV，D64（6点） | 将pT转FP16和dV dot移到dK之后 | 四组前端保留顺序；D64/D128均.text一致 | [源码](../attention_dkdv/source_port/frozen_port.py) / [报告](../attention_dkdv/source_port/report.md) |
| 大N partial Top-K（6点） | 循环外编码有序uint32 key，循环内更新key和稳定index | 前端保留整数循环状态；local/merge均.text一致 | [源码](../partial_topk/source_port/source_v1.py) / [报告](../partial_topk/source_port/report.md) |

这里的指令一致性专指 ELF `.text` 代码段。源码文件位置、SSA命名和调试信息会改变IR文本或完整cubin，
不能把`.text`一致偷换成完整cubin逐字节一致。源码自身导出TTIR再编译的一致性是另一项pipeline核验。

## 不回避原方案的限制和负结果

Attention共测原12点；上表只汇总有收益的D64六点。D128的源码回写同样复现了原TTIR方案的不足：

| Attention组 | 独立复测baseline/source | 独立复测IR/source | 使用建议 |
|---|---:|---:|---|
| d128_c0 | 0.9987× | 1.000168× | 保留原baseline |
| d128_c1 | 0.9408× | 0.999849× | 保留原baseline |

Norm仍存在大hidden、小rows的局部回退，源码回写不会消除原算法的shape取舍。
INT8沿用预先允许相对FP64最多差1 code的合同；Top-K values数值精确、indices严格精确，±0等价。
INT4、MoE和Attention保留原有逐位输出对照与数值门槛，没有为了源码版放宽精度。

## 本轮怎样比较

B=原baseline TTIR文件编译产物；I=原优化TTIR文件编译产物；S=公开 `@triton.jit` 源码直接产生的CompiledKernel。
**S路线实际启动源码JIT产物**，没有把导出TTIR的重编译版本冒充源码路线。
源码CompiledKernel的Python绑定层保留constexpr槽位；底层设备参数ABI、grid和完整backend options与I一致。
复用原41个性能点、冻结tile/warp/stage配置、kernel数和workspace，不重新autotune，不额外搜索不同算法。

官方Triton3.7.0/H100，五张独占GPU并行；每项先通过原数值/压力检查，后做主测7轮、独立进程5轮、rep20ms。
同轮随机三路顺序，保存全部原始样本、输入seed/哈希、源码/IR/产物哈希、资源和A/A。
计时是预分配CUDA graph设备序列，不含编译、分配、Python；未锁频，未主动清cache。
A/A采用[0.97,1.03]的预先门槛。Attention旧main/repeat有大D64点超门槛，已保留并排除；
统一对三路增加预热后重跑完整矩阵，最终汇总只采用合格main_v2/repeat_v2，未混入旧样本。

所有新源码路径完成数值验证、memcheck和racecheck，均无错误，racecheck无警告；源码/原TTIR的静态比较另行保存。
各目录保留早期launcher适配失败和测试harness修复记录，未修改原实验baseline或候选来迁就源码。
这些性能数字是固定矩阵的A级证据，机器码/结构事实是C级；不外推最强库实现、其他硬件或其他输入合同。

## 对TTIR价值的判断

本轮可以确认：已经知道这些变换后，公开Triton源码足以表达它们，正常前端也保留了这些实例所需的结构。
所以这些加速不应被解释为TTIR独有的表达能力优势。TTIR仍可用于观察数据流、定位重复计算和控制实验截点，
但本轮没有测量‘只看源码’与‘看TTIR’两组在不知道答案时发现优化的概率、时间或搜索效率。
这是忠实回写/编译路径对照，不是盲化A/B/C优化能力试验。

[共同协议](source_port_plan.md) · [机器可读汇总](source_port_comparison.json) · [上一轮方法报告](optimization_methods_and_triton.md)
