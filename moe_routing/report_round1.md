# Grouped / biased MoE routing：直接 TTIR 优化结果

最终采用 `membership`：完整8点主测几何平均 **1.1502×**，
独立进程复测 **1.1502×**。
性能证据为 **A 级，
仅限本轮冻结8点、FP32和两阶段设备序列**；
全部8点获益，
最小主测/复测分别1.0501×/1.0499×。
冻结选择见 [selection.json](selection.json)，
精确原始样本见 [主测](results/main/performance.json) 和 [复测](results/repeat/performance.json)。


## 真实 baseline 与测量范围

baseline 是当前本地 FlagGems NVIDIA/Hopper grouped_topk 的两个原样 kernel（[来源与快照哈希](upstream/provenance.json)）。
输入是 logits 和 correction bias，
完整执行 tanh 形式 sigmoid、biased top2 group score、选择4组、组内全局top8专家、原始未加bias sigmoid权重提取、归一化和 weights/indices 双输出。
G=8，
连续等长分组，
E=256/512，
K=8，
FP32，
ties按小索引优先。
tokens为正16倍数（冻结IR的真实属性），
无别名。
此处“完整routing”不包含router线性层、token dispatch/scatter或expert计算。


该baseline已融合组分数和组内选择所需的计算，
保留两个 kernel；
不是只用 torch 分步操作作性能对照。
每个性能点先测 stage2 4/8 warps（各3轮），
均选4；
stage1固定4，
num_stages=3。
没有穷尽配置或比较其他routing库，
也未声称它是H100最强实现。


官方 Triton 3.7.0、PyTorch 2.12.0+cu130、GPU2 NVIDIA H100 80GB，
UUID `GPU-1b49d645-5a41-1372-f736-40978ddba030`。
每次主/复测都对8个映射核对源码与TTIR文件入口 **LLVM/PTX/cubin字节一致**、完整backend metadata一致（cache key除外）、三个输出逐位一致；
候选均从TTIR文件经相同原生编译和launcher执行。
完整环境和编译哈希在对应 `pipeline.json`、`accuracy.json`、`performance.json`；
安装包本地RECORD核验见 [distribution_record.json](results/distribution_record.json)，
不把RECORD核验称作远程发行来源认证。


计时为预分配输出的 **完整两kernel CUDA graph replay**，
主测每点7轮、复测每点5轮、rep=20ms，
随机配对内顺序。
未计Python launch、分配、编译和加载；
未清cache，
未锁时钟。
复测额外A/A各点中位比范围 `0.998944–1.000702`，
几何 `0.999810`。


## 逐点时间

微秒为各版本样本中位数；
加速比为配对比值中位数，
故不必恰等于两列中位时间之比。


| tokens | experts | baseline μs | membership μs | 主测加速 | 独立复测 |
|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 7.166 | 6.824 | 1.0501× | 1.0505× |
| 128 | 256 | 7.570 | 7.211 | 1.0502× | 1.0499× |
| 512 | 256 | 10.649 | 9.556 | 1.1147× | 1.1144× |
| 4096 | 256 | 48.648 | 41.306 | 1.1783× | 1.1789× |
| 16 | 512 | 8.072 | 7.198 | 1.1217× | 1.1183× |
| 128 | 512 | 8.662 | 7.773 | 1.1145× | 1.1155× |
| 512 | 512 | 12.959 | 10.562 | 1.2269× | 1.2282× |
| 4096 | 512 | 63.811 | 46.269 | 1.3791× | 1.3793× |

## 改动与机制证据

`membership`直接把冻结TTIR里的 `[E,8]` 专家/组索引比较、AND和sum归约，
改成 `[8,E/8]` 的group_selected broadcast，
再reshape到 `[E]`。
本轮分组连续且等长，
映射等价。
没有重新生成Python候选，
也不改kernel数、grid、签名、options或workspace。
组内原始分数加载依然是索引 `tt.load`；
此改写不涉及 `tt.gather efficient_layout`。


静态证据（C级）：E256/E512的stage2 shared分别从1024/2048B降到16B，
寄存器32分别降到25/28，
spill始终0。
TTGIR convert_layout计数2→1；
PTX静态shuffle计数90/138→42，
静态div计数18/34→2，
bar.sync计数仍7。
消除了group membership的大中间量和运行时整除是直接结构事实；
各项资源变化的独立因果贡献未经进一步消融，
不把静态指令计数当动态执行次数。
详见 [static_analysis.json](results/static_analysis.json) 和逐阶段产物哈希。


## 保留的负结果

`argmax`把每轮 expert max + tie-index min 两个归约改为稳定 value/index 联合归约，
精度全部通过。
主测全矩阵几何 **0.9975×**，
4096×256/512约0.909×；
它把静态bar.sync从7减到4，
却没有改善整个任务，
说明少一个归约不保证更快。
`combined`同时应用两项，
主测几何 **1.1390×**，
低于membership。
两者只有主测，
没有独立性能复测，
故各自性能为 **B级**。
它们的TTIR、全部配对样本、编译资源及数值证据均保留。
没有按shape挑最有利策略来扩大收益；
单一membership策略在主测后冻结，
复测不重新择优。


## 数值与sanitizer

初始与主测各192项（4版本×8形状×6分布）；
独立复测96项（baseline/membership）；
memcheck和racecheck各96项。
全部PASS，
总计672项版本/用例检查。
分布为normal、全tie、group tie、bias主导、饱和和全负饱和。
每项检查indices相对baseline/CPU参考严格相等，
weights及group workspace相对baseline逐位相等，
CPU weights满足rtol=2e-5/atol=2e-6，
并检查输入不变、sentinel覆盖、有限性及尾部canary。


本轮CPU参考刻意采用原kernel的 `0.5*tanh(0.5*x)+0.5` 浮点表达式和原归一化epsilon；
不把它等同所有sigmoid实现的逐位合同。
未测试NaN/Inf、不同dtype、不同K/G或非16整除tokens。
memcheck **0 errors**；
racecheck **0 errors、0 warnings**。
日志：[memcheck](results/memcheck.log)、[racecheck](results/racecheck.log)。


## 复现与解释边界

见 [README](README.md) 的新输出目录命令。
`make_candidates.py`可逐字节重建全部候选，
`report.py`从原始JSON检查接受门槛并生成本报告。
`optimized_s2.ttir`只是已接受membership的字节副本，
stage1始终使用冻结baseline。


本轮证明此固定routing契约中的TTIR结构改写产生可复现收益。
这种broadcast/reshape也能在Triton源码表达；
未开展独立盲化A/B/C实验，
不能由此断言TTIR相对源码拥有独占优化能力。
两个kernel仍有融合或不同选择算法的进一步空间，
本轮未测，
不作收益承诺。


新增的实际源码回写比较见 [source_port/report.md](source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。
