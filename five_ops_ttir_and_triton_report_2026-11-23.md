# 五类算子 TTIR 优化与 Triton 源码回写报告

> 最新更新（2026-11-24）：已完成[第六轮TTIR优化](complex_suite/round6_report.md)。下文同时保留r3–r5、第一轮及源码回写历史；r2–r6新候选尚未做源码回写计时。

## 第六轮完成后的最新结果（2026-11-24）

[全部152个shape的加速比明细](complex_suite/round6_per_shape.md) · [CSV表格](complex_suite/round6_per_shape.csv)

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

[完整第六轮报告](complex_suite/round6_report.md) · [机器可读汇总](complex_suite/round6_summary.json)

## r3–r5结束时的结果（历史）

表中均为最后一轮实际有效路线的独立复测几何均值，来自同轮直接配对计时，不由历史加速比相乘得到。

| 算子／范围 | 点数 | 最终保留版本 | 相对第二轮 | 相对原始baseline |
|---|---:|---|---:|---:|
| INT4反量化＋GEMM | 30 | r3 `group128` | 1.030268× | 2.517380× |
| Grouped／biased MoE | 30 | r5 `pair_integer_constants` | 1.071690× | 1.507741× |
| Norm＋动态INT8 | 30 | 第二轮 `reciprocal_early` | 1.000204× | 1.435725× |
| Attention dK／dV | 32 | 第二轮分组选择 | 1.000010× | 1.759590× |
| 大N partial Top-K | 30 | r5 `contract` | 1.022347× | 1.232306× |

INT4新增固定GROUP128除法转右移；MoE新增双值top2归约与整数常量特化组合；Top-K利用既定有限值、grid和完整块条件消除冗余保护。Norm和Attention的新候选没有达到晋升门槛，保留parent；接近1的相对第二轮比值是重复计时波动。Attention在SM1200MHz下完成正式实验并恢复自动频率，D64／D128各16点的累计复测分别为1.126879×／2.747550×。

全部最终选择均有数值、A/A及sanitizer资格；MoE未晋升的r3/r4候选没有新增sanitizer，Norm／Top-K的sanitizer使用缩减行数覆盖30个映射。详细覆盖、每轮负结果、Triton表达方式及原始数据见[多轮报告](complex_suite/multiround_report.md)和[机器可读汇总](complex_suite/multiround_summary.json)。

---


实验日期：2026-11-23。本文汇总五份算子优化报告，并收录已有 TTIR 改动直接回写为 Triton 源码后的实测结果。

## Triton 源码回写实测

已将上一轮五类最终 TTIR 改动忠实写回公开 Triton 源码；没有重新 autotune 或增加算法优化。41 个原性能点中，源码与对应优化 TTIR 的可执行 ELF `.text` 均相同，实际源码 JIT 对象的设备时间也与 TTIR 优化版基本一致。完整 cubin 可能因调试信息不同，不能混称字节一致。

下表为各自冻结矩阵内独立复测的几何平均，属于该范围内的 A 级性能证据。Attention 实际测试 12 点，表中仅列有收益的 D64 六点；D128 的负结果保留在详细报告中。

| 算子／范围 | 独立复测 baseline／源码 | 独立复测 baseline／TTIR | 独立复测 TTIR 时间／源码时间 |
|---|---:|---:|---:|
| INT4反量化＋GEMM（7点） | 1.0955× | 1.0947× | 1.000816× |
| Grouped/biased MoE（8点） | 1.1507× | 1.1509× | 1.000107× |
| Norm＋动态INT8（8点） | 1.0558× | 1.0550× | 1.000569× |
| Attention dK/dV，D64（6点） | 1.1239× | 1.1218× | 1.001609× |
| 大N partial Top-K（6点） | 1.0823× | 1.0830× | 0.999474× |

这些比值都来自本轮三路同输入、同配置、随机交错配对，主测 7 轮、独立复测 5 轮，每轮 20ms，不能由旧报告比值推算。最后一列接近 1 表示两优化路径近似同速，大于 1 表示源码版更快。Attention 首轮 A/A 超门槛，旧轮保留并排除，最终采用统一 500ms 预热后的合格完整重测；D128 仍保留负结果，不混入 D64 收益。

新源码均通过原数值／压力合同与 memcheck、racecheck：0 errors，racecheck 0 warnings。源码计时使用正常前端产生的 `CompiledKernel`，导出 TTIR 重编译只用于 pipeline 核验。

结论由“源码可以表达但尚未测量”推进为“本轮已知优化可忠实回写，源码复现机器指令与收益”。这仍不是盲化 A/B/C 搜索效率对照，不能据此判断看 TTIR 是否提高发现优化的概率。

[完整三路性能与源码入口](complex_suite/source_port_comparison.md) · [机器可读汇总](complex_suite/source_port_comparison.json) · [优化方法及 Triton 表达能力分析](complex_suite/optimization_methods_and_triton.md)

## 原始报告索引

| 算子 | TTIR 优化报告 | Triton 源码回写报告 |
|---|---|---|
| INT4 反量化＋GEMM | [原始报告](int4_gemm/report_round1.md) | [源码回写实测](int4_gemm/source_port/report.md) |
| Grouped／biased MoE routing | [原始报告](moe_routing/report_round1.md) | [源码回写实测](moe_routing/source_port/report.md) |
| Norm＋动态 INT8 | [原始报告](norm_quant/report_round1.md) | [源码回写实测](norm_quant/source_port/report.md) |
| Attention backward dK／dV | [原始报告](attention_dkdv/report_round1.md) | [源码回写实测](attention_dkdv/source_port/report.md) |
| 大 N partial Top-K | [原始报告](partial_topk/report_round1.md) | [源码回写实测](partial_topk/source_port/report.md) |

下文收录五份 TTIR 优化详细报告。各报告中的原轮次性能保留其原始测量口径，不能与上表的源码回写轮次混算；原文的相对链接已调整为相对于本文件。

## 1. INT4 反量化＋GEMM

来源：[int4_gemm/report_round1.md](int4_gemm/report_round1.md)。

**最终选择 `scalar_unsigned_v1`：冻结 7 点矩阵的配对加速比几何均值为
1.094808×，独立进程复测 1.095816×；最差点分别为 1.063898× / 1.068932×。**
这是本矩阵的 A 级性能证据，所有候选的数值、资源及原始样本均保留；不外推为
其他 dtype、量化格式、硬件或形状的性能，也不是对 A/B/C 信息消融的证据。

### 基线与唯一变量

来源为 FlagGems commit `ef8aea78fed8321c3e02a15bf31e847cd0579e4f` 的
`_weight_int4pack_mm_with_scales_and_zeros.py`。原始快照在
[upstream/source.py.snapshot](int4_gemm/upstream/source.py.snapshot)，冻结提取版
[tutorial.py](int4_gemm/tutorial.py) 保留设备函数的完整计算语句，仅去掉 `libentry`、主机包装与
FlagGems 依赖，将 `tle.program_id` 绑定到 upstream `tl.program_id`。
源码哈希、提取说明与每点配置在 [baseline.json](int4_gemm/baseline.json)。

基线已经融合 INT4 nibble 解包、groupwise 反量化与 Tensor Core dot。
没有以显存完整反量化后再 GEMM 作为唯一基线。按预定的四组 tile 配置逐点筛选并冻结；
这是有限配置调优，不能称全面调优或最强库基线。FP16 输入/输出、FP32 反量化中间计算
及累积，group=128，连续 packed `[N,K/2]` uint8，低 nibble 是偶 K。
具体支持域与数值容差见 [optimization_plan.md](int4_gemm/optimization_plan.md)。

优化只修改导出的 TTIR；输入、ABI、grid、完整后端 options、launch 和计时完全匹配。
冻结基线/候选都使用 `/root/venvs/ir-ops-dev/bin/python`、官方 Triton 3.7.0 和其
原生 `triton.compile(path)`→CompiledKernel→CUDA 链。源码与文件入口的 LLVM/PTX/cubin
在全部 7 点上字节一致，源码与基线输出逐位一致。最终 alias 与候选同 options 且输出
逐位一致，见 [最终 pipeline 核验](int4_gemm/results/pipeline_final.json)。

设备是独占物理 GPU 0，H100 80GB HBM3，UUID
`GPU-cc248ebf-79d0-42d7-da65-99d1ef6f4ed1`；同卡无并行实验。
未锁时钟，未控制冷缓存。环境、驱动、工具链哈希及实际编译 metadata 随结果记录。

### 直接 TTIR 变换

[make_candidates.py](int4_gemm/make_candidates.py) 以冻结 baseline 为唯一输入，确定性修改 SSA：

| 候选 | 结构变化 | 7 点初筛几何均值 | 状态 |
|---|---|---:|---|
| `shared_scale_v1` | 删除 high 分支 group/scale/zero 加载，直接使用 low 分支值 | 1.076263× | 完整主测 1.076005×，未独立复测，B 级 |
| `group_scalar_v1` | 在共享基础上用标量 `k0/group` 再广播，去掉恒真的 K mask | 1.082623× | 仅初筛，B 级 |
| `unsigned_v1` | 对有非负证明的索引除余改用 unsigned | 1.033726× | 仅初筛，B 级 |
| `scalar_unsigned_v1` | 合并 group 标量化、共享、unsigned | 1.096218× | 最终选择，A 级 |
| `cast_after_v1` | FP32 join/trans/reshape 后再转 FP16 | 0.999993× | 无收益；负结果保留，B 级 |

共享依赖 group 为偶数：同 byte 的偶/奇 K 位置属于同一 group；K 精确整除 128，
两者的有效 K mask 相同。标量化进一步依赖 BK 为 64 或 128，且每次循环起点与
group/BK 对齐，因此一个 tile 内 group 恒定。此事实不成立的 group、K 尾部或 BK
不在支持域内。unsigned 只用于非负 program id、正尺寸及非负 K 索引。
没有改反量化减法、乘法、FP16 舍入点或 dot 累积次序。

cast_after 假设没有在当前编译链中产生收益。7/7 shape 的 ELF `.text` 哈希与 baseline
相同，寄存器/shared 也相同；完整 PTX/cubin 哈希因位置/调试信息不同而不一致。
因此不能称“移动 cast 发现后端缺陷”，也不能把源码注释中的可能风险视为本轮已证实。
机器指令证据在 [analysis.json](int4_gemm/results/analysis.json)。

### 数值与内存检查

- 第一进程：7 shape × random/extreme_nibble/zero_scale/zero/constant ×
  baseline 与 5 候选，共 210 次版本检查全部通过。
- 独立 seed=123：baseline 与最终候选共 70 次检查全部通过。
- 对照 FP32 ATen matmul：反量化权重先按合同舍入至 FP16，使用
  `atol=0.035, rtol=0.003`；首进程最大绝对误差 0.03125。
- 两个小 shape 额外对照 CPU FP64 matmul。所有候选还要求与基线逐位相同，
  检查输出有限、末尾 64 元素 canary 未改、全部输入未修改。
- 最终 baseline/候选在完整 7×5 分布上分别运行 memcheck/racecheck，
  各 70 次版本检查均通过：memcheck **0 errors**；racecheck
  **0 hazards，0 errors，0 warnings**。

原始证据：[数值](int4_gemm/results/accuracy.json)、[独立数值](int4_gemm/results/accuracy_repeat.json)、
[memcheck](int4_gemm/results/memcheck.log)、[racecheck](int4_gemm/results/racecheck.log)。
这些是明确有限分布与形状的检查，不是对支持域内所有输入的证明。

### 完整主测与独立复测

主指标为预分配输出的单 fused kernel CUDA graph replay，排除数据构造、参考、分配、
编译、加载和 Python launch。主测 7 轮/点，独立进程 seed=123 复测 5 轮/点；
每个 sample `rep=20ms`，随机配对顺序，采用每点配对加速比中位数。
初筛为 3 轮、10ms。每轮原始时间和顺序均保留，未只取最快样本。

| M,N,K | 冻结 BM,BN,BK | baseline μs | 最终候选 μs | 主测加速比 | 复测加速比 |
|---|---|---:|---:|---:|---:|
| 16,4096,4096 | 32,64,128 | 74.319 | 64.675 | 1.149121× | 1.147382× |
| 128,11008,4096 | 32,64,64 | 326.716 | 298.316 | 1.095187× | 1.095235× |
| 1,256,128 | 16,64,64 | 3.779 | 3.546 | 1.066774× | 1.068932× |
| 32,1024,1024 | 16,64,64 | 19.992 | 18.337 | 1.090165× | 1.089031× |
| 64,2048,4096 | 16,64,64 | 74.340 | 66.972 | 1.110139× | 1.110531× |
| 128,4096,4096 | 32,64,64 | 117.241 | 110.201 | 1.063898× | 1.071649× |
| 17,80,256 | 16,64,64 | 6.052 | 5.549 | 1.090610× | 1.089872× |

时间列为各版本样本中位数，不能用四舍五入时间之比替代配对加速比。
所有配置均为 4 warps、3 stages、1 CTA。全矩阵几何均值主/复测
**1.094808× / 1.095816×**，算术均值 **1.095128× / 1.096090×**。
A/A 主/复测几何均值 **1.000034× / 1.000069×**，各点接近 1。

固定七形状、仅重采样各形状配对样本的 bootstrap 95% 区间为
主测 `[1.094468,1.095178]`，复测 `[1.095595,1.096029]`。
这仅描述本轮 graph 样本内波动，不包含形状分布、跨时钟或跨日期不确定性；
两独立进程本身的差异更能反映这项局限。

选择依据是主测全矩阵几何均值，要求 ≥1.02 且最差点 ≥0.95；
[预复测选择](int4_gemm/results/selection_pre_repeat.json) 在复测前冻结。
复测与 sanitizer 仅负责验收，没有重新按复测收益选候选。
没有逐点拼接最有利候选；例如 `(128,4096,4096)` 的 shared_scale 主测为 1.133123×，
确实快于最终全局候选的 1.063898×，这一局部不足保留。

配对原始数据：[初筛](int4_gemm/results/screen.json)、[主测](int4_gemm/results/main.json)、
[复测](int4_gemm/results/repeat.json)，重算脚本 [analyze.py](int4_gemm/analyze.py)。

### 资源证据与归因边界

| 分组 | baseline registers | 最终 registers | shared bytes | spill |
|---|---:|---:|---:|---:|
| BM32,BN64,BK128 | 250 | 210 | 40960→40960 | 0→0 |
| BM32,BN64,BK64 | 168 | 138 | 20480→20480 | 0→0 |
| BM16,BN64,BK64 | 140 | 106 | 16384→16384 | 0→0 |

TTIR 可见的重复 group/scale/zero 数据流在冻结基线中确实存在，直接删去后最终机器码、
寄存器与性能均变化。组合还包含 group 标量化、K mask 简化与 unsigned 索引，因此
不能将 1.095× 全部归于 scale 加载、寄存器或某个单项。没有 NCU 指标或逐项独立复测，
资源下降是 C 级静态事实；组合性能才是本矩阵 A 级证据。

文章讨论的偶/奇两路 dot 已存在于另一份 FlagGems 实现，本轮未重新实现或将其称为
新优化；未开展 INT4 独立反量化吞吐实验、residual epilogue、其他 packing、BF16/FP8
或其他硬件实验。当前改写可用 Triton 源码回写，但本轮没有 A/B/C 盲测，不能证明
“直接 TTIR 独占收益”或“看到 TTIR 比只看源码更好”。

最终映射和文件哈希见 [selection_final.json](int4_gemm/results/selection_final.json)；
每组 `optimized.ttir` 是 `scalar_unsigned_v1.ttir` 字节副本。未修改 FlagGems 源码或注册。
搭建过程失败记录在 [setup_failures.md](int4_gemm/results/setup_failures.md)。

新增的实际源码回写比较见 [source_port/report.md](int4_gemm/source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。

## 2. Grouped／biased MoE routing

来源：[moe_routing/report_round1.md](moe_routing/report_round1.md)。

最终采用 `membership`：完整8点主测几何平均 **1.1502×**，
独立进程复测 **1.1502×**。
性能证据为 **A 级，
仅限本轮冻结8点、FP32和两阶段设备序列**；
全部8点获益，
最小主测/复测分别1.0501×/1.0499×。
冻结选择见 [selection.json](moe_routing/selection.json)，
精确原始样本见 [主测](moe_routing/results/main/performance.json) 和 [复测](moe_routing/results/repeat/performance.json)。


### 真实 baseline 与测量范围

baseline 是当前本地 FlagGems NVIDIA/Hopper grouped_topk 的两个原样 kernel（[来源与快照哈希](moe_routing/upstream/provenance.json)）。
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
安装包本地RECORD核验见 [distribution_record.json](moe_routing/results/distribution_record.json)，
不把RECORD核验称作远程发行来源认证。


计时为预分配输出的 **完整两kernel CUDA graph replay**，
主测每点7轮、复测每点5轮、rep=20ms，
随机配对内顺序。
未计Python launch、分配、编译和加载；
未清cache，
未锁时钟。
复测额外A/A各点中位比范围 `0.998944–1.000702`，
几何 `0.999810`。


### 逐点时间

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

### 改动与机制证据

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
详见 [static_analysis.json](moe_routing/results/static_analysis.json) 和逐阶段产物哈希。


### 保留的负结果

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


### 数值与sanitizer

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
日志：[memcheck](moe_routing/results/memcheck.log)、[racecheck](moe_routing/results/racecheck.log)。


### 复现与解释边界

见 [README](moe_routing/README.md) 的新输出目录命令。
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


新增的实际源码回写比较见 [source_port/report.md](moe_routing/source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。

## 3. Norm＋动态 INT8

来源：[norm_quant/report_round1.md](norm_quant/report_round1.md)。

最终采用 `stream_v1`。冻结8点几何平均：主测 **1.0548×**，独立进程复测 **1.0521×**；最差点分别 **0.9883× / 0.9859×**。性能证据为A，仅限本矩阵和本研究baseline。

融合研究baseline一次完成残差加法、RMSNorm、amax、INT8量化，并写出FP32 residual、scale和INT8。它参考FlagGems结构，但不是生产kernel原样提取；每个点只在4/8 warps中预调优，未穷尽配置。候选把max(abs(rho*u))改为rho*max(abs(u))，将sum(z²)/max(abs(u))合为tuple reduction，再对输出使用cs cache hint；单kernel、ABI/grid/options不变。

| shape | baseline μs | candidate μs | 主测 | 复测 |
|---|---:|---:|---:|---:|
| (1, 4096) | 2.968 | 2.858 | 1.0364× | 1.0361× |
| (128, 1024) | 2.219 | 2.097 | 1.0583× | 1.0603× |
| (128, 4096) | 3.695 | 3.606 | 1.0222× | 1.0246× |
| (2048, 4096) | 29.695 | 27.791 | 1.0686× | 1.0699× |
| (512, 8192) | 15.985 | 13.388 | 1.1941× | 1.1718× |
| (513, 4097) | 14.645 | 14.463 | 1.0132× | 1.0120× |
| (32, 16384) | 8.076 | 8.162 | 0.9883× | 0.9859× |
| (64, 257) | 2.238 | 2.078 | 1.0697× | 1.0663× |

时间为样本中位数，加速比为配对比值中位数，两者不必恰好相除。

### 协议与证据

固定官方 Triton3.7.0 / PyTorch2.12.0+cu130 / H100。基线先从冻结源码导出，再与文件入口核对LLVM/PTX/cubin字节一致；每次主/复测在计时前重复核验，实际完整backend options一致。所有候选从TTIR文件编译，未改TTGIR/LLVM/PTX/cubin。

主测7轮、独立进程复测5轮、每轮20ms；配对内随机次序、每轮A/A。预分配完整设备序列CUDA graph时间，不含编译、分配和Python launch；cache未清理、时钟未锁定。每shape保存计时输入seed与SHA256。初筛后冻结单一候选，不按独立复测择优。

固定shape、重采样配对轮次的95% bootstrap区间：主测 `[1.0540127834869335, 1.0561889960576514]`，复测 `[1.0506641106664159, 1.0528457296864264]`。它只反映这些样本的波动，不表示跨模型/设备的不确定性。

主证据：[main_v2](norm_quant/results/main_v2/results.json)、[repeat_v2](norm_quant/results/repeat_v2/results.json)、[manifest](norm_quant/manifest.json)、[选择和哈希](norm_quant/selection.json)。memcheck **0 errors**，racecheck **0 errors/0 warnings**，只过滤检查本实验kernel；[memcheck日志](norm_quant/results/logs/memcheck_final.log)、[racecheck日志](norm_quant/results/logs/racecheck_final.log)。

初版main/repeat也保留；审阅发现它们的timing tensor在seed初始化前分配，配对公平但无法按所记seed完整重建输入。本报告改用已修复seed并保存输入哈希的main_v2/repeat_v2，没有把旧样本混入平均。

### 数值、资源与负结果

正式主/复测共224项版本/分布检查通过；另保留各次初筛与sanitizer内检查。所有输出有canary、sentinel及输入不修改检查。

| shape | regs baseline→candidate | spill baseline→candidate | shared B baseline→candidate |
|---|---|---|---|
| (1, 4096) | 56→56 | 0→0 | 4096→4096 |
| (128, 1024) | 30→29 | 0→0 | 32→64 |
| (128, 4096) | 56→56 | 0→0 | 4096→4096 |
| (2048, 4096) | 56→56 | 0→0 | 4096→4096 |
| (512, 8192) | 93→93 | 0→0 | 4096→4096 |
| (513, 4097) | 120→117 | 0→0 | 4096→4096 |
| (32, 16384) | 156→170 | 0→0 | 4096→4096 |
| (64, 257) | 28→28 | 0→0 | 2048→2048 |

资源是C级静态证据；Top-K表列local阶段，merge资源在原始JSON。不把寄存器或同步减少直接解释为全部性能收益。

量化合同预先允许Q相对FP64至多差1 code、scale rtol=3e-5/atol=1e-9，另验证重构误差；**不声称INT8逐位等价**。Z精确、Q显式限制在[-127,127]，因此-128 sentinel不能蒙混过关。覆盖normal/zero/constant/offset/cancellation/tiny/signed-weight，小rows另有CPU FP64。

4种候选的完整初筛保留在 `results/screen/results.json`（B）：因式分解、联合统计、消去rho、输出cache hint；stream收益是组合结果。N=16384、rows32约1.4%回退仍保留，未用逐点挑选掩盖。未涵盖FP8、BF16、任意非连续layout、NaN/Inf。初次setup曾错误比较metadata的cache key，修正为完整backend options比较；失败日志保留，没有放过LLVM/PTX/cubin一致性门槛。

### 复现

先查看 `nvidia-smi` 并设置独占 `CUDA_VISIBLE_DEVICES`，Python固定 `/root/venvs/ir-ops-dev/bin/python`。在仓库根执行下列命令，输出目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES=6 /root/venvs/ir-ops-dev/bin/python norm_quant/experiment.py \
  --run my_recheck --names baseline,stream_v1 --rounds 5 --rep 20 --seed 234
```

脚本先自动执行pipeline/数值门槛，再计时；`--check-only`只验证。候选生成器对已有IR做字节一致性断言，拒绝覆盖不同内容。已冻结manifest时不要再次`--setup`。

新增的实际源码回写比较见 [source_port/report.md](norm_quant/source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。

## 4. Attention backward dK／dV

来源：[attention_dkdv/report_round1.md](attention_dkdv/report_round1.md)。

固定官方 fused dK/dV 提取基线下，`r1_dv_late` 在 D64 的 6 个点稳定加速；D128 保留 baseline。不是完整 Attention backward 的端到端收益。

基线来自 [官方 Triton v3.7.0 fused-attention](https://github.com/triton-lang/triton/blob/v3.7.0/python/tutorials/06-fused-attention.py)，将 `_attn_bwd` 截在 dQ 段之前，概率恢复、dP/dS、dK/dV 四个 dot 和因果分段保持原样。`frozen_source.py` 同时保存未截断完整 backward。12 点源码/文件入口 LLVM、PTX、cubin 全部字节一致；提取输出与完整官方 backward 的 dK/dV 逐位一致。

环境：官方 Triton 3.7.0 / PyTorch 2.12.0+cu130 / 独占物理 GPU4 H100 80GB，4 warps、5 stages、1 CTA。官方固定配置未经逐 shape autotune，尤其 D128 存在明显 spill；结果不证明战胜调优后的其他实现。冻结代码和来源哈希见 [provenance](attention_dkdv/upstream/provenance.json)，契约见 [optimization_plan.md](attention_dkdv/optimization_plan.md)。

### A 级性能结果

指标为预分配输出的 CUDA graph 设备 kernel 时间；不含 forward、K 预缩放、delta、分配、编译和 Python。每点主测7对/复测5对，rep=20ms，随机对内顺序；复测为独立进程与新输入seed=123。未锁频，无主动清缓存。所有原始样本和顺序在 [主测](attention_dkdv/results/main/benchmark.json) 与 [复测](attention_dkdv/results/repeat/benchmark.json)。

| 组 | 主测几何均值 | 复测几何均值 | 主测最差 | 最终选择 |
|---|---:|---:|---:|---|
| d64_c0 | 1.1255× | 1.1209× | 1.0999× | `r1_dv_late` |
| d64_c1 | 1.1227× | 1.1219× | 1.0748× | `r1_dv_late` |
| d128_c0 | 0.9974× | 0.9981× | 0.9578× | `baseline` |
| d128_c1 | 0.9375× | 0.9373× | 0.8980× | `baseline` |

**D64 六点合计：主测 1.1241×，独立复测 1.1214×。** 表中 D128 是真实负结果，不能用回退选择的1×代替实测，也不把混合策略的合成数值称为全12点实测。

| (B,H,N,D,causal) | 主测 baseline μs | 主测候选 μs | 主测加速 | 复测加速 |
|---|---:|---:|---:|---:|
| (1, 4, 256, 64, False) | 8.593 | 7.813 | 1.0999× | 1.1000× |
| (1, 4, 1024, 64, False) | 25.436 | 22.269 | 1.1422× | 1.1437× |
| (2, 8, 2048, 64, False) | 72.682 | 64.121 | 1.1350× | 1.1195× |
| (1, 4, 256, 64, True) | 9.727 | 9.046 | 1.0748× | 1.0733× |
| (1, 4, 1024, 64, True) | 26.558 | 23.678 | 1.1217× | 1.1226× |
| (2, 8, 2048, 64, True) | 69.454 | 59.165 | 1.1736× | 1.1720× |
| (1, 4, 256, 128, False) | 37.455 | 36.632 | 1.0224× | 1.0217× |
| (1, 4, 1024, 128, False) | 129.894 | 128.184 | 1.0133× | 1.0152× |
| (2, 8, 2048, 128, False) | 542.082 | 565.944 | 0.9578× | 0.9587× |
| (1, 4, 256, 128, True) | 42.829 | 46.029 | 0.9304× | 0.9304× |
| (1, 4, 1024, 128, True) | 127.942 | 142.469 | 0.8980× | 0.8951× |
| (2, 8, 2048, 128, True) | 454.737 | 461.101 | 0.9861× | 0.9888× |

A/A 每点3对的中位比范围：主测 0.9888–1.0003，复测 0.9958–1.0006。这些检查只量化本次波动，未证明全部系统噪声被消除。

### 修改与机制边界

`r1_dv_late` 直接移动 TTIR 循环中 `pT -> fp16` 的 cast 和 dV dot，使其在 dK dot 后执行；causal 的两个循环都修改。数学公式、FP16转换位置相对于各自消费者、每个 accumulator 的加法顺序、签名、grid、编译options都保持一致。没有从优化 Python kernel 重新导出候选。

C 级静态证据见 [static_artifacts.json](attention_dkdv/results/static_artifacts.json)：D64非因果寄存器255→250，spill均0；因果255→255，spill均0，SASS指令数1378→1354。所有组 HGMMA 与 barrier 的静态数量不变。机器码和调度改变，但没有NCU消融证据把约12%的收益归因给某个单一资源量。

D128非因果 spill 178→204、因果182→160；后者 spill 下降却仍明显变慢。这是“减少spill不保证加速”的本次反例，不能仅凭寄存器/溢出数字选择方案。

### 初筛与负结果（B 级）

7个候选均保留在各组 ir/：unsigned div/rem、delta提前、dV后移、dP提前、load.cg、连续head地址折叠，以及 dV后移+unsigned 组合。初筛为完整12点但每点仅3对，选拔记录见 [screen](attention_dkdv/results/screen/benchmark.json) 与 [selection_pre_main.json](attention_dkdv/selection_pre_main.json)。D64非因果 unsigned 初筛几何0.9246×，dV后移+unsigned仅1.0562×，没有继承单项dV后移的全部收益。load.cg接近1×；delta/dP提前在D128退步。没有丢弃不利候选或用重复运行挑最小耗时。

### 验证与范围

初轮224项通过（28输入 × baseline与7候选），独立seed复测56项通过。候选输出均与baseline逐位相同；FP64原始注意力数学参考门槛固定为rtol=0.03, atol=0.008。包含N128/256/1024/2048、causal/noncausal、D64/128、zero_dO、constant和large QK；每轮4个小例另做CPU FP64 autograd。检查输入不修改、NaN sentinel完全覆盖及有限性。

最终两个版本各在完整28输入运行memcheck/racecheck：[memcheck](attention_dkdv/results/memcheck_v2.log) **0 errors**；[racecheck](attention_dkdv/results/racecheck_v2.log) **0 hazards / 0 errors / 0 warnings**，各含56项数值检查。早先filter参数写成`regex:`被工具拒绝，未执行kernel；失败原日志保留，修正为`regex=`后重新运行，未覆盖原结果。

仅覆盖连续FP16、N为128倍数、无dropout、无别名、预先具备LSE/delta和预缩放K的dK/dV子算子。固定配置结果不外推ragged序列、其他D/dtype、其他GPU或整个训练step。这是直接TTIR候选实验，没有构造文章中的A/B/C盲测，因此不能据此证明TTIR信息优于源码信息；dV调度变换也可能回写Triton源码表达。

复现入口见 [README.md](attention_dkdv/README.md)。冻结选择与IR哈希见 [selection.json](attention_dkdv/selection.json)，编译器/libtriton/ptxas哈希见 [toolchain.json](attention_dkdv/results/toolchain.json)。

新增的实际源码回写比较见 [source_port/report.md](attention_dkdv/source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。

## 5. 大 N partial Top-K

来源：[partial_topk/report_round1.md](partial_topk/report_round1.md)。

最终采用 `ordered_carry_v3`。冻结6点几何平均：主测 **1.0827×**，独立进程复测 **1.0816×**；最差点分别 **1.0242× / 1.0155×**。性能证据为A，仅限本矩阵和本研究baseline。

固定两阶段研究baseline：每1024输入保留K候选，再合并成全局Top-K；沿用FlagGems分块重复选择思路，未全排序。输出value/原始index，值降序，同分按较小index优先。候选把FP32 score映射为单调uint32 key，以unsigned max归约，并将编码移出K循环、key作为loop-carried SSA值保留；min-index tie规则保留。kernel数、workspace、ABI/grid和4warps/3stages不变，计时覆盖两kernel。此baseline配置固定，未逐点autotune。

| shape | baseline μs | candidate μs | 主测 | 复测 |
|---|---:|---:|---:|---:|
| (1, 65536, 8) | 7.802 | 6.832 | 1.1420× | 1.1415× |
| (16, 65536, 8) | 11.458 | 10.561 | 1.0855× | 1.0925× |
| (128, 65536, 8) | 45.946 | 44.863 | 1.0242× | 1.0155× |
| (16, 262144, 8) | 27.122 | 25.983 | 1.0439× | 1.0579× |
| (16, 262144, 16) | 52.787 | 49.081 | 1.0757× | 1.0548× |
| (4, 65537, 8) | 8.431 | 7.459 | 1.1298× | 1.1326× |

时间为样本中位数，加速比为配对比值中位数，两者不必恰好相除。

### 协议与证据

固定官方 Triton3.7.0 / PyTorch2.12.0+cu130 / H100。基线先从冻结源码导出，再与文件入口核对LLVM/PTX/cubin字节一致；每次主/复测在计时前重复核验，实际完整backend options一致。所有候选从TTIR文件编译，未改TTGIR/LLVM/PTX/cubin。

主测7轮、独立进程复测5轮、每轮20ms；配对内随机次序、每轮A/A。预分配完整设备序列CUDA graph时间，不含编译、分配和Python launch；cache未清理、时钟未锁定。每shape保存计时输入seed与SHA256。初筛后冻结单一候选，不按独立复测择优。

固定shape、重采样配对轮次的95% bootstrap区间：主测 `[1.0823705798047045, 1.0831040625325354]`，复测 `[1.0563513698092808, 1.0934704411250469]`。它只反映这些样本的波动，不表示跨模型/设备的不确定性。

主证据：[main_v2](partial_topk/results/main_v2/results.json)、[repeat_v2](partial_topk/results/repeat_v2/results.json)、[manifest](partial_topk/manifest.json)、[选择和哈希](partial_topk/selection.json)。memcheck **0 errors**，racecheck **0 errors/0 warnings**，只过滤检查本实验kernel；[memcheck日志](partial_topk/results/logs/memcheck_final.log)、[racecheck日志](partial_topk/results/logs/racecheck_final.log)。

初版main/repeat也保留；审阅发现它们的timing tensor在seed初始化前分配，配对公平但无法按所记seed完整重建输入。本报告改用已修复seed并保存输入哈希的main_v2/repeat_v2，没有把旧样本混入平均。

### 数值、资源与负结果

正式主/复测共192项版本/分布检查通过；另保留各次初筛与sanitizer内检查。所有输出有canary、sentinel及输入不修改检查。

| shape | regs baseline→candidate | spill baseline→candidate | shared B baseline→candidate |
|---|---|---|---|
| (1, 65536, 8) | 30→40 | 0→0 | 16→16 |
| (16, 65536, 8) | 30→40 | 0→0 | 16→16 |
| (128, 65536, 8) | 30→40 | 0→0 | 16→16 |
| (16, 262144, 8) | 30→40 | 0→0 | 16→16 |
| (16, 262144, 16) | 30→40 | 0→0 | 16→16 |
| (4, 65537, 8) | 30→38 | 0→0 | 16→16 |

资源是C级静态证据；Top-K表列local阶段，merge资源在原始JSON。不把寄存器或同步减少直接解释为全部性能收益。

有限FP32的value数值精确、indices严格精确；**±0等价，不要求保留零符号位**，baseline的max结果本来就不保证选中index处的零位型。v3先归一化±0以保持同分index规则。已选值/padding使用-inf的有序key，小于所有finite key；尾块不足K也不能胜过真实输入。主测扩展到正/负subnormal及最低有限值，另含normal/全tie/整数tie/±0/递增；小rows对CPU FP64稳定排序。

负结果：联合value/index argmax导致全6点约0.810×；仅改有序max但每轮重算key约0.980×。两者在旧main完整7轮有B级证据，见 `results/main/results.json`；joint/local_joint/cg消融在screen。v3的改变是循环状态表示，不能把全部收益简单归因于少一次reduce。未实施packed64候选；未测试NaN/Inf、不同K、不同chunk配置或动态workspace协议。

### 复现

先查看 `nvidia-smi` 并设置独占 `CUDA_VISIBLE_DEVICES`，Python固定 `/root/venvs/ir-ops-dev/bin/python`。在仓库根执行下列命令，输出目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python partial_topk/experiment.py \
  --run my_recheck --names baseline,ordered_carry_v3 --rounds 5 --rep 20 --seed 234
```

脚本先自动执行pipeline/数值门槛，再计时；`--check-only`只验证。候选生成器对已有IR做字节一致性断言，拒绝覆盖不同内容。已冻结manifest时不要再次`--setup`。

新增的实际源码回写比较见 [source_port/report.md](partial_topk/source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。

