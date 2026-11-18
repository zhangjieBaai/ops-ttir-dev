# LayerNorm 第二轮：有效与无效经验

这是一次已完成实验的复盘，用来指导其他算子选候选，而不是把 LayerNorm 的配置当成通用最佳配置。实验代码和原始记录保持不变；本次只整理知识。

## 适用范围和证据

H100/SM90，官方 Triton 3.7.0、无 FlagTree；FP16 输入/输出、FP32 累加。固定 8 warps、1 CTA、3 stages，同一编译入口和 launcher，只改 TTIR。性能矩阵是 M=4096、N=1024…15872、间隔512，共30点。指标为预分配输出的 CUDA graph kernel 时间；不是端到端或冷缓存收益。

| 结果 | 数字 | 证据 |
|---|---|---|
| 第一轮 parent，同轮完整矩阵 | 几何1.2952× | A，见下方主测 |
| 最终 `r2_mixed_stream` | 几何1.4397×，算术1.4995× | A，30点/7轮 |
| 最终版独立进程复测 | 几何1.4395×，算术1.4991× | A，30点/5轮，seed=123 |
| 最终资源 | 80→48 registers/thread，spill 2→0 | C，编译/加载资源记录 |
| 相对 parent 的分布 | 22/30点更快，最大回退3.63% | A，不能只报告平均数 |

[A级主测](../layer_norm/results/round2/full_stable.json)、[独立复测](../layer_norm/results/round2/independent_repeat.json)、[资源](../layer_norm/results/round2/resources.json)、[选择记录](../layer_norm/results/round2/selection.json)、[完整报告](../layer_norm/report_round2.md)。主测 JSON 中 `optimized` 是**当时的第一轮 parent**，不能按今天的 `ir/optimized.ttir` 解释它。

## 有效经验

### E1. 一起控制加载宽度、累加宽度、输出分组和尺寸范围

**怎么做**：从8192宽的累加状态，改为1024/2048宽分组。分组之间先累加，再做行级规约；输出也按分组处理。对常见 N 范围静态展开，对更大 N 保留动态循环。

最终候选使用1024/2048/3072/4096/6144/8192/10240/12288/14336/16384范围；N≤3072用1024分组，其他用2048分组，N≤1024的实际宽度为1024。这些阈值来自本轮数据，并非其他算子的默认配置。

**证据**：完整组合为A级1.4397×；资源为C级48寄存器、无spill。构造方法见 [`grouped_body()` / `grouped()`](../layer_norm/make_round2.py)，产物见 [`r2_mixed_stream.ttir`](../layer_norm/ir/r2_mixed_stream.ttir)。

**可迁移判断**：不要只问“tile多大”，还要问多少中间值同时存活、循环有多少次、尾部有多少无效工作。窄分组可能降低状态量，却增加循环和地址成本，必须在同一矩阵验证。

**未证明**：没有完整单因素消融，不能把组合总收益全部分给“缩小累加器”，也不能仅凭寄存器变化认定因果。

### E2. 多保留一次 load，可能比长时间保留整行数据划算

**怎么做**：mean用显式 `cacheModifier=ca` 的普通load，variance用默认属性。当前编译器下不同属性可以让两次读取保留，而不是公共子表达式合并后长时间保留输入。

**证据边界**：最终采用这种写法的组合有A级性能证据；`.ca` 单独候选只有B级初筛（6点几何1.4901×）。[独立能力探针](evidence-index.md#kernelforge-知识核查)支持本编译器/探针中默认与显式ca混用保留两次load这一C级现象，不证明对所有程序都成立。

**可迁移判断**：减少load条数和减少寄存器存活可能相互冲突。检查TTGIR/PTX/SASS实际保留的加载、寄存器、spill，再比较完整计时。`.ca`是缓存提示，**不是通用的禁止CSE指令**；volatile也不是“每次都读DRAM”的保证。

### E3. 对差距最大的尺寸做有边界的细化

**怎么做**：在2048分组流式输出候选基础上，仅把N≤3072的分组改为1024。比较两者同一完整矩阵，保留大N不受逻辑宽度修改的范围。

| N | 2048分组流式版 µs | 混合分组版 µs |
|---:|---:|---:|
| 1536 | 11.846 | 11.558 |
| 2048 | 12.463 | 12.230 |
| 2560 | 16.526 | 15.264 |
| 3072 | 20.340 | 20.077 |
| 4096 | 26.314 | 26.321 |

**证据**：B级同会话完整矩阵对照，2048流式版尚未独立复测；混合版的最终总收益另有A级复测。两版均48寄存器、无spill。改动范围以外也可能因全局编译布局变化而有细小差异，不能假设机器码绝对不变。

来源：[full_stable.json](../layer_norm/results/round2/full_stable.json)、[生成器](../layer_norm/make_round2.py)。**可迁移判断**：先按逐shape差距挑范围，再比较整体资源档位和全部测试点，避免用一个宽分支损害其他分支。

### E4. 缓存提示要随数据用途设计，但单项收益需要单独对照

最终组合对Y使用 `cacheModifier=cs` 的store；生成的PTX中可见对应流式store。完整2048流式版几何1.4322×（B级，未独立复测），混合版1.4397×（A级）。

这支持“该组合值得采用”，**不支持“.cs单独贡献了某个百分比”**。`.ca`版与`.ca+.cs`版没有可接受的完整同会话消融：首次同时包含它们的测量因基线漂移被整轮弃用，不能再抽取其中有利点作归因。

**可迁移判断**：流式输出和后续会重复读的数据应分别考虑。先提出缓存假设，再做有效的同会话对照；不能把LLIR缓存组合的收益直接贴到TTIR某个modifier上。

### E5. 先用NCU区分“资源问题”和“供给问题”，再用benchmark决定

相同NCU采集条件（base clock、cache-control=all）下，最终版实际occupancy约55%–57%，parent约33%–35%。

| N | parent NCU µs | 最终版 NCU µs | parent / 最终版 DRAM Tbyte/s |
|---:|---:|---:|---:|
| 4096 | 31.360 | 24.576 | 1.613 / 1.905 |
| 8704 | 62.752 | 50.144 | 2.011 / 2.417 |
| 15872 | 89.664 | 97.696 | 2.723 / 2.494 |

**证据为C级机制线索**：[NCU摘要](../layer_norm/results/round2/ncu_finalists_summary.json)、[原始报告](../layer_norm/results/round2/ncu_finalists.ncu-rep)。中等N变快，大N却变慢；因此occupancy不能单独作为目标。不能将NCU固定时钟/清缓存耗时替换成常规CUDA graph计时。

可复用的顺序：编译资源 → 实际occupancy和访存/依赖指标 → 检查目标指令 → 正常配对计时 → 完整矩阵与独立复测。工具用法见 [diagnosis.md](diagnosis.md)。

## 无效、收益不足和不采用的尝试

“无效”是指本轮没有成为最终选择，不等于所有算子上都无效。下表B级均为6点/3轮/rep=10ms的初筛；正确性通过不等于性能值得采用。

| 尝试 | 本轮观测 | 为什么未采用 / 下一次怎么判断 |
|---|---|---|
| 只做1024/2048动态窄循环 | 32寄存器、无spill；最差相对baseline仅0.7836×/0.7508×（B） | 资源很好但大N回退；检查循环、地址和访存成本，不只看寄存器 |
| `r2_fold2048`：8192加载，reshape/reduce折叠为2048累加 | 39寄存器；初筛几何1.2493×、最差1.0301×（B） | 提供可行表达，但本轮整体不足以入选；未证明形状折叠普遍无效 |
| 延后Mean/Rstd写出以复用输入 | `r2_delay`初筛几何1.4017×，80寄存器、4 spill（B/C） | 部分点改善但活跃状态仍大；无别名契约下才允许重排 |
| 延后store再叠加affine/FMA改写 | `r2_delay_affine_v2`初筛几何1.2008×、最差0.9600×，94寄存器（B/C） | “各自合理”的变换组合后仍会退步；需重新验证组合，不将收益相加 |
| 用volatile保持重载 | `r2_fine1024_reload`为40寄存器，但N=15872在NCU下108.416µs，parent89.504µs（C） | occupancy升高仍慢，long-scoreboard/issue为18.90 vs 5.67；机制未单独消融，不推出“volatile总是错误” |
| 直接写张量`arith.negf` | 两个affine初版解析后lowering失败（C） | 先最小编译探针；本轮改用`-0.0-x`后通过，不把它外推为所有入口/版本不支持 |
| 只删调试loc | `.text`相同（C），初筛约1.000×（B） | 是阅读/比较辅助，不是算术或访存优化；完整cubin可因debug不同 |
| 只做倒数复用/rsqrt | 首轮`arithmetic`初筛约1.009×（B） | 可作为廉价候选，不能预设是主要瓶颈；压力误差检查仍必要 |

来源：[初筛1](../layer_norm/results/round2/screen_batch1.json)、[初筛2](../layer_norm/results/round2/screen_batch2.json)、[资源](../layer_norm/results/round2/resources.json)、[NCU初批](../layer_norm/results/round2/ncu_batch1_summary.json)、[编译失败日志](../layer_norm/results/round2/compile_v2.log)、[首轮初筛](../layer_norm/results/screening.json)。

## 流程中的教训

### P1. 舍弃不稳定测量，不能只删掉不喜欢的样本

第一次完整测试在N=1024上出现同一baseline约31–68µs的漂移，原因未确定。该次运行被中止并整轮排除，原始部分结果仍保留。后续所有版本都加相同的untimed graph稳定阶段，再重跑整个矩阵并独立复测。

证据：[measurement_validity.json](../layer_norm/results/round2/measurement_validity.json)、[被弃用样本](../layer_norm/results/round2/full_benchmark.json)、[有效主测](../layer_norm/results/round2/full_stable.json)。预热参数是 `--stabilize-ms 100`，实际调用graph benchmark的rep参数，不是保证固定100ms墙钟等待。预热不证明无干扰，也不能倒推最初漂移一定由NCU、时钟或其他进程造成。

### P2. 明确baseline与parent两个参照

最终版相对原始baseline的30点加速都大于1，但相对上一版只有22点更快，最大回退3.63%。这两句话同时成立。

在N=13312，parent约77.775µs，最终版约80.598µs；不能因为全矩阵平均改善就宣称逐点不退步。新算子应提前声明优化目标是整体平均、特定分布，还是逐点无回退；本轮没有承诺逐点无回退。

### P3. 平均逼近LLIR，不等于证明TTIR层级优于LLIR

B级补充同协议观测中，用同一原生baseline重算，LLIR几何约1.412×、本轮TTIR约1.440×，优化版耗时几何比约0.981。LLIR候选仍来自旧离线后端，比较没有独立重复，不能据此判定层级的普遍优劣。不得再拿旧LLIR自身较慢baseline的1.476×与新TTIR的1.440×直接比较。

来源：[llir_comparison.json](../layer_norm/results/round2/llir_comparison.json)、[baseline审计](../layer_norm/baseline_audit.md)。

### P4. 测试入口也会失败，不能把脚本失败记成kernel失败

独立复测第一次因验证文件没有parent名称 `tile_dispatch_math_v2` 而在计时前抛出 `KeyError`。修复时核对它与当时`optimized`的TTIR字节/哈希一致，登记验证别名并使用新验证文件，随后重新运行；没有修改kernel或补造性能样本。

来源：[失败日志](../layer_norm/results/round2/independent_repeat.log)、[别名来源](../layer_norm/results/round2/accuracy_finalists.json)、[成功复测](../layer_norm/results/round2/independent_repeat_v2.log)。验证记录应以**不可变候选名和内容哈希**为身份；`optimized.ttir`是可更新别名，不能单独作为长期身份。

## 给新算子的行动建议

1. 先固定同链路baseline，确认文件入口和源码路径的下游产物一致，再分析任何加速比。
2. 从逐shape差距选少量假设，写清控制项、预期的下游变化及可能退步的范围。
3. 看逻辑状态、循环、布局、访存和整体资源，不把某一个指标设为唯一目标。
4. 每个候选先全数值验证，再初筛；组合候选重新验证，不继承父版本PASS。
5. 对最终版跑完整矩阵、独立复测、分支边界和sanitizer；同时报告对baseline和parent的结果。
6. 保留失败、弃用测量和不可归因部分，给出源路径、哈希和证据等级。

MM需要重新分析`tt.dot`、数据复用、pipeline与shared资源，不能直接复制LayerNorm的1024/2048分组或8-warps配置。
