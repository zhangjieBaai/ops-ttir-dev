# Attention dK/dV：现有 TTIR 优化的 Triton 源码回写

源码回写完整复现原 `r1_dv_late`：四组编译后的 ELF `.text` 与直接 TTIR 候选逐字节相同，三路配对测量的源码/IR差异在计时波动量级。D64收益与D128退步都保留；这说明本次优化可由公开Triton源码表达，不构成TTIR独占能力。

唯一回写是把概率FP16 cast与dV dot移到dK dot之后。没有进一步调优tile、warp、stage、数学表达或测试数据。原文件不变；本目录[frozen_port.py](frozen_port.py)在black/isort后冻结，哈希见[source_manifest.json](source_manifest.json)。

## 三路性能（A级，仅本12点固定配置）

比值定义为前者耗时/后者耗时；IR/source>1表示源码更快。每点先取同轮比值中位数，再计算跨点几何均值。

baseline=原baseline.ttir；IR=原r1_dv_late.ttir，四组都测；source=正常Python JIT生成的CompiledKernel。source计时没有偷换成导出TTIR重编译版本。三路设备ABI、grid及全部后端options一致；Python绑定层为AST产物传递constexpr，但这些不是额外设备参数，计时不包含Python。

独占物理GPU4/H100，官方Triton3.7.0，4warps/5stages，完整原12点；每点同轮随机三路顺序，主测7轮、独立seed123复测5轮，rep=20ms CUDA graph，每路500ms预热，无锁频、无主动缓存清理。不含forward、预缩放K、delta、分配、编译、Python。全部原始时间/顺序/输入与产物哈希见[主测](results/main_v2/benchmark.json)及[复测](results/repeat_v2/benchmark.json)。

| 组 | baseline/IR 主测 / 复测 | baseline/source 主测 / 复测 | IR/source 主测 / 复测 |
|---|---:|---:|---:|
| d64_c0 | 1.1203 / 1.1214 | 1.1181 / 1.1253 | 0.9967 / 1.0031 |
| d64_c1 | 1.1239 / 1.1222 | 1.1241 / 1.1224 | 1.0001 / 1.0001 |
| d128_c0 | 0.9968 / 0.9986 | 0.9968 / 0.9987 | 1.0000 / 1.0002 |
| d128_c1 | 0.9408 / 0.9409 | 0.9407 / 0.9408 | 1.0000 / 0.9998 |
| 全12点 | 1.0424 / 1.0428 | 1.0419 / 1.0437 | 0.9992 / 1.0008 |

D64六点：baseline/source几何 **1.1211× / 1.1239×**；baseline/IR为1.1221× / 1.1218×。

全12点IR/source几何 **0.9992× / 1.0008×**。D128两路线都存在退步，正式使用应延续父实验D128保留baseline的选择；没有用人为1×覆盖负结果。

| (B,H,N,D,causal) | baseline μs | IR μs | source μs | baseline/IR 主/复 | baseline/source 主/复 | IR/source 主/复 |
|---|---:|---:|---:|---:|---:|---:|
| (1, 4, 256, 64, False) | 8.592 | 7.810 | 7.816 | 1.1001 / 1.0998 | 1.1001 / 1.1012 | 0.9997 / 1.0001 |
| (1, 4, 1024, 64, False) | 25.437 | 22.263 | 22.263 | 1.1426 / 1.1425 | 1.1425 / 1.1425 | 1.0000 / 1.0000 |
| (2, 8, 2048, 64, False) | 73.580 | 65.160 | 66.160 | 1.1186 / 1.1223 | 1.1121 / 1.1327 | 0.9904 / 1.0094 |
| (1, 4, 256, 64, True) | 9.723 | 9.054 | 9.049 | 1.0739 / 1.0734 | 1.0747 / 1.0737 | 1.0002 / 1.0003 |
| (1, 4, 1024, 64, True) | 26.502 | 23.625 | 23.625 | 1.1218 / 1.1221 | 1.1217 / 1.1222 | 1.0000 / 1.0000 |
| (2, 8, 2048, 64, True) | 69.671 | 59.136 | 59.130 | 1.1783 / 1.1734 | 1.1783 / 1.1734 | 1.0001 / 1.0000 |
| (1, 4, 256, 128, False) | 37.404 | 36.633 | 36.627 | 1.0211 / 1.0226 | 1.0211 / 1.0226 | 1.0002 / 1.0000 |
| (1, 4, 1024, 128, False) | 129.984 | 128.063 | 128.074 | 1.0150 / 1.0183 | 1.0150 / 1.0187 | 1.0000 / 1.0001 |
| (2, 8, 2048, 128, False) | 539.984 | 565.059 | 565.202 | 0.9557 / 0.9562 | 0.9555 / 0.9562 | 0.9998 / 1.0004 |
| (1, 4, 256, 128, True) | 42.837 | 45.983 | 45.987 | 0.9316 / 0.9310 | 0.9315 / 0.9312 | 1.0000 / 1.0001 |
| (1, 4, 1024, 128, True) | 127.951 | 140.713 | 140.717 | 0.9093 / 0.9097 | 0.9093 / 0.9097 | 1.0000 / 1.0000 |
| (2, 8, 2048, 128, True) | 452.574 | 460.442 | 460.391 | 0.9830 / 0.9833 | 0.9829 / 0.9830 | 1.0001 / 0.9995 |

同机器码不保证每轮耗时完全相等。保留全部配对样本，不把计时波动解释为不同的编译调度。
原main/repeat的A/A在大D64非因果点分别为1.0361/1.0307，超出预声明[0.97,1.03]门槛，整轮排除。
本报告仅使用完整main_v2/repeat_v2：所有路线统一500ms预热，仍保持7轮/5轮、rep20ms和原输入seed。

A/A各点3对的中位比范围：主测0.9948–1.0007，复测0.9998–1.0031。

## Pipeline与机器码证据

12点逐点验证原baseline源码与原TTIR的LLVM/PTX/cubin字节一致。新源码导出的TTIR与其重编译产物的LLVM/PTX/cubin也逐字节一致，输出逐位相同，见[pipeline](results/pipeline_main_v3/pipeline.json)和[独立pipeline](results/pipeline_repeat/pipeline.json)。

新源码与直接修改的IR文本因SSA命名及源码位置等存在差异，完整cubin也可能含不同调试信息；这里明确只宣称它们的ELF.text机器代码段完全一致。前端保留的loop顺序是QK→dP→dK→pT cast→dV，与直接TTIR变换一致，而baseline为QK→cast→dV→dP→dK。各组ELF.text哈希和完整产物哈希在pipeline中，逐组原始TTIR diff和SASS静态计数见[structural_analysis.json](results/structural_analysis.json)。

| 组 | baseline regs / spill | IR和source regs / spill | 三路shared bytes |
|---|---:|---:|---:|
| d64_c0 | 255 / 0 | 250 / 0 | 74752 |
| d64_c1 | 255 / 0 | 255 / 0 | 74752 |
| d128_c0 | 255 / 178 | 255 / 204 | 148480 |
| d128_c1 | 255 / 182 | 255 / 160 | 148480 |

资源/指令为C级静态证据，不把寄存器或spill的改变单独归因为性能因果；未运行NCU。

## 验证、失败记录与边界

原精度合同完全继承：原28输入、原分布及FP64数学参考、rtol=0.03/atol=0.008；主测84项和独立seed复测84项（三路线×28）全通过且source/IR与baseline输出逐位相同。每轮4个小例另做CPU FP64 autograd；检查NaN sentinel覆盖、有限性及输入不修改，没有删压力case或放宽容差。

新源码完整28输入：[memcheck](results/memcheck.log) **0 errors**，[racecheck](results/racecheck.log) **0 hazards / 0 errors / 0 warnings**；各28项数值检查通过。测试setup为获取JIT产物会额外启动4次源码kernel，同样接受sanitizer检查。

早期harness失败完整保留：[debug_dispositions.json](results/debug_dispositions.json)。首次错误地给AST CompiledKernel传TTIR绑定列表导致启动异常，随后修正constexpr绑定；v2精度全PASS但原失败进程未退出，主动排除并隔离重做v3。pipeline v2因tuple/list options比较产生false negative，改为逐字段JSON规范化比较后全部通过。没有任何数值失败通过放宽门槛处理，也未从失败/重叠运行采集正式性能。

原baseline未经逐shape autotune；结果只覆盖连续FP16、N为128倍数、D64/128、causal/noncausal及既有LSE/delta/预缩放K，不是完整attention backward或训练step收益。不支持本实验未声明的ragged、dropout或其它dtype。

复现命令见[README.md](README.md)。本目录没有覆盖父实验的baseline、候选、选择或报告。
