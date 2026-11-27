# Attention backward dK/dV：直接 TTIR 优化实验

固定官方 fused dK/dV 提取基线下，`r1_dv_late` 在 D64 的 6 个点稳定加速；D128 保留 baseline。不是完整 Attention backward 的端到端收益。

基线来自 [官方 Triton v3.7.0 fused-attention](https://github.com/triton-lang/triton/blob/v3.7.0/python/tutorials/06-fused-attention.py)，将 `_attn_bwd` 截在 dQ 段之前，概率恢复、dP/dS、dK/dV 四个 dot 和因果分段保持原样。`frozen_source.py` 同时保存未截断完整 backward。12 点源码/文件入口 LLVM、PTX、cubin 全部字节一致；提取输出与完整官方 backward 的 dK/dV 逐位一致。

环境：官方 Triton 3.7.0 / PyTorch 2.12.0+cu130 / 独占物理 GPU4 H100 80GB，4 warps、5 stages、1 CTA。官方固定配置未经逐 shape autotune，尤其 D128 存在明显 spill；结果不证明战胜调优后的其他实现。冻结代码和来源哈希见 [provenance](upstream/provenance.json)，契约见 [optimization_plan.md](optimization_plan.md)。

## A 级性能结果

指标为预分配输出的 CUDA graph 设备 kernel 时间；不含 forward、K 预缩放、delta、分配、编译和 Python。每点主测7对/复测5对，rep=20ms，随机对内顺序；复测为独立进程与新输入seed=123。未锁频，无主动清缓存。所有原始样本和顺序在 [主测](results/main/benchmark.json) 与 [复测](results/repeat/benchmark.json)。

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

## 修改与机制边界

`r1_dv_late` 直接移动 TTIR 循环中 `pT -> fp16` 的 cast 和 dV dot，使其在 dK dot 后执行；causal 的两个循环都修改。数学公式、FP16转换位置相对于各自消费者、每个 accumulator 的加法顺序、签名、grid、编译options都保持一致。没有从优化 Python kernel 重新导出候选。

C 级静态证据见 [static_artifacts.json](results/static_artifacts.json)：D64非因果寄存器255→250，spill均0；因果255→255，spill均0，SASS指令数1378→1354。所有组 HGMMA 与 barrier 的静态数量不变。机器码和调度改变，但没有NCU消融证据把约12%的收益归因给某个单一资源量。

D128非因果 spill 178→204、因果182→160；后者 spill 下降却仍明显变慢。这是“减少spill不保证加速”的本次反例，不能仅凭寄存器/溢出数字选择方案。

## 初筛与负结果（B 级）

7个候选均保留在各组 ir/：unsigned div/rem、delta提前、dV后移、dP提前、load.cg、连续head地址折叠，以及 dV后移+unsigned 组合。初筛为完整12点但每点仅3对，选拔记录见 [screen](results/screen/benchmark.json) 与 [selection_pre_main.json](selection_pre_main.json)。D64非因果 unsigned 初筛几何0.9246×，dV后移+unsigned仅1.0562×，没有继承单项dV后移的全部收益。load.cg接近1×；delta/dP提前在D128退步。没有丢弃不利候选或用重复运行挑最小耗时。

## 验证与范围

初轮224项通过（28输入 × baseline与7候选），独立seed复测56项通过。候选输出均与baseline逐位相同；FP64原始注意力数学参考门槛固定为rtol=0.03, atol=0.008。包含N128/256/1024/2048、causal/noncausal、D64/128、zero_dO、constant和large QK；每轮4个小例另做CPU FP64 autograd。检查输入不修改、NaN sentinel完全覆盖及有限性。

最终两个版本各在完整28输入运行memcheck/racecheck：[memcheck](results/memcheck_v2.log) **0 errors**；[racecheck](results/racecheck_v2.log) **0 hazards / 0 errors / 0 warnings**，各含56项数值检查。早先filter参数写成`regex:`被工具拒绝，未执行kernel；失败原日志保留，修正为`regex=`后重新运行，未覆盖原结果。

仅覆盖连续FP16、N为128倍数、无dropout、无别名、预先具备LSE/delta和预缩放K的dK/dV子算子。固定配置结果不外推ragged序列、其他D/dtype、其他GPU或整个训练step。这是直接TTIR候选实验，没有构造文章中的A/B/C盲测，因此不能据此证明TTIR信息优于源码信息；dV调度变换也可能回写Triton源码表达。

复现入口见 [README.md](README.md)。冻结选择与IR哈希见 [selection.json](selection.json)，编译器/libtriton/ptxas哈希见 [toolchain.json](results/toolchain.json)。

新增的实际源码回写比较见 [source_port/report.md](source_port/report.md)，其中同轮比较原基线、TTIR优化版与正常Triton源码编译版。
