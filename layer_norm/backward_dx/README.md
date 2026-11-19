# LayerNorm backward dX TTIR

本目录独立于上一轮前向实验。baseline来自官方Triton3.7.0教程`_layer_norm_bwd_dx_fused`的**dX计算段提取**：保留该段的加载、两次规约和dX写出语句，双方均不计算dW/dB、不使用梯度累加锁。不是完整原版fused backward的性能测试。

- [首轮结果报告](report.md)：有限组级收益，中间两组保留baseline。
- 契约、候选预算及验收：[optimization_plan.md](optimization_plan.md)。
- 原始语句来源：[upstream/provenance.json](upstream/provenance.json)。
- 源码/TTIR下游一致性：[results/pipeline_baseline.json](results/pipeline_baseline.json)。
- 提取版dX与未修改完整fused教程dX逐位一致：[results/extraction_check.json](results/extraction_check.json)。
- 五组基线：`ir/b1024`、`b2048`、`b4096`、`b8192`、`b16384`；按N的next-power-of-two选择（最小1024）。组内baseline和候选同options、grid、ABI、加载和launcher。

## 测试入口

从仓库根目录运行，先用nvidia-smi选空闲GPU，结果目录必须是新目录：

```bash
CUDA_VISIBLE_DEVICES=7 bash layer_norm/backward_dx/test_accuracy.sh
CUDA_VISIBLE_DEVICES=7 bash layer_norm/backward_dx/test_performance.sh
```

Python固定`/root/venvs/ir-ops-dev/bin/python`（upstream Triton3.7.0，无FlagTree）。FP16 X/DY/W/DX，FP32保存统计量，N与stride为16倍数且N≤16384。计时只包括dX kernel的CUDA graph replay，不计计算保存统计量、前向、autograd、分配/编译/加载或dW/dB。

对照native ATen autograd和基于实际保存统计量的FP64公式；小用例额外CPU FP64 autograd，记录误差、有限性、canary和输入不修改。135个用例/版本。数值容差与前向不同，详见契约，不把梯度大绝对值误差直接称失败或放宽容差以接受候选。

候选名不可变，`optimized.ttir`是验收后选择的便捷副本；未达到组内主/复测至少1.02×及最差0.95×条件则保留该组baseline。失败或未采用候选均保留。固定family不同于每个shape重新autotune，不外推任意dtype/layout/行宽。


首轮候选与原始证据位于`results/`；完整主测`full.json`、独立复测`repeat.json`、A/A控制、NCU和失败日志均保留。`finalize.py`按主测冻结组级候选，复测与sanitizer只负责验收，不重新选择复测中最有利的候选。

## 第二轮瓶颈与负结果

[第二轮报告](report_round2.md)：新增8个候选均未优于第一轮parent，保留当前分组选择。报告包含同条件NCU、实际DRAM读写字节数、寄存器/同步反汇编、完整30点和独立复测；原始证据位于`results/round2/`。
