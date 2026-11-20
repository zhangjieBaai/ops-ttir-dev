# var_mean：直接 TTIR 优化实验

H100 / 官方 Triton 3.7.0，连续 FP32 输入最后一维按行归约。30 个 shape 主测几何平均加速 **1.2928×**，独立进程复测 **1.2947×**；计时口径为预分配输出的 CUDA graph kernel 时间。

最终候选是 `r2_short_hybrid_aligned_v1`，与 `ir/optimized.ttir` 字节一致。短行使用单 tile，长行采用256宽两遍中心化循环，并在 TTIR 声明契约保证的 N 16整除前提。两版均为32 registers/thread、0 spill、16 bytes shared。

[完整报告与30点结果](report_round2.md)包含配对区间、数值容差、源码/TTIR编译链核查、负结果、memcheck/racecheck和适用范围。源码及helper的int64行索引语义经过AST核查；早期提取遗漏该转换的试验已归档，全部性能结论使用修正后的 `var_mean-row-f32-v3` 契约与 r2 数据。

```bash
cd /root/ops-ttir-dev
CUDA_VISIBLE_DEVICES=5 bash var_mean/test_accuracy.sh
CUDA_VISIBLE_DEVICES=5 bash var_mean/test_performance.sh
```

脚本创建新结果目录，要求独占空闲 GPU。性能脚本依次进行编译链核查、正确性、计时控制、30点配对主测和独立进程复测、两项 sanitizer 检查。默认设备为物理GPU5，可通过CUDA_VISIBLE_DEVICES指定另一张空闲H100，其他卡的数据应单独报告。

性能矩阵为 M∈{1,17,128,1024,4096} × N∈{128,512,1040,4096,8192,16384}。支持契约要求1≤M≤4096、16≤N≤16384、N是16的倍数、correction=0/1，无别名。全张量两阶段路径、FP16/BF16、非连续输入和端到端 API 性能不在本轮结论范围内。30点已有历史暴露，不作为新 holdout。

契约：[contract.json](contract.json)。原始证据：[results/runs/r2](results/runs/r2/)。最终选择：[selection.json](results/runs/r2/selection.json)。基线冻结提交：`3c65f20`。KernelForge的审查记录位于 `/root/KernelForge-workspace/runs/var-mean-ttir-r2/`；本实验未添加其算子执行适配器。

三项改动的完整2³消融和上游Triton可表达性验证见 [消融报告](ablation_v1/report_final.md)。其中包含单项收益、最终上下文中的负交互、全部12个条件比较，以及入口参数提示被丢弃的源码反例。
