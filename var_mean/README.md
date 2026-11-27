# var_mean：直接 TTIR 优化实验

当前统一展示见[完整结果](report.md)：每个shape只列基线时间、优化时间、加速比，保留算术/几何平均和方法收益。

H100 / 官方 Triton 3.7.0，连续 FP32 输入最后一维按行归约。30 个 shape 主测几何平均加速 **1.3416×**，独立进程复测 **1.3408×**；计时口径为预分配输出的 CUDA graph kernel 时间。

当前候选为完整消融中的 **A+C**（`ab4_101`），与 `ir/optimized.ttir` 字节一致：短行使用单 tile并复用输入，长行保留1024宽两遍中心化循环，并在 TTIR 声明契约保证的 N 16整除前提。基线与当前版均为32 registers/thread、0 spill、16 bytes shared。以上为GPU4固定30点矩阵及独立进程复测的A级性能证据，原始记录见[完整消融报告](ablation_v1/report_final.md)。

晋升移除了B（长行256宽循环）：在A+C上加入B会使主测耗时增加3.46%。原A+B+C保留在 `ir/r2_short_hybrid_aligned_v1.ttir` 和 `ir/ab4_111.ttir`。原r2选择、消融manifest和报告保持冻结；历史报告中的optimized指当时的A+B+C，当前身份以[selection.json](selection.json)为准。

晋升后在GPU4重新运行完整性能脚本：30点主测/独立复测为 **1.3412× / 1.3399×**，编译链、两seed正确性、计时控制及memcheck/racecheck均通过。新optimized的LLVM/PTX/cubin与原消融A+C逐字节一致。记录见[晋升验证](results/runs/promotion_ac_v1/outcome.json)，与上面的原消融统计分开保存。

[上一轮A+B+C报告与30点结果](report_round2.md)包含配对区间、数值容差、源码/TTIR编译链核查、负结果、memcheck/racecheck和适用范围。源码及helper的int64行索引语义经过AST核查；早期提取遗漏该转换的试验已归档，当前性能结论使用修正后的 `var_mean-row-f32-v3` 契约与完整消融数据。

```bash
cd /root/ops-ttir-dev
CUDA_VISIBLE_DEVICES=5 bash var_mean/test_accuracy.sh
CUDA_VISIBLE_DEVICES=5 bash var_mean/test_performance.sh
```

脚本创建新结果目录，要求独占空闲 GPU。性能脚本依次进行编译链核查、正确性、计时控制、30点配对主测和独立进程复测、两项 sanitizer 检查。默认设备为物理GPU5，可通过CUDA_VISIBLE_DEVICES指定另一张空闲H100，其他卡的数据应单独报告。

性能矩阵为 M∈{1,17,128,1024,4096} × N∈{128,512,1040,4096,8192,16384}。支持契约要求1≤M≤4096、16≤N≤16384、N是16的倍数、correction=0/1，无别名。全张量两阶段路径、FP16/BF16、非连续输入和端到端 API 性能不在本轮结论范围内。30点已有历史暴露，不作为新 holdout。

契约：[contract.json](contract.json)。原始证据：[results/runs/r2](results/runs/r2/)。当前选择：[selection.json](selection.json)。上一轮选择：[r2/selection.json](results/runs/r2/selection.json)。基线冻结提交：`3c65f20`。KernelForge的审查记录位于 `/root/KernelForge-workspace/runs/var-mean-ttir-r2/`；本实验未添加其算子执行适配器。

三项改动的完整2³消融和上游Triton可表达性验证见 [消融报告](ablation_v1/report_final.md)。其中包含单项收益、最终上下文中的负交互、全部12个条件比较，以及入口参数提示被丢弃的源码反例。
