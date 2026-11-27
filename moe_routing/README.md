# MoE routing TTIR 实验

[最终版逐项消融](report_ablation.md)：完整30点、逐项加入、最终逐项移除、两阶段交叉消融，主测及独立复测。[当前优化报告](report.md)将最终结果与历史搜索分开；[消融契约](ablation_plan.md)、[机器可读审计](ablation_v1/summary.json)、[全部版本逐shape CSV](ablation_v1/all_shapes.csv)。

来自本地 FlagGems Hopper `grouped_topk.py` 的两个冻结设备 kernel：sigmoid → 每组biased top2之和 → top4 groups → top8 experts → 原始sigmoid权重及归一化。原baseline与最终TTIR使用同一官方Triton原生编译和同一launcher。候选只直接编辑冻结TTIR，不改Python kernel/TTGIR/LLVM/PTX/cubin。

## 测试入口

在仓库根目录先用`nvidia-smi`选择空闲且独占的H100。Python固定`/root/venvs/ir-ops-dev/bin/python`（官方Triton 3.7.0），同一设备不并发计时与其他GPU任务。以下以GPU2为例：

```bash
export CUDA_VISIBLE_DEVICES=2

# baseline及最终版和逐项消融：16路线×30点×8分布=3840项
bash moe_routing/test_accuracy.sh

# 全部30点性能：先pipeline及数值验证，再5轮配对计时
bash moe_routing/test_performance.sh

# 显式指定新目录、seed和轮数：逐项消融全流程
bash moe_routing/run_experiment.sh moe_routing/ablation_v1/results/my_main 42 5
```

默认按UTC时间建立独立目录，脚本拒绝覆盖旧目录。可用`RESULTS_DIR=moe_routing/ablation_v1/results/my_run`指定路径。独立复测可在另一张空闲H100运行：

```bash
CUDA_VISIBLE_DEVICES=3 bash moe_routing/run_experiment.sh \
  moe_routing/ablation_v1/results/my_repeat 123 3
/root/venvs/ir-ops-dev/bin/python moe_routing/report_ablation.py \
  --main moe_routing/ablation_v1/results/my_main/measurement \
  --repeat moe_routing/ablation_v1/results/my_repeat/measurement
```

核心入口：`verify_pipeline.py`核查冻结源码和TTIR的LLIR/PTX/cubin字节一致；`make_ablation.py`生成不可变逐项对照且核对final与r6已验收IR相同；`ablation_benchmark.py`逐30点pipeline、全部正确性和随机交错计时；`report_ablation.py`审计两次原始样本、IR/产物哈希、A/A和最终选择，生成报告及CSV。每次测试保留全部原始样本、输入种子/哈希、环境、资源与下游哈希。

## 实验契约

H100/SM90、连续FP32 logits与bias；E=256/512，连续等长G=8、选4组和8个expert，tokens为正16倍数。性能矩阵的tokens为16/32/64/128/256/384/512/768/1024/1536/2048/3072/4096/6144/8192，各配两个E，共30点。4 warps / 3 stages / 1 CTA；两个kernel、原grid与工作区；host仅分配和launch。sigmoid固定`0.5*tanh(0.5*x)+0.5`，tie选小索引，weight取未加bias分数。只覆盖有限输入，不外推NaN/Inf、其他E/G/K/dtype/stride。

正确性检查group scores和weights的FP32原始位型、精确indices、CPU稳定参考、canary、输出覆盖和输入不变。计时是预分配CUDA graph的完整两kernel设备时间，不含分配、编译、CPU launch，不代表端到端。缓存未清、自动频率。最终版已有[memcheck/racecheck](multiround/r6/report.md)；新增消融路线额外sanitizer的实际覆盖见[消融报告](report_ablation.md)。

可再生后端产物在忽略的`artifacts/`中；输入TTIR、冻结谱系和原始数值/性能JSON在`ablation_v1/`，纳入Git。冻结源在`frozen_kernels.py`与`upstream/`，禁止修改或格式化。新增输入范围或算法需要另建合同。

## 历史实验与当前选择

当前已验收选择为[r6 batch4](multiround/r6/selection.json)，各E两阶段IR在`multiround/r6/ir/e*_w4/optimized_s*.ttir`。本轮消融不修改该选择。

- [首轮协议](optimization_plan.md)、[上游来源和哈希](upstream/provenance.json)、[冻结性能映射](baseline.json)、[首轮8点报告](report_round1.md)。
- [第二轮扩展30点](round2/report.md)、[第三至五轮](multiround/report.md)、[第六轮](multiround/r6/report.md)。各轮结果只用于本轮对照，不能把历史平均增量相乘为最终加速比。
- [首轮源码回写实测](source_port/report.md)仅覆盖当时membership；最终组合的源码可表达性与实测边界见[消融报告](report_ablation.md)。
- 历史6点阶段消融在`stage_ablation/main`和`repeat`，保留为子集B级证据。本轮16路线的完整30点实验替代该子集作为逐项解释入口。
