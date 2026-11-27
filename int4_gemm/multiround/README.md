# INT4 三轮连续TTIR优化

[总报告](report.md) · [机器可读汇总](summary.json) · [冻结协议](optimization_plan.md)

每轮同一30点、同配置，原baseline和第二轮anchor固定。parent仅在完整验收后前移；r3/r4/r5是不同结构实验，不是多seed计时。每轮完整数据见各自report.md和results/，选中IR映射见selection.json。

先用nvidia-smi隔离空闲GPU，官方Python固定 `/root/venvs/ir-ops-dev/bin/python`。例如重新检验r3（新输出目录必须不存在）：

```bash
CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python int4_gemm/multiround/run.py \
  --stage r3 --run independent_new --names baseline,anchor,parent,group128 \
  --rounds 5 --rep 20 --seed 234
```

该入口自动先核查全部30点source/file baseline pipeline、实际options和全部数值/压力，再计时。`--check-only`跳过计时。计时对象全部为TTIR文件编译产物，源代码只用于baseline pipeline核查。

`generate.py r3/r4/r5`可再生并逐字节检查候选，拒绝覆盖不同内容。各阶段manifest.json仅保存原始baseline的shape/options来源；实际parent来源和哈希在candidates.json，接受/保留链在selection.json。`workflow.py`是本轮已执行的一次性顺序任务记录，不应在既有results上重复运行。

GPU产物可再生，原始配对样本、候选TTIR、负结果和工具日志均保留。r5的精度检查在原浮点完全相等基础上额外核对FP16原始int16位型；数值合同未放宽。未修改旧round2或源代码回写实验。
