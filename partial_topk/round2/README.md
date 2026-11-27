# partial_topk 第二轮：扩展30点

[报告](report.md) · [汇总](summary.json) · [冻结30点映射](expanded/manifest.json) · [选择](expanded/selection.json)

parent为 `ordered_carry_v3`，新候选为 `static_unroll_v2`。父目录原baseline/optimized及source_port保持不变；新的正式证据和optimized别名在expanded/。

从仓库根使用官方venv并先隔离空闲GPU；已有manifest时不要重跑round2_expand.py。可重建候选并核对字节：

```bash
/root/venvs/ir-ops-dev/bin/python partial_topk/round2/make_candidates.py --expanded
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python complex_suite/round2_runner.py \
  --op partial_topk --expanded --run independent_new --names static_unroll_v2 --rounds 5 --rep 20 --seed 234
```

结果目录必须不存在。脚本在所有候选完整数值验证后才计时，每次执行原baseline源码/文件pipeline和完整options核验。`--check-only`仅验证。

小矩阵pilot与失败文件保留；最终报告不混入其样本。Root脚本和本轮共同门槛见 [round2_plan.md](../../complex_suite/round2_plan.md)。
