# MoE grouped/biased routing TTIR 第二轮

[最终报告](report.md) · [冻结30点协议](optimization_plan.md) · [矩阵](manifest30.json) · [候选生成器](make_candidates.py) · [选择](selection.json) · [机器可读汇总](summary.json)

本目录不修改父实验文件。父版本为membership，新候选来自其冻结TTIR；两个kernel的ABI/grid/options/workspace不变。原8点是历史证据，当前主/复测只汇总冻结30点。

先确认GPU空闲，使用官方venv与独占设备；在仓库根运行，结果目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES=2 /root/venvs/ir-ops-dev/bin/python moe_routing/round2/run.py \
  --run user_repeat --variants r2_rank_ordered_gather --rounds 5 --rep 20 --seed 234
```

脚本自动先核验原baseline源码/TTIR pipeline并做完整数值校验，然后再计时。`--check-only`只做pipeline和准确性；`--compile-only`只编译。`make_candidates.py`可重建并逐字节核对不可变候选。

完整CUDA graph计时包括两个kernel，不含Python/分配/编译。增量是parent/candidate，累计是baseline/candidate，均由同轮直接时间获得；不能用旧8点收益乘新30点增量。

所有原始样本、输入seed/hash、pipeline/准确性、完整编译metadata/哈希和sanitizer日志保存在results/。artifacts/可从TTIR重建。更新本文报告使用`/root/venvs/ir-ops-dev/bin/python moe_routing/round2/report.py`，仅接受完整有效证据。
