# Attention第三至第五轮

[最终报告](report.md) · [汇总](summary.json) · [协议](optimization_plan.md)

每轮32点，r3/r4/r5是真实的不同候选搜索。parent只从上一轮已验收选择接续，anchor始终为第二轮已验收版本。所有kernel候选从TTIR文本变换产生；原历史源码、IR、选择和报告不改。

`run_locked.py`是一次性完整编排，含GPU4 SM1200MHz锁频和finally恢复、子进程清理。已有结果后不可原地重跑；需新实验目录/运行ID。单步复现时设置`ATTENTION_MR_ROUND=r3`（或r4/r5），使用官方venv与独占GPU，并沿用原锁频/恢复协议，例如：

```bash
bash <<'SH'
set -e
ATTENTION_GPU=4
nvidia-smi -i "$ATTENTION_GPU" --lock-gpu-clocks=1200,1200
trap 'nvidia-smi -i "$ATTENTION_GPU" --reset-gpu-clocks' EXIT
export CUDA_VISIBLE_DEVICES="$ATTENTION_GPU"
export ATTENTION_MR_ROUND=r5 ATTENTION_MR_LOCK_SM_MHZ=1200
/root/venvs/ir-ops-dev/bin/python attention_dkdv/multiround/verify_pipeline.py \
  --output attention_dkdv/multiround/r5/results/new_pipeline
/root/venvs/ir-ops-dev/bin/python attention_dkdv/multiround/validate.py \
  --selection attention_dkdv/multiround/r5/selection_pre_main.json --seed 123 \
  --output attention_dkdv/multiround/r5/results/new_accuracy
/root/venvs/ir-ops-dev/bin/python attention_dkdv/multiround/benchmark.py \
  --selection attention_dkdv/multiround/r5/selection_pre_main.json --seed 123 --rounds 5 --rep 20 \
  --accuracy attention_dkdv/multiround/r5/results/new_accuracy/accuracy.json \
  --pipeline attention_dkdv/multiround/r5/results/new_pipeline/pipeline.json \
  --output attention_dkdv/multiround/r5/results/new_performance
SH
```

性能命令见benchmark.py的`--help`，使用新的输出目录。`report_all.py`只从已验收数据重建报告，不运行GPU。
