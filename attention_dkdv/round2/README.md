# Attention dK/dV TTIR 第二轮

[完整结果](report.md) · [机器可读汇总](summary.json) · [实验协议](optimization_plan.md) · [32点矩阵](expanded_matrix.json) · [候选映射](active_manifest.json)

本目录继承父实验的数值、布局、ABI/grid、固定4warps/5stages。D64 parent为r1_dv_late，D128 parent为baseline；所有新候选只直接修改TTIR，原目录的冻结文件不变。

使用 `/root/venvs/ir-ops-dev/bin/python`，先通过nvidia-smi选择空闲卡并设置CUDA_VISIBLE_DEVICES。本次独占物理GPU4。输出目录必须不存在。

原始baseline32点pipeline核验：

```bash
CUDA_VISIBLE_DEVICES=4 /root/venvs/ir-ops-dev/bin/python attention_dkdv/round2/verify_pipeline.py \
  --output attention_dkdv/round2/results/my_pipeline
```

全候选完整44输入精度：

```bash
CUDA_VISIBLE_DEVICES=4 /root/venvs/ir-ops-dev/bin/python attention_dkdv/round2/validate.py \
  --output attention_dkdv/round2/results/my_accuracy
```

正式选择固定后复测：

```bash
CUDA_VISIBLE_DEVICES=4 /root/venvs/ir-ops-dev/bin/python attention_dkdv/round2/validate.py \
  --selection attention_dkdv/round2/selection_pre_main.json --seed 123 \
  --output attention_dkdv/round2/results/my_selected_accuracy
CUDA_VISIBLE_DEVICES=4 /root/venvs/ir-ops-dev/bin/python attention_dkdv/round2/benchmark.py \
  --selection attention_dkdv/round2/selection_pre_main.json --seed 123 --rounds 5 --rep 20 \
  --accuracy attention_dkdv/round2/results/my_selected_accuracy/accuracy.json \
  --pipeline attention_dkdv/round2/results/my_pipeline/pipeline.json \
  --output attention_dkdv/round2/results/my_performance
```

候选可按 `make_candidates.py` → `make_rows_v2.py` → `make_rows_v3.py` → `make_unroll.py` → `make_unroll_v5.py` 顺序从冻结TTIR重建，各文件对同名候选执行不可覆盖检查。v1/v2和展开v4的解析失败有意保留，不属于最终选择；`active_manifest.json`只列已修复候选。

基准同时记录原baseline、当前parent、新candidate，并分别报告累计baseline/candidate与增量parent/candidate。A/A超门槛的运行不能进入最终结论；性能初筛不能当作最终收益。候选虽从TTIR实现，其输出子块、循环展开和指针重算都可由公开Triton语法表达；未在本轮执行源码回写。

## 固定频率的正式协议

自动频率下，大D64非因果点在多轮A/A中超门槛；固定1200MHz后控制稳定。最终仅采用固定频率的完整32点筛选/主测/复测，不与自动频率数据混合。`clocks_locked=True`、requested1200MHz及观测频率随环境记录。预热是可复用32-launch CUDA graph，初始实际>=500ms、每次计时前实际>=100ms；性能计时仍调用原triton.testing.do_bench_cudagraph。

`run_locked.py`是本轮一次性完整编排，保存设置前后、返回码和恢复记录，try/finally保证正常或异常退出时重置GPU4自动频率。复现性能必须同样锁定独占设备并在子shell退出时恢复，例如：

```bash
bash <<'SH'
set -e
R2_GPU=4
nvidia-smi -i "$R2_GPU" --lock-gpu-clocks=1200,1200
trap 'nvidia-smi -i "$R2_GPU" --reset-gpu-clocks' EXIT
export CUDA_VISIBLE_DEVICES="$R2_GPU"
export ATTENTION_R2_LOCK_SM_MHZ=1200
/root/venvs/ir-ops-dev/bin/python attention_dkdv/round2/verify_pipeline.py --output attention_dkdv/round2/results/my_fixed_pipeline
/root/venvs/ir-ops-dev/bin/python attention_dkdv/round2/validate.py --selection attention_dkdv/round2/selection_pre_main.json --seed 123 --output attention_dkdv/round2/results/my_fixed_accuracy
/root/venvs/ir-ops-dev/bin/python attention_dkdv/round2/benchmark.py --selection attention_dkdv/round2/selection_pre_main.json --seed 123 --rounds 5 --rep 20 --accuracy attention_dkdv/round2/results/my_fixed_accuracy/accuracy.json --pipeline attention_dkdv/round2/results/my_fixed_pipeline/pipeline.json --output attention_dkdv/round2/results/my_fixed_performance
SH
```

所有输出目录必须是新目录；先确认GPU4无其他工作负载。只设环境标记而未实际锁频不符合本协议。
