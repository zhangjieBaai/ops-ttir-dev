# LayerNorm TTIR 实验

官方 Triton v3.7.0 LayerNorm 前向实现导出为冻结的 `ir/baseline.ttir`。`ir/optimized.ttir` 是当前选中的 TTIR 优化版。两者经相同 Triton 原生后端编译和相同 launcher 执行；不通过独立 `llc → ptxas` 编译。

## 测试入口

所有命令从仓库根目录执行。先用 `nvidia-smi` 选择一张空闲 H100，下面以 GPU 7 为例。Python 环境固定为 Triton 3.7.0 的 `/root/venvs/ir-ops-dev/bin/python`，不使用 FlagTree 环境。

```bash
export CUDA_VISIBLE_DEVICES=7

# 正确性测试：baseline 和 optimized 各 142 个 shape/压力/边界用例
bash layer_norm/test_accuracy.sh

# 完整性能测试：30 个 shape，每点 7 轮，配对比较 baseline 与 optimized
bash layer_norm/test_performance.sh

# 从基线再生候选，依次执行正确性、候选筛选、全量 benchmark、复测和 sanitizer
bash layer_norm/run_experiment.sh
```

每次运行会在 `layer_norm/results/runs/<UTC 时间戳>/` 建立独立目录。性能测试只接受哈希与正确性结果相符的验证文件；输出目录已存在时会退出，不覆盖旧结果。也可以指定空闲设备或独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=3 RESULTS_DIR=layer_norm/results/runs/manual_01 \
  bash layer_norm/test_performance.sh
```

核心入口：

- `validate.py`：数值、压力、边界、stride padding 检查；`--suite smoke|full|sanitizer`。
- `benchmark.py`：预分配输出的 CUDA graph 配对计时；`--full` 跑 30 个 shape。
- `verify_pipeline.py`：确认环境、Triton 编译选项、baseline 下游产物和教程源码编译一致。
- `run_experiment.sh`：复现历史首轮候选搜索；当前版本复测使用 `test_performance.sh`。
- `make_round2.py`、`finish_round2.py`：第二轮候选生成、证据验收及选择；计划见 `optimization_plan_v2.md`。

## 当前结果

- [第二轮优化报告](report_round2.md)：当前推荐版本及完整复测、LLIR 对照、NCU 分析。
- [第一轮性能复测](results/perf_recheck_20261118T120303Z/report.md)：历史版本，几何平均 1.293×，算术平均 1.314×。
- [正确性与环境/编译链核查](results/native_pipeline_accuracy.json)、[pipeline 核查](results/native_pipeline_check.json)。
- [优化方法与完整首轮报告](report.md)。
- [TTIR 与 LLIR 的对照分析](llir_comparison.md)、[baseline 一致性核查](baseline_audit.md)。
- [当前推荐 TTIR](ir/optimized.ttir)、[冻结 baseline](ir/baseline.ttir)。

## 实验契约

H100 / SM90；FP16 输入、权重、偏置、输出；FP32 累加和统计量。8 warps、1 CTA、3 stages。输入列连续，X/Y 行 stride 相同，N 和 stride 是 16 的倍数，不支持别名。性能矩阵为 M=4096、N=1024…15872、间隔 512。计时是预分配输出的重复 CUDA graph kernel 时间，不含 CPU launch 或分配时间。

`artifacts/` 保存可再生编译阶段，已由 Git 忽略。正确性和性能原始样本、环境信息、IR/cubin 哈希保存在 `results/` 并纳入 Git。

## Backward dX 实验

前向实验之外，独立的教程dX计算段提取与TTIR优化位于 [backward_dx/](backward_dx/README.md)。该实验不包含dW/dB或梯度累加锁，不能与完整fused backward耗时混比。
