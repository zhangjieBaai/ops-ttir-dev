# TTIR experiments

- 当前实验文档是 `layer_norm/optimization_plan.md`。先读契约，再运行实验。
- 不修改 `layer_norm/tutorial.py`、上游快照或 `ir/baseline.ttir`；新优化只能直接改 TTIR。
- 同名候选不可静默重写。失败候选和结果保留，用新版本名修正。
- 固定官方 Triton 环境 `/root/venvs/ir-ops-dev/bin/python`。先隔离空闲 GPU；同一设备不并行跑 benchmark/sanitizer/其他 GPU 任务。
- 所有候选先正确性后性能；计时必须对照同路径重编译的 baseline，记录全部配对样本和 IR 哈希。
- 本轮 FP16、N/stride 都为 16 的倍数、无别名。新增范围/dtype 必须单独声明和验证，不能外推现有结论。
- Python 参考计算仅用于测试；候选 launch 仅使用从 TTIR 编译的 kernel。
- 提交前 `pre-commit run --all-files`。教程提取文件为冻结源码，已排除格式化。

- 每次主实验先运行 `verify_pipeline.py`：baseline.ttir 与 optimized.ttir 同用 Triton 3.7.0 原生后端，baseline 的 LLVM/PTX/cubin 必须与教程源码路径一致。禁止混入独立 llc 路径。`common.compile_ir` 会拒绝系统环境、FlagTree/FlagPrism 或错误 Triton 版本。
