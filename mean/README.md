# mean_dim TTIR 实验

本轮只优化 `K>=1024` 的向量循环路径；连续 FP16 输入 `[M,N,K]` 沿 N 取均值，FP32 累积、FP16 输出。性能集固定30点，另有12个边界正确性点。最终结果与限制见 [首轮报告](report_round1.md)，原始证据见 `results/runs/final_gpu6/`。

同一组的 baseline/optimized 只改变TTIR文件内容，使用同一原生 Triton 3.7.0 编译函数、配置、grid、CompiledKernel launcher及CUDA graph计时。`baseline.json`冻结42个shape到26个TTIR/配置组的映射；每个组的最终文件为`ir/<group>/optimized.ttir`。源码参考为用户指定本地FlagGems mean.py快照。

先用 `nvidia-smi` 选空闲设备，再从 `/root/ops-ttir-dev` 运行：

```bash
CUDA_VISIBLE_DEVICES=6 bash mean/test_accuracy.sh
CUDA_VISIBLE_DEVICES=6 bash mean/test_performance.sh
```

脚本每次创建新目录，不覆盖历史证据。第二个入口包含pipeline、正确性、30点7轮主测与独立进程5轮复测。最终内存检查可单独运行，输出路径必须未存在：

```bash
CUDA_VISIBLE_DEVICES=6 compute-sanitizer --tool memcheck --error-exitcode 86 \
  /root/venvs/ir-ops-dev/bin/python mean/validate.py optimized --sanitizer \
  --output /root/tmp/mean-memcheck-new.json
CUDA_VISIBLE_DEVICES=6 compute-sanitizer --tool racecheck --error-exitcode 87 \
  /root/venvs/ir-ops-dev/bin/python mean/validate.py optimized --sanitizer \
  --output /root/tmp/mean-racecheck-new.json
```

`export_baseline.py`、`make_candidates.py`、`select.py` 为写一次的建档入口，已有冻结文件时拒绝覆盖。检查候选直接生成的可重现性可阅读 `verify_candidates.py` 及 `results/generator_reproduction.json`，所有候选来自TTIR文本操作，没有候选Python kernel。`report_results.py`从原始结果派生报告。

GPU7中止的两次正式计时（`final/`和`final_clean/`）因其他计算进程干扰全部排除。`setup_attempts/line_info/` 是统一关闭调试行号前的历史基线，常规执行不会选用。
