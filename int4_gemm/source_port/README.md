本轮实测已知 TTIR 优化直接回写为 Triton 源码后的性能。

- [结果报告](report.md)：同轮 baseline / optimized TTIR / source JIT 三路比较。
- [冻结源码](source_v1.py)：只使用公开 Triton API，源码指纹在 [source_manifest.json](source_manifest.json)。
- [实验协议](optimization_plan.md)、[三路原始样本](results/main_v2/results.json)、[独立进程复测](results/repeat/results.json)。
- [汇总JSON](summary.json)、[静态机器码与资源核查](static_analysis.json)、[编译环境指纹](toolchain.json)。

在仓库根目录运行，先用 nvidia-smi 选择空闲GPU。新结果目录不得存在：

```bash
CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python int4_gemm/source_port/run.py \
  --run my_main --rounds 7 --rep 20 --seed 42
CUDA_VISIBLE_DEVICES=0 /root/venvs/ir-ops-dev/bin/python int4_gemm/source_port/run.py \
  --run my_repeat --rounds 5 --rep 20 --seed 123
```

每次自动核验两个 pipeline 和三路完整options、执行全数值矩阵，之后才计时。`--check-only`跳过计时；可以在compute-sanitizer下使用。源码路计时持有原始JIT产生的ASTSource CompiledKernel；导出的TTIR文件版只参与pipeline验证。

分析重建：`/root/venvs/ir-ops-dev/bin/python int4_gemm/source_port/report.py`。它读取历史JSON和sanitizer日志，不进行GPU测试。
