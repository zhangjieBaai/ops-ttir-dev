# Attention dK/dV 源码回写

[三路报告](report.md)：原直接TTIR变换忠实回写为 [frozen_port.py](frozen_port.py)。四个组的源码产物 ELF `.text` 与原 `r1_dv_late` 完全相同，复现D64收益和D128退步；没有额外调优。关键数值见 [summary.json](summary.json)。

- [冻结源码与父候选哈希](source_manifest.json)
- [三路主测](results/main_v2/benchmark.json) / [独立复测](results/repeat_v2/benchmark.json)
- [12点pipeline](results/pipeline_main_v3/pipeline.json)：原baseline门槛与新source->TTIR重编译门槛均通过
- [数值验证](results/accuracy_main_v3/accuracy.json) / [独立数值](results/accuracy_repeat/accuracy.json)：各84项，合同未变
- [memcheck](results/memcheck.log)：0 errors；[racecheck](results/racecheck.log)：0 errors、0 warnings
- [早期harness失败与排除原因](results/debug_dispositions.json)

从仓库根目录运行；先选空闲GPU，输出目录必须不存在。固定官方Triton3.7.0 venv：

```bash
CUDA_VISIBLE_DEVICES=4 /root/venvs/ir-ops-dev/bin/python attention_dkdv/source_port/verify_pipeline.py \
  --output attention_dkdv/source_port/results/my_pipeline
CUDA_VISIBLE_DEVICES=4 /root/venvs/ir-ops-dev/bin/python attention_dkdv/source_port/validate_source.py \
  --output attention_dkdv/source_port/results/my_accuracy
CUDA_VISIBLE_DEVICES=4 /root/venvs/ir-ops-dev/bin/python attention_dkdv/source_port/benchmark_threeway.py \
  --pipeline attention_dkdv/source_port/results/my_pipeline/pipeline.json \
  --accuracy attention_dkdv/source_port/results/my_accuracy/accuracy.json \
  --output attention_dkdv/source_port/results/my_perf --rounds 7 --rep 20 --warmup 500
```

原main/repeat因A/A门槛未通过已整轮排除；当前结果使用统一每路500ms预热的完整v2重测。

独立复测给验证器和计时器都加 `--seed 123`，计时 `--rounds 5`。三路由同一函数分别启动原TTIR baseline、原TTIR候选和正常Python JIT产物；source route使用AST CompiledKernel所需的constexpr绑定，设备ABI/grid不变，CUDA graph时长不含Python绑定成本。

sanitizer：

```bash
CUDA_VISIBLE_DEVICES=4 compute-sanitizer --tool memcheck --error-exitcode 90 \
  --kernel-name regex=_attn_bwd_dkdv_only \
  /root/venvs/ir-ops-dev/bin/python attention_dkdv/source_port/validate_source.py \
  --source-only --seed 99 --output attention_dkdv/source_port/results/my_memcheck
```

racecheck另用新目录、`--tool racecheck --seed 100`。`frozen_port.py`已在冻结前格式化，禁止继续修改；它忠实保留上游wrapper的未用LN2常量，lint对该冻结文件豁免F841。其余文件black/isort/flake8通过。所有新增内容仅在本目录，不改变父实验既有选择或结果。
