# Vector add: direct TTIR optimization

当前统一展示见[完整结果](report.md)：每个shape只列基线时间、优化时间、加速比，保留算术/几何平均和方法收益。

当前 `optimized.ttir` 已按用户接受局部收益的指示，采用 `r1_store_cs`。
原始 GPU 7 实验在 N=2^22 有约 1.206x 局部收益；原 1.02x 整体晋升门槛
仍记为未达到，此次选择明确记录了用户授权的政策变更。
当前正式对比见 [promotion_report_cs.md](promotion_report_cs.md)，选择记录见
`results/selection.json`。此前未晋升实验的历史报告见 [report.md](report.md)。

NCU 诊断和待执行的第二轮计划见 [optimization_plan_v2.md](optimization_plan_v2.md)：
小尺寸网格不足，中等尺寸有缓存/分组机会，大尺寸 DRAM 吞吐已约 92%。

Official tutorial snapshot: `upstream/01-vector-add.py.txt`; source and commit
hashes: `upstream/provenance.json`. Frozen contract: `optimization_plan.md`.
FP32, 16-byte aligned, non-aliasing contiguous vectors; positive lengths divisible
by 16 up to 2^27. Fixed 1024 elements/CTA and four warps. No autotune.

Use `/root/venvs/ir-ops-dev/bin/python`, explicitly isolating an idle H100:

```bash
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python vector_add/export_baseline.py --verify
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python vector_add/verify_pipeline.py --output /root/tmp/vector-add-pipeline-new.json
CUDA_VISIBLE_DEVICES=7 /root/venvs/ir-ops-dev/bin/python vector_add/validate.py baseline --output /root/tmp/vector-add-accuracy-new.json
```

KernelForge supplies workflow, optimization, evidence and independent review;
its current CLI has no vector_add execution adapter. Actual kernels compile
from TTIR through upstream Triton's native file entry, runtime and launcher.
CUDA graph kernel timings exclude compilation/allocation/Python issue and do
not reproduce the tutorial's default cache-clearing `do_bench` metric.
