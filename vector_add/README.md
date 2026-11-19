# Vector add: direct TTIR optimization

本轮完整矩阵未达到预设 1.02x 晋升门槛，`optimized.ttir` 保留 baseline。
streaming store 在 N=2^22 有约 1.206x 局部收益，16 点几何均值仅 1.0131x。
实验细节和证据见 [report.md](report.md)；最终选择见 `results/selection.json`。

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
