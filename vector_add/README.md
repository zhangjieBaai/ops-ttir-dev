# Vector add: direct TTIR optimization

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
