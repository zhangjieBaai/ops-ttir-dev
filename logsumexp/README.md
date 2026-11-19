# logsumexp direct TTIR experiment

Baseline is the user-specified FlagGems `ops/logsumexp.py`, not
`special_logsumexp.py`. See [optimization_plan.md](optimization_plan.md) for the
frozen FP16 inner/non-inner contract, 30 performance shapes and 21 extra
correctness boundaries. Original NVIDIA heuristics are frozen per shape;
51 points map to 29 byte-verified TTIR/configuration groups in baseline.json.
No autotune is used. Source function bodies and int64 program_id are verified
against immutable snapshots by AST. Only registration decorators are isolated.

Use `/root/venvs/ir-ops-dev/bin/python` with one idle GPU selected through
`CUDA_VISIBLE_DEVICES`. `common.compile_ir` is the common upstream Triton
file compiler; `common.launch` is the common native CompiledKernel launcher.
The host wrapper performs checks/allocation/launch only. PyTorch compute is
used exclusively by data preparation and oracle checks.

Baseline receipts are in `results/runs/baseline_v1/`. The pipeline checks every
point: source TTIR, full backend options, normalized TTGIR, byte-identical
LLVM/PTX/cubin, and bitwise output equality. Installation RECORD attestation
is local; it is not independent remote wheel provenance. Artifacts are ignored,
but every compilation regenerates and records stage hashes and resources.

Kernel graph replay excludes allocation, Python submission, compilation and
loading. This experiment does not modify or qualify FlagGems public APIs.
First round complete: [report_round1.md](report_round1.md),
[selection](results/selection.json), [acceptance](results/acceptance.json).
Selected `r1_trim`: 30-case geometric speedup 1.0297x, independent repeat
1.0283x; non-inner benefit, inner machine code unchanged. Worst case
0.9880x / 0.9819x. Every `ir/<group>/optimized.ttir` matches the selected
immutable candidate. All 51-point pipeline, two-seed accuracy and final
memcheck/racecheck passed. Full paired samples and negative candidates retained.

```bash
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_accuracy.sh
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_performance.sh
```

Do not rerun export_baseline.py over the existing freeze. make_candidates.py and
make_mixed.py use exclusive creation and refuse overwriting candidates. To audit
generation without changing files, run audit_artifacts.py with a saved benchmark
and a fresh --output path. Exact historical generator snapshots are retained in
results/generators/; current generators produce the same candidate bytes.
