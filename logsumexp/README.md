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
Current result: [report_round4.md](report_round4.md),
[selection](results/selection_round4.json), [acceptance](results/acceptance_round4.json).
Selected `r4_focus` targets previously flat shapes: 8-row CTA batching for
small inner inputs (same host grid, masked unique row ownership), 32-lane
streams for one-row small inputs, and bounded i32 non-inner indexing.
Whole 30-case geometric speedup vs original baseline: 1.1221x main /
1.1234x repeat; arithmetic 1.1401x / 1.1417x. The separately frozen 15-case
subset and every per-shape result are reported without hiding unchanged points.
All 51-point pipeline, two-seed accuracy and sanitizers pass. Failed candidates
and the invalid first screen are retained; no claim that every flat case improved.

Historical reports: [round one](report_round1.md), [round two](report_round2.md),
[round three](report_round3.md). Previous selections/acceptance are archived
in the corresponding `results/runs/r*_setup/` directories. Original baseline,
IR candidates, options and shape mapping remain immutable.

```bash
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_accuracy.sh
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_performance.sh
```

Do not rerun export_baseline.py over the existing freeze. make_candidates.py and
make_mixed.py use exclusive creation and refuse overwriting candidates. To audit
generation without changing files, run audit_artifacts.py with a saved benchmark
and a fresh --output path. Exact historical generator snapshots are retained in
results/generators/; current generators produce the same candidate bytes.
