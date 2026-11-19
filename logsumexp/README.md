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
Current round: [report_round2.md](report_round2.md),
[selection](results/selection_round2.json), [acceptance](results/acceptance_round2.json).
Selected `r2_select_after` preserves round-one short-reduction improvements and
uses one exponential per online recurrence update. All 30 cases: geometric
speedup 1.0615x vs original baseline, independent repeat 1.0605x. Direct paired
gain vs `r1_trim`: 1.0311x / 1.0309x. Worst vs parent: 0.9966x / 0.9929x.
All 51-point pipeline, two-seed accuracy, memcheck/racecheck passed. Raw samples
and all negative candidates retained. Old 20/10 split is reused regression
coverage, not a fresh holdout. Every optimized.ttir matches selection_round2.

Historical round one: [report_round1.md](report_round1.md),
[selection archive](results/runs/r2_setup/parent_selection.json),
[acceptance archive](results/runs/r2_setup/parent_acceptance.json).
Original baseline and immutable r1_trim files remain available.

```bash
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_accuracy.sh
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_performance.sh
```

Do not rerun export_baseline.py over the existing freeze. make_candidates.py and
make_mixed.py use exclusive creation and refuse overwriting candidates. To audit
generation without changing files, run audit_artifacts.py with a saved benchmark
and a fresh --output path. Exact historical generator snapshots are retained in
results/generators/; current generators produce the same candidate bytes.
