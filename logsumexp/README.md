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
Current result: [report_round3.md](report_round3.md),
[selection](results/selection_round3.json), [acceptance](results/acceptance_round3.json).
Selected `r3_mix` meets the requested >10% average speedup on the frozen 30 cases:
arithmetic 1.1212x main / 1.1195x independent repeat; geometric 1.1028x / 1.1014x.
The main geometric 95% interval crosses 1.10; point-estimate attainment and
interval uncertainty are reported separately. Relative to round-two parent,
geometric gain is 1.0391x / 1.0383x; worst case 0.9376x / 0.9325x.
Large inner rows use a FP16 maximum before FP32 exponential/sum, and a guarded
full65536 tile replaces the online path. Small inner and all non-inner retain
round-two code. All 51-point pipeline, two-seed accuracy and sanitizers pass.
Two first repeat timings failed A/A and were excluded wholesale; their raw
receipts and invalidity records remain. Final accepted repeats use *_v2.json.
Old 20/10 partition is reused regression coverage, never a fresh holdout.

Historical reports: [round one](report_round1.md), [round two](report_round2.md).
Previous parent selections/acceptance are archived in `results/runs/r2_setup/`
and `results/runs/r3_setup/`. Original baseline and candidates remain immutable.

```bash
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_accuracy.sh
CUDA_VISIBLE_DEVICES=0 bash logsumexp/test_performance.sh
```

Do not rerun export_baseline.py over the existing freeze. make_candidates.py and
make_mixed.py use exclusive creation and refuse overwriting candidates. To audit
generation without changing files, run audit_artifacts.py with a saved benchmark
and a fresh --output path. Exact historical generator snapshots are retained in
results/generators/; current generators produce the same candidate bytes.
