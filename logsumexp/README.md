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
Results and the selected candidate will be linked here after the first round.
