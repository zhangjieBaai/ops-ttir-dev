# Attention r6

[Report](report.md) · [Summary](summary.json) · [Plan](optimization_plan.md)

This round modifies only TTIR. It keeps the r5 accepted parent, round2 anchor, original32 shapes, ABI/grid/configuration and output contract. D64 is an explicit unchanged fallback. D128 candidates move dV late or split the two output accumulations into serial passes inside the same CTA.

- `../r6_generate.py`: deterministic generation, asserting byte identity when output exists.
- `../r6_run_locked.py`: full screen/main/repeat/sanitizer workflow with GPU4 clock restore in `finally`.
- `../r6_finalize.py`: qualification and immutable optimized aliases.
- `../r6_report.py`: read-only evidence audit and report regeneration after clock restore.
- `results/`, `selection_pre_main.json`, `selection.json`: raw evidence and frozen decisions.
- `failures.json`, `manifest_v1.json`, `generate_v1.py.snapshot`, `results/pilot/`: retained first-parser failures. Corrected candidate has a distinct `_v2` name.

Run from `/root/ops-ttir-dev` with `/root/venvs/ir-ops-dev/bin/python`. Existing result directories must not be overwritten. The complete orchestrator is a one-shot workflow; reproduction requires an independent new experiment directory/run ID. Individual verification can use the existing reusable scripts with new output paths and `ATTENTION_MR_ROUND=r6`; measurements must also use the documented exclusive GPU4 / fixed1200MHz protocol.

```bash
CUDA_VISIBLE_DEVICES=4 ATTENTION_MR_ROUND=r6 \
/root/venvs/ir-ops-dev/bin/python attention_dkdv/multiround/validate.py \
  --selection attention_dkdv/multiround/r6/selection_pre_main.json \
  --seed 123 --output attention_dkdv/multiround/r6/results/new_accuracy

/root/venvs/ir-ops-dev/bin/python attention_dkdv/multiround/r6_report.py
```

No source-port performance experiment is included. Public Triton expression order and serial loops can express the two structures, but machine-code/performance equivalence to such source has not been tested.
