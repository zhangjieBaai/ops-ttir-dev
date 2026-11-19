# Superseded setup: int32 program_id

This complete trial used an extraction that omitted FlagGems ext.program_id's int64 cast. Its numerical checks do not establish fidelity to the requested source baseline. All performance values here are excluded from the corrected experiment's conclusions. The final run was interrupted during independent repetition after the provenance defect was found; no final sanitizer acceptance was granted.

Original absolute `var_mean/ir/...`, `var_mean/tutorial.py`, `var_mean/baseline.json`, `var_mean/contract.json` and `var_mean/results/runs/r1/...` references resolve to this archive's `ir/`, `tutorial.py.snapshot`, `baseline.json`, `contract.json`, and `results/` respectively. Raw JSON and IR bytes were moved unchanged. Earlier contract-v1 and line-info setup attempts remain in sibling directories.

The corrected source is the current `var_mean/tutorial.py`, contract var_mean-row-f32-v3, evidence `var_mean/results/runs/r2/`. Do not mix the two baselines or reuse the archived speedups as corrected-source results.
