# logsumexp TTIR round 2

Continue v1: all operator semantics, baseline source, 51-point mapping, full
backend options, launcher, tolerances and 30 performance shapes remain frozen
by optimization_plan.md and baseline.json. Parent is immutable r1_trim, recorded
by results/selection.json before this round. Do not overwrite historical files.
Round budget: at most six initial candidates, 20-point tuning screen (3 rounds),
then one frozen finalist against baseline AND parent on all 30 points (7 rounds)
and an independent process/seed repeat (5 rounds), followed by final sanitizers.
Stop if no stable incremental improvement; negative outcomes are deliverables.

Round 1 already exposed all shapes. The old 20/10 partition is reused for tuning
and regression checks only, not a fresh holdout. Equal per-case weighting stays
fixed. Retention requires full-matrix geometric >1.01 relative to parent and
worst-case >=0.90 against parent in both main and repeat, plus v1 criteria vs
baseline, accuracy, all-point pipeline equivalence and memcheck/racecheck.
Ratios vs parent must come from direct randomized within-process pairs;
no quotient of historical benchmark medians. Same physical GPU throughout each
paired run; no cross-device absolute latency claims.

Hypotheses (D, not measured):
1. Online max is either old m or x. Let q=exp(-abs(x-m)); update z as
   z*q+1 when x>m, else z+q. Thus two exponentials become one. Both -inf
   still uses the original guard preserving z; finite/-inf semantics unchanged.
2. A select-before-FMA formulation or select-after-arithmetic formulation may
   lower differently; validate rounding within unchanged tolerances.
3. Widen the 128-row non-inner online tile to 256/512 with matching loop step,
   reduce recurrence iterations but risk registers/shared memory. Grid/options
   remain fixed; this is direct TTIR tensor/loop rewriting, not autotuning.
4. Loop unroll hints may expose independent scheduling but increase code size.

All transformations start from parent TTIR, preserving its small-N improvement.
No inline PTX, no source re-export, no lower-level IR editing. Artifacts, negative
results, all raw pairs, generator identity and environment receipts retained.
Use common.compile_ir/common.launch. Formal round2 evidence additionally binds
this file hash and archived parent selection. Existing baseline receipts remain
historical; run new all-point pipeline check before this round's timing.
