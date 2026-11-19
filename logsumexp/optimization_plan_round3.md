# Round3: >10% speedup on the original 30 cases

User target: average speedup >1.10 relative to original FlagGems baseline.
Require both equal-case geometric and arithmetic means >1.10 in complete
7-round main and 5-round independent process/seed repeat. Report direct paired
comparison against immutable parent r2_select_after too, worst-case threshold
>=0.90 against parent. No changing case membership, weights, numerical scope,
tolerance, original baseline, mapping, parsed compiler options, runtime ABI,
grid, shared compiler/launcher or timing method. No cold-cache/end-to-end claim.
The old 20/10 partition is reused tuning/regression, never a fresh holdout.

A TTIR rewrite may change tensor shapes, loop structure, algebra, and device
control flow under the fixed launcher/grid. Every CTA keeps its original output
slice ownership in this round. No grid consolidation or workspace protocols.
No source re-export, inline PTX or downstream IR editing. Per-group constants
may differ inside TTIR; host shape->group mapping is unchanged. Runtime N guards
protect single-tile specializations; invalid lanes masked and integer offsets
proved bounded (<2^31). Output sentinel/guards and original all-inf semantics
must pass all51points x8distributions before timing. Final sanitizers all51points.

Hypotheses (D):
- Replace online inner reductions with a full single-tile stable max/sum for
  large N, or static bounded range dispatch; allows FP16 values resident during
  reduction rather than loop state and recurring exponentials.
- Narrow non-inner online tile to32/64 (rather than failed widening) to reduce
  state and change scheduling, fixed column ownership. Costs more iterations.
- Reshape/split large single-tile reductions to alter backend lane/warp lowering.
- Express exponential rescaling in base2/FMA form and test cache load hints.

Initial budget up to8 named candidates, 20-case tuning at3paired rounds with
A/A. Adaptive second batch allowed only after recording new evidence and what
it changes. Once finalist frozen: all30case direct parent pairs and original
baseline pairs at7rounds, then independent process/seed repeat at5rounds on
same physical GPU; save all samples/order and during-run occupancy checks.
Accept only accuracy+pipeline+target+memcheck/racecheck+review; report any unmet
target explicitly. Preserve all failed/negative candidates. Historical parent
selection/acceptance saved in r3_setup, never overwrite immutable source/IR.
