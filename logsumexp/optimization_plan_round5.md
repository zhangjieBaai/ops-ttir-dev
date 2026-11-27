# Round5: elementary-function cost

Keep original51point/30performance matrix, FP16 finite/-inf contract, tolerance,
options/grid/ABI/compiler/launcher and timing boundary. Parent r4_focus immutable.
No source re-export/inline ISA/downstream editing. Hypothesis: math.log currently
lowers to long libdevice polynomial; use existing libdevice fast_logf intrinsic
from TTIR, or log2+ln2 to improve short reductions. Approximation must pass
unchanged FP64/native tolerance and -inf guard; no nnan/ninf assumptions.
Existing libdevice extern library/compiler options remain identical. Safe finite
sums are>=1 for supported slices; all-inf output guard remains. No dtype/layout
extension. Add near1 sums and random wide distributions as stress if accepted.
Budget: three candidates initially, full51x8 accuracy before tuning20case5rounds.
Finalist main30x7 and independent repeat30x5 vs parent AND original baseline,
A/A all30, raw pairs and telemetry, no rejected-run sample cherry-picking.
Promote if geometric>1.01 vs parent, worst>=.90 and original full mean>1.10;
otherwise retain parent. Final sanitizers, pipeline, precommit and close review.
Cases already exposed, no independent holdout claim. No cold-cache/end-to-end
claim. Parent selection archived; negative candidates retained.
