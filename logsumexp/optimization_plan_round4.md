# Round4: improve previously flat shapes

User requests actual attempts on shapes that have not accelerated. Retain frozen
FP16 contract, exact original source/51-point mapping/30 performance cases,
parsed options/ABI/grid/common compile_ir+launch and original tolerances.
Parent immutable r3_mix; parent selection/acceptance copied into r4_setup.
No inline ISA, source re-export or editing TTGIR/LLVM/PTX/cubin. Negative outcomes
are useful; never claim every shape improved or promise a win without data.

Primary target class, frozen before trials:
- inner K=1, M={1,64,1024}, N={128,1008,4096,8192} (12 cases).
- non-inner (1,8192,64), (16,2048,64), (256,128,64) (3 cases).
Report this15-case subset separately and all30 equal-case original metrics.
20/10 split is reused tuning/regression, not fresh holdout. Selection uses tuning
results only then full main+independent process repeat. Original all30 arithmetic
and geometric means should remain >1.10; preserve parent improvements. Require
whole-domain parent worst>=.90 both processes; no acceptance of unsafe candidates.
Target subset geometric >1.01 vs parent in both full runs for promotion; if no
candidate qualifies, keep parent and document the attempted mechanisms.

Hypotheses (D):
1. Under the SAME host grid(M,1,1), group2/4/8 adjacent inner rows in a CTA and
   let only pid%R==0 compute. That CTA owns rows pid+[0,R); other CTAs return.
   Coverage: every r is uniquely (R*floor(r/R))+r%R. All row/input/output loads
   and stores are masked by r<M, preserving N masks. No host launch, grid,
   runtime signature, workspace or configuration change. Unlike round3, this
   round allows TTIR-only redistribution of output ownership among existing
   CTAs, with the proof above and full guard/sanitizer validation. M=1 groups
   keep parent. Relevant runtime M=3 boundaries test incomplete row groups.
2. Swap logical tensor axes for non-inner and reduce axis1 instead of0; actual
   address/ownership unchanged. Compiler chooses physical layout; check real
   code equivalence rather than assuming transposition helps.
3. Narrow proven-bounded non-inner address arithmetic to i32 under the new
   parent algorithm. Previous global i32 trial was neutral but this new
   composition changes online arithmetic and resources; test separately.

Budget: up to6 initial candidates; all51x8 distributions before performance,
20-case tuning at3 rounds withA/A. Skip code-identical duplicate candidates.
Record adaptive follow-up only if results identify a changed premise. Freeze
one finalist; 30 cases, 7-round main /5-round independent seed/process repeat
against original baseline and direct parent, same physicalGPU. All raw pairs,
A/A, telemetry retained; whole invalid runs excluded and rerun with new IDs.
Final all51point memcheck/racecheck, pipeline consistency, precommit and review.
All performance is graph-replay one-kernel device latency, no cold-cache,
allocation/submission/JIT or full API claim. Clock lock state stays unknown.
