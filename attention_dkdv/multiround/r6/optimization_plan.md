# Attention r6: D128 live-state tradeoffs

Freeze the original 32 shapes, 4 warps / 5 stages, exact output contract, ABI, grid, kernel count and workspace. Parent is r5/ir/<group>/optimized.ttir; anchor is round2/ir/<group>/optimized.ttir. No source port, inline assembly, downstream rewrite, or pipeline-depth tuning.

Predeclared candidates (before generation or timing):

1. `r6_dv_late`: D128 moves the probability FP16 cast and dV dot after the dK dot on the accepted 64-key parents. The r1 D128 experiment used the original 128-key baseline, so this is a new composition on a different accepted structure.
2. `r6_fission_v_then_k`: D128 computes each 64-key tile in a dV-only query pass, stores dV, then a dK-only pass and stores dK. Both passes keep the original masked/unmasked query order and dot accumulation order. QK/probability and some loads are recomputed to avoid simultaneous dK/dV accumulator state. Same CTA and grid, no temporary global storage. Extra work can outweigh reduced state.

Both candidate labels use byte-identical parent code for D64, which is a declared fallback/control, not a new optimization. The no-alias contract permits dV store before the independent dK pass. A dependency slice only removes statements unused by that pass and unused loop-carried state; no arithmetic reassociation.

Each candidate must pass full44-input accuracy before screening, including exact FP16 output bits and CPU FP64 references at N128. Baseline source/file LLVM/PTX/cubin gate covers all32. Screen3 rep10, frozen group selection, main7 rep20 and independent seed123 repeat5 rep20; A/A per-point median [0.97,1.03]. Promotion requires both group incremental geometric means >=1.02 and worst>=0.95 plus clean memcheck/racecheck. Record raw paired times, input and IR hashes, static resources, and actual selected final-route ratios against parent/anchor/original. Failed/negative candidates remain recorded.

GPU4 exclusive, fixed SM1200MHz, equal event-verified500ms initial and100ms per-measurement warmup, unchanged native do_bench_cudagraph. A guarded workflow restores automatic clocks in finally and records before/lock/after/reset evidence. Reports distinguish static C evidence from qualified A timing. If all candidates fail/are identity, report that honestly.
