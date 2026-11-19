# mean_dim TTIR experiment v1

Contract: one contiguous FP16 input [M,N,K], mean along dimension 1, FP32 accumulation and FP16 output [M,K]. Positive dimensions, distinct aligned storage, product < 2^31. No dtype promotion, noncontiguous inputs, empty reduction, multi-axis reduction or full mean in this round. NaN/Inf propagation checked separately. Original three paths and exact per-point heuristic configurations; no autotune in these paths. Grid, signature and full backend options fixed for each group.

Performance: exactly the 30 shapes in common.PERF_SHAPES, plus 12 accuracy-only boundary shapes. First 12 cover inner/non-inner; last 18 cover vector loop. Kernel-only CUDA graph replay, preallocated tensors, randomized paired order, 7 rounds x 20ms; independent process 5 rounds x 20ms. Same-device A/A control, bootstrap conditional intervals, all raw samples retained. No clock lock or explicit cache flush. Selection uses tuning subset (even indices); odd indices held out until final measurement. Any dispatch must use structural path rules, never benchmark shape lookup.

Accuracy: each frozen shape, normal/constant/offset/alternating/zero/outlier, CPU FP64 reference rounded to FP16 and native ATen. atol=0.002, rtol=0.002. Fresh NaN output sentinels and finite guard canaries; special values equal_nan=True. Final memcheck and racecheck mandatory.

Directions (hypotheses, D): vector path serial N loop causes dependency/latency costs; directly synthesize 2D TTIR tiles 4/16 lanes along N, reduce after loop while preserving K tile=1024, grid and 8 warps. Also test 4-way sequential unrolling and cache policy. Inner/non-inner controls unchanged initially. Maximum 5 candidates, one screening stage, final comparison and repeat. Stop after negative screening or stable useful finalist. Source bodies never edited to generate candidates; use TTIR text generation only.

Acceptance: source/file TTIR and LLIR/PTX/cubin identical for every frozen point; same native compiler and launcher; correctness before timing; retain failures; report scoped speedups and regressions. Final independent review via KernelForge. Evidence grades follow knowledge/README.md.
