# var_mean TTIR experiment r1

Contract owner: contract.json (var_mean-row-f32-v2). Source: upstream/provenance.json.
30 preregistered shapes = M in {1,17,128,1024,4096} x N in {128,512,1040,4096,8192,16384}.
12 tuning points specified by zero-based indices in contract; remaining 18 validation points held until candidate selection. Equal point weights. After exposure no claim of untouched holdout for later revisions.

Baseline has no autotune: BLOCK_N=1024, warps=4, stages=3, CTAs=1, fusion enabled for all points. Freeze one runtime-N/M/correction TTIR, prove source/file backend stage identity at every shape. Strict native Triton 3.7 venv; isolate physical GPU7 and hold cooperative device lock. Recheck interference before timing.

Only candidate TTIR changes. No inline PTX, downstream editing, launcher/grid/config/shape mapping changes. Initial candidate batch: tile256, tile4096, tile8192; then bounded single-tile runtime dispatch to reuse input across mean and centered variance. Keep two-pass centered algorithm. Tensor widths may affect register allocation, occupancy, loop overhead and load reuse; these are hypotheses until products/timing exist.

Budget: <=8 TTIR candidates, <=2 screening batches, each 12 points x3 paired rounds x10ms. Validate numerical correctness before timing. Choose one candidate on tuning, then evaluate all 18 validation points. Stop/reassess on severe regressions or repeated <5% new gains. Full final:30x7 paired rounds, independent seed123 process30x5 rounds. Same native file compiler/launcher/options; randomized baseline/candidate order. Record every raw pair, conditional bootstrap95% intervals, actual IR and product hashes/resources. A/A and deliberately duplicated launch controls on representative small/large shapes; independent process includes baseline A/A cross-process comparison.

Accuracy: full matrix normal inputs; boundary N16,112,128,144,496,512,528,1008,1024,1040,4080,4096,4112,8176,8192,8208,16368,16384. M1/17; correction0/1. Stress constant,offset10000,low variance,zero mean,alternating magnitudes,first-element outlier; CPU FP64 authoritative reference and native GPU FP32 diagnostic. Reset output sentinels and check output canaries. Final memcheck/racecheck boundary/stress subset with error-exitcode1.

Archive immutable baseline, all candidates and negative results, exact commands/versions and Git commits. Scope is an experimental row kernel, not an ATen integration or global var_mean claim. Follow KernelForge evidence and independent close-review workflow; it lacks a var_mean execution adapter, so use local native-Triton scripts rather than claiming framework qualification.

Pipeline setup finding: default line-info file parser maps one unknown location to input line45; PTX/cubin identical but LLVM debug metadata differs. Attempt archived under setup_attempts/debug_locations. Freeze TRITON_DISABLE_LINE_INFO=1 symmetrically before source export/file compile, requiring exact LLVM/PTX/cubin identity in final baseline.

Setup numerical finding: original baseline passed CPU FP64 tolerance but strict GPU FP32 native gate failed offset10000 (relative3.7e-4). Contract-v1 and pipeline archived; v2 keeps CPU tolerance unchanged, uses CPU FP64 as authoritative, records native GPU discrepancies without gating on its less accurate offset result. Complete baseline accuracy and A/A plus duplicated-launch timing checkpoint before ranking candidates.
