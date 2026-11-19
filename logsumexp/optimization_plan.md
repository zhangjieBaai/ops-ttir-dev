# logsumexp TTIR v1: frozen protocol

Baseline: user-specified `/root/FlagGems/src/flag_gems/ops/logsumexp.py`,
commit and byte hashes in upstream/provenance.json. Both inner and non-inner
kernel bodies remain unchanged. Registration decorators are isolated; the exact
NVIDIA heuristics and int64 program_id helper are snapshotted and evaluated
explicitly. No autotune. This is a kernel experiment, not full FlagGems API qualification.

Contract: FP16 input/output, contiguous positive M,N,K (K=1 inner or K=64
non-inner), reduce axis=1 of (M,N,K), FP32 statistics. Separate aligned input
and output allocations, pointers 16-byte aligned, N divisible by 16; integer
address products below 2^31. Inputs finite or -inf, each output slice either has
at least one finite element or is entirely -inf. NaN and +inf are unsupported
because the original implementation does not implement native ATen semantics
for every such case. Empty tensors, arbitrary strides, other dtypes and
multidimensional reductions are outside this experiment. Unmapped shapes refuse.

30 performance cases, equal weight per shape:
- inner: M in {1,64,1024}, N in {128,1008,4096,8192,32768,65536}, K=1 (18).
- non-inner: M in {1,16,256}, N in {128,512,2048,8192}, K=64 (12).
Tuning: inner M={1,1024}, non-inner M={1,256} (20).
Validation: inner M=64, non-inner M=16 (10), first exposed after finalist freeze.
This is a caller-selected suite, not an official tutorial performance suite.
Additional correctness cases: M=3 inner N={16,144,496,528,2032,2064,8176,
8208,32752,32784,65520,65552}; non-inner M=3,K=64,
N={16,112,144,496,528,2032,2064,8176,8208}.
Every point is exported and checked; only identical TTIR AND complete options
may share a group. Freeze shape -> group -> TTIR SHA -> parsed options.

Options: exact original NVIDIA heuristic TILE_N,TILE_K,ONE_TILE_PER_CTA and
num_warps for each point; num_stages=3,num_ctas=1,enable_fp_fusion=True.
Source and TTIR file use identical complete parsed backend options and SM90.
File input starts at make_ttgir. No override/inspection hooks; no custom backend,
inline PTX or edits to TTGIR/LLVM/PTX/cubin. Signature, grid, launcher and mapping
are fixed per group. Only candidate TTIR content may differ.

Correctness: FP64 torch.logsumexp oracle, rounded to FP16; rtol=1e-3,
atol=2e-3, checked elementwise, matching -inf. Also compare native FP32 ATen.
Distributions: normal, constant zero, offset +/-1000, alternating +/-80,
all -inf, sparse finite amid -inf, near-zero cancellation of max+log(sum).
Guard prefix/suffix output, reset sentinel before every kernel; check finite
outputs for finite-reference slices, bitwise source/file output equality,
input preservation, and final memcheck/racecheck on all frozen shapes.

Primary metric: complete one-kernel device time with preallocated outputs,
triton.testing.do_bench_cudagraph(rep=20ms,return_mode=all), no cache clearing.
Compilation, allocation, Python submission and loading excluded. Same physical
GPU, isolated explicitly; advisory flock held across GPU entry processes.
20 launches + graph warmup per variant, synchronize before timed samples;
7 paired rounds main, 5 paired rounds independent process/seed. Randomize pair
order and candidate/control order. Save every replay sample and pair. Include
per-shape A/A controls, accept apparatus if median ratio within [0.97,1.03].
Save telemetry during execution and before/after; clocks not actively locked,
clock lock state unknown unless queried. Detect other compute processes on the
selected GPU before and after timing; contamination invalidates whole run.
Report per-case paired ratio medians and arithmetic/geometric means; paired
round bootstrap 95% intervals (5000 resamples, conditional on frozen cases).
No per-case no-regression promise; selection requires validation geometric
mean >1.01 and no point <0.90 in both full main and independent repeat. If no
candidate qualifies, keep baseline and report negative findings. Finalist is
fixed before validation timing; no per-shape cherry-picking after seeing results.

Budget: at most 6 first-round candidates, 20-case tuning screen (3 rounds),
one finalist full matrix and independent repeat. Main lever hypotheses:
reduce excess padding/work in non-inner heuristics, reduce online recurrence
cost for multi-tile rows using stable two-pass max/sum, and narrow proven-bounded
integer address arithmetic. Each remains a hypothesis until measured. Compilation,
correctness and negative outcomes retained. Initial freeze and final results
are separate Git commits; unrelated existing work is not staged.
