# Vector add TTIR experiment v1

Frozen before export/optimization. Baseline: user-supplied official main tutorial,
pinned source in upstream/provenance.json. Only TTIR changes are candidates.

## Contract

- FP32 contiguous one-dimensional CUDA tensors, independent storages, pointers
  aligned to 16 bytes; positive N divisible by 16, N <= 2^27. No alias, other
  dtype, negative/zero length, strided or misaligned inputs. This is one fixed
  specialization, not arbitrary-length support. Tail cases refer to CTA tails.
- BLOCK_SIZE=1024, grid=(ceil(N/1024),1,1), num_warps=4, num_ctas=1,
  num_stages=3, enable_fp_fusion=True. Runtime ABI=(x,y,out,N:i32).
- Official upstream Triton 3.7.0 in /root/venvs/ir-ops-dev; H100 SM90.
  No inline assembly, overrides, modified compiler stages, or lower-IR edits.
- Finite FP32 results exactly equal native torch.add and source kernel; signed
  zero bits checked. NaN comparison uses equal_nan (payload not promised).
  Inf, overflow, subnormal and cancellation cases are tested. Output sentinel
  reset per launch; prefix/suffix canaries and input immutability checked.
- Correctness covers all official 16 lengths plus 16,32,1008,1024,1040,2032,
  2048,2064,98432,4194288,4194320; pressure patterns on boundary/mid sizes.
  CPU FP32 reference additionally checked for small sizes. Two process seeds.

## Frozen performance protocol

- Primary metric for this experiment: CUDA graph replay device kernel latency,
  preallocated output, triton.testing.do_bench_cudagraph(rep=20ms). Excludes
  allocation, Python issue, compilation and loading. Does not reproduce tutorial
  do_bench cache clearing. Cache not cleared; clocks not locked by this experiment.
  This explicit TTIR protocol takes precedence over KernelForge's generic primary
  metric definition. No broader KernelForge operator qualification is claimed.
- All official x_vals=[2**i for i in range(12,28)] (16 lengths), equal shape
  weights; no removed points. Tuning exponents=12,13,16,17,20,21,24,25;
  validation=14,15,18,19,22,23,26,27 (adjacent-pair groups kept together).
  Validation performance first exposed after finalist identity is fixed.
- Baseline/candidate same input and output buffers, one shared compiler/launcher,
  untimed warmup, randomized candidate and pair order. Save all pair means and
  per-replay samples returned by timer; A/A baseline control each shape/round.
- Screening 3 paired rounds on tuning sizes; full matrix 9 rounds; independent
  process 7 rounds, different data/order seed. Paired median per shape, arithmetic
  and geometric aggregate, 95% paired-round bootstrap intervals (5000 draws).
- Accept a performance replacement only if full and repeat geometric speedups
  >=1.02, both aggregate intervals exclude 1, validation geo>=1.01, and worst
  per-shape median >=0.95. A/A per-shape median within [0.97,1.03]; otherwise
  mark measurement uncertain, investigate and preserve the whole run.
- If no candidate qualifies, optimized.ttir remains baseline with an explicit
  no-replacement decision; local gains remain local evidence.

## Search and completion

Initial budget <=8 direct TTIR candidates: store streaming, load cg, combinations,
eviction priority, splitting a CTA's 1024 values into two independent 512 tensors,
and (if justified) full-CTA mask fast path. Prior LLIR results in
/root/ir-ops-dev/vector_add are background only, never this experiment's gains.
Stop a branch when compilation produces identical machine code or reliable
measurements show no meaningful gain. Preserve failures and regressions.
Total initial GPU search budget <=15 minutes, compilation/validation <=20 minutes;
final required evidence is separate. Record any justified extension before use.

Pipeline gate before search: exact exported TTIR; full LLVM/PTX/cubin byte
identity source vs file; TTGIR parser-normalized identity; bitwise outputs; full
parsed backend options/target identical. Final candidate requires memcheck and
racecheck. Results get unique run directories; source, baseline and candidates
immutable. Scoped Git baseline commit precedes candidates; final commit retains
raw evidence. pre-commit all-files and independent KernelForge close review.
