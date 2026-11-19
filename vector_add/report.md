# Vector add TTIR experiment

## Result

The upstream Triton tutorial baseline was exported and replayed through the same
Triton 3.7.0 native TTIR file compiler and launcher as each candidate. Pipeline
identity, numerical tests, paired timing controls, and baseline memory/sync
checks passed. No candidate met the preregistered 1.02x promotion threshold, so
`ir/optimized.ttir` remains byte-identical to the frozen baseline.

Full-matrix measurement and an independent process repeat found a local cache-policy effect at
N=4,194,304. `cacheModifier=cs` measured 1.2055x and `evictionPolicy=evict_first`
measured 1.2057x there. Their full 16-size geometric means were only 1.0131x and
1.0125x; their 95% paired-round bootstrap intervals were [1.0116, 1.0142] and
[1.0113, 1.0139]. Independent-process replay (seed 137, seven rounds) measured geometric means
1.0119x [1.0103, 1.0140] and 1.0136x [1.0115, 1.0146], respectively. Both runs
passed all A/A controls. These are A-grade performance observations over this
frozen matrix, but neither candidate meets the 1.02x promotion criterion.
The N=2^22 local gains repeated at 1.2058x and 1.2061x. Validation-subset
geometric means were 1.0249/1.0234x for streaming store and 1.0230/1.0263x for
evict_first (main/repeat); overall promotion still fails. At the other 15 points, both candidates were close to
baseline; the worst median speedups were 0.9934x and 0.9956x at N=32768.

The split-512 candidate improved N=1,048,576 (1.0716x) and N=2,097,152 (1.1344x)
but regressed the four smallest screening points to 0.8731x-0.9199x and had a
0.9665x eight-point geometric mean. It was rejected at screening and not taken
into the full matrix. `r1_load_last_cs` had an 0.9900x screening geometric mean
and a 0.9649x point, so it was rejected as well. Failed split512 v1 SSA syntax
is preserved in its compile receipt; corrected v2 candidates compiled and
passed all 36 correctness cases.

## Method and scope

Source is the official vector-add tutorial pinned to commit
`22459e3dbcef2b403aab7ff2a3981309a3537e39`; source SHA256 is recorded in
`upstream/provenance.json`. The frozen specialization is FP32, contiguous,
16-byte aligned, independent buffers, positive N divisible by 16, and N<=2^27.
The launch is 1024 elements per CTA, four warps. This is not a claim for other
dtypes, lengths, aliasing, alignment or layouts.

Pipeline verification compared source and file entry paths at all 16 official
lengths and four boundaries. LLVM IR, PTX and complete cubin bytes matched; the
TTGIR parser-normalized forms and outputs matched. Correctness covered 36 cases,
including signed zero, finite overflow, infinities, NaN, subnormal values,
cancellation, output canaries and input immutability. Six unsupported argument
classes were refused. Candidate accuracy used the same FP32 reference and source
kernel before any performance testing.

The timing metric is Triton 3.7.0 `do_bench_cudagraph` kernel replay with
preallocated input/output. It excludes allocation, compilation, loading, Python
launch and host synchronization and does not match the tutorial's cache-clearing
default `do_bench`. Cache was not explicitly cleared and clocks were not locked.
Baseline and candidates used the same target, parsed compiler options, grid,
launcher and runtime ABI. Each full-matrix point has nine randomized paired
rounds plus an A/A control; raw replay samples are retained in
`results/runs/r1_20261119/full.json`.

## Evidence

- `results/setup/pipeline.json`: baseline source/file compiler identity and toolchain.
- `results/runs/r1_20261119/pipeline.json`: candidate pipeline and cubin identities.
- `results/setup/accuracy.json`: pre-candidate baseline correctness (superseded for pressure cases).
- `results/runs/r1_20261119/accuracy.json`: corrected baseline and candidate correctness.
- `results/runs/r1_20261119/screen.json`: eight-point tuning-set screening, including rejected variants (B-grade screening evidence).
- `results/runs/r1_20261119/full.json`: full 16-point paired measurements and controls.
- `results/selection.json`: final candidate selection and matching TTIR hash.

KernelForge supplied task discovery, optimization/evidence methodology and
review workflow. Its installed task CLI has no vector_add execution adapter;
this report makes no KernelForge operator-qualification claim.

## Per-size main/repeat timings

Times below are medians of paired round means, in microseconds. Speedup uses
the median of within-round baseline/candidate ratios. Intervals above resample
paired rounds at each fixed workload; they do not estimate unseen workloads
or all cross-session machine variation.

| N | CS baseline us | CS candidate us | CS main x | CS repeat x | Evict-first main x | Evict-first repeat x |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1.2533 | 1.2543 | 0.99899 | 1.00028 | 1.00051 | 1.00031 |
| 8192 | 1.2910 | 1.2829 | 1.00013 | 0.99982 | 0.99946 | 1.00249 |
| 16384 | 1.3014 | 1.2924 | 1.00665 | 0.99997 | 1.00026 | 1.01124 |
| 32768 | 1.3095 | 1.3151 | 0.99335 | 1.00986 | 0.99559 | 1.01254 |
| 65536 | 1.3973 | 1.3945 | 1.00357 | 1.00009 | 1.00553 | 1.00001 |
| 131072 | 1.5389 | 1.5404 | 0.99973 | 1.00019 | 1.00013 | 1.00010 |
| 262144 | 1.6965 | 1.6970 | 1.00058 | 0.99206 | 0.99983 | 0.99330 |
| 524288 | 2.0101 | 1.9947 | 1.00787 | 0.99418 | 0.99764 | 1.00170 |
| 1048576 | 2.7567 | 2.7425 | 1.00111 | 0.99950 | 1.00356 | 1.00053 |
| 2097152 | 4.8553 | 4.8237 | 1.00694 | 1.00359 | 1.00653 | 1.00366 |
| 4194304 | 16.7254 | 13.8710 | 1.20550 | 1.20581 | 1.20573 | 1.20608 |
| 8388608 | 34.7698 | 34.7475 | 1.00071 | 1.00074 | 1.00075 | 1.00085 |
| 16777216 | 67.2307 | 67.2048 | 1.00036 | 1.00045 | 1.00037 | 1.00042 |
| 33554432 | 132.0117 | 131.9671 | 1.00035 | 1.00029 | 1.00033 | 1.00031 |
| 67108864 | 261.5025 | 261.3973 | 1.00038 | 1.00036 | 1.00044 | 1.00040 |
| 134217728 | 520.8777 | 520.6281 | 1.00041 | 1.00050 | 1.00047 | 1.00042 |

## Static artifacts and limits

All candidates change TTIR only; they use the same parsed options, signature,
grid and launcher. Compilation artifacts and resource counts are in
`results/runs/r1_20261119/compile_v3.json`. SASS `.text` hashes are of executable
section bytes, not complete cubin identity. The two rejected split512 v1 files
remain immutable, and corrected versions have distinct v2 names.

The shared compiler and TTIR baseline did not change during optimization.
Pre-search harness fixes are recorded in baseline manifest predecessor records;
source/TTIR remained unchanged. The special-value correction was a review
finding; corrected accuracy receipts supersede initial pressure-case coverage.
Installed Triton RECORD hashes were checked. This uses the existing project's
venv and does not independently attest a freshly downloaded wheel. Dynamic
library paths are recorded; torch components loaded from the specified venv
despite external LD_LIBRARY_PATH entries.

Final baseline, optimized alias and both measured cache candidates were checked
with memcheck and racecheck over 23 cases each. Logs and identity-bound accuracy
receipts are in `memcheck_final.*` and `racecheck_final.*`. No candidate was
promoted, and no claim is made about a different cache state or application time.
