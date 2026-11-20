# Triton source versus selected TTIR comparison

Question: can the methods in selected immutable r4_focus be expressed using
upstream Triton3.7 Python, and does their device time match direct TTIR?
This is a NEW SOURCE-vs-IR experiment explicitly requested by user; do not label
modified-source exports as direct TTIR optimization. Historical baseline/IR,
manifest and selection stay unchanged. No fastlog from unfinished r5 included.

Fixed original51 correctness shapes and30 performance shapes, FP16 contiguous,
finite/-inf, tolerances rtol1e-3 atol2e-3, same frozen heuristic configuration,
full parsed backend options,target,grid,ABI,data and native CompiledKernel launcher.
Source implementation uses only public tl APIs and @triton.jit, no inline ISA,
internal builder, binary replacement or compiler override. Preserve all algorithm
branches incl unexecuted fallbacks under original config. Custom tl.reduce for
FP16 max avoids tl.max implicit FP32 promotion. No retuning/reconfiguring.

Both implementations compiled/loaded before timing; source starts at Python
frontend and TTIR file starts after make_ttir. Save every actual stage and
resource identity. Check original source-baseline pipeline anew, and source
optimized native output vs its own TTIR file roundtrip: downstream options,
LLVM/PTX/cubin identical (or disclose exact difference), normalizedTTGIR.
Compare optimized-source vs optimized-TTIR ELF.text, PTX instruction body,
registers/spills/shared, output bitwise agreement plus independent FP64/native
oracle and guards. No claim of source/TTIR IR byte equality unless checked.

Accuracy all51x8 stress distributions, seeds42/123. Include additional nearzero,
near1 and wider dynamic-range distributions if needed. Outputs reset eachlaunch,
guards/input preservation, all-inf sign matching. Final memcheck/racecheck all51.

Performance: all30, equal-case arithmetic/geometric means and everycase times,
7-round main and5-round independent process/seed repeat. Same physicalGPU,
randomize pairs and within-pair order. Pair originalbaseline/source, original
baseline/TTIR, and DIRECT TTIR/source in eachprocess; A/A uses sameTTIR twice.
Same triton.testing.do_bench_cudagraph rep20ms return_mode=all; warmup20launches
plus100ms graph. Originaldevice latency only, preallocated buffers, no host
submission/alloc/JIT/loading. No cacheclearing; clock lock unknown/not set.
Check foreignGPUprocess between pairs, hold shared UUID lock, telemetry200ms.
A/A median must be[.97,1.03] percase; failedwhole runs retained/excluded and new
run IDs required. bootstrap5000 pair-round draws,95% CI conditional on shapes.
Ratio TTIR/source>1 means SOURCE FASTER. Ratios close to1 comparable to A/A
spread are no established difference, even if artifact bytes differ.
Record compile time separately, never mix into device speedups. Full matrix is
reused regression, no freshholdout. Report limits rather than universal claims.

Launcher detail: ASTSource CompiledKernel retains Python constexpr argument
placeholders, IRSource exposes runtime-only args. launch_native adapts these
native call signatures; actual device ABI/grid/stream unchanged. Passing only
runtime values to AST launcher was a harness bug caught before timing in pilot
v2, not an algorithm failure. Correct native signature is used for source
timing; no metadata patch or binary replacement. Both use the same Triton
native launcher implementation. Python wrapper overhead is excluded by graphs.
