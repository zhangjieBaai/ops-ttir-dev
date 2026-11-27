# INT4 r6: nibble representation and conversion

Frozen before GPU timing. Parent is multiround/r5 optimized.ttir (the accepted r3 group128); anchor remains round2 r2_group_pair. Keep all30 shape/config mappings, original ABI/grid/workspace, official Triton3.7 native compiler, FP16 inputs/outputs, FP32 dequantization and dot accumulation. No autotune, stage tuning, source port or inline assembly.

Three new candidates:

1. `interleave_u8_v1`: join low/high integer nibbles, transpose and reshape to [BK,BN] before conversion and dequantization. This changes the representation and possible lowering, while every element still evaluates `(float(nibble)-float(zero))*float(scale)` and rounds once to FP16 before the unchanged dot.
2. `magic_fp32_v1`: exact conversion of decoded nibbles0..15 via zero-extension, OR0x4b000000, FP32 bitcast and subtraction of8388608. These IEEE FP32 values are exactly8388608+n, so the result including +0 is identical to uint8-to-FP32. Subtraction/multiplication by zero/scale remain unchanged. Potential instruction savings or added costs are hypotheses.
3. `interleave_magic_v1`: combine the two new transformations.

These do not repeat prior cast-after (which only moved FP16 truncation), PID mapping, preload or K-loop unroll. All parent source/IR/history remains frozen. Retain compile/numerical failures under their original names; any syntax correction gets a versioned new name.

All30 original-source/file baseline pipelines must match LLVM/PTX/cubin; all30 x5 distributions and CPU FP64 small cases must pass before timing, including raw FP16 bits, output canaries and unchanged inputs. Compare nonempty executable ELF.text to parent before timing; executable identities are C-level facts, not improvements. If no new executable candidate survives, report disposition without new performance claims.

For surviving nonidentity candidates: full30 screen3 rounds rep10ms; choose highest incremental geomean among worst>=.95 if available. Freeze choice, then main7 rounds rep20ms and independent seed123 repeat5 rounds rep20ms with original baseline, r2 anchor, current parent and finalist, random order. Per-shape parent A/A median must be[.97,1.03]; invalid complete runs retained and excluded. Promote only if main AND repeat incremental geo>=1.02 and worst>=.95 plus memcheck/racecheck pass. Negative finalist may also be sanitized; retain parent if rejected. Final effective metrics use actual final-route times in the same run, not multiplication of old ratios.

GPU0 exclusive; no clock lock or cold-cache claims. Public Triton can express all three changes with join/trans/reshape, cast/bitcast and integer operators, but no source performance equivalence is claimed in r6.
