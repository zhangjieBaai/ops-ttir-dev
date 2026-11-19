# Streaming-store candidate promotion

The user explicitly accepts the previously verified approximately 1.206x local
gain and authorizes promoting that candidate. The current optimized alias will
therefore use `ir/r1_store_cs.ttir`, including for shapes where it is effectively
equal to baseline. No per-shape host dispatch or new specialization is added.

This is an explicit selection-policy revision: the original v1 requirement for
both aggregate geometric means to reach 1.02 remains historically unmet and is
not rewritten as passed. Source, baseline, runtime ABI, BLOCK_SIZE=1024, four
warps, options, shape set, numerical checks and timing method remain unchanged.
New measurements use the actual `optimized` TTIR file entry and preserve full
parsed options, artifact hashes and all 16 per-shape results.

The preceding selection record and optimized alias are archived unchanged under
`results/runs/promote_cs_20261119/history/` before updating the alias. The promoted
TTIR must have SHA256 `ddd8ec671f57abe0e0538e70b6e38dcdfc3f494ec33b73b87073cb3fe6e25c0e`,
the exact immutable candidate already validated and checked by both sanitizers.

GPU 7 is occupied by an unrelated experiment. This new paired session uses the
idle GPU 0, with baseline and optimized interleaved on that same physical GPU.
Its independent-process repeat also uses GPU 0. Historical GPU 7 samples remain
separate; they are not pooled with the new device session.

Validation: native-chain pipeline check, two full 36-case numerical runs, full
16-size graph paired comparison with 9 main rounds / 7 repeat rounds, rep=20ms,
different seeds, all raw samples and A/A controls. The explicit user selection
can accept a local benefit and modest overall improvement; meaningful correctness
failures, sanitizer failures or invalid timing still prevent a completed claim.
Existing sanitizer evidence is reusable only after verifying identical TTIR and
cubin hashes; it must not be relabeled as a fresh sanitizer run. Final reporting
states the observed overall and local effects independently and lists regressions.
