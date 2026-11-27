# MoE sequential TTIR r3/r4/r5

[Combined report](report.md) · [Summary](summary.json) · [Protocol](optimization_plan.md)

The accepted chain is round2 anchor -> retained r3 -> retained r4 -> r5 pair_integer_constants.
All30 points and exact routing semantics remain unchanged. Only this directory was written.

To rerun final selection from the repository root, isolate an idle GPU and use a new result ID:

```bash
CUDA_VISIBLE_DEVICES=2 /root/venvs/ir-ops-dev/bin/python moe_routing/multiround/run.py \
  --round r5 --run independent_new --variants pair_integer_constants \
  --rounds 5 --rep 20 --seed 234
```

`--check-only` executes pipeline and accuracy checks without timing. Generation refuses to overwrite different existing TTIR. Do not rerun workflow scripts over existing result IDs.

r3 contains failed BoolAttr gather and unordered concat versions plus corrected unit-attribute and ordered concat variants. r4 preserves failed SSA-shadowing tuple candidate and a superseded initial selection/main; only screen_v2/main_v2/repeat_v2 support its final decision. r5 starts from the retained accepted parent, and applies the validated tuple top2 plus constant specialization as a new combination.

Artifacts are reproducible compiler outputs; original source, parent copies, candidate hashes, manifests, input hashes, pipeline gates and raw timings are retained.
