# MoE r6 reproduction

Read [optimization_plan.md](optimization_plan.md), [report.md](report.md) and [summary.json](summary.json).
The accepted r5 TTIR is the parent; all30 shapes and launch configurations remain frozen. Only files under r6 and sibling r6_* helpers belong to this round.

From `/root/ops-ttir-dev`, with GPU2 idle and reserved:

```bash
/root/venvs/ir-ops-dev/bin/python moe_routing/multiround/r6_generate.py
CUDA_VISIBLE_DEVICES=2 /root/venvs/ir-ops-dev/bin/python moe_routing/multiround/r6_run.py --round r6 --run reproduction_screen
CUDA_VISIBLE_DEVICES=2 /root/venvs/ir-ops-dev/bin/python moe_routing/multiround/r6_run.py --round r6 --run reproduction_main --variants batch4 --rounds 7 --rep 20 --seed 42
CUDA_VISIBLE_DEVICES=2 /root/venvs/ir-ops-dev/bin/python moe_routing/multiround/r6_run.py --round r6 --run reproduction_repeat --variants batch4 --rounds 5 --rep 20 --seed 123
CUDA_VISIBLE_DEVICES=2 compute-sanitizer --tool memcheck --error-exitcode 99 --kernel-name regex=topk /root/venvs/ir-ops-dev/bin/python moe_routing/multiround/r6_run.py --round r6 --run reproduction_memcheck --variants batch4 --check-only
CUDA_VISIBLE_DEVICES=2 compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name regex=topk /root/venvs/ir-ops-dev/bin/python moe_routing/multiround/r6_run.py --round r6 --run reproduction_racecheck --variants batch4 --check-only
/root/venvs/ir-ops-dev/bin/python moe_routing/multiround/r6_report.py
```

Run sequentially on GPU2; do not overlap timing and sanitizer. Each new result directory must not already exist. The generator asserts byte equality for already-existing immutable TTIR. `r6_report.py` audits and regenerates reports from original qualified run IDs; reproduction run IDs do not silently replace the historical evidence.

`r6_workflow.py` documents exact original selection/qualification subprocesses; its original log/result paths are immutable and it intentionally refuses to rerun over them. New kernel bytes or protocol changes require new candidate/run IDs.
