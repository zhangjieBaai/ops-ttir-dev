#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Select one idle GPU explicitly}"
run="mean/results/runs/ablation-reproduce-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$run"
py=/root/venvs/ir-ops-dev/bin/python
variants=(optimized ab_f0i1 ab_f1i0 ab_f1i1 ab_u0g0t0 ab_u0g0t1 ab_u0g1t0 ab_u0g1t1 ab_u1g0t0 ab_u1g0t1 ab_u1g1t0 ab_u1g1t1)
"$py" mean/verify_pipeline.py "${variants[@]}" --output "$run/pipeline.json"
"$py" mean/validate.py "${variants[@]}" --output "$run/accuracy.json"
"$py" mean/benchmark_ablation.py --accuracy "$run/accuracy.json" --pipeline "$run/pipeline.json" --output "$run/main.json"
"$py" mean/validate.py "${variants[@]}" --seed 99 --output "$run/accuracy_repeat.json"
"$py" mean/benchmark_ablation.py --accuracy "$run/accuracy_repeat.json" --pipeline "$run/pipeline.json" --rounds 5 --seed 99 --output "$run/repeat.json"
compute-sanitizer --tool memcheck --error-exitcode 86 "$py" mean/validate.py "${variants[@]}" --sanitizer --output "$run/memcheck.json" 2>&1 | tee "$run/memcheck.log"
compute-sanitizer --tool racecheck --error-exitcode 87 "$py" mean/validate.py "${variants[@]}" --sanitizer --output "$run/racecheck.json" 2>&1 | tee "$run/racecheck.log"
"$py" mean/verify_triton_syntax.py --output "$run/public_syntax.json"
"$py" mean/branch_probe_ablation.py --output "$run/branch_probes.json"
"$py" mean/analyze_ablation.py --run "$run" --output "$run/analysis.json"
