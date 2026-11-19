#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU explicitly}"
run="mean/results/runs/round2-reproduce-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$run"
py=/root/venvs/ir-ops-dev/bin/python
"$py" mean/verify_pipeline.py optimized r2_final --output "$run/pipeline.json"
"$py" mean/validate.py optimized r2_final --output "$run/accuracy.json"
if [[ "${1:-performance}" == accuracy ]]; then
    exit 0
fi
"$py" mean/benchmark_round2.py r2_final --parent optimized --validation "$run/accuracy.json" --pipeline "$run/pipeline.json" --output "$run/main.json"
"$py" mean/validate.py optimized r2_final --seed 99 --output "$run/accuracy_repeat.json"
"$py" mean/benchmark_round2.py r2_final --parent optimized --seed 99 --rounds 5 --validation "$run/accuracy_repeat.json" --pipeline "$run/pipeline.json" --output "$run/repeat.json"
compute-sanitizer --tool memcheck --error-exitcode 86 "$py" mean/validate.py r2_final --sanitizer --output "$run/memcheck.json" 2>&1 | tee "$run/memcheck.log"
compute-sanitizer --tool racecheck --error-exitcode 87 "$py" mean/validate.py r2_final --sanitizer --output "$run/racecheck.json" 2>&1 | tee "$run/racecheck.log"
