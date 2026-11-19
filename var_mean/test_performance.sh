#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TRITON_DISABLE_LINE_INFO=1
PY=/root/venvs/ir-ops-dev/bin/python
RUN="${1:-$(mktemp -d var_mean/results/runs/performance_XXXXXX)}"
mkdir -p "$RUN"
bash var_mean/test_accuracy.sh "$RUN"
"$PY" var_mean/benchmark.py --names aa double --suite controls --rounds 5 --accuracy "$RUN/accuracy.json" --pipeline "$RUN/pipeline.json" --output "$RUN/controls.json"
"$PY" var_mean/benchmark.py --names optimized --accuracy "$RUN/accuracy.json" --pipeline "$RUN/pipeline.json" --output "$RUN/full.json"
"$PY" var_mean/validate.py --names baseline optimized --seed 123 --output "$RUN/repeat_accuracy.json"
"$PY" var_mean/benchmark.py --names optimized --rounds 5 --seed 123 --accuracy "$RUN/repeat_accuracy.json" --pipeline "$RUN/pipeline.json" --output "$RUN/repeat.json"
"$PY" var_mean/benchmark.py --names aa double --suite controls --seed 123 --rounds 5 --accuracy "$RUN/repeat_accuracy.json" --pipeline "$RUN/pipeline.json" --output "$RUN/repeat_controls.json"
for TOOL in memcheck racecheck; do
    compute-sanitizer --tool "$TOOL" --error-exitcode 1 "$PY" var_mean/validate.py --names baseline optimized --sanitizer --output "$RUN/${TOOL}_accuracy.json" > "$RUN/${TOOL}.log" 2>&1
done
