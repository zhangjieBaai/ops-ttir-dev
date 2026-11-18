#!/usr/bin/env bash
# New evidence directory for every invocation; GPU selection stays explicit.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one idle physical GPU}"
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME
export LD_LIBRARY_PATH=""
export PYTHONNOUSERSITE=1
PY=/root/venvs/ir-ops-dev/bin/python
source mm/device_guard.sh
mkdir -p mm/results/runs
RUN=$(mktemp -d "mm/results/runs/baseline_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
exec > >(tee "$RUN/run.log") 2>&1
trap 'status=$?; echo "exit_status=$status" > "$RUN/exit_status.txt"' EXIT
set -x
date -u --iso-8601=seconds
git rev-parse HEAD
git status --short
nvidia-smi
run_step "$PY" mm/export_baseline.py
run_step "$PY" mm/verify_pipeline.py --output "$RUN/pipeline.json"
run_step "$PY" mm/validate.py --pipeline "$RUN/pipeline.json" --output "$RUN/accuracy.json"
run_step "$PY" mm/benchmark.py --pipeline "$RUN/pipeline.json" --validation "$RUN/accuracy.json" --output "$RUN/benchmark.json"
run_step "$PY" mm/validate.py --seed 123 --pipeline "$RUN/pipeline.json" --output "$RUN/accuracy_repeat.json"
run_step "$PY" mm/benchmark.py --seed 123 --pipeline "$RUN/pipeline.json" --validation "$RUN/accuracy_repeat.json" --output "$RUN/benchmark_repeat.json"
for TOOL in memcheck racecheck; do
    run_step compute-sanitizer --tool "$TOOL" --error-exitcode 99 --log-file "$RUN/$TOOL.log" \
        "$PY" mm/validate.py --suite sanitizer --pipeline "$RUN/pipeline.json" --output "$RUN/${TOOL}_accuracy.json"
done
echo "Baseline complete: $RUN"
