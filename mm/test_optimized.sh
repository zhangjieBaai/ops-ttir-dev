#!/usr/bin/env bash
# Replay the retained six-family selection without autotuning.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Choose one idle physical GPU}"
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=""
export KERNELFORGE_PLATFORM=h100
PY=/root/venvs/ir-ops-dev/bin/python
source mm/device_guard.sh
RUN=$(mktemp -d "mm/results/runs/optimized_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
NAMES=mm/results/runs/mm_opt_r1_20261118/selected_names.json
exec > >(tee "$RUN/run.log") 2>&1
trap 'status=$?; echo "exit_status=$status" > "$RUN/exit_status.txt"' EXIT
set -x
run_step "$PY" mm/verify_pipeline.py --output "$RUN/pipeline.json"
run_step "$PY" mm/experiment_round1.py check --names r1_canonical optimized --pipeline "$RUN/pipeline.json" --output "$RUN/check"
run_step "$PY" mm/experiment_round1.py bench --names r1_canonical optimized --full --rounds 7 \
    --pipeline "$RUN/pipeline.json" --validation "$RUN/check/summary.json" --output "$RUN/full.json"
run_step "$PY" mm/experiment_round1.py check --seed 123 --names r1_canonical optimized --pipeline "$RUN/pipeline.json" --output "$RUN/repeat_check"
run_step "$PY" mm/experiment_round1.py bench --seed 123 --names r1_canonical optimized --full --rounds 5 \
    --pipeline "$RUN/pipeline.json" --validation "$RUN/repeat_check/summary.json" --output "$RUN/repeat.json"
for TOOL in memcheck racecheck; do
    run_step compute-sanitizer --tool "$TOOL" --error-exitcode 99 --log-file "$RUN/$TOOL.log" \
        "$PY" mm/experiment_round1.py sanitize --selection "$NAMES" --output "$RUN/${TOOL}_accuracy.json"
done
run_step "$PY" mm/check_round1_dispatch.py --output "$RUN/dispatch.json"
echo "Raw optimized selection replay: $RUN (inspect validity markers before claiming a gain)"
