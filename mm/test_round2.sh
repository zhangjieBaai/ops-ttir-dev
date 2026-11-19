#!/usr/bin/env bash
# Revalidate immutable round2 candidates; never overwrite old evidence or aliases.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Choose one idle H100 GPU}"
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=""
export KERNELFORGE_PLATFORM=h100
PY=/root/venvs/ir-ops-dev/bin/python
source mm/device_guard.sh
RUN=$(mktemp -d mm/results/runs/mm_opt_r2_replay_XXXXXX)
exec > >(tee "$RUN/run.log") 2>&1
trap 'status=$?; echo "exit_status=$status" > "$RUN/exit_status.txt"' EXIT
set -x
NAMES=(r2_pid_direct r2_zero_pad r2_absolute r2_zero_absolute r2_zero_pid r2_absolute_pid r2_zero_absolute_pid r2_k_offset)
run_step "$PY" mm/verify_pipeline.py --output "$RUN/pipeline.json"
run_step "$PY" mm/experiment_round2.py check --names "${NAMES[@]}" --pipeline "$RUN/pipeline.json" --output "$RUN/check"
run_step "$PY" mm/experiment_round2.py bench --names "${NAMES[@]}" --pipeline "$RUN/pipeline.json" \
    --validation "$RUN/check/summary.json" --output "$RUN/screen.json"
run_step "$PY" mm/experiment_round2.py bench --full --rounds 7 --names r2_k_offset --pipeline "$RUN/pipeline.json" \
    --validation "$RUN/check/summary.json" --output "$RUN/full.json"
run_step "$PY" mm/experiment_round2.py check --seed 123 --names r2_k_offset --pipeline "$RUN/pipeline.json" --output "$RUN/repeat_check"
run_step "$PY" mm/experiment_round2.py bench --full --rounds 5 --seed 123 --names r2_k_offset --pipeline "$RUN/pipeline.json" \
    --validation "$RUN/repeat_check/summary.json" --output "$RUN/repeat.json"
for TOOL in memcheck racecheck; do
    run_step compute-sanitizer --tool "$TOOL" --error-exitcode 99 --log-file "$RUN/$TOOL.log" \
        "$PY" mm/experiment_round2.py sanitize --names baseline r2_pid_direct r2_zero_pad r2_absolute r2_k_offset \
        --output "$RUN/${TOOL}_accuracy.json"
done
run_step "$PY" mm/check_round1_dispatch.py --output "$RUN/dispatch.json"
echo "Round2 raw replay: $RUN; report all negative and validity outcomes"
