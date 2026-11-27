#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?Select an idle H100}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=""
PY=/root/venvs/ir-ops-dev/bin/python
TASK=mm/host_allshape_v1
source mm/device_guard.sh
run_step "$PY" "$TASK/experiment.py" bench --validation "$TASK/results/check_v4.json" --output "$TASK/results/main.json" > "$TASK/results/main.log" 2>&1
run_step "$PY" "$TASK/experiment.py" check --seed 123 --output "$TASK/results/repeat_check.json" > "$TASK/results/repeat_check.log" 2>&1
run_step "$PY" "$TASK/experiment.py" bench --seed 123 --rounds 5 --validation "$TASK/results/repeat_check.json" --output "$TASK/results/repeat.json" > "$TASK/results/repeat.log" 2>&1
run_step compute-sanitizer --tool memcheck --error-exitcode 99 --log-file "$TASK/results/memcheck.log" \
    "$PY" "$TASK/experiment.py" sanitize --output "$TASK/results/memcheck_check.json" > "$TASK/results/memcheck_stdout.log" 2>&1
ensure_idle
set +e
compute-sanitizer --tool racecheck --error-exitcode 99 --log-file "$TASK/results/racecheck.log" \
    "$PY" "$TASK/experiment.py" sanitize --output "$TASK/results/racecheck_check.json" > "$TASK/results/racecheck_stdout.log" 2>&1
RACE_STATUS=$?
set -e
ensure_idle
printf '%s\n' "$RACE_STATUS" > "$TASK/results/racecheck_exit_status.txt"
printf 'Measurements and tools finished; racecheck exit %s (warnings retained)\n' "$RACE_STATUS"
