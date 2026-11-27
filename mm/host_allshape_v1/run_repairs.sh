#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?Select the original H100}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=""
PY=/root/venvs/ir-ops-dev/bin/python
TASK=mm/host_allshape_v1
source mm/device_guard.sh
mapfile -t REPAIR_SIZES < <("$PY" -c 'import json; print("\n".join(map(str,json.load(open("mm/host_allshape_v1/results/repair_plan.json"))["sizes"])))')
run_step "$PY" "$TASK/experiment.py" bench --rounds 7 --sizes "${REPAIR_SIZES[@]}" --no-wall \
    --validation "$TASK/results/check_v4.json" --output "$TASK/results/repair_main.json" > "$TASK/results/repair_main.log" 2>&1
run_step "$PY" "$TASK/experiment.py" bench --rounds 7 --seed 123 --sizes "${REPAIR_SIZES[@]}" --no-wall \
    --validation "$TASK/results/repeat_check.json" --output "$TASK/results/repair_repeat.json" > "$TASK/results/repair_repeat.log" 2>&1
