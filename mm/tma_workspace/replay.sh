#!/usr/bin/env bash
# Reproduce the matched-workspace trial; tool failures remain nonzero and recorded.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Choose an idle H100 GPU}"
cd "$(dirname "$0")/../.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=""
PY=/root/venvs/ir-ops-dev/bin/python
W=mm/tma_workspace/experiment.py
RUN=$(mktemp -d mm/tma_workspace/results/replay_XXXXXX)
exec > >(tee "$RUN/run.log") 2>&1
set -x
"$PY" "$W" pipeline --output "$RUN/pipeline.json"
"$PY" "$W" check --pipeline "$RUN/pipeline.json" --output "$RUN/check"
"$PY" "$W" bench --full --rounds 7 --pipeline "$RUN/pipeline.json" --validation "$RUN/check/summary.json" --output "$RUN/full.json"
"$PY" "$W" check --seed 123 --pipeline "$RUN/pipeline.json" --output "$RUN/repeat_check"
"$PY" "$W" bench --full --rounds 5 --seed 123 --pipeline "$RUN/pipeline.json" --validation "$RUN/repeat_check/summary.json" --output "$RUN/repeat.json"
final_status=0
for TOOL in memcheck racecheck; do
    set +e
    compute-sanitizer --tool "$TOOL" --error-exitcode 99 --log-file "$RUN/$TOOL.log" \
        "$PY" "$W" check --seed 123 --pipeline "$RUN/pipeline.json" --output "$RUN/${TOOL}_check"
    tool_status=$?
    set -e
    echo "$tool_status" > "$RUN/${TOOL}_exit_status.txt"
    if (( tool_status != 0 )); then final_status=$tool_status; fi
 done
"$PY" mm/tma_workspace/verify_api.py --output "$RUN/api.json"
echo "Raw matched-workspace experiment: $RUN; final tool status=$final_status (no automatic promotion)"
exit "$final_status"
