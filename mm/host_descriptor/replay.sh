#!/usr/bin/env bash
# Feasibility only: no timing/promotion while racecheck remains unresolved.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Choose one idle H100 GPU}"
cd "$(dirname "$0")/../.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=""
PY=/root/venvs/ir-ops-dev/bin/python
W=mm/host_descriptor/probe.py
REG=mm/host_descriptor/results/prototype_inputs_v2.json
RUN=$(mktemp -d mm/host_descriptor/results/replay_XXXXXX)
exec > >(tee "$RUN/run.log") 2>&1
set -x
"$PY" "$W" --full --names envelope_baseline envelope_host_v1 --output "$RUN/check"
final_status=0
for TOOL in memcheck racecheck; do
    set +e
    compute-sanitizer --tool "$TOOL" --error-exitcode 99 --log-file "$RUN/$TOOL.log" \
        "$PY" "$W" --full --names envelope_baseline envelope_host_v1 --output "$RUN/${TOOL}_check"
    tool_status=$?
    set -e
    echo "$tool_status" > "$RUN/${TOOL}_exit_status.txt"
    if (( tool_status != 0 )); then final_status=$tool_status; fi
 done
# This deliberately different per-loop schedule is a diagnostic, not a silent
# switch of the baseline contract or a performance candidate.
set +e
compute-sanitizer --tool racecheck --error-exitcode 99 --log-file "$RUN/serial_racecheck.log" \
    "$PY" "$W" --full --names envelope_baseline host_serial --registry "$REG" --output "$RUN/serial_check"
serial_status=$?
set -e
printf '%s\n' "$serial_status" > "$RUN/serial_exit_status.txt"
if (( serial_status != 0 )); then final_status=$serial_status; fi
"$PY" mm/host_descriptor/final_checks.py --output "$RUN/final_checks.json"
echo "Feasibility evidence: $RUN; tool status=$final_status. No automatic optimization promotion."
exit "$final_status"
