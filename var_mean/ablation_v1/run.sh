#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TRITON_DISABLE_LINE_INFO=1
PY=/root/venvs/ir-ops-dev/bin/python
ROOT=var_mean/ablation_v1
RUN="${1:-$(mktemp -d "$ROOT/results/reproduction_XXXXXX")}"
mkdir -p "$RUN"
NAMES=(baseline ab4_000 ab4_001 ab4_010 ab4_011 ab4_100 ab4_101 ab4_110 ab4_111)
"$PY" var_mean/verify_pipeline.py --candidate ab4_111 --output "$RUN/pipeline.json"
"$PY" var_mean/validate.py --names "${NAMES[@]}" --output "$RUN/accuracy.json"
"$PY" var_mean/benchmark.py --names aa double --suite controls --rounds 5 --accuracy "$RUN/accuracy.json" --pipeline "$RUN/pipeline.json" --output "$RUN/controls.json"
"$PY" "$ROOT/benchmark.py" --accuracy "$RUN/accuracy.json" --output "$RUN/main.json"
"$PY" var_mean/validate.py --names "${NAMES[@]}" --seed 123 --output "$RUN/repeat_accuracy.json"
"$PY" "$ROOT/benchmark.py" --accuracy "$RUN/repeat_accuracy.json" --rounds 5 --seed 123 --output "$RUN/repeat.json"
"$PY" var_mean/benchmark.py --names aa double --suite controls --rounds 5 --seed 123 --accuracy "$RUN/repeat_accuracy.json" --pipeline "$RUN/pipeline.json" --output "$RUN/repeat_controls.json"
"$PY" "$ROOT/verify_source.py" --source-module source_probes --output "$RUN/source_feasibility.json"
"$PY" "$ROOT/verify_source.py" --source-module source_probes_v2 --output "$RUN/source_feasibility_v2.json"
"$PY" "$ROOT/verify_source.py" --output "$RUN/source_feasibility_v3.json"
for TOOL in memcheck racecheck; do
    compute-sanitizer --tool "$TOOL" --error-exitcode 1 "$PY" "$ROOT/sanitize.py" --output "$RUN/${TOOL}_accuracy.json" > "$RUN/${TOOL}.log" 2>&1
done
"$PY" "$ROOT/audit_source.py" --results-dir "$RUN"
"$PY" "$ROOT/analyze.py" --results-dir "$RUN" --output "$RUN/report.md"
