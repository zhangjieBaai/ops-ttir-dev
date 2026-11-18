#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one idle GPU first}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/ir-ops-dev/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-layer_norm/results/reruns/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RESULTS_DIR"
if [[ -e "$RESULTS_DIR/full_benchmark.json" ]]; then
    echo "Use a new RESULTS_DIR to preserve existing evidence" >&2
    exit 2
fi

"$PYTHON_BIN" layer_norm/export_baseline.py --verify
"$PYTHON_BIN" layer_norm/verify_pipeline.py --output "$RESULTS_DIR/native_pipeline_check.json"
"$PYTHON_BIN" layer_norm/make_candidates.py
"$PYTHON_BIN" layer_norm/validate.py baseline canonical arithmetic single_tile \
    tile_dispatch_v2 tile_dispatch_math_v2 --suite full \
    --output "$RESULTS_DIR"/accuracy_full.json > "$RESULTS_DIR"/accuracy_full.log 2>&1
"$PYTHON_BIN" layer_norm/benchmark.py canonical arithmetic single_tile tile_dispatch_v2 tile_dispatch_math_v2 \
    --rounds 3 --rep 30 --validation "$RESULTS_DIR"/accuracy_full.json \
    --output "$RESULTS_DIR"/screening.json > "$RESULTS_DIR"/screening.log 2>&1
"$PYTHON_BIN" layer_norm/benchmark.py tile_dispatch_v2 tile_dispatch_math_v2 \
    --full --rounds 7 --rep 20 --validation "$RESULTS_DIR"/accuracy_full.json \
    --output "$RESULTS_DIR"/full_benchmark.json > "$RESULTS_DIR"/full_benchmark.log 2>&1
"$PYTHON_BIN" layer_norm/finalize.py --select --results-dir "$RESULTS_DIR"
winner="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["kernel"])' "$RESULTS_DIR/selection.json")"
"$PYTHON_BIN" layer_norm/benchmark.py "$winner" --full --rounds 5 --rep 10 --seed 123 \
    --validation "$RESULTS_DIR"/accuracy_full.json \
    --output "$RESULTS_DIR"/independent_repeat.json > "$RESULTS_DIR"/independent_repeat.log 2>&1
compute-sanitizer --tool memcheck --error-exitcode 86 --log-file "$RESULTS_DIR"/memcheck.log \
    "$PYTHON_BIN" layer_norm/validate.py baseline "$winner" --suite sanitizer \
    --output "$RESULTS_DIR"/memcheck_accuracy.json > "$RESULTS_DIR"/memcheck_accuracy.log 2>&1
compute-sanitizer --tool racecheck --error-exitcode 86 --log-file "$RESULTS_DIR"/racecheck.log \
    "$PYTHON_BIN" layer_norm/validate.py baseline "$winner" --suite sanitizer \
    --output "$RESULTS_DIR"/racecheck_accuracy.json > "$RESULTS_DIR"/racecheck_accuracy.log 2>&1
"$PYTHON_BIN" layer_norm/finalize.py --results-dir "$RESULTS_DIR"
