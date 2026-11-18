#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one idle GPU first}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/ir-ops-dev/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${RESULTS_DIR:-layer_norm/results/runs/${RUN_ID}_accuracy}"
if [[ -e "$RESULTS_DIR" ]]; then
    echo "Results directory exists; choose a new RESULTS_DIR" >&2
    exit 2
fi
mkdir -p "$RESULTS_DIR"

"$PYTHON_BIN" layer_norm/export_baseline.py --verify
"$PYTHON_BIN" layer_norm/verify_pipeline.py --output "$RESULTS_DIR/pipeline.json"
"$PYTHON_BIN" layer_norm/validate.py baseline optimized --suite full \
    --output "$RESULTS_DIR/accuracy.json" | tee "$RESULTS_DIR/accuracy.log"
