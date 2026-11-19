#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU}"
DX_PY=/root/venvs/ir-ops-dev/bin/python
DX_OUT="${RESULTS_DIR:-layer_norm/backward_dx/results/runs/$(date -u +%Y%m%dT%H%M%SZ)_accuracy}"
if [[ -e "$DX_OUT" ]]; then
    echo 'Choose a new RESULTS_DIR' >&2
    exit 2
fi
mkdir -p "$DX_OUT"
"$DX_PY" layer_norm/backward_dx/export_baseline.py --check-selected --output "$DX_OUT/pipeline.json"
"$DX_PY" layer_norm/backward_dx/validate.py baseline optimized --output "$DX_OUT/accuracy.json" | tee "$DX_OUT/accuracy.log"
