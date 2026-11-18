#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one idle physical GPU}"
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME
export LD_LIBRARY_PATH=""
export PYTHONNOUSERSITE=1
PY=/root/venvs/ir-ops-dev/bin/python
source mm/device_guard.sh
mkdir -p mm/results/runs
RUN=$(mktemp -d "mm/results/runs/accuracy_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
exec > >(tee "$RUN/run.log") 2>&1
run_step "$PY" mm/export_baseline.py --verify
run_step "$PY" mm/verify_pipeline.py --output "$RUN/pipeline.json"
run_step "$PY" mm/validate.py --pipeline "$RUN/pipeline.json" --output "$RUN/accuracy.json"
echo "Accuracy evidence: $RUN"
