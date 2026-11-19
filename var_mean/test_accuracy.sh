#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TRITON_DISABLE_LINE_INFO=1
PY=/root/venvs/ir-ops-dev/bin/python
RUN="${1:-$(mktemp -d var_mean/results/runs/accuracy_XXXXXX)}"
mkdir -p "$RUN"
"$PY" var_mean/verify_pipeline.py --candidate optimized --output "$RUN/pipeline.json"
"$PY" var_mean/validate.py --names baseline optimized --output "$RUN/accuracy.json"
