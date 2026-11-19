#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Select one idle GPU explicitly}"
run_dir=$(mktemp -d sum/results/runs/accuracy_XXXXXXXX)
python_bin=/root/venvs/ir-ops-dev/bin/python
"$python_bin" sum/verify_pipeline.py optimized --output "$run_dir/pipeline.json"
"$python_bin" sum/validate.py baseline optimized --output "$run_dir/accuracy.json"
echo "$run_dir"
