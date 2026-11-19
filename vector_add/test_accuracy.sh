#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Select one idle H100 explicitly}"
cd "$(dirname "$0")/.."
run_dir=$(mktemp -d "$PWD/vector_add/results/runs/accuracy_XXXXXX")
py=/root/venvs/ir-ops-dev/bin/python
"$py" vector_add/verify_pipeline.py --names optimized --output "$run_dir/pipeline.json"
"$py" vector_add/validate.py baseline optimized --output "$run_dir/accuracy.json"
