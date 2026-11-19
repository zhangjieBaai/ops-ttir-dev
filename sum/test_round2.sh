#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Select one idle GPU explicitly}"
run_dir=$(mktemp -d sum/results/runs/round2_reproduce_XXXXXXXX)
python_bin=/root/venvs/ir-ops-dev/bin/python
"$python_bin" sum/verify_pipeline.py --output "$run_dir/source_pipeline.json"
"$python_bin" sum/round2.py pipeline --pipeline "$run_dir/source_pipeline.json" --output "$run_dir/pipeline.json"
"$python_bin" sum/round2.py validate --output "$run_dir/accuracy.json"
"$python_bin" sum/round2.py bench --pipeline "$run_dir/pipeline.json" \
  --validation "$run_dir/accuracy.json" --output "$run_dir/benchmark.json"
echo "$run_dir"
