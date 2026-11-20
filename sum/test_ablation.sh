#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Select one idle GPU explicitly}"
run_dir=$(mktemp -d sum/ablation/results/reproduce_XXXXXXXX)
python_bin=/root/venvs/ir-ops-dev/bin/python
# Existing frozen manifest/IR is reused; all compiled identities are rechecked.
"$python_bin" sum/ablation_run.py validate --output "$run_dir/accuracy.json"
"$python_bin" sum/ablation_run.py pipeline --output "$run_dir/pipeline.json"
"$python_bin" sum/ablation_run.py bench --pipeline "$run_dir/pipeline.json" \
  --validation "$run_dir/accuracy.json" --output "$run_dir/main.json"
echo "$run_dir"
