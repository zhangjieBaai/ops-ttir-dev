#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?请先指定独占空闲GPU}"
run_name=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
seed=${2:-42}
rounds=${3:-7}
experiment_python=/root/venvs/ir-ops-dev/bin/python
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid run name" >&2; exit 2; }
experiment_root=int4_gemm/ablation
test ! -e "$experiment_root/results/$run_name"
test ! -e "$experiment_root/results/${run_name}_pipeline.json"
"$experiment_python" "$experiment_root/generate.py"
"$experiment_python" "$experiment_root/verify_pipeline.py" \
  --seed "$seed" --output "$experiment_root/results/${run_name}_pipeline.json"
"$experiment_python" "$experiment_root/run.py" \
  --run "$run_name" --seed "$seed" --rounds "$rounds" --rep 20
