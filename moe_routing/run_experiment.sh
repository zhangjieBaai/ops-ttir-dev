#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one idle, exclusively reserved GPU}"
PY=/root/venvs/ir-ops-dev/bin/python
OUT=${1:-moe_routing/ablation_v1/results/run_$(date -u +%Y%m%dT%H%M%SZ)}
SEED=${2:-42}
ROUNDS=${3:-5}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
test ! -e "$OUT"
mkdir -p "$OUT"
"$PY" moe_routing/make_ablation.py
"$PY" moe_routing/verify_pipeline.py --output "$OUT/pipeline_original.json"
"$PY" moe_routing/ablation_benchmark.py --output "$OUT/measurement" --seed "$SEED" --rounds "$ROUNDS" --rep 10
