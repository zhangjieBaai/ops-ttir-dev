#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${RESULTS_DIR:-moe_routing/ablation_v1/results/performance_$(date -u +%Y%m%dT%H%M%SZ)}
exec bash moe_routing/run_experiment.sh "$OUT" "${SEED:-42}" "${ROUNDS:-5}"
