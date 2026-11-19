#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU explicitly}"
run="mean/results/runs/accuracy-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$run"
/root/venvs/ir-ops-dev/bin/python mean/verify_pipeline.py optimized --output "$run/pipeline.json"
/root/venvs/ir-ops-dev/bin/python mean/validate.py optimized --output "$run/accuracy.json"
