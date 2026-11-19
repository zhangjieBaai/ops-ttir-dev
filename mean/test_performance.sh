#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU explicitly}"
run="mean/results/runs/performance-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$run"
/root/venvs/ir-ops-dev/bin/python mean/verify_pipeline.py optimized --output "$run/pipeline.json"
/root/venvs/ir-ops-dev/bin/python mean/validate.py optimized --output "$run/accuracy.json"
/root/venvs/ir-ops-dev/bin/python mean/benchmark.py optimized --validation "$run/accuracy.json" --pipeline "$run/pipeline.json" --output "$run/main.json"
/root/venvs/ir-ops-dev/bin/python mean/benchmark.py optimized --seed 99 --rounds 5 --validation "$run/accuracy.json" --pipeline "$run/pipeline.json" --output "$run/repeat.json"
