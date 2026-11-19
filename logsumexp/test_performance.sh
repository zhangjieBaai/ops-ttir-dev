#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir=$(mktemp -d logsumexp/results/runs/performance_XXXXXXXX)
/root/venvs/ir-ops-dev/bin/python logsumexp/verify_pipeline.py optimized --output "$run_dir/pipeline.json"
/root/venvs/ir-ops-dev/bin/python logsumexp/validate.py optimized --output "$run_dir/accuracy.json"
/root/venvs/ir-ops-dev/bin/python logsumexp/benchmark.py optimized --rounds 7 --round-protocol logsumexp/optimization_plan_round3.md --pipeline "$run_dir/pipeline.json" --validation "$run_dir/accuracy.json" --output "$run_dir/benchmark.json"
