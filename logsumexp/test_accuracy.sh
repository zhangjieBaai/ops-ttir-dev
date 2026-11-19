#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir=$(mktemp -d logsumexp/results/runs/accuracy_XXXXXXXX)
/root/venvs/ir-ops-dev/bin/python logsumexp/verify_pipeline.py optimized --output "$run_dir/pipeline.json"
/root/venvs/ir-ops-dev/bin/python logsumexp/validate.py optimized --output "$run_dir/accuracy.json"
