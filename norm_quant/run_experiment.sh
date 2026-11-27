#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an idle exclusive GPU}"
PYTHON=/root/venvs/ir-ops-dev/bin/python
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
"$PYTHON" norm_quant/make_ablation.py
"$PYTHON" norm_quant/ablation_benchmark.py --run "${RUN_ID}_main" --rounds 7 --rep 20 --seed 42
"$PYTHON" norm_quant/ablation_benchmark.py --run "${RUN_ID}_repeat" --rounds 5 --rep 20 --seed 123
"$PYTHON" norm_quant/report_ablation.py --main "${RUN_ID}_main" --repeat "${RUN_ID}_repeat"
