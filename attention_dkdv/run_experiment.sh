#!/usr/bin/env bash
# Reproduce current final and its ablation without replacing historical results.
set -euo pipefail
gpu=${1:?usage: run_experiment.sh GPU SEED ROUNDS FRESH_OUTPUT_DIRECTORY}
seed=${2:?input and timing seed}
rounds=${3:?paired rounds (main 7, repeat 5)}
output=${4:?fresh output directory}
mkdir -p "$(dirname "$output")"
mkdir "$output"
output=$(realpath "$output")
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=
python_bin=/root/venvs/ir-ops-dev/bin/python
"$python_bin" attention_dkdv/make_ablation.py
nvidia-smi -i "$gpu" -lgc 1200,1200
trap 'nvidia-smi -i "$gpu" -rgc' EXIT
export ATTENTION_R2_LOCK_SM_MHZ=1200
"$python_bin" attention_dkdv/round2/verify_pipeline.py --output "$output/pipeline"
"$python_bin" attention_dkdv/ablation_benchmark.py --seed "$seed" --rounds "$rounds" \
  --pipeline "$output/pipeline/pipeline.json" --output "$output/measurement"
