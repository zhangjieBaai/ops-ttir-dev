#!/usr/bin/env bash
# Run on one idle GPU, starting with automatic SM clocks.
set -eu
gpu=${1:?GPU index}
seed=${2:?input and timing seed}
rounds=${3:?number of paired rounds}
output=${4:?fresh output directory}
mkdir "$output"
output=$(realpath "$output")
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH=
python_bin=/root/venvs/ir-ops-dev/bin/python
nvidia-smi -i "$gpu" -lgc 1200,1200
export ATTENTION_ABLATION_SM_MHZ=1200
trap 'nvidia-smi -i "$gpu" -rgc' EXIT
"$python_bin" round2/verify_pipeline.py --output "$output/pipeline"
"$python_bin" ablation/benchmark.py --seed "$seed" --rounds "$rounds" --output "$output/benchmark.json"
