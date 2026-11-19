#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU}"
: "${NCU_OUTPUT:?Choose a new output prefix}"
ncu --target-processes all --kernel-name-base function --kernel-name regex:layer_norm_bwd_dx \
    --section LaunchStats --section Occupancy --section WarpStateStats --section SpeedOfLight \
    --section MemoryWorkloadAnalysis --clock-control base --cache-control all -o "$NCU_OUTPUT" \
    /root/venvs/ir-ops-dev/bin/python layer_norm/backward_dx/benchmark.py joint_coefficients grouped2048 \
    --profile --validation layer_norm/backward_dx/results/accuracy_all.json --output "$NCU_OUTPUT-launches.json"
