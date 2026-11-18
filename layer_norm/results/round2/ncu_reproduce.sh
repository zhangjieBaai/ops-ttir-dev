#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
: "${CUDA_VISIBLE_DEVICES:?Set one idle GPU}"
NCU_OUTPUT="${NCU_OUTPUT:?Choose a new report output prefix}"
ncu --target-processes all --kernel-name-base function --kernel-name regex:_layer_norm_fwd_fused \
  --section LaunchStats --section Occupancy --section SpeedOfLight --section WarpStateStats \
  --section MemoryWorkloadAnalysis --clock-control base --cache-control all -o "$NCU_OUTPUT" \
  /root/venvs/ir-ops-dev/bin/python layer_norm/profile_ttir.py \
  tile_dispatch_math_v2 r2_fine2048_stream r2_mixed_stream
