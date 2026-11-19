#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU}"
: "${RESULTS_DIR:?Choose a new output directory}"
if [[ -e "$RESULTS_DIR" ]]; then
    echo 'Choose a new RESULTS_DIR to preserve evidence' >&2
    exit 2
fi
DX_PY=/root/venvs/ir-ops-dev/bin/python
mkdir -p "$RESULTS_DIR"
"$DX_PY" layer_norm/backward_dx/export_baseline.py --check-selected --output "$RESULTS_DIR/pipeline.json"
"$DX_PY" layer_norm/backward_dx/make_round2.py
"$DX_PY" layer_norm/backward_dx/validate.py baseline r1_parent r2_range2048 r2_range4096 \
    --output "$RESULTS_DIR/accuracy.json" > "$RESULTS_DIR/accuracy.log" 2>&1
"$DX_PY" layer_norm/backward_dx/benchmark.py r1_parent r2_range2048 r2_range4096 \
    --full --rounds 7 --rep 20 --validation "$RESULTS_DIR/accuracy.json" \
    --output "$RESULTS_DIR/full.json" | tee "$RESULTS_DIR/full.log"
# Profiles are serialized after timings. Both use the same native kernels and fixed NCU conditions.
ncu --target-processes all --kernel-name-base function --kernel-name regex:layer_norm_bwd_dx \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct \
    --clock-control base --cache-control all -o "$RESULTS_DIR/traffic" \
    "$DX_PY" layer_norm/backward_dx/benchmark.py r1_parent r2_loop2048 --profile \
    --validation layer_norm/backward_dx/results/round2/accuracy_batch1.json \
    --output "$RESULTS_DIR/traffic_launches.json" > "$RESULTS_DIR/traffic.log" 2>&1
