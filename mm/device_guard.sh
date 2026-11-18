# Sourced from the repository root. Check between GPU processes, when ours exited.
ensure_idle() {
    local occupants
    occupants=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
        --query-compute-apps=pid,process_name --format=csv,noheader)
    if [[ -n "$occupants" ]]; then
        echo "GPU $CUDA_VISIBLE_DEVICES has an existing compute process; aborting this run." >&2
        echo "$occupants" >&2
        return 1
    fi
}

run_step() {
    ensure_idle
    "$@"
    ensure_idle
}
