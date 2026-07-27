#!/usr/bin/env bash
set -euo pipefail

seed=${SEED:-2021}
run_id=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
log_root="logs/TimeBase/seed_${seed}"
mkdir -p "$log_root"

run_worker() {
    local gpu=$1
    local dataset pred
    while read -r dataset pred; do
        echo "WORKER gpu=${gpu} task=${dataset}-${pred}"
        NO_SUMMARY=1 bash scripts/TimeBase/run.sh "$dataset" "$gpu" "$pred"
    done < <(job_schedule)
}

# Descending seed-2021 epoch time. Shared traversal plus per-job locks lets the
# first free GPU take the next expensive horizon instead of waiting by dataset.
job_schedule() {
    printf '%s\n' \
        'traffic 720' 'traffic 192' 'electricity 720' 'traffic 96' \
        'traffic 336' 'electricity 336' 'electricity 192' 'electricity 96' \
        'ETTm2 336' 'ETTm2 96' 'ETTm2 192' 'weather 96' \
        'weather 192' 'weather 720' 'weather 336' 'ETTm2 720' \
        'ETTm1 720' 'ETTm1 336' 'ETTm1 96' 'ETTh2 96' \
        'ETTh1 96' 'ETTm1 192' 'ETTh2 192' 'ETTh1 336' \
        'ETTh1 192' 'ETTh1 720' 'ETTh2 336' 'ETTh2 720'
}

if [ "${1:-}" = "--worker" ]; then
    gpu=${2:?GPU index is required}
    run_worker "$gpu"
    exit 0
fi

exec {run_lock_fd}> "$log_root/.run.lock"
if ! flock -n "$run_lock_fd"; then
    echo "Another TimeBase seed ${seed} run is already active" >&2
    exit 1
fi

worker_pids=()
stop_workers() {
    local pid
    trap - INT TERM EXIT
    for pid in "${worker_pids[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    for pid in "${worker_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap stop_workers INT TERM EXIT

# Each GPU owns one process group and dynamically claims individual horizons.
setsid bash "$0" --worker 0 \
    > "$log_root/worker_0_${run_id}.log" 2>&1 & worker_pids+=("$!")
setsid bash "$0" --worker 1 \
    > "$log_root/worker_1_${run_id}.log" 2>&1 & worker_pids+=("$!")
setsid bash "$0" --worker 2 \
    > "$log_root/worker_2_${run_id}.log" 2>&1 & worker_pids+=("$!")

status=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || status=1
done
worker_pids=()
trap - INT TERM EXIT

"${PYTHON:-/root/miniconda3/bin/python}" summarize_results.py
exit "$status"
