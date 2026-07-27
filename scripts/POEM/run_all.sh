#!/usr/bin/env bash
set -euo pipefail

model_id=POEM
model=POEM
output_family=POEM
run_tag=poem
seed=${SEED:-2021}
python_bin=${PYTHON:-/root/miniconda3/bin/python}
gpu_ids=${GPU_IDS:-0}
dataset_filter=${DATASET_FILTER:-all}
pred_filter=${PRED_FILTER:-all}
. scripts/POEM/formal_config.sh

is_complete() {
    local dataset=$1 pred=$2 slot data_type data_path d_model period layers
    local dropout affine epochs patience batch lr lradj enc_in result_dir setting
    read -r slot dataset data_type data_path pred d_model period layers dropout \
        affine epochs patience batch lr lradj < <(formal_job_for "$dataset" "$pred")
    enc_in=$(enc_in_for "$dataset")
    setting="${model_id}_${model}_${dataset}_ftM_sl720_pl${pred}_test_seed${seed}"
    result_dir="results/${output_family}/seed_${seed}/${dataset}/pred_${pred}/${setting}"
    [ -f "$result_dir/metrics.npy" ] && [ -f "$result_dir/settings.json" ] || return 1
    "$python_bin" - "$result_dir/settings.json" "$data_type" "$data_path" \
        "$d_model" "$period" "$layers" "$dropout" "$affine" "$epochs" \
        "$patience" "$batch" "$lr" "$lradj" "$enc_in" "$pred" "$seed" <<'PY'
import json
import math
import sys

settings = json.load(open(sys.argv[1]))
expected = {
    "model_id": "POEM",
    "output_family": "POEM",
    "run_tag": "poem",
    "data": sys.argv[2],
    "data_path": sys.argv[3],
    "d_model": int(sys.argv[4]),
    "period_len": int(sys.argv[5]),
    "mixer_layers": int(sys.argv[6]),
    "mixer_dropout": float(sys.argv[7]),
    "affine": int(sys.argv[8]),
    "train_epochs": int(sys.argv[9]),
    "patience": int(sys.argv[10]),
    "batch_size": int(sys.argv[11]),
    "learning_rate": float(sys.argv[12]),
    "lradj": sys.argv[13],
    "enc_in": int(sys.argv[14]),
    "pred_len": int(sys.argv[15]),
    "seed": int(sys.argv[16]),
}
for key, expected_value in expected.items():
    actual_value = settings.get(key)
    if isinstance(expected_value, float):
        if actual_value is None or not math.isclose(float(actual_value), expected_value):
            raise SystemExit(1)
    elif actual_value != expected_value:
        raise SystemExit(1)
PY
}

# Descending measured/estimated runtime. Three workers traverse the same list
# and use per-job locks, so a free GPU immediately claims the next horizon.
job_schedule() {
    local dataset pred
    while read -r dataset pred; do
        if { [ "$dataset_filter" = all ] || [ "$dataset_filter" = "$dataset" ]; } \
            && { [ "$pred_filter" = all ] || [ "$pred_filter" = "$pred" ]; }; then
            printf '%s %s\n' "$dataset" "$pred"
        fi
    done < <(printf '%s\n' \
        'traffic 96' 'traffic 720' 'traffic 336' 'traffic 192' \
        'electricity 96' 'electricity 720' 'electricity 336' 'electricity 192' \
        'weather 336' 'weather 192' 'weather 720' 'weather 96' \
        'ETTm2 720' 'ETTm2 336' 'ETTm2 192' 'ETTm2 96' \
        'ETTm1 720' 'ETTm1 336' 'ETTm1 192' 'ETTm1 96' \
        'ETTh1 720' 'ETTh1 336' 'ETTh1 192' 'ETTh1 96' \
        'ETTh2 720' 'ETTh2 336' 'ETTh2 192' 'ETTh2 96')
}

run_worker() {
    local gpu=$1
    local slot dataset data_type data_path pred d_model period layers dropout affine
    local epochs patience batch lr lradj
    local wanted_dataset wanted_pred lock_fd

    while read -r wanted_dataset wanted_pred; do
        read -r slot dataset data_type data_path pred d_model period layers dropout affine epochs patience batch lr lradj \
            < <(formal_job_for "$wanted_dataset" "$wanted_pred")
        if is_complete "$dataset" "$pred"; then
            echo "SKIP complete: ${dataset}-${pred}"
            continue
        fi

        exec {lock_fd}> "$lock_root/${dataset}_${pred}.lock"
        if ! flock -n "$lock_fd"; then
            echo "SKIP active on another worker: ${dataset}-${pred}"
            exec {lock_fd}>&-
            continue
        fi
        if is_complete "$dataset" "$pred"; then
            echo "SKIP complete after lock: ${dataset}-${pred}"
            exec {lock_fd}>&-
            continue
        fi

        local log_dir="logs/${output_family}/seed_${seed}/${dataset}"
        local log_path="${log_dir}/pred_${pred}.log"
        mkdir -p "$log_dir"
        echo "START gpu=${gpu}: ${dataset}-${pred}"
        "$python_bin" -u run_longExp.py \
            --is_training 1 --root_path ./dataset/ --data_path "$data_path" \
            --model_id "$model_id" --data "$data_type" --features M \
            --model "$model" --output_family "$output_family" \
            --seq_len 720 --pred_len "$pred" --period_len "$period" \
            --enc_in "$(enc_in_for "$dataset")" --d_model "$d_model" \
            --mixer_layers "$layers" --mixer_dropout "$dropout" \
            --revin 1 --affine "$affine" \
            --train_epochs "$epochs" --patience "$patience" --batch_size "$batch" \
            --learning_rate "$lr" --weight_decay 1e-4 --gradient_clip 1.0 \
            --loss mae --lradj "$lradj" --run_tag "$run_tag" --seed "$seed" --gpu "$gpu" \
            > "$log_path" 2>&1
        echo "DONE gpu=${gpu}: ${dataset}-${pred}"
        exec {lock_fd}>&-
    done < <(job_schedule)
}

log_root="logs/${output_family}/seed_${seed}"
lock_root="$log_root/.locks"
mkdir -p "$log_root"
mkdir -p "$lock_root"

if [ "${1:-}" = "--worker" ]; then
    gpu=${2:?GPU index is required}
    run_worker "$gpu"
    exit 0
fi

exec {run_lock_fd}> "$log_root/.run.lock"
if ! flock -n "$run_lock_fd"; then
    echo "Another POEM seed ${seed} run is already active" >&2
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

read -r -a gpu_array <<< "$gpu_ids"
for gpu in "${gpu_array[@]}"; do
    setsid env DATASET_FILTER="$dataset_filter" PRED_FILTER="$pred_filter" \
        bash "$0" --worker "$gpu" \
        > "$log_root/worker_${gpu}.log" 2>&1 & worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || status=1
done
worker_pids=()
trap - INT TERM EXIT

if [ "${SKIP_SUMMARY:-0}" != 1 ]; then
    "$python_bin" summarize_results.py
fi
exit "$status"
