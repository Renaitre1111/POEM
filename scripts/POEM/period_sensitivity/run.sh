#!/usr/bin/env bash
set -euo pipefail

seed=${SEED:-2021}
pred_len=96
python_bin=${PYTHON:-/root/miniconda3/bin/python}
gpu=${GPU_ID:-0}
output_family=period_sensitivity
log_root="logs/${output_family}/seed_${seed}"
. scripts/POEM/formal_config.sh

if [ "$seed" != 2021 ]; then
    echo "POEM period sensitivity is configured only for seed 2021" >&2
    exit 1
fi

# The default periods (ETTh1=24, ETTm1=96, weather=24) reuse formal POEM runs.
job_table() {
    cat <<'EOF'
ETTh1 12
ETTh1 18
ETTh1 36
ETTh1 42
ETTm1 48
ETTm1 72
ETTm1 144
ETTm1 168
weather 12
weather 18
weather 36
weather 42
EOF
}

is_complete() {
    local dataset=$1 period=$2 data_type=$3 data_path=$4 d_model=$5 layers=$6
    local dropout=$7 affine=$8 epochs=$9 patience=${10} batch=${11} lr=${12}
    local lradj=${13} enc_in=${14} model_id="PeriodLen${period}"
    local run_tag="period_sensitivity_${period}" setting result_dir
    setting="${model_id}_POEM_${dataset}_ftM_sl720_pl${pred_len}_test_seed${seed}"
    result_dir="results/${output_family}/seed_${seed}/${dataset}/pred_${pred_len}/${setting}"

    [ -f "$result_dir/metrics.npy" ] && [ -f "$result_dir/settings.json" ] || return 1
    "$python_bin" - "$result_dir/settings.json" "$model_id" "$run_tag" \
        "$data_type" "$data_path" "$d_model" "$period" "$layers" "$dropout" \
        "$affine" "$epochs" "$patience" "$batch" "$lr" "$lradj" "$enc_in" \
        "$pred_len" "$seed" <<'PY'
import json
import math
import sys

settings = json.load(open(sys.argv[1]))
expected = {
    "model_id": sys.argv[2],
    "model": "POEM",
    "output_family": "period_sensitivity",
    "run_tag": sys.argv[3],
    "data": sys.argv[4],
    "data_path": sys.argv[5],
    "d_model": int(sys.argv[6]),
    "period_len": int(sys.argv[7]),
    "mixer_layers": int(sys.argv[8]),
    "mixer_dropout": float(sys.argv[9]),
    "affine": int(sys.argv[10]),
    "train_epochs": int(sys.argv[11]),
    "patience": int(sys.argv[12]),
    "batch_size": int(sys.argv[13]),
    "learning_rate": float(sys.argv[14]),
    "lradj": sys.argv[15],
    "enc_in": int(sys.argv[16]),
    "pred_len": int(sys.argv[17]),
    "seed": int(sys.argv[18]),
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

mkdir -p "$log_root"
exec {run_lock_fd}> "$log_root/.run.lock"
if ! flock -n "$run_lock_fd"; then
    echo "Another POEM period-sensitivity run is already active" >&2
    exit 1
fi

while read -r dataset period; do
    read -r _slot _dataset data_type data_path _formal_pred d_model _formal_period \
        layers dropout affine epochs patience batch lr lradj \
        < <(formal_job_for "$dataset" "$pred_len")
    enc_in=$(enc_in_for "$dataset")
    model_id="PeriodLen${period}"
    run_tag="period_sensitivity_${period}"

    if is_complete "$dataset" "$period" "$data_type" "$data_path" "$d_model" \
        "$layers" "$dropout" "$affine" "$epochs" "$patience" "$batch" \
        "$lr" "$lradj" "$enc_in"; then
        echo "SKIP complete: ${dataset}-period${period}"
        continue
    fi

    log_dir="$log_root/$dataset"
    log_path="$log_dir/period_${period}.log"
    mkdir -p "$log_dir"
    echo "START gpu=${gpu}: ${dataset}-period${period}"
    "$python_bin" -u run_longExp.py \
        --is_training 1 --root_path ./dataset/ --data_path "$data_path" \
        --model_id "$model_id" --data "$data_type" --features M \
        --model POEM --output_family "$output_family" \
        --seq_len 720 --pred_len "$pred_len" --period_len "$period" \
        --enc_in "$enc_in" --d_model "$d_model" \
        --mixer_layers "$layers" --mixer_dropout "$dropout" \
        --phase_rank 4 --harmonics 2 --revin 1 --affine "$affine" \
        --train_epochs "$epochs" --patience "$patience" --batch_size "$batch" \
        --learning_rate "$lr" --weight_decay 1e-4 --gradient_clip 1.0 \
        --loss mae --lradj "$lradj" --run_tag "$run_tag" \
        --seed "$seed" --gpu "$gpu" \
        > "$log_path" 2>&1
    echo "DONE gpu=${gpu}: ${dataset}-period${period}"
done < <(job_table)

"$python_bin" summarize_period_sensitivity.py
