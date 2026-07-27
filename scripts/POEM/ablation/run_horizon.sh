#!/usr/bin/env bash
set -euo pipefail

pred_len=${1:?Prediction length is required}
seed=${SEED:-2021}
python_bin=${PYTHON:-/root/miniconda3/bin/python}
gpu_ids=${GPU_IDS:-0}
log_root="logs/ablation/seed_${seed}"
lock_root="$log_root/.locks"
. scripts/POEM/formal_config.sh

if [ "$seed" != 2021 ]; then
    echo "POEM ablations are configured only for seed 2021" >&2
    exit 1
fi
if [ "$pred_len" != 96 ] && [ "$pred_len" != 192 ] \
    && [ "$pred_len" != 336 ] && [ "$pred_len" != 720 ]; then
    echo "Supported prediction lengths are 96, 192, 336, and 720" >&2
    exit 1
fi

# model_id phase_mixer harmonic_mixer vanilla_mixer global_forecast geometry_type run_tag output_family
variant_table() {
    cat <<'EOF'
POEM 1 1 0 1 fixed poem POEM
NoPhaseInteraction 0 1 0 1 fixed nophaseinteraction ablation
NoHarmonicModulation 1 0 0 1 fixed noharmonicmodulation ablation
LinearPhaseBackbone 0 0 0 1 fixed linearphasebackbone ablation
VanillaMLPMixer 0 0 1 1 fixed vanillamlpmixer ablation
NoGlobalForecast 1 1 0 0 fixed noglobalforecast ablation
NoGeometry 1 1 0 1 none nogeometry ablation
LearnableGeometry 1 1 0 1 learnable learnablegeometry ablation
EOF
}

job_schedule() {
    local model_id _ dataset
    while read -r model_id _; do
        for dataset in ETTh1 ETTm1 weather traffic; do
            printf '%s %s\n' "$model_id" "$dataset"
        done
    done < <(variant_table)
}

is_complete() {
    local model_id=$1 dataset=$2 data_type=$3 data_path=$4 d_model=$5 period=$6
    local layers=$7 dropout=$8 affine=$9 epochs=${10} patience=${11} batch=${12}
    local lr=${13} lradj=${14} enc_in=${15} phase_mixer=${16} harmonic_mixer=${17}
    local vanilla_mixer=${18} global_forecast=${19} geometry_type=${20} run_tag=${21}
    local output_family=${22} result_dir setting

    if [ "$output_family" = POEM ]; then
        setting="POEM_POEM_${dataset}_ftM_sl720_pl${pred_len}_test_seed${seed}"
        result_dir="results/POEM/seed_${seed}/${dataset}/pred_${pred_len}/${setting}"
    else
        result_dir="results/ablation/seed_${seed}/${model_id}/${dataset}/pred_${pred_len}"
    fi

    [ -f "$result_dir/metrics.npy" ] && [ -f "$result_dir/settings.json" ] || return 1
    "$python_bin" - "$result_dir/settings.json" \
        "$model_id" "$data_type" "$data_path" "$d_model" "$period" "$layers" \
        "$dropout" "$affine" "$epochs" "$patience" "$batch" "$lr" "$lradj" "$enc_in" \
        "$phase_mixer" "$harmonic_mixer" "$vanilla_mixer" "$global_forecast" \
        "$geometry_type" "$run_tag" "$output_family" "$pred_len" "$seed" <<'PY'
import json
import math
import sys

settings = json.load(open(sys.argv[1]))
expected = {
    "model_id": sys.argv[2],
    "data": sys.argv[3],
    "data_path": sys.argv[4],
    "d_model": int(sys.argv[5]),
    "period_len": int(sys.argv[6]),
    "mixer_layers": int(sys.argv[7]),
    "mixer_dropout": float(sys.argv[8]),
    "affine": int(sys.argv[9]),
    "train_epochs": int(sys.argv[10]),
    "patience": int(sys.argv[11]),
    "batch_size": int(sys.argv[12]),
    "learning_rate": float(sys.argv[13]),
    "lradj": sys.argv[14],
    "enc_in": int(sys.argv[15]),
    "use_phase_interaction": int(sys.argv[16]),
    "use_harmonic_modulation": int(sys.argv[17]),
    "use_vanilla_mixer": int(sys.argv[18]),
    "use_global_forecast": int(sys.argv[19]),
    "geometry_type": sys.argv[20],
    "run_tag": sys.argv[21],
    "output_family": sys.argv[22],
    "pred_len": int(sys.argv[23]),
    "seed": int(sys.argv[24]),
}
actual_geometry = settings.get("geometry_type", "fixed")
for key, expected_value in expected.items():
    if key == "geometry_type":
        actual_value = actual_geometry
    elif key == "use_global_forecast":
        actual_value = settings.get(key, 1)
    else:
        actual_value = settings.get(key)
    if isinstance(expected_value, float):
        if actual_value is None or not math.isclose(float(actual_value), expected_value):
            raise SystemExit(1)
    elif actual_value != expected_value:
        raise SystemExit(1)
PY
}

run_worker() {
    local gpu=$1
    local wanted_model wanted_dataset
    local model_id phase_mixer harmonic_mixer vanilla_mixer global_forecast
    local geometry_type run_tag output_family slot dataset data_type data_path formal_pred
    local d_model period layers dropout affine epochs patience batch lr lradj enc_in
    local lock_fd log_dir log_path

    while read -r wanted_model wanted_dataset; do
        read -r model_id phase_mixer harmonic_mixer vanilla_mixer global_forecast \
            geometry_type run_tag output_family \
            < <(variant_table | awk -v m="$wanted_model" '$1 == m { print; exit }')
        read -r slot dataset data_type data_path formal_pred d_model period layers \
            dropout affine epochs patience batch lr lradj \
            < <(formal_job_for "$wanted_dataset" "$pred_len")
        enc_in=$(enc_in_for "$dataset")

        if is_complete "$model_id" "$dataset" "$data_type" "$data_path" \
            "$d_model" "$period" "$layers" "$dropout" "$affine" "$epochs" \
            "$patience" "$batch" "$lr" "$lradj" "$enc_in" "$phase_mixer" \
            "$harmonic_mixer" "$vanilla_mixer" "$global_forecast" "$geometry_type" \
            "$run_tag" "$output_family"; then
            echo "SKIP complete: ${model_id}-${dataset}-${pred_len}"
            continue
        fi

        exec {lock_fd}> "$lock_root/${model_id}_${dataset}_${pred_len}.lock"
        if ! flock -n "$lock_fd"; then
            echo "SKIP active on another worker: ${model_id}-${dataset}-${pred_len}"
            exec {lock_fd}>&-
            continue
        fi

        log_dir="$log_root/$model_id/$dataset"
        log_path="$log_dir/pred_${pred_len}.log"
        mkdir -p "$log_dir"
        echo "START gpu=${gpu}: ${model_id}-${dataset}-${pred_len}"
        "$python_bin" -u run_longExp.py \
            --is_training 1 --root_path ./dataset/ --data_path "$data_path" \
            --model_id "$model_id" --data "$data_type" --features M \
            --model POEM --output_family "$output_family" \
            --seq_len 720 --pred_len "$pred_len" --period_len "$period" \
            --enc_in "$enc_in" --d_model "$d_model" \
            --mixer_layers "$layers" --mixer_dropout "$dropout" \
            --use_phase_interaction "$phase_mixer" \
            --use_harmonic_modulation "$harmonic_mixer" \
            --use_vanilla_mixer "$vanilla_mixer" \
            --use_global_forecast "$global_forecast" --geometry_type "$geometry_type" \
            --revin 1 --affine "$affine" --train_epochs "$epochs" --patience "$patience" \
            --batch_size "$batch" --learning_rate "$lr" --weight_decay 1e-4 \
            --gradient_clip 1.0 --loss mae --lradj "$lradj" \
            --run_tag "$run_tag" --seed "$seed" --gpu "$gpu" \
            > "$log_path" 2>&1
        echo "DONE gpu=${gpu}: ${model_id}-${dataset}-${pred_len}"
        exec {lock_fd}>&-
    done < <(job_schedule | awk -v worker="$WORKER_INDEX" -v total="$WORKER_COUNT" \
        'NR % total == worker')
}

mkdir -p "$lock_root"

if [ "${2:-}" = "--worker" ]; then
    gpu=${3:?GPU index is required}
    WORKER_INDEX=${4:?Worker index is required}
    WORKER_COUNT=${5:?Worker count is required}
    run_worker "$gpu"
    exit 0
fi

exec {run_lock_fd}> "$log_root/.run_${pred_len}.lock"
if ! flock -n "$run_lock_fd"; then
    echo "Another POEM ablation run for horizon ${pred_len} is active" >&2
    exit 1
fi

read -r -a gpu_array <<< "$gpu_ids"
worker_count=${#gpu_array[@]}
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

for worker_index in "${!gpu_array[@]}"; do
    gpu=${gpu_array[$worker_index]}
    setsid bash "$0" "$pred_len" --worker "$gpu" "$worker_index" "$worker_count" \
        > "$log_root/worker_${gpu}_pred_${pred_len}.log" 2>&1 & worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || status=1
done
worker_pids=()
trap - INT TERM EXIT
exit "$status"
