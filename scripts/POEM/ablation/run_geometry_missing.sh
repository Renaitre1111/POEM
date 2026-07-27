#!/usr/bin/env bash
set -euo pipefail

# Run missing NoGeometry + LearnableGeometry ablations on
# ETTh2, ETTm2, electricity  (4 pred_lens × 3 datasets × 2 variants = 24 jobs)

seed=2021
python_bin=${PYTHON:-/root/miniconda3/bin/python}
gpu=${GPU_ID:-0}
output_family=ablation
log_root="logs/ablation/seed_${seed}"
lock_root="$log_root/.locks"
pred_lens=(96 192 336 720)
datasets=(ETTh2 ETTm2 electricity)

. scripts/POEM/formal_config.sh

# geometry_type → model_id prefix → run_tag
declare -A model_id_map=([none]=NoGeometry [learnable]=LearnableGeometry)
declare -A run_tag_map=([none]=nogeometry [learnable]=learnablegeometry)

mkdir -p "$log_root" "$lock_root"

for dataset in "${datasets[@]}"; do
    enc_in=$(enc_in_for "$dataset")
    for pred_len in "${pred_lens[@]}"; do
        read -r _slot _ds data_type data_path _formal_pred d_model _period \
            layers dropout affine epochs patience batch lr lradj \
            < <(formal_job_for "$dataset" "$pred_len")

        for geometry_type in none learnable; do
            model_id="${model_id_map[$geometry_type]}"
            run_tag="${run_tag_map[$geometry_type]}"

            setting="${model_id}_POEM_${dataset}_ftM_sl720_pl${pred_len}_test_seed${seed}"
            result_dir="results/ablation/seed_${seed}/${model_id}/${dataset}/pred_${pred_len}"

            if [ -f "$result_dir/metrics.npy" ] && [ -f "$result_dir/settings.json" ]; then
                echo "SKIP complete: ${model_id}-${dataset}-${pred_len}"
                continue
            fi

            lock_path="$lock_root/${model_id}_${dataset}_${pred_len}.lock"
            exec {lock_fd}> "$lock_path"
            if ! flock -n "$lock_fd"; then
                echo "SKIP locked: ${model_id}-${dataset}-${pred_len}"
                exec {lock_fd}>&-
                continue
            fi

            log_dir="$log_root/${model_id}/${dataset}"
            log_path="$log_dir/pred_${pred_len}.log"
            mkdir -p "$log_dir"
            echo "START: ${model_id}-${dataset}-${pred_len}"
            "$python_bin" -u run_longExp.py \
                --is_training 1 --root_path ./dataset/ --data_path "$data_path" \
                --model_id "$model_id" --data "$data_type" --features M \
                --model POEM --output_family "$output_family" \
                --seq_len 720 --pred_len "$pred_len" --period_len "$_period" \
                --enc_in "$enc_in" --d_model "$d_model" \
                --mixer_layers "$layers" --mixer_dropout "$dropout" \
                --use_phase_interaction 1 --use_harmonic_modulation 1 \
                --use_vanilla_mixer 0 --use_global_forecast 1 \
                --geometry_type "$geometry_type" \
                --revin 1 --affine "$affine" --train_epochs "$epochs" \
                --patience "$patience" --batch_size "$batch" \
                --learning_rate "$lr" --weight_decay 1e-4 \
                --gradient_clip 1.0 --loss mae --lradj "$lradj" \
                --run_tag "$run_tag" --seed "$seed" --gpu "$gpu" \
                > "$log_path" 2>&1
            echo "DONE: ${model_id}-${dataset}-${pred_len}"
            exec {lock_fd}>&-
        done
    done
done

echo "All missing geometry ablations complete."
"$python_bin" analysis/ablation.py
