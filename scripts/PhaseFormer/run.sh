#!/usr/bin/env bash
set -euo pipefail

dataset_filter=${1:-all}
gpu=${2:-0}
pred_filter=${3:-all}
python_bin=${PYTHON:-/root/miniconda3/bin/python}
seed=${SEED:-2021}
lock_root="logs/PhaseFormer/seed_${seed}/.locks"
mkdir -p "$lock_root"

# Values mirror neumyor/PhaseFormer's run_*.py files. The only intentional
# protocol change is the shared POEM repository seed above.
job_table() {
    printf '%s\n' \
        'ETTh1 ETTh1 ETTh1.csv 96 256 3 4 16 32 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTh1 ETTh1 ETTh1.csv 192 256 3 4 16 32 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTh1 ETTh1 ETTh1.csv 336 256 3 4 16 32 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTh1 ETTh1 ETTh1.csv 720 256 3 32 128 256 16 2 0.0 0.00015 70 14 0.3 0' \
        'ETTh2 ETTh2 ETTh2.csv 96 256 1 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTh2 ETTh2 ETTh2.csv 192 256 1 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTh2 ETTh2 ETTh2.csv 336 256 1 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTh2 ETTh2 ETTh2.csv 720 256 1 4 8 8 4 1 0.1 0.001 30 8 1.0 1' \
        'ETTm1 ETTm1 ETTm1.csv 96 16 2 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTm1 ETTm1 ETTm1.csv 192 16 2 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTm1 ETTm1 ETTm1.csv 336 16 1 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTm1 ETTm1 ETTm1.csv 720 16 2 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTm2 ETTm2 ETTm2.csv 96 16 2 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTm2 ETTm2 ETTm2.csv 192 16 1 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTm2 ETTm2 ETTm2.csv 336 16 1 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'ETTm2 ETTm2 ETTm2.csv 720 16 1 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'weather custom weather.csv 96 16 3 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'weather custom weather.csv 192 16 2 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'weather custom weather.csv 336 16 2 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'weather custom weather.csv 720 16 2 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'electricity custom electricity.csv 96 16 2 8 32 64 8 1 0.1 0.002 30 8 1.0 1' \
        'electricity custom electricity.csv 192 16 1 128 16 32 4 4 0.1 0.001 30 8 1.0 1' \
        'electricity custom electricity.csv 336 16 2 8 32 64 8 1 0.1 0.001 30 8 1.0 1' \
        'electricity custom electricity.csv 720 16 1 128 16 32 4 4 0.1 0.001 30 8 1.0 1' \
        'traffic custom traffic.csv 96 16 2 32 64 128 1 8 0.1 0.001 30 8 1.0 1' \
        'traffic custom traffic.csv 192 16 1 128 16 32 4 4 0.1 0.001 30 8 1.0 1' \
        'traffic custom traffic.csv 336 16 1 128 16 32 4 4 0.1 0.001 30 8 1.0 1' \
        'traffic custom traffic.csv 720 16 1 128 16 32 4 4 0.1 0.001 30 8 1.0 1'
}

enc_in_for() {
    case "$1" in
        ETTh1|ETTh2|ETTm1|ETTm2) echo 7 ;;
        weather) echo 21 ;;
        electricity) echo 321 ;;
        traffic) echo 862 ;;
    esac
}

while read -r dataset data_type data_path pred batch layers latent encoder_hidden \
    predictor_hidden routers heads dropout lr epochs patience huber_delta epoch_test; do
    if [ "$dataset_filter" != all ] && [ "$dataset_filter" != "$dataset" ]; then
        continue
    fi
    if [ "$pred_filter" != all ] && [ "$pred_filter" != "$pred" ]; then
        continue
    fi
    setting="PhaseFormer_PhaseFormer_${dataset}_ftM_sl720_pl${pred}_test_seed${seed}"
    result_file="results/PhaseFormer/seed_${seed}/${dataset}/pred_${pred}/${setting}/metrics.npy"
    if [ -f "$result_file" ]; then
        echo "SKIP complete: ${dataset}-${pred}"
        continue
    fi

    lock_path="${lock_root}/${dataset}_${pred}.lock"
    exec {lock_fd}> "$lock_path"
    if ! flock -n "$lock_fd"; then
        echo "SKIP active on another worker: ${dataset}-${pred}"
        exec {lock_fd}>&-
        continue
    fi
    if [ -f "$result_file" ]; then
        echo "SKIP complete after lock: ${dataset}-${pred}"
        exec {lock_fd}>&-
        continue
    fi

    log_dir="logs/PhaseFormer/seed_${seed}/${dataset}"
    log_path="${log_dir}/pred_${pred}.log"
    mkdir -p "$log_dir"
    echo "START gpu=${gpu}: ${dataset}-${pred}"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run_longExp.py \
        --is_training 1 --model PhaseFormer --model_id PhaseFormer \
        --output_family PhaseFormer --run_tag phaseformer_original \
        --root_path ./dataset/ --data_path "$data_path" --data "$data_type" \
        --features M --seq_len 720 --pred_len "$pred" --period_len 24 \
        --enc_in "$(enc_in_for "$dataset")" \
        --latent_dim "$latent" --phase_encoder_hidden "$encoder_hidden" \
        --predictor_hidden "$predictor_hidden" --phase_layers "$layers" \
        --phase_attn_heads "$heads" --phase_attn_dropout "$dropout" \
        --phase_attn_use_relpos 1 --phase_num_routers "$routers" \
        --phase_use_pos_embed 1 --phase_pos_dropout 0.0 \
        --use_revin 1 --revin_affine 0 --revin_eps 1e-5 \
        --train_epochs "$epochs" --patience "$patience" --batch_size "$batch" \
        --learning_rate "$lr" --optimizer adam --weight_decay 0 \
        --gradient_clip 0 --loss huber --huber_delta "$huber_delta" \
        --lradj type3 --scheduler_mode phaseformer --drop_last_train 1 \
        --num_workers 6 --test_last 1 --sanity_val_steps 2 \
        --eval_test_each_epoch "$epoch_test" --deterministic 1 \
        --matmul_precision medium --seed "$seed" --gpu 0 \
        > "$log_path" 2>&1
    echo "DONE gpu=${gpu}: ${dataset}-${pred}"
    exec {lock_fd}>&-
done < <(job_table)

if [ "${NO_SUMMARY:-0}" != 1 ]; then
    "$python_bin" summarize_results.py
fi
