#!/usr/bin/env bash
set -euo pipefail

dataset_filter=${1:-all}
gpu=${2:-0}
pred_filter=${3:-all}
python_bin=${PYTHON:-/root/miniconda3/bin/python}
seed=${SEED:-2021}
lock_root="logs/TimeBase/seed_${seed}/.locks"
mkdir -p "$lock_root"

# dataset data_type data_path pred period basis individual orth_weight batch lr
# Values mirror hqh0728/TimeBase at commit 369b330, except for seed=2021.
job_table() {
    printf '%s\n' \
        'ETTh1 ETTh1 ETTh1.csv 96 24 6 0 0.16 64 1e-1' \
        'ETTh1 ETTh1 ETTh1.csv 192 24 6 0 0.16 256 4e-1' \
        'ETTh1 ETTh1 ETTh1.csv 336 24 6 0 0.08 256 4e-1' \
        'ETTh1 ETTh1 ETTh1.csv 720 24 6 0 0.12 64 5e-2' \
        'ETTh2 ETTh2 ETTh2.csv 96 24 6 0 0.20 512 2e-1' \
        'ETTh2 ETTh2 ETTh2.csv 192 24 4 0 0.08 512 6e-2' \
        'ETTh2 ETTh2 ETTh2.csv 336 24 6 0 0.12 64 4e-1' \
        'ETTh2 ETTh2 ETTh2.csv 720 24 6 0 0.12 64 4e-1' \
        'ETTm1 ETTm1 ETTm1.csv 96 4 18 0 0.04 512 2e-2' \
        'ETTm1 ETTm1 ETTm1.csv 192 4 20 0 0.04 256 2e-2' \
        'ETTm1 ETTm1 ETTm1.csv 336 4 20 0 0.08 256 2e-2' \
        'ETTm1 ETTm1 ETTm1.csv 720 6 20 0 0.12 128 1e-2' \
        'ETTm2 ETTm2 ETTm2.csv 96 4 20 0 0.04 64 1e-2' \
        'ETTm2 ETTm2 ETTm2.csv 192 4 20 0 0.04 64 1e-2' \
        'ETTm2 ETTm2 ETTm2.csv 336 4 20 0 0.04 64 1e-2' \
        'ETTm2 ETTm2 ETTm2.csv 720 6 20 0 0.04 64 1e-2' \
        'weather custom weather.csv 96 24 6 0 0.04 512 2e-2' \
        'weather custom weather.csv 192 24 6 0 0.04 512 2e-2' \
        'weather custom weather.csv 336 24 6 0 0.08 512 5e-2' \
        'weather custom weather.csv 720 24 6 0 0.04 512 5e-2' \
        'electricity custom electricity.csv 96 24 6 0 0.04 128 2e-2' \
        'electricity custom electricity.csv 192 24 6 0 0.04 128 2e-2' \
        'electricity custom electricity.csv 336 24 6 0 0.04 128 2e-2' \
        'electricity custom electricity.csv 720 24 6 0 0.08 128 8e-2' \
        'traffic custom traffic.csv 96 24 6 0 0.04 128 5e-2' \
        'traffic custom traffic.csv 192 24 8 0 0.04 128 3e-2' \
        'traffic custom traffic.csv 336 24 15 0 0.08 128 3e-2' \
        'traffic custom traffic.csv 720 24 8 0 0.04 128 3e-2'
}

enc_in_for() {
    case "$1" in
        ETTh1|ETTh2|ETTm1|ETTm2) echo 7 ;;
        weather) echo 21 ;;
        electricity) echo 321 ;;
        traffic) echo 862 ;;
    esac
}

while read -r dataset data_type data_path pred period basis individual orth_weight batch lr; do
    if [ "$dataset_filter" != all ] && [ "$dataset_filter" != "$dataset" ]; then
        continue
    fi
    if [ "$pred_filter" != all ] && [ "$pred_filter" != "$pred" ]; then
        continue
    fi
    setting="TimeBase_TimeBase_${dataset}_ftM_sl720_pl${pred}_test_seed${seed}"
    result_file="results/TimeBase/seed_${seed}/${dataset}/pred_${pred}/${setting}/metrics.npy"
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

    log_dir="logs/TimeBase/seed_${seed}/${dataset}"
    log_path="${log_dir}/pred_${pred}.log"
    mkdir -p "$log_dir"
    echo "START gpu=${gpu}: ${dataset}-${pred}"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run_longExp.py \
        --is_training 1 --model TimeBase --model_id TimeBase \
        --output_family TimeBase --run_tag timebase_original \
        --root_path ./dataset/ --data_path "$data_path" --data "$data_type" \
        --features M --seq_len 720 --pred_len "$pred" --period_len "$period" \
        --enc_in "$(enc_in_for "$dataset")" --basis_num "$basis" \
        --use_period_norm 1 --use_orthogonal 1 \
        --orthogonal_weight "$orth_weight" --individual "$individual" \
        --train_epochs 30 --patience 5 --batch_size "$batch" \
        --learning_rate "$lr" --optimizer adam --weight_decay 0 \
        --gradient_clip 0 --loss mse --lradj type3 \
        --scheduler_mode timebase --pct_start 0.3 --drop_last_train 0 \
        --num_workers 10 --test_last 0 --sanity_val_steps 0 \
        --eval_test_each_epoch 1 --deterministic 0 \
        --matmul_precision highest --seed "$seed" --gpu 0 \
        > "$log_path" 2>&1
    echo "DONE gpu=${gpu}: ${dataset}-${pred}"
    exec {lock_fd}>&-
done < <(job_table)

if [ "${NO_SUMMARY:-0}" != 1 ]; then
    "$python_bin" summarize_results.py
fi
