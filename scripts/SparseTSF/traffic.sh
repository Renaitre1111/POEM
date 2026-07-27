#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-0}
python_bin=${PYTHON:-/root/miniconda3/bin/python}
seed=${SEED:-2021}
pred_len=96
setting="SparseTSF_SparseTSF_traffic_ftM_sl720_pl${pred_len}_test_seed${seed}"
result_file="results/SparseTSF/seed_${seed}/traffic/pred_${pred_len}/${setting}/metrics.npy"
log_dir="logs/SparseTSF/seed_${seed}/traffic"
log_path="${log_dir}/pred_${pred_len}.log"

if [ -f "$result_file" ]; then
    echo "SKIP complete: traffic-${pred_len}"
    exit 0
fi

mkdir -p "$log_dir"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u run_longExp.py \
    --is_training 1 --model SparseTSF --model_id SparseTSF \
    --output_family SparseTSF --run_tag sparsetsf_mlp_official \
    --root_path ./dataset/ --data_path traffic.csv --data custom \
    --features M --seq_len 720 --pred_len "$pred_len" --enc_in 862 \
    --period_len 24 --model_type mlp --d_model 512 \
    --train_epochs 30 --patience 5 --batch_size 128 \
    --learning_rate 0.01 --optimizer adam --weight_decay 0 \
    --gradient_clip 0 --loss mse --lradj type3 \
    --scheduler_mode sparsetsf --pct_start 0.3 --drop_last_train 1 \
    --num_workers 10 --test_last 0 --sanity_val_steps 0 \
    --eval_test_each_epoch 1 --deterministic 0 \
    --matmul_precision highest --seed "$seed" --gpu 0 \
    > "$log_path" 2>&1

echo "DONE traffic-${pred_len}: ${log_path}"
