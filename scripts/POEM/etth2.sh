#!/usr/bin/env bash
set -euo pipefail

seed=${SEED:-2021}
gpu=${GPU:-0}
python_bin=${PYTHON:-python3}

data=ETTh2
data_path=ETTh2.csv
period_len=24
enc_in=7

for pred_len in 96 192 336 720; do
    case $pred_len in
        96)  d_model=12; layers=1; dropout=0.1; epochs=30; patience=8; batch=256; lr=0.001;    lradj=constant ;;
        192) d_model=16; layers=1; dropout=0.1; epochs=30; patience=8; batch=64;  lr=0.0008;   lradj=type3 ;;
        336) d_model=16; layers=1; dropout=0.1; epochs=30; patience=8; batch=64;  lr=0.000704; lradj=constant ;;
        720) d_model=24; layers=1; dropout=0.1; epochs=30; patience=8; batch=256; lr=0.001;    lradj=type3 ;;
    esac

    echo "=== POEM ${data} pred_len=${pred_len} ==="
    $python_bin -u run_longExp.py \
        --is_training 1 --root_path ./dataset/ --data_path "$data_path" \
        --model_id POEM --data "$data" --features M \
        --model POEM --output_family POEM \
        --seq_len 720 --pred_len "$pred_len" --period_len "$period_len" \
        --enc_in "$enc_in" --d_model "$d_model" \
        --mixer_layers "$layers" --mixer_dropout "$dropout" \
        --revin 1 --affine 0 \
        --train_epochs "$epochs" --patience "$patience" --batch_size "$batch" \
        --learning_rate "$lr" --weight_decay 1e-4 --gradient_clip 1.0 \
        --loss mae --lradj "$lradj" --run_tag poem --seed "$seed" --gpu "$gpu"
done
echo "=== POEM ${data} done ==="
