#!/usr/bin/env bash
set -euo pipefail

for pred_len in 96 192 336 720; do
    bash scripts/POEM/ablation/run_horizon.sh "$pred_len"
done

"${PYTHON:-/root/miniconda3/bin/python}" analysis/ablation.py
