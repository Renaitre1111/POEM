#!/usr/bin/env bash
set -euo pipefail

export GPU_IDS=${GPU_IDS:-"0 1"}
export DATASET_FILTER=all
export PRED_FILTER=all
export SKIP_SUMMARY=0

echo "[1/3] Repairing formal POEM results"
bash scripts/POEM/run_all.sh

echo "[2/3] Completing POEM ablations"
bash scripts/POEM/ablation/run_all.sh

echo "[3/3] Completing POEM sensitivity experiments"
bash scripts/POEM/sensitivity/run_all.sh
