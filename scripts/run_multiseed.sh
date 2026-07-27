#!/usr/bin/env bash
set -euo pipefail

python_bin=${PYTHON:-/root/miniconda3/bin/python}
mkdir -p logs
exec {pipeline_lock_fd}> logs/.multiseed_pipeline.lock
if ! flock -n "$pipeline_lock_fd"; then
    echo "Another multi-seed pipeline is already active" >&2
    exit 1
fi

if [ "${RETRAIN_PHASEFORMER:-0}" = 1 ]; then
    echo "START model=PhaseFormer seed=2021 protocol rerun"
    SEED=2021 RUN_ID="phaseformer_protocol" \
        bash scripts/PhaseFormer/run_all.sh
    echo "DONE model=PhaseFormer seed=2021 protocol rerun"
fi

for seed in 2022 2023; do
    for model in POEM PhaseFormer TimeBase; do
        echo "START model=${model} seed=${seed}"
        SEED="$seed" RUN_ID="multiseed_${model}" \
            bash "scripts/${model}/run_all.sh"
        echo "DONE model=${model} seed=${seed}"
    done
done

"$python_bin" analysis/multiseed.py
