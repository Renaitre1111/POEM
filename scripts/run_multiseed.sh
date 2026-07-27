#!/usr/bin/env bash
set -euo pipefail

python_bin=${PYTHON:-/root/miniconda3/bin/python}
mkdir -p logs
exec {pipeline_lock_fd}> logs/.multiseed_pipeline.lock
if ! flock -n "$pipeline_lock_fd"; then
    echo "Another multi-seed pipeline is already active" >&2
    exit 1
fi

for seed in 2022 2023; do
    echo "START model=POEM seed=${seed}"
    SEED="$seed" RUN_ID="multiseed_POEM" \
        bash scripts/POEM/run_all.sh
    echo "DONE model=POEM seed=${seed}"
done

"$python_bin" analysis/multiseed.py
