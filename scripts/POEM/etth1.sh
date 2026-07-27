#!/usr/bin/env bash
set -euo pipefail

DATASET_FILTER=ETTh1 exec bash scripts/POEM/run_all.sh "$@"
