#!/usr/bin/env bash
set -euo pipefail

DATASET_FILTER=ETTh2 exec bash scripts/POEM/run_all.sh "$@"
