#!/usr/bin/env bash
set -euo pipefail

DATASET_FILTER=ETTm2 exec bash scripts/POEM/run_all.sh "$@"
