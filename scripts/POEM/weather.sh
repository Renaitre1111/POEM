#!/usr/bin/env bash
set -euo pipefail

DATASET_FILTER=weather exec bash scripts/POEM/run_all.sh "$@"
