#!/usr/bin/env bash
set -euo pipefail

DATASET_FILTER=traffic exec bash scripts/POEM/run_all.sh "$@"
