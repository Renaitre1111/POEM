#!/usr/bin/env bash
set -euo pipefail

DATASET_FILTER=electricity exec bash scripts/POEM/run_all.sh "$@"
