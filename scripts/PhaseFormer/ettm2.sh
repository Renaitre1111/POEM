#!/usr/bin/env bash
set -euo pipefail

bash scripts/PhaseFormer/run.sh ETTm2 "${1:-0}"
