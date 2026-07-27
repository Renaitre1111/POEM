#!/usr/bin/env bash
set -euo pipefail

bash scripts/PhaseFormer/run.sh ETTh2 "${1:-0}"
