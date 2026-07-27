#!/usr/bin/env bash
set -euo pipefail

bash scripts/PhaseFormer/run.sh traffic "${1:-0}"
