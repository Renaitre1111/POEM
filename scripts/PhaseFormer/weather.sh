#!/usr/bin/env bash
set -euo pipefail

bash scripts/PhaseFormer/run.sh weather "${1:-0}"
