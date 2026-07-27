#!/usr/bin/env bash
set -euo pipefail

bash scripts/PhaseFormer/run.sh electricity "${1:-0}"
