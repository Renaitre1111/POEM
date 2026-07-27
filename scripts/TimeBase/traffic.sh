#!/usr/bin/env bash
set -euo pipefail

bash scripts/TimeBase/run.sh traffic "${1:-0}"
