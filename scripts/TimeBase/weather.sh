#!/usr/bin/env bash
set -euo pipefail

bash scripts/TimeBase/run.sh weather "${1:-0}"
