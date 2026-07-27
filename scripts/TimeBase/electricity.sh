#!/usr/bin/env bash
set -euo pipefail

bash scripts/TimeBase/run.sh electricity "${1:-0}"
