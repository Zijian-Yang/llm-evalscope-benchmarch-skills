#!/usr/bin/env bash
set -euo pipefail
python3 "$(dirname "$0")/model_benchmark.py" run --scenario stability "$@"
