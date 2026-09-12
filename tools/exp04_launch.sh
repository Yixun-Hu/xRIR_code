#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
exec /home/yixunhu/miniconda3/envs/xRIR/bin/python -m tools.exp04_launcher "$@"
