#!/usr/bin/env bash
# Unattended full-run template (replace each <...> with its recorded value):
# nohup setsid tools/exp04_launch.sh full --gpu <gpu> --reviewed-commit <sha> \
#   --probe-json <receipt> --log-dir <logs> > <launcher.log> 2>&1 &
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
exec /home/yixunhu/miniconda3/envs/xRIR/bin/python -m tools.exp04_launcher "$@"
