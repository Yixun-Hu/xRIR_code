#!/usr/bin/env bash
# Certified invocation: nohup setsid (preserves inherited ignored signals).
# Unattended full-run template (replace each <...> with its recorded value):
# nohup setsid tools/exp07_launch.sh full --backbone <b> [--yaw-aug 1] --gpu <gpu> \
#   --reviewed-commit <sha> --probe-json <receipt> --log-dir <logs> > <launcher.log> 2>&1 &
# Fit probe: tools/exp07_launch.sh probe --backbone <b> [--yaw-aug 1] --gpu <gpu> \
#   --reviewed-commit <sha>
# Recovery: tools/exp07_launch.sh finalize <attempt> --launcher-log <launcher.log>
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
exec /home/yixunhu/miniconda3/envs/xRIR/bin/python -m tools.exp07_launcher "$@"
