#!/usr/bin/env bash
# Certified invocation: nohup setsid (preserves inherited ignored signals).
# Unattended full-run template (replace each <...> with its recorded value):
# nohup setsid tools/exp04_launch.sh full --gpu <gpu> --reviewed-commit <sha> \
#   --probe-json <receipt> --log-dir <logs> > <launcher.log> 2>&1 &
# Recovery: tools/exp04_launch.sh finalize <attempt> --launcher-log <launcher.log>
# exp_07 seen arms (every argument is passed through unchanged):
#   tools/exp04_launch.sh probe --protocol seen --backbone <b> [--yaw-aug 1] --gpu <gpu> \
#     --reviewed-commit <sha>
#   tools/exp04_launch.sh full  --protocol seen --backbone <b> [--yaw-aug 1] --gpu <gpu> \
#     --reviewed-commit <sha> --probe-json <receipt>
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
exec /home/yixunhu/miniconda3/envs/xRIR/bin/python -m tools.exp04_launcher "$@"
