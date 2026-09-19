#!/bin/bash
# exp_05 fail-closed producers: five registered profiles over the certified evaluation runs.
set -uo pipefail
cd /home/yixunhu/codespace/xRIR_code
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED=0 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python; OUT=ckpt/exp05/results; mkdir -p $OUT
LOG=worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_$(date +%Y%m%dT%H%M%S)_producers.log
ARMS="S_simple S_cylindrical M_simple M_cylindrical L_simple L_cylindrical"
run() { local prof=$1; shift; echo "$(date -Is) START $prof ($# runs)" >> "$LOG"; "$PY" tools/param_curve.py --profile $prof --runs "$@" --json $OUT/$prof.json --summary $OUT/$prof.txt >> "$LOG" 2>&1; echo "$(date -Is) END $prof exit=$?" >> "$LOG"; }
k8=(); k1=(); yaw=(); for a in $ARMS; do for s in 42 43 44 45 46; do k8+=(ckpt/exp05/eval/${a}_k8_seed${s}_k0); k1+=(ckpt/exp05/eval/${a}_k1_seed${s}_k0); done; yaw+=(ckpt/exp05/eval/${a}_k8_seed42_yaw); done
run CURVE_K8 "${k8[@]}"; run CURVE_K1 "${k1[@]}"; run TARGETS_K8 "${k8[@]}"; run TARGETS_K1 "${k1[@]}"; run YAW_K8_SEED42 "${yaw[@]}"
echo "PRODUCERS DONE $(date -Is)" >> "$LOG"; echo "$LOG"
