#!/bin/bash
# exp_10 yaw_pilot — sequential queue of the remaining arms on one GPU (after the cylindrical-dinov3 session's ping).
# usage: exp10_gpu_queue.sh <gpu_index>
set -uo pipefail
GPU=${1:-1}
cd /home/yixunhu/codespace/xRIR_code
E=worklog/worklog_yixun/exp_10_yaw_pilot_claude
S=$E/yaw_pilot_results_assets/scripts/exp10_arm_chain.sh
QLOG=$E/yaw_pilot_$(date +%Y-%m-%d_%H:%M:%S)_gpu${GPU}_queue.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
say "queue start on GPU $GPU, free memory ${FREE} MiB"
[ "$FREE" -lt 8000 ] && { say "REFUSED: < 8 GB free on GPU $GPU"; exit 9; }
M8=ckpt/yaw_rotation/reference_manifest.json; H8=47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d
M1=ckpt/yaw_rotation/reference_manifest_k1.json; H1=f6d71f86d5d313f2a982fe6f8cb801116a20ea1c37b0a65d290c088934fd73e3  # v1.1: hard-coded (v1 computed it with the system python3 outside the conda env → empty → refused)
say "K=1 manifest hash $H1"
run() { say "=== arm $1"; "$S" "$@" 2>&1 | tail -3 | tee -a "$QLOG"; local rc=${PIPESTATUS[0]}; say "=== arm $1 exit $rc"; return $rc; }
run released_k1 simple checkpoints/xRIR_unseen.pth "$M1" "$H1" 1 cuda "$GPU" none
run control_k8 simple ckpt/xRIR_simple_8_shot/epoch_12.pth "$M8" "$H8" 8 cuda "$GPU" ckpt/yaw_rotation/sweep_control/per_sample_yaw.json
run cyl_k8 cylindrical ckpt/xRIR_cyl_8_shot/epoch_12.pth "$M8" "$H8" 8 cuda "$GPU" ckpt/yaw_rotation/sweep_cyl/per_sample_yaw.json
say "QUEUE DONE"
