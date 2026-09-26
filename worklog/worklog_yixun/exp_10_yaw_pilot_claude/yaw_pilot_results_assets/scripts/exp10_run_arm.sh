#!/bin/bash
# exp_10 yaw_pilot — one arm (probe or full) with a teed timestamped log. DRAFT until the Coder's CLI is final.
# usage: exp10_run_arm.sh <arm> <backbone> <checkpoint> <manifest> <manifest_hash> <num_shot> <batches: probe|all> <device> [gpu_index]
set -euo pipefail
ARM=$1; BB=$2; CK=$3; MAN=$4; HASH=$5; K=$6; BATCHES=$7; DEV=$8; GPU=${9:-1}
cd /home/yixunhu/codespace/xRIR_code
source ~/miniconda3/etc/profile.d/conda.sh; conda activate xRIR
export PYTHONPATH=$PYTHONPATH:$(pwd); export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
if [ "$DEV" = "cuda" ]; then export CUDA_VISIBLE_DEVICES=$GPU; else export CUDA_VISIBLE_DEVICES=""; fi
E=worklog/worklog_yixun/exp_10_yaw_pilot_claude
TAG=${ARM}_${BATCHES}
LOG=$E/yaw_pilot_$(date +%Y-%m-%d_%H:%M:%S)_${TAG}.log
OUT=ckpt/exp10/${ARM}_${BATCHES}
[ -e "$OUT" ] && { echo "refusing: $OUT exists" | tee "$LOG"; exit 2; }
[ -n "$(git status --porcelain | grep -v '^ M worklog/\|^?? worklog/')" ] && { echo "refusing: tree dirty outside worklog/" | tee "$LOG"; exit 3; }
echo "HEAD $(git rev-parse HEAD)  arm=$ARM backbone=$BB ckpt=$CK manifest=$MAN K=$K batches=$BATCHES device=$DEV gpu=$GPU  started $(date -Is)" | tee "$LOG"
python tools/exp10_yaw_pilot.py --arm "$ARM" --backbone "$BB" --checkpoint "$CK" --manifest "$MAN" --manifest-hash "$HASH" --num-shot "$K" \
  --ks 0,64,128,256,384 --batches "$BATCHES" --controls --device "$DEV" --threads 16 --out-dir "$OUT" 2>&1 | tee -a "$LOG"
echo "EXIT ${PIPESTATUS[0]}  ended $(date -Is)" | tee -a "$LOG"
