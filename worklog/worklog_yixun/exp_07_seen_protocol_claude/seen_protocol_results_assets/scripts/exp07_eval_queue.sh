#!/bin/bash
# exp_07 seen evaluations for one trained arm: K=8 and K=1 x seeds 42..46 (10 runs), binding the arm's final attempt.
# usage: exp07_eval_queue.sh <gpu> <role: seen_simple|seen_cyl|seen_aug>
set -uo pipefail
GPU=${1:?gpu}; ROLE=${2:?role}
cd /home/yixunhu/codespace/xRIR_code
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python; REC=worklog/worklog_yixun/exp_07_seen_protocol_claude
export PYTHONPATH="$PWD" XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONHASHSEED=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
case $ROLE in seen_simple|seen_aug) BB=simple;; seen_cyl) BB=cylindrical;; *) echo "bad role"; exit 2;; esac
ATT=$(readlink -f ckpt/exp07/$ROLE/final); CK=$ATT/epoch_012.pth
[ -f "$CK" ] || { echo "no certified checkpoint for $ROLE"; exit 3; }
if git status --short --untracked-files=all | grep -v '^.. worklog/' | grep -q .; then echo "ABORT dirty outside worklog"; exit 4; fi
SHA=$(git rev-parse HEAD)
QLOG=$REC/seen_protocol_$(date +%Y%m%dT%H%M%S)_eval_queue_${ROLE}_gpu${GPU}.log; mkdir -p ckpt/exp07/eval
echo "$(date -Is) QUEUE START role=$ROLE gpu=$GPU reviewed=$SHA attempt=$ATT" >> "$QLOG"
for K in 8 1; do for seed in 42 43 44 45 46; do
  label=${ROLE}_k${K}_seed${seed}_k0; out=ckpt/exp07/eval/$label
  [ -e "$out" ] && { echo "$(date -Is) SKIP existing $out" >> "$QLOG"; continue; }
  man=ckpt/exp07/reference_manifest_seen_k${K}_seed${seed}.json
  digest=$("$PY" -c 'import sys; from tools.reference_manifest import load_manifest, manifest_hash; print(manifest_hash(load_manifest(sys.argv[1])))' "$man")
  echo "$(date -Is) START $label" >> "$QLOG"
  "$PY" tools/exp07_eval_launch.py --split seen --backbone $BB --checkpoint "$CK" --num-shot $K \
    --manifest "$man" --manifest-hash "$digest" --gl-seed $seed --batch-size 16 --conditions P \
    --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --decomposition-batches 0 --max-samples 0 \
    --num-workers 12 --threads 2 --out-dir "$out" --run-label "$label" --reviewed-commit "$SHA" \
    --data-root "$XRIR_DATA_PATH" --log-dir "$REC" --gpu "$GPU" \
    --bind-input "train_manifest=$ATT/train_manifest.json" --bind-input "train_completion=$ATT/completion.json" \
    --bind-input "train_args=$ATT/args.json" > "$REC/seen_protocol_$(date +%Y%m%dT%H%M%S)_evallauncher_${label}.log" 2>&1; rc=$?
  echo "$(date -Is) END $label exit=$rc" >> "$QLOG"; [ $rc -eq 0 ] || echo "$(date -Is) FAILED $label (see its evallauncher log)" >> "$QLOG"
done; done
echo "EVAL_QUEUE_DONE role=$ROLE gpu=$GPU $(date -Is)" >> "$QLOG"; echo "$QLOG"
