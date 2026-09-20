#!/bin/bash
# exp_07 pre-training GPU phase: parity receipt -> seen audit -> 5 released K=8 evals -> calibration.
# Usage: exp07_pretrain_phase.sh <gpu>    (reviewed commit = HEAD of the main tree at launch; tree must be clean outside worklog/)
set -uo pipefail
GPU=${1:?gpu}
cd /home/yixunhu/codespace/xRIR_code
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
REC=worklog/worklog_yixun/exp_07_seen_protocol_claude
OUT=ckpt/exp07/results
export PYTHONPATH="$PWD" XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export PYTHONHASHSEED=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
MERGE=$(git rev-parse HEAD)
STAMP=$(date +%Y%m%dT%H%M%S)
LOG=$REC/seen_protocol_${STAMP}_pretrain_phase_gpu${GPU}.log
mkdir -p "$OUT" ckpt/exp07/eval
{
echo "PHASE START $(date -Iseconds) gpu=$GPU reviewed=$MERGE"
git status --short | grep -v '^.. worklog/' | grep -v '^?? ' && { echo "PHASE ABORT dirty outside worklog"; exit 2; }
PARITY="$REC/gpu_parity_${MERGE}.log"
echo "STEP parity $(date -Iseconds)"
"$PY" tools/exp07_parity.py --log "$PARITY" --reviewed-commit "$MERGE" --gpu "$GPU"; rc=$?
echo "STEP parity exit=$rc receipt=${PARITY%.log}.receipt.json"
[ $rc -eq 0 ] || { echo "PHASE ABORT parity failed"; exit 3; }
echo "STEP audit $(date -Iseconds)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/exp07_audit.py --protocol seen --n-batches 3 --batch-size 32 --seed 0 --num-workers 12 --out ckpt/exp07/alignment_audit_seen.json; rc=$?
echo "STEP audit exit=$rc"
[ $rc -eq 0 ] || { echo "PHASE ABORT audit failed"; exit 4; }
for seed in 42 43 44 45 46; do
  manifest="ckpt/exp07/reference_manifest_seen_k8_seed${seed}.json"
  digest=$("$PY" -c 'import sys; from tools.reference_manifest import load_manifest, manifest_hash; print(manifest_hash(load_manifest(sys.argv[1])))' "$manifest")
  echo "STEP eval released_seen k8 seed$seed $(date -Iseconds)"
  "$PY" tools/exp07_eval_launch.py --split seen --backbone simple \
    --checkpoint checkpoints/xRIR_seen.pth --num-shot 8 --manifest "$manifest" \
    --manifest-hash "$digest" --gl-seed "$seed" --batch-size 16 \
    --conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols \
    --decomposition-batches 0 --max-samples 0 --num-workers 12 --threads 2 \
    --out-dir "ckpt/exp07/eval/released_seen_k8_seed${seed}_k0" \
    --run-label "released_seen_k8_seed${seed}_k0" --reviewed-commit "$MERGE" \
    --data-root "$XRIR_DATA_PATH" --log-dir "$REC" --gpu "$GPU"; rc=$?
  echo "STEP eval released_seen k8 seed$seed exit=$rc"
  [ $rc -eq 0 ] || { echo "PHASE ABORT eval seed $seed failed"; exit 5; }
done
echo "STEP calibration $(date -Iseconds)"
"$PY" tools/exp07_calibration.py --reviewed-commit "$MERGE" \
  --runs ckpt/exp07/eval/released_seen_k8_seed{42..46}_k0 \
  --json "$OUT/CALIBRATION_SEEN_V1.json"; rc=$?
echo "STEP calibration exit=$rc"
echo "PHASE DONE $(date -Iseconds) calibration_exit=$rc"
} > "$LOG" 2>&1
echo "$LOG"
