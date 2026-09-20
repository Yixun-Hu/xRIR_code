#!/bin/bash
# GPU-0 chain: wait for L_simple certification -> exp_05 L-tier evaluations -> exp_07 seen arms 1 (simple) and 2 (cylindrical).
set -uo pipefail
cd /home/yixunhu/codespace/xRIR_code
S=/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad
REC=worklog/worklog_yixun/exp_07_seen_protocol_claude
export PYTHONPATH="$PWD" XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONHASHSEED=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
LOG=$REC/seen_protocol_$(date +%Y%m%dT%H%M%S)_gpu0_chain.log
say() { echo "$(date -Is) $*" >> "$LOG"; }
clean_or_abort() { if git status --short | grep -v '^.. worklog/' | grep -qv '^?? '; then say "CHAIN ABORT: tree dirty outside worklog"; exit 2; fi; }
say "CHAIN START pid $$"
# 1. wait for L_simple certification (final symlink) and GPU 0 free of the trainer
until [ -L ckpt/exp05/L_simple/final ]; do sleep 60; done
say "STEP L_simple certified -> $(readlink ckpt/exp05/L_simple/final)"
until [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep 9c72ac69)" ]; do sleep 30; done
say "STEP GPU 0 free"
# 2. exp_05 L-tier evaluations (22 runs + 2 yaw blocks)
clean_or_abort; SHA=$(git rev-parse HEAD); say "STEP exp05 evals start reviewed=$SHA"
bash $S/exp05_eval_queue.sh 0 $SHA L_cylindrical L_simple >> "$LOG" 2>&1; say "STEP exp05 evals exit=$?"
# 3. exp_07 seen arms, sequential
for spec in "seen_simple simple 0" "seen_cyl cylindrical 0"; do
  set -- $spec; role=$1; bb=$2; yaw=$3
  clean_or_abort; SHA=$(git rev-parse HEAD); stamp=$(date +%Y%m%dT%H%M%S)
  flags=(--backbone $bb --yaw-aug $yaw --gpu 0 --reviewed-commit $SHA --log-dir $REC)
  say "STEP $role smoke reviewed=$SHA stamp=$stamp"
  tools/exp07_launch.sh smoke "${flags[@]}" --timestamp ${stamp}smoke >> "$LOG" 2>&1; rc=$?; say "STEP $role smoke exit=$rc"; [ $rc -eq 0 ] || { say "CHAIN ABORT: $role smoke failed"; exit 3; }
  say "STEP $role probe"
  tools/exp07_launch.sh probe "${flags[@]}" --timestamp ${stamp}probe >> "$LOG" 2>&1; rc=$?; say "STEP $role probe exit=$rc"; [ $rc -eq 0 ] || { say "CHAIN ABORT: $role probe failed"; exit 4; }
  receipt=ckpt/exp07/$role/_probe_${stamp}probe_${role}.json; [ -f "$receipt" ] || { say "CHAIN ABORT: probe receipt $receipt missing"; exit 4; }
  say "STEP $role full launch receipt=$receipt"
  tools/exp07_launch.sh full "${flags[@]}" --timestamp ${stamp}full --probe-json "$receipt" > "$REC/${role}_${stamp}full_launcher.log" 2>&1; rc=$?
  say "STEP $role full exit=$rc final=$(readlink ckpt/exp07/$role/final 2>/dev/null || echo none)"
  [ $rc -eq 0 ] && [ -L ckpt/exp07/$role/final ] || { say "CHAIN ABORT: $role full not certified"; exit 5; }
done
say "CHAIN DONE"
