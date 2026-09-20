#!/bin/bash
# Launch one exp_07 seen arm (smoke -> probe -> full) on a GPU once it has been free of compute processes for 5 min (not before a given time) and the disk gate holds.
# usage: exp07_arm_chain.sh <gpu> <role> <backbone> <yaw> <not_before 'YYYY-MM-DD HH:MM'>
set -uo pipefail
GPU=$1; ROLE=$2; BB=$3; YAW=$4; NB=$(date -d "$5" +%s)
cd /home/yixunhu/codespace/xRIR_code
S=/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad
REC=worklog/worklog_yixun/exp_07_seen_protocol_claude
export PYTHONPATH="$PWD" XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONHASHSEED=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
UU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
LOG=$REC/seen_protocol_$(date +%Y%m%dT%H%M%S)_${ROLE}_chain_gpu${GPU}.log
say() { echo "$(date -Is) $*" >> "$LOG"; }
say "ARM CHAIN START role=$ROLE gpu=$GPU not_before=$5 pid $$"
while [ $(date +%s) -lt $NB ]; do sleep 60; done
clear=0; while [ $clear -lt 5 ]; do if [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$UU")" ]; then clear=$((clear+1)); else clear=0; fi; sleep 60; done
say "STEP GPU $GPU clear for 5 min"
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9'); [ "$free_gb" -ge 52 ] || { say "CHAIN ABORT: only ${free_gb} GB free (< 52)"; exit 2; }
if git status --short --untracked-files=all | grep -v '^.. worklog/' | grep -q .; then say "CHAIN ABORT: tree dirty outside worklog"; exit 3; fi
SHA=$(git rev-parse HEAD); stamp=$(date +%Y%m%dT%H%M%S)
flags=(--backbone $BB --yaw-aug $YAW --gpu $GPU --reviewed-commit $SHA --log-dir $REC)
say "STEP $ROLE smoke reviewed=$SHA stamp=$stamp"
tools/exp07_launch.sh smoke "${flags[@]}" --timestamp ${stamp}smoke >> "$LOG" 2>&1; rc=$?; say "STEP $ROLE smoke exit=$rc"; [ $rc -eq 0 ] || { say "CHAIN ABORT: smoke failed"; exit 4; }
say "STEP $ROLE probe"
tools/exp07_launch.sh probe "${flags[@]}" --timestamp ${stamp}probe >> "$LOG" 2>&1; rc=$?; say "STEP $ROLE probe exit=$rc"; [ $rc -eq 0 ] || { say "CHAIN ABORT: probe failed"; exit 5; }
receipt=ckpt/exp07/$ROLE/_probe_${stamp}probe_${ROLE}.json; [ -f "$receipt" ] || { say "CHAIN ABORT: receipt missing"; exit 5; }
say "STEP $ROLE full launch receipt=$receipt"
tools/exp07_launch.sh full "${flags[@]}" --timestamp ${stamp}full --probe-json "$receipt" > "$REC/${ROLE}_${stamp}full_launcher.log" 2>&1; rc=$?
say "STEP $ROLE full exit=$rc final=$(readlink ckpt/exp07/$ROLE/final 2>/dev/null || echo none)"
[ $rc -eq 0 ] && [ -L ckpt/exp07/$ROLE/final ] || { say "CHAIN ABORT: full not certified"; exit 6; }
say "STEP $ROLE evaluations"; bash $S/exp07_eval_queue.sh $GPU $ROLE >> "$LOG" 2>&1; say "CHAIN DONE role=$ROLE"
