#!/bin/bash
# Wait until <role> is certified AND the main tree is clean (incl. untracked outside worklog/), then run its 10 seen evaluations on <gpu>.
# usage: exp07_eval_waiter.sh <gpu> <role>
set -uo pipefail
GPU=$1; ROLE=$2; cd /home/yixunhu/codespace/xRIR_code
S=/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad
LOG=worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_$(date +%Y%m%dT%H%M%S)_${ROLE}_eval_waiter_gpu${GPU}.log
say() { echo "$(date -Is) $*" >> "$LOG"; }
say "WAITER START role=$ROLE gpu=$GPU pid $$"
until [ -L ckpt/exp07/$ROLE/final ]; do sleep 60; done; say "STEP $ROLE certified"
n=0; while git status --short --untracked-files=all | grep -v '^.. worklog/' | grep -q .; do n=$((n+1)); [ $((n % 10)) -eq 1 ] && say "WAIT tree dirty outside worklog: $(git status --short --untracked-files=all | grep -v '^.. worklog/' | tr '\n' ' ' | cut -c1-120)"; sleep 60; done
say "STEP tree clean; running evaluations"
bash $S/exp07_eval_queue.sh $GPU $ROLE >> "$LOG" 2>&1; say "WAITER DONE role=$ROLE exit=$?"
