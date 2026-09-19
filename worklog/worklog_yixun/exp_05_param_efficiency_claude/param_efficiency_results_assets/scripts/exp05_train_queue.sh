#!/bin/bash
# usage: exp05_train_queue.sh <gpu> <reviewed_commit> <tier:backbone> [<tier:backbone> ...]
# Runs exp_05 confirmatory trainings sequentially on one GPU; each launch needs its clean receipt; stops the queue on the first non-certified attempt.
cd /home/yixunhu/codespace/xRIR_code
GPU=$1; SHA=$2; shift 2; P=worklog/worklog_yixun/exp_05_param_efficiency_claude
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
QLOG=$P/param_efficiency_$(date +%Y%m%dT%H%M%S)_train_queue_gpu${GPU}.log
for arm in "$@"; do
  tier=${arm%%:*}; bb=${arm##*:}; TS=$(date +%Y%m%dT%H%M%S)
  receipt=$(ls -t ckpt/exp05/${tier}_${bb}/_probe_*_${tier}_${bb}.json | head -1)
  echo "$(date -Is) START ${tier}_${bb} receipt=$receipt ts=$TS" >> "$QLOG"
  nohup setsid tools/exp04_launch.sh full --tier "$tier" --backbone "$bb" --gpu "$GPU" --reviewed-commit "$SHA" --probe-json "$receipt" --timestamp "$TS" > "$P/param_efficiency_${TS}_launcher_${tier}_${bb}.log" 2>&1
  rc=$?
  final=ckpt/exp05/${tier}_${bb}/final
  if [ $rc -eq 0 ] && [ -L "$final" ] && [ -f "$final/completion.json" ]; then
    echo "$(date -Is) CERTIFIED ${tier}_${bb} -> $(readlink $final)" >> "$QLOG"
  else
    echo "$(date -Is) NOT_CERTIFIED ${tier}_${bb} exit=$rc (queue stopped)" >> "$QLOG"; break
  fi
done
echo "TRAIN_QUEUE_DONE gpu=$GPU $(date -Is)" >> "$QLOG"; echo "$QLOG"
