#!/bin/bash
# usage: exp05_probe_run.sh <gpu> <reviewed_commit> <tier> <backbone> ; runs one exp_05 fit/timing probe synchronously
cd /home/yixunhu/codespace/xRIR_code
GPU=$1; SHA=$2; TIER=$3; BB=$4; TS=$(date +%Y%m%dT%H%M%S); P=worklog/worklog_yixun/exp_05_param_efficiency_claude
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
LOG=$P/param_efficiency_${TS}_probe_${TIER}_${BB}_launcher.log
nohup setsid tools/exp04_launch.sh probe --tier "$TIER" --backbone "$BB" --gpu "$GPU" --reviewed-commit "$SHA" --timestamp "${TS}" > "$LOG" 2>&1
echo "EXP05_PROBE_EXIT=$? tier=$TIER backbone=$BB receipt=$(ls -t ckpt/exp05/${TIER}_${BB}/_probe_*.json 2>/dev/null | head -1)" >> "$LOG"
echo "$LOG"
