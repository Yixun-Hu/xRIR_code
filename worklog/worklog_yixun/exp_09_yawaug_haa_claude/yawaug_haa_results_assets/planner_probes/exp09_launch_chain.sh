#!/bin/bash
# exp_09 launch chain (armed after the code is merged, re-filled, suite-checked and Codex-approved):
# waits for the GO marker (written by the Planner once the peer pings "GPU 1 released"), then for GPU 1
# to be empty for 3 consecutive minutes, launches the HAA queue through the runbook, waits for it,
# runs the summariser and the results generators. Every step fails closed via the runbook.
#   nohup setsid bash exp09_launch_chain.sh > <record>/yawaug_haa_<ts>_launch_chain.log 2>&1 &
set -uo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
E=$R/worklog/worklog_yixun/exp_09_yawaug_haa_claude
RB=$E/yawaug_haa_results_assets/planner_probes/exp09_runbook.sh
GO=$E/yawaug_haa_results_assets/planner_probes/GO_GPU1
GPU=1
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) EXP09-CHAIN $*"; }
say "waiting for GO marker $GO"
while [ ! -f "$GO" ]; do sleep 60; done
say "GO seen ($(cat "$GO")); waiting for GPU $GPU to be empty 3 min"
empty=0
while [ $empty -lt 3 ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU)" ]; then empty=$((empty+1)); else empty=0; fi
  sleep 60
done
say "GPU $GPU empty for 3 min; launching"
bash "$RB" launch $GPU || { say "launch refused (exit $?)"; exit 2; }
PIDF=$(ls -t "$E"/yawaug_haa_*_haa_gpu${GPU}.pid | head -1); PID=$(cat "$PIDF"); say "pipeline pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
say "pipeline exited: $(grep -hE 'QUEUE_DONE|QUEUE_FAILED' "$E"/yawaug_haa_*_launch.log 2>/dev/null | tail -1)"
say "GPU $GPU free — ping the peer 'GPU 1 released'"
bash "$RB" summarize || { say "summarize refused (exit $?)"; exit 3; }
bash "$RB" results || { say "results refused (exit $?)"; exit 4; }
say "DONE"
