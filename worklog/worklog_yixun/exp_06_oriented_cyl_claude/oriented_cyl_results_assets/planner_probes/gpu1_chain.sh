#!/bin/bash
# exp_06 GPU-1 chain (2026-09-18): HAA queue -> H3 sim queue back-to-back, so the card never idles
# for 5 minutes after 17:30 (the peer's seen_aug watcher would take it). Run detached:
#   nohup setsid bash gpu1_chain.sh <haa jobs...> > <record>/oriented_cyl_<ts>_gpu1_chain.log 2>&1 &
# Preconditions are the runbook's own (approvals gate, tree clean outside worklog, GPU empty, receipt present).
set -uo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
E=$R/worklog/worklog_yixun/exp_06_oriented_cyl_claude
RB=$E/oriented_cyl_results_assets/planner_probes/posttrain_runbook.sh
export GPU=1
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) GPU1-CHAIN $*"; }
say "start jobs=$*"
GPU=1 bash "$RB" haa "$@" || { say "haa step refused (exit $?)"; exit 2; }
PIDF=$(ls -t "$E"/oriented_cyl_*_posttrain_haa_gpu1.pid | head -1)
PID=$(cat "$PIDF"); say "HAA pipeline pid $PID ($PIDF)"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
say "HAA pipeline exited; queue log: $(grep -hE 'QUEUE_DONE|QUEUE_FAILED' "$E"/oriented_cyl_*_posttrain_haa.log 2>/dev/null | tail -1)"
# straight into the sim queue (seconds of gap): approvals gate, receipt check and the 10 evaluations
EXP06_F2_LANDED=1 GPU=1 bash "$RB" sim || { say "sim step refused (exit $?)"; exit 3; }
PIDF=$(ls -t "$E"/oriented_cyl_*_posttrain_sim_gpu1.pid | head -1)
PID=$(cat "$PIDF"); say "SIM queue pid $PID ($PIDF)"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
say "SIM queue exited: $(grep -hE 'SIM_QUEUE_DONE|FAILED' "$E"/oriented_cyl_*_posttrain_sim.log 2>/dev/null | tail -3 | tr '\n' ' ')"
say "GPU 1 free — ping the peer 'GPU 1 released'"
