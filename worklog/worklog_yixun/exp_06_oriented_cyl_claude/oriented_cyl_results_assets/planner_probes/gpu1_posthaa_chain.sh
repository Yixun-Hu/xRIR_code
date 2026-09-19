#!/bin/bash
# exp_06 (2026-09-18): after BOTH HAA pipelines exit, re-fill the two sim-path approvals digests and launch the
# H3 sim queue on GPU 1 within minutes (the peer's seen_aug watcher takes GPU 1 after 17:30 once idle 5 min).
# The re-fill must wait for every HAA child to finalise: children bind the approvals blob at their reviewed commit.
set -uo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
E=$R/worklog/worklog_yixun/exp_06_oriented_cyl_claude
RB=$E/oriented_cyl_results_assets/planner_probes/posttrain_runbook.sh
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) POSTHAA-CHAIN $*"; }
say "waiting for HAA pipelines 1005097 (GPU 0) and 1010000 (GPU 1)"
while kill -0 1005097 2>/dev/null || kill -0 1010000 2>/dev/null; do sleep 20; done
say "both pipelines exited: $(grep -hE 'QUEUE_DONE|QUEUE_FAILED' "$E"/oriented_cyl_2026-09-18_12:1*_posttrain_haa.log | tr '\n' ' ')"
sleep 5
say "refill3"
EXP06_REFILL_EXPECT="compare:43b3461f,eval_launch:0a73f84a" bash "$RB" refill3 || { say "refill3 refused (exit $?)"; exit 2; }
say "sim launch on GPU 1"
EXP06_F2_LANDED=1 GPU=1 bash "$RB" sim || { say "sim step refused (exit $?)"; exit 3; }
PIDF=$(ls -t "$E"/oriented_cyl_*_posttrain_sim_gpu1.pid | head -1); PID=$(cat "$PIDF"); say "SIM queue pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
say "SIM queue exited: $(grep -hE 'SIM_QUEUE_DONE|FAILED|SKIP' "$E"/oriented_cyl_*_posttrain_sim.log 2>/dev/null | tail -3 | tr '\n' ' ')"
say "GPU 1 free — ping the peer 'GPU 1 released'"
