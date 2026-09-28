#!/bin/bash
# exp_11 PHASE-1 chain (armed 2026-09-28): waits for the marker GO_GPU (content: the gpu index, written by the
# Planner on the owner's "card released" ping), then: 3 consecutive empty-GPU minutes → exp11_runbook.sh smoke <gpu>
# → launch <gpu> (queue yawaug_hf:0 yawaug_hf:1 yawaug_hf:2 yawaug_hf:zeroshot) → wait for the pipeline → summarize
# (phase 1 → ckpt/exp11/phase1/) → results → writes PHASE1_DONE. Every step fails closed via the runbook.
#   nohup setsid bash exp11_phase1_chain.sh > <record>/orientation_cue_fairness_<ts>_phase1_chain.log 2>&1 &
set -uo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
E=$R/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude
P=$E/orientation_cue_fairness_results_assets/planner_probes
RB=$P/exp11_runbook.sh
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) EXP11-P1 $*"; }
say "waiting for marker $P/GO_GPU"
while [ ! -f "$P/GO_GPU" ]; do sleep 60; done
GPU=$(tr -d '[:space:]' < "$P/GO_GPU"); [[ "$GPU" =~ ^[01]$ ]] || { say "bad GO_GPU content '$GPU'"; exit 2; }
say "GO for GPU $GPU; requiring 3 consecutive empty minutes"
empty=0; while [ $empty -lt 3 ]; do if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU")" ]; then empty=$((empty+1)); else empty=0; fi; sleep 60; done
[ "$(df --output=avail -BG / | tail -1 | tr -dc '0-9')" -ge 25 ] || { say "disk below 25 GB; refusing"; exit 3; }
bash "$RB" smoke "$GPU" || { say "smoke refused/failed"; exit 4; }
bash "$RB" launch "$GPU" || { say "launch refused"; exit 5; }
PIDF=$(ls -t "$E"/orientation_cue_fairness_*_haa_gpu${GPU}.pid | head -1); PID=$(cat "$PIDF"); say "pipeline pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
say "pipeline exited: $(grep -hE 'QUEUE_DONE|QUEUE_FAILED' "$E"/orientation_cue_fairness_*_launch.log 2>/dev/null | tail -1)"
bash "$RB" summarize || { say "summarize refused (exit $?)"; exit 6; }
bash "$RB" results || { say "results refused (exit $?)"; exit 7; }
echo "PHASE1_DONE $(date -Is) gpu $GPU head $(git rev-parse HEAD)" > "$P/PHASE1_DONE"
say "PHASE1_DONE — GPU $GPU stays reserved for Phase 1b (Planner arms the next chain)"
