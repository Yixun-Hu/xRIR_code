#!/bin/bash
# exp_11 PHASE-1b chain (armed 2026-09-28): waits for PHASE1_DONE and GO_GPU, then on that gpu: 3 empty minutes →
# exp11_runbook_phase2.sh smoke11 H <gpu> (the exp_11 launcher's smoke: oriented training/eval rungs + adapter rungs e/f)
# → launch11 <gpu> (J and K: control_adapter/yawaug_adapter × seeds 0,1,2 + zeroshot) → wait → summarize11 phase1b →
# results11 phase1b → PHASE1B_DONE. Every step fails closed via the runbook.
set -uo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
E=$R/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude
P=$E/orientation_cue_fairness_results_assets/planner_probes
RB=$P/exp11_runbook_phase2.sh
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) EXP11-P1B $*"; }
say "waiting for PHASE1_DONE and GO_GPU"
while [ ! -f "$P/PHASE1_DONE" ] || [ ! -f "$P/GO_GPU" ]; do sleep 60; done
GPU=$(tr -d '[:space:]' < "$P/GO_GPU"); [[ "$GPU" =~ ^[01]$ ]] || { say "bad GO_GPU"; exit 2; }
say "Phase 1 done; GPU $GPU; requiring 3 consecutive empty minutes"
empty=0; while [ $empty -lt 3 ]; do if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU")" ]; then empty=$((empty+1)); else empty=0; fi; sleep 60; done
[ "$(df --output=avail -BG / | tail -1 | tr -dc '0-9')" -ge 25 ] || { say "disk below 25 GB; refusing"; exit 3; }
bash "$RB" smoke11 H "$GPU" || { say "smoke11 refused/failed"; exit 4; }
JOBS="control_adapter:0 control_adapter:1 control_adapter:2 control_adapter:zeroshot yawaug_adapter:0 yawaug_adapter:1 yawaug_adapter:2 yawaug_adapter:zeroshot"
bash "$RB" launch11 "$GPU" $JOBS || { say "launch11 refused"; exit 5; }
PIDF=$(ls -t "$E"/orientation_cue_fairness_*_haa11_gpu${GPU}.pid | head -1); PID=$(cat "$PIDF"); say "pipeline pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
say "pipeline exited: $(grep -hE 'QUEUE_DONE|QUEUE_FAILED' "$E"/orientation_cue_fairness_*_launch11.log 2>/dev/null | tail -1)"
bash "$RB" summarize11 phase1b || { say "summarize11 refused (exit $?)"; exit 6; }
bash "$RB" results11 phase1b || { say "results11 refused (exit $?)"; exit 7; }
echo "PHASE1B_DONE $(date -Is) gpu $GPU head $(git rev-parse HEAD)" > "$P/PHASE1B_DONE"
say "PHASE1B_DONE — GPU $GPU stays reserved for the H pretraining (next chain)"
