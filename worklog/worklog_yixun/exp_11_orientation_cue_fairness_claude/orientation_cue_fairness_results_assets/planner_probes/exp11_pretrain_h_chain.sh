#!/bin/bash
# exp_11 PRETRAIN-H chain (armed 2026-09-28): waits for PHASE1B_DONE and GO_GPU, then on that gpu: 3 empty minutes →
# disk ≥ 30 GB → exp11_runbook_phase2.sh pretrain H <gpu> → waits for the launcher to exit and for
# ckpt/exp11/pretrain/xRIR_simpor_8_shot/final/completion.json → PRETRAIN_H_DONE. It does NOT pin or launch the HAA
# queue: the Planner reviews the completion, pins (pin11 H) and arms the next chain.
set -uo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
E=$R/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude
P=$E/orientation_cue_fairness_results_assets/planner_probes
RB=$P/exp11_runbook_phase2.sh
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) EXP11-PH $*"; }
say "waiting for PHASE1B_DONE and GO_GPU"
while [ ! -f "$P/PHASE1B_DONE" ] || [ ! -f "$P/GO_GPU" ]; do sleep 60; done
GPU=$(tr -d '[:space:]' < "$P/GO_GPU"); [[ "$GPU" =~ ^[01]$ ]] || { say "bad GO_GPU"; exit 2; }
say "Phase 1b done; GPU $GPU; requiring 3 consecutive empty minutes"
empty=0; while [ $empty -lt 3 ]; do if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU")" ]; then empty=$((empty+1)); else empty=0; fi; sleep 60; done
[ "$(df --output=avail -BG / | tail -1 | tr -dc '0-9')" -ge 30 ] || { say "disk below 30 GB; refusing"; exit 3; }
bash "$RB" pretrain H "$GPU" || { say "pretrain refused"; exit 4; }
PIDF=$(ls -t "$E"/orientation_cue_fairness_*_pretrain_H_gpu${GPU}.pid | head -1); PID=$(cat "$PIDF"); say "launcher pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 300; done
ROOT=ckpt/exp11/pretrain/xRIR_simpor_8_shot
if [ -f "$ROOT/final/completion.json" ]; then echo "PRETRAIN_H_DONE $(date -Is) gpu $GPU head $(git rev-parse HEAD) final $(readlink -f $ROOT/final)" > "$P/PRETRAIN_H_DONE"; say "PRETRAIN_H_DONE — Planner reviews the completion, pins and arms H's HAA queue"; else say "launcher exited WITHOUT a promoted final/completion.json — inspect $ROOT"; exit 5; fi
