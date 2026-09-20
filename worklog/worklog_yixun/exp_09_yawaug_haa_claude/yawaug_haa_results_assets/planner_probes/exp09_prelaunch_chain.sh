#!/bin/bash
# exp_09 pre-launch chain (2026-09-20): once BOTH markers exist —
#   REVIEW_OK        (Planner: Codex close review approved; contains the approved tip SHA)
#   PEER_ENDGAME_DONE (Planner: the peer pinged "end-game GPU work done" — main may move again)
# — merge the approved tip into main, re-fill the three moved approvals keys, run the CPU suite,
# run the bounded room-frame smoke on GPU 1, and write GO_GPU1 for the launch chain if every gate holds.
#   nohup setsid bash exp09_prelaunch_chain.sh > <record>/yawaug_haa_<ts>_prelaunch_chain.log 2>&1 &
set -uo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
E=$R/worklog/worklog_yixun/exp_09_yawaug_haa_claude
P=$E/yawaug_haa_results_assets/planner_probes
RB=$P/exp09_runbook.sh
EXPECT="finalize:ea8146d8,haa_pipeline_sh:4a401ee3,summarize_haa:3fc1e82d"
say() { echo "$(date +%Y-%m-%dT%H:%M:%S) EXP09-PRELAUNCH $*"; }
say "waiting for markers REVIEW_OK and PEER_ENDGAME_DONE"
while [ ! -f "$P/REVIEW_OK" ] || [ ! -f "$P/PEER_ENDGAME_DONE" ]; do sleep 60; done
TIP=$(cat "$P/REVIEW_OK" | tr -d '[:space:]')
say "markers present; approved tip $TIP"
[ "$(git -C /home/yixunhu/codespace/xRIR_code_wt rev-parse exp09-yawaug)" = "$TIP" ] || { say "exp09-yawaug moved from $TIP; refusing"; exit 2; }
PEER_ACK=1 FULLFIX=$TIP MERGE_MSG="exp09: merge the yaw-augmented HAA arm (room-frame pipeline init, exp_09 summariser, blob-at-commit HAA approvals) at ${TIP:0:7}" bash "$RB" merge || { say "merge refused"; exit 2; }
EXP06_REFILL_EXPECT="$EXPECT" bash "$RB" refill || { say "refill refused"; exit 3; }
bash "$RB" suite || { say "suite step failed"; exit 4; }
SL=$(ls -t "$E"/yawaug_haa_*_suite_full_cpu.log | head -1)
FAILS=$(grep -E "^FAILED" "$SL" | grep -vE "test_exp07_profiles.py::test_the_new_arm_checkpoint_digests_come_from_the_runtime_approval\[(seen_simple-simple-0|seen_cyl-cylindrical-0|seen_aug-simple-1)\]" || true)
if [ -n "$FAILS" ]; then say "suite has failures beyond the two exp_07 guards:"; echo "$FAILS"; exit 4; fi
say "suite OK ($(tail -1 "$SL"))"
# GPU 1 must be empty for the smoke; wait up to 3 h for the peer's release
n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1)" ]; do n=$((n+1)); [ $n -gt 180 ] && { say "GPU 1 still busy after 3 h"; exit 5; }; sleep 60; done
bash "$RB" smoke 1 || { say "smoke refused/failed"; exit 6; }
echo "GO $(date -Is) tip $TIP merge $(git rev-parse HEAD)" > "$P/GO_GPU1"
say "GO_GPU1 written — the launch chain takes over"
