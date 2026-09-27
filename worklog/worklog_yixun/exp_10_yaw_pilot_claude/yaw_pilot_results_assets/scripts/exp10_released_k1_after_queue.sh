#!/bin/bash
# exp_10 — run the released_k1 arm on GPU <gpu> after the queue log reports QUEUE DONE (waits; then one arm chain).
# The chain's exit status is this script's exit status (Codex tooling review, finding 13): it used
# to log the failure and return success.
# env: EXP10_REPO_ROOT — repository root override for the tests.
set -uo pipefail
GPU=${1:-1}; QLOG=${2:?usage: exp10_released_k1_after_queue.sh <gpu_index> <queue log>}
cd "${EXP10_REPO_ROOT:-/home/yixunhu/codespace/xRIR_code}"
until grep -q "QUEUE DONE" "$QLOG"; do sleep 60; done
E=worklog/worklog_yixun/exp_10_yaw_pilot_claude
"$E/yaw_pilot_results_assets/scripts/exp10_arm_chain.sh" released_k1 simple checkpoints/xRIR_unseen.pth ckpt/yaw_rotation/reference_manifest_k1.json f6d71f86d5d313f2a982fe6f8cb801116a20ea1c37b0a65d290c088934fd73e3 1 cuda "$GPU" none
rc=$?
echo "RELEASED_K1_EXIT=$rc" >> "$QLOG"
if [ "$rc" -ne 0 ]; then echo "RELEASED_K1 FAILED (see the arm chain log)" >> "$QLOG"; fi
exit "$rc"
