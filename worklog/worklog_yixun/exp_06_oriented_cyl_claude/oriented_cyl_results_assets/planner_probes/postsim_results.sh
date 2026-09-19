#!/bin/bash
# exp_06 (2026-09-18 evening): after the H3 sim queue exits, produce the canonical outputs in one pass.
#   1. comparer (H3) with the current approvals (committed at c8611fc: compare/eval_launch re-filled for the sim path)
#   2. summariser (H1/H1b/H2/D/side split) against the approvals version the HAA children ran under:
#      the working-tree approvals are restored to 8ed7684's bytes (byte-identical to the blob at every
#      child's reviewed commit) for the duration of the run and put back afterwards; see the notebook
#      entry of 19:0x for why (re-verification binds the working file to the child's commit).
#   3. results markdown from the producer outputs only.
# Every step fails closed; outputs are exclusive-create; the tree is left exactly as found.
set -euo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
E=$R/worklog/worklog_yixun/exp_06_oriented_cyl_claude
AP=worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json
SIM_COMMIT=c8611fc772f336e2ee4d95a2f2cf8d30bfa895df      # approvals in force for the sim launches
HAA_COMMIT=2e43887a361eb59ea19ea5a7ab8d5710410fa5d9      # a child's reviewed commit; approvals blob == 8ed7684's
HAA_BLOB_COMMIT=8ed76848df41ee833eeb4f2b0919a12b5c342536
TS=$(date +%Y-%m-%d_%H:%M:%S); LOG=$E/oriented_cyl_${TS}_postsim_results.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
refuse() { log "REFUSE: $*"; exit 2; }
log "POSTSIM start HEAD=$(git rev-parse HEAD)"
git status --porcelain | grep -v '^.. worklog/' | grep -q . && refuse "tree dirty outside worklog/"
[ "$(find ckpt/exp06/sim_eval -name completion.json | wc -l)" -eq 10 ] || refuse "sim_eval has $(find ckpt/exp06/sim_eval -name completion.json | wc -l)/10 completions"
[ ! -e ckpt/exp06/stats.json ] && [ ! -e ckpt/exp06/summary.txt ] || refuse "summariser outputs exist"
# --- 1. comparer (skipped if its outputs already exist from an earlier pass) --------------------------
[ "$(git rev-parse HEAD:$AP)" = "$(git rev-parse $SIM_COMMIT:$AP)" ] || refuse "HEAD's approvals blob differs from $SIM_COMMIT's"
cmp -s "$AP" <(git show $SIM_COMMIT:$AP) || refuse "working-tree approvals differ from $SIM_COMMIT's blob before the comparer"
if [ -e ckpt/exp06/h3.json ]; then log "comparer outputs exist (h3.json $(sha256sum ckpt/exp06/h3.json | cut -c1-12)); skipping"; else
log "comparer (H3) with approvals at $SIM_COMMIT"
$PY tools/exp06_compare.py --runs-a ckpt/yaw_aug/eval/control_k8_seed{42,43,44,45,46}_k0 \
    --runs-b ckpt/exp06/sim_eval/cyl/seed{42,43,44,45,46} --runs-c ckpt/exp06/sim_eval/cyl_or/seed{42,43,44,45,46} \
    --approved "$AP" --approved-commit "$SIM_COMMIT" --json ckpt/exp06/h3.json --summary ckpt/exp06/h3.txt 2>&1 | grep -vE "UserWarning|warnings.warn" | tee -a "$LOG"
test -f ckpt/exp06/h3.json || refuse "comparer wrote no h3.json"
fi
# --- 2. summariser against the HAA-time approvals ----------------------------------------------------
[ "$(git rev-parse $HAA_COMMIT:$AP)" = "$(git rev-parse $HAA_BLOB_COMMIT:$AP)" ] || refuse "blob mismatch $HAA_COMMIT vs $HAA_BLOB_COMMIT"
log "restoring working-tree approvals to the blob at $HAA_BLOB_COMMIT (== every HAA child's reviewed-commit blob) for the summariser"
git show $HAA_BLOB_COMMIT:$AP > "$AP"
trap 'git checkout -q HEAD -- "$AP"; log "approvals restored to HEAD ($(git rev-parse --short HEAD))"' EXIT
$PY tools/exp06_summarize_haa.py --approved "$AP" --approved-commit "$HAA_COMMIT" --gate-g1 ckpt/exp06/gate_g1.json \
    --json ckpt/exp06/stats.json --summary ckpt/exp06/summary.txt 2>&1 | grep -vE "UserWarning|warnings.warn" | tee -a "$LOG"
test -f ckpt/exp06/stats.json || refuse "summariser wrote no stats.json"
git checkout -q HEAD -- "$AP"; trap - EXIT; log "approvals restored to HEAD ($(git rev-parse --short HEAD))"
cmp -s "$AP" <(git show HEAD:$AP) || refuse "approvals not identical to HEAD after restore"
# --- 3. results markdown -------------------------------------------------------------------------------
$PY "$E/oriented_cyl_results_assets/make_results_md.py" --stats ckpt/exp06/stats.json --h3 ckpt/exp06/h3.json --gate ckpt/exp06/gate_g1.json 2>&1 | tee -a "$LOG"
log "POSTSIM OK: h3 $(sha256sum ckpt/exp06/h3.json | cut -c1-12) stats $(sha256sum ckpt/exp06/stats.json | cut -c1-12)"
