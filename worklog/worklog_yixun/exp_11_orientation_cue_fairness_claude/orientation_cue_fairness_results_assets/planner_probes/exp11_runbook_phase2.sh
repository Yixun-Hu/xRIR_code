#!/bin/bash
# exp_11 runbook — PHASE 1b / PHASE 2 (the exp_11-owned family: arms J, K via adapters; H, I via new pretrainings).
# One step per invocation; every step fails closed; logs to <record>/orientation_cue_fairness_<ts>_<step>.log
# Steps:
#   merge2   FULLFIX=<approved exp11-cue tip> PEER_ACK=1 MERGE_MSG=...      merge round 2 into main
#   refill2  EXP06_REFILL_EXPECT=summarize_haa:<prefix8>                    re-fill the moved exp_06 key
#   fill11                                                                  first fill of exp_11's record: 8 code keys + reused (artifacts stay null)
#   suite                                                                   full CPU suite to a log
#   dryrun11 [gpu] <jobs...>                                                exp11 pipeline dry-run
#   smoke11 <H|I> <gpu>                                                     tools/exp11_launch.sh smoke (all rungs incl. adapter e/f)
#   launch11 <gpu> <jobs...>                                                exp11 HAA queue (e.g. control_adapter:0 ... yawaug_adapter:zeroshot)
#   pretrain <H|I> <gpu>                                                    tools/exp11_launch.sh full (detached, 36 h ceiling)
#   pin11 <H|I>                                                             fill artifacts.<key> from the promoted final/completion.json
#   summarize11 <phase1b|final>                                             exp_11 summariser for the phase
#   results11 <phase1b|final>                                               render results md/html for the phase
set -euo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
E=$R/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude
AP06=worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json
AP11=worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_results_assets/approved_digests.json
TRAILER="Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
OUT=ckpt/exp11/sim2real
step=${1:-help}; shift || true; TS=$(date +%Y-%m-%d_%H:%M:%S); LOG=$E/orientation_cue_fairness_${TS}_${step}.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
refuse() { log "REFUSE: $*"; exit 2; }
clean_outside_worklog() { if git status --porcelain | grep -v '^.. worklog/' | grep -q .; then git status --porcelain | grep -v '^.. worklog/' | tee -a "$LOG"; refuse "tree dirty outside worklog/"; fi; }
gpu_empty() { local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$1") || refuse "GPU $1 query failed"; [ -z "$apps" ] || refuse "GPU $1 busy: $apps"; log "GPU $1 empty (explicit census)"; }
arm_root() { case $1 in H) echo ckpt/exp11/pretrain/xRIR_simpor_8_shot;; I) echo ckpt/exp11/pretrain/xRIR_simpor_yawaug_8_shot;; *) refuse "arm must be H or I";; esac; }
arm_key() { case $1 in H) echo simpor_epoch_012;; I) echo simpor_yaw_epoch_012;; esac; }
case $step in
  merge2)
    log "MERGE2 $TS HEAD=$(git rev-parse HEAD)"; [ "$(git branch --show-current)" = main ] || refuse "not on main"
    [ "${PEER_ACK:-0}" = 1 ] || refuse "message the peer first, then PEER_ACK=1"
    clean_outside_worklog; [ -n "${FULLFIX:-}" ] && [ -n "${MERGE_MSG:-}" ] || refuse "set FULLFIX=<40-hex approved tip> MERGE_MSG=<subject>"
    git merge-base --is-ancestor "$FULLFIX" exp11-cue || refuse "$FULLFIX is not on exp11-cue"
    git merge --no-ff "$FULLFIX" -m "$MERGE_MSG

Reviewed by Codex (worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_codex_code_round2*_review.md). Merged at $FULLFIX.

$TRAILER" 2>&1 | tee -a "$LOG"
    $PY -m py_compile tools/exp06_*.py tools/exp11_*.py model/*.py && bash -n tools/exp11_launch.sh tools/exp11_haa_pipeline.sh tools/exp06_haa_pipeline.sh && git diff --check HEAD~1 HEAD | tee -a "$LOG"
    log "MERGE2 OK: $(git rev-parse HEAD)" ;;
  refill2)
    log "REFILL2 $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    [ -n "${EXP06_REFILL_EXPECT:-}" ] || refuse "set EXP06_REFILL_EXPECT=summarize_haa:<prefix8>"
    EXP06_REFILL_EXPECT="$EXP06_REFILL_EXPECT" CUDA_VISIBLE_DEVICES='' $PY - "$R" "$AP06" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
repo, rel = sys.argv[1:3]
from tools import exp06_profiles as ap, exp06_approvals_api as api, provenance
path = repo + '/' + rel; head = provenance.git_state(repo)['HEAD']
a = json.load(open(path)); fresh = ap.compute_code_digests(repo, head)
changed = sorted(k for k in a['code'] if a['code'][k] != fresh.get(k))
expected = dict(item.split(':') for item in os.environ['EXP06_REFILL_EXPECT'].split(','))
assert set(changed) == set(expected), ('changed keys differ from the reviewed set', changed, sorted(expected))
for k, pre in expected.items():
    assert fresh[k].startswith(pre), (k, fresh[k][:12], 'expected prefix', pre)
a['code'] = {k: fresh[k] for k in a['code']}
api.validate(a); open(path, 'w').write(json.dumps(a, sort_keys=True, indent=2) + '\n')
print('code changed:', changed); print({k: fresh[k][:16] for k in changed})
PY
    git add "$AP06" && git commit -q -m "exp06/exp11: approvals re-fill after the exp_11 round-2 merge (${EXP06_REFILL_EXPECT//:*,/, })

$TRAILER" && log "committed $(git rev-parse HEAD)"; log "REFILL2 OK" ;;
  fill11)   # first fill of exp_11's own record: code keys at HEAD + reused identities; artifacts stay null
    log "FILL11 $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    CUDA_VISIBLE_DEVICES='' $PY - "$R" "$AP11" "$AP06" <<'PY' 2>&1 | tee -a "$LOG"
import hashlib, json, sys
repo, rel11, rel06 = sys.argv[1:4]
from tools import exp11_profiles as p11, provenance
head = provenance.git_state(repo)['HEAD']
path = repo + '/' + rel11; a = json.load(open(path))
assert all(v is None for v in a['code'].values()), 'exp_11 code keys already filled'
fresh = p11.compute_code_digests(repo, head)
assert set(fresh) == set(a['code']), (sorted(fresh), sorted(a['code']))
a['code'] = {k: fresh[k] for k in a['code']}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
a['reused']['approved_digests_exp04'] = sha(repo + '/worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_results_assets/approved_digests.json')
a['reused']['approved_digests_exp06'] = sha(repo + '/' + rel06)
a['reused']['legacy_receipt'] = {'path': 'ckpt/exp06/legacy_receipt.json', 'sha256': sha(repo + '/ckpt/exp06/legacy_receipt.json')}
for k in a['artifacts']:
    assert all(v is None for v in a['artifacts'][k].values()), 'artifacts must stay null at the first fill'
p11.validate(a); open(path, 'w').write(json.dumps(a, sort_keys=True, indent=2) + '\n')
print({k: v[:16] for k, v in a['code'].items()}); print('reused:', {k: (v if isinstance(v, str) else v['sha256'])[:16] for k, v in a['reused'].items()})
PY
    git add "$AP11" && git commit -q -m "exp11: first approvals fill (eight code keys at HEAD, reused identities; artifacts null)

$TRAILER" && log "committed $(git rev-parse HEAD)"; log "FILL11 OK" ;;
  suite)
    log "SUITE $TS HEAD=$(git rev-parse HEAD)"
    CUDA_VISIBLE_DEVICES='' $PY -m pytest tests -q -p no:cacheprovider -rf --durations=5 > "$E/orientation_cue_fairness_${TS}_suite_full_cpu.log" 2>&1 || true
    tail -1 "$E/orientation_cue_fairness_${TS}_suite_full_cpu.log" | tee -a "$LOG"; grep -E "^FAILED|^ERROR" "$E/orientation_cue_fairness_${TS}_suite_full_cpu.log" | tee -a "$LOG" || true ;;
  dryrun11)
    GPU=${1:-1}; shift || true; log "DRYRUN11 $TS HEAD=$(git rev-parse HEAD) jobs=$*"
    EXP11_HAA_RECORD=$E bash tools/exp11_haa_pipeline.sh "$GPU" "$@" --dry-run 2>&1 | tee -a "$LOG" | grep -cE "^RUN|^JOB" ;;
  smoke11)
    ARM=${1:?H|I}; GPU=${2:-1}; log "SMOKE11 $TS arm=$ARM GPU=$GPU HEAD=$(git rev-parse HEAD)"; clean_outside_worklog; gpu_empty "$GPU"
    EXP11_RECORD=$E bash tools/exp11_launch.sh smoke --arm "$ARM" --gpu "$GPU" --reviewed-commit "$(git rev-parse HEAD)" 2>&1 | grep -vE "UserWarning|warnings.warn" | tee -a "$LOG"
    log "SMOKE11 OK (diagnostic)" ;;
  launch11)
    GPU=${1:-1}; shift || true; [ $# -ge 1 ] || refuse "give the jobs"; log "LAUNCH11 $TS GPU=$GPU HEAD=$(git rev-parse HEAD) jobs=$*"; clean_outside_worklog; gpu_empty "$GPU"
    for j in "$@"; do init=${j%%:*}; [ ! -d "$OUT/$init/${j#*:}" ] && [ ! -d "$OUT/$init/seed${j#*:}" ] || refuse "$OUT/$init/${j#*:} exists"; done
    df -h / | tail -1 | awk '{print "free on /:", $4}' | tee -a "$LOG"
    EXP11_HAA_RECORD=$E bash tools/exp11_haa_pipeline.sh "$GPU" "$@" --dry-run 2>&1 | grep -E "PIPELINE|^JOB|REFUS|refus" | tee -a "$LOG"
    PIDF=$E/orientation_cue_fairness_${TS}_haa11_gpu${GPU}.pid
    EXP11_HAA_RECORD=$E nohup setsid bash tools/exp11_haa_pipeline.sh "$GPU" "$@" < /dev/null >> "$LOG" 2>&1 &
    echo $! > "$PIDF"; sleep 30
    kill -0 "$(cat "$PIDF")" 2>/dev/null && log "exp11 HAA queue launched: pid $(cat "$PIDF")" || { tail -20 "$LOG"; refuse "queue exited early"; } ;;
  pretrain)
    ARM=${1:?H|I}; GPU=${2:-1}; log "PRETRAIN $TS arm=$ARM GPU=$GPU HEAD=$(git rev-parse HEAD)"; clean_outside_worklog; gpu_empty "$GPU"
    ROOT=$(arm_root "$ARM"); [ ! -e "$ROOT/final" ] || refuse "$ROOT/final exists"
    EXP11_RECORD=$E bash tools/exp11_launch.sh full --arm "$ARM" --gpu "$GPU" --reviewed-commit "$(git rev-parse HEAD)" --dry-run 2>&1 | grep -E "REFUS|refus|attempt|timeout" | head -8 | tee -a "$LOG"
    PIDF=$E/orientation_cue_fairness_${TS}_pretrain_${ARM}_gpu${GPU}.pid
    EXP11_RECORD=$E nohup setsid bash tools/exp11_launch.sh full --arm "$ARM" --gpu "$GPU" --reviewed-commit "$(git rev-parse HEAD)" < /dev/null >> "$LOG" 2>&1 &
    echo $! > "$PIDF"; sleep 60
    kill -0 "$(cat "$PIDF")" 2>/dev/null && log "pretraining $ARM launched: pid $(cat "$PIDF")" || { tail -30 "$LOG"; refuse "launcher exited early"; } ;;
  pin11)   # fill artifacts.<key> for one arm from its promoted final/completion.json
    ARM=${1:?H|I}; log "PIN11 $TS arm=$ARM HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    ROOT=$(arm_root "$ARM"); KEY=$(arm_key "$ARM"); test -f "$ROOT/final/completion.json" || refuse "no $ROOT/final/completion.json"
    CUDA_VISIBLE_DEVICES='' $PY - "$R" "$AP11" "$ROOT" "$KEY" <<'PY' 2>&1 | tee -a "$LOG"
import hashlib, json, os, sys
repo, rel, root, key = sys.argv[1:5]
from tools import exp11_profiles as p11
path = repo + '/' + rel; a = json.load(open(path))
assert all(v is None for v in a['artifacts'][key].values()), key + ' already pinned'
c = json.load(open(os.path.join(repo, root, 'final', 'completion.json')))
assert c.get('child_exit') == 0 and c.get('run_type') in ('exp11_train', 'full'), c.get('run_type')
ck = os.path.join(repo, root, 'final', 'epoch_012.pth'); sha = hashlib.sha256(open(ck, 'rb').read()).hexdigest()
rec = c.get('sha256') or c.get('artifacts') or {}
recorded = rec.get('epoch_012.pth') if isinstance(rec, dict) else None
assert recorded in (None, sha), ('completion records a different epoch_012.pth digest', recorded[:16], sha[:16])
a['artifacts'][key] = {'epoch': 12, 'path': os.path.relpath(os.path.realpath(ck), repo), 'sha256': sha}
p11.validate(a); open(path, 'w').write(json.dumps(a, sort_keys=True, indent=2) + '\n'); print(key, a['artifacts'][key])
PY
    git add "$AP11" && git commit -q -m "exp11: pin artifacts.$KEY (arm $ARM, epoch_012.pth of the finalized attempt)

$TRAILER" && log "committed $(git rev-parse HEAD)"; log "PIN11 OK" ;;
  summarize11)
    PH=${1:?phase1b|final}; log "SUMMARIZE11 $TS phase=$PH HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    case $PH in phase1b) J=ckpt/exp11/phase1b/stats.json; S=ckpt/exp11/phase1b/summary.txt;; final) J=ckpt/exp11/stats.json; S=ckpt/exp11/summary.txt;; *) refuse "phase";; esac
    [ ! -e "$J" ] && [ ! -e "$S" ] || refuse "outputs exist"; mkdir -p "$(dirname "$J")"
    CUDA_VISIBLE_DEVICES='' $PY tools/exp06_summarize_haa.py --experiment exp11 --phase "$PH" --exp11-root $OUT --exp09-root ckpt/exp09/sim2real --approved "$AP06" --approved-commit "$(git rev-parse HEAD)" --gate-g1 ckpt/exp06/gate_g1.json --json "$J" --summary "$S" 2>&1 | grep -vE "UserWarning|warnings.warn" | tee -a "$LOG"
    test -f "$J" || refuse "no stats.json"; log "SUMMARIZE11 OK $(sha256sum "$J" | cut -c1-12)" ;;
  results11)
    PH=${1:?phase1b|final}; case $PH in phase1b) J=ckpt/exp11/phase1b/stats.json;; final) J=ckpt/exp11/stats.json;; *) refuse "phase";; esac
    log "RESULTS11 $TS phase=$PH"; CUDA_VISIBLE_DEVICES='' $PY "$E/orientation_cue_fairness_results_assets/make_results_md.py" --stats "$J" --out "$E/orientation_cue_fairness_results_${PH}.md" 2>&1 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES='' $PY "$E/orientation_cue_fairness_results_assets/make_results_html.py" --stats "$J" 2>&1 | tee -a "$LOG"; log "RESULTS11 OK" ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
