#!/bin/bash
# exp_09 yawaug_haa runbook (2026-09-19). One step per invocation; every step fails closed; logs to
#   <record>/yawaug_haa_<YYYY-MM-DD_HH:MM:SS>_<step>.log
# Steps: merge (FULLFIX=<approved exp09-yawaug tip>, PEER_ACK=1, MERGE_MSG) | refill (EXP06_REFILL_EXPECT=key:prefix8,...) |
#        suite | dryrun | smoke | launch [gpu] | summarize | results
set -euo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
E=$R/worklog/worklog_yixun/exp_09_yawaug_haa_claude
E06=$R/worklog/worklog_yixun/exp_06_oriented_cyl_claude
AP=worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json
TRAILER="Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
YAWAUG_CKPT=ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth
OUT=ckpt/exp09/sim2real
export EXP06_HAA_OUT=$OUT EXP06_HAA_RECORD=$E EXP09_YAWAUG_CKPT=$YAWAUG_CKPT EXP06_HEADING_DIR=ckpt/exp06/heading
step=${1:-help}; shift || true; TS=$(date +%Y-%m-%d_%H:%M:%S); LOG=$E/yawaug_haa_${TS}_${step}.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
refuse() { log "REFUSE: $*"; exit 2; }
clean_outside_worklog() { if git status --porcelain | grep -v '^.. worklog/' | grep -q .; then git status --porcelain | grep -v '^.. worklog/' | tee -a "$LOG"; refuse "tree dirty outside worklog/"; fi; }
gpu_empty() { local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$1") || refuse "GPU $1 query failed"; [ -z "$apps" ] || refuse "GPU $1 busy: $apps"; log "GPU $1 empty"; }
case $step in
  merge)
    log "MERGE $TS HEAD=$(git rev-parse HEAD)"; [ "$(git branch --show-current)" = main ] || refuse "not on main"
    [ "${PEER_ACK:-0}" = 1 ] || refuse "message the peer first, then PEER_ACK=1"
    clean_outside_worklog; [ -n "${FULLFIX:-}" ] && [ -n "${MERGE_MSG:-}" ] || refuse "set FULLFIX=<40-hex approved tip> MERGE_MSG=<subject>"
    git merge-base --is-ancestor "$FULLFIX" exp09-yawaug || refuse "$FULLFIX is not on exp09-yawaug"
    git merge --no-ff "$FULLFIX" -m "$MERGE_MSG

Reviewed by Codex (worklog/worklog_yixun/exp_09_yawaug_haa_claude/yawaug_haa_codex_code_*_review.md). Merged at $FULLFIX.

$TRAILER" 2>&1 | tee -a "$LOG"
    $PY -m py_compile tools/exp06_*.py && bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh && git diff --check HEAD~1 HEAD | tee -a "$LOG"
    log "MERGE OK: $(git rev-parse HEAD)" ;;
  refill)
    log "REFILL $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    [ -n "${EXP06_REFILL_EXPECT:-}" ] || refuse "set EXP06_REFILL_EXPECT=key:prefix8,..."
    EXP06_REFILL_EXPECT="$EXP06_REFILL_EXPECT" CUDA_VISIBLE_DEVICES='' $PY - "$R" "$AP" <<'PY' 2>&1 | tee -a "$LOG"
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
    git add "$AP" && git commit -q -m "exp06/exp09: approvals re-fill after the exp_09 merge (${EXP06_REFILL_EXPECT//:*,/, })

Keys recomputed at HEAD after the reviewed merge of exp09-yawaug; the other keys unchanged.

$TRAILER" && log "committed $(git rev-parse HEAD)"; log "REFILL OK" ;;
  suite)   # CPU-only, to a log; the only permitted failure is the peer's exp_07 guard
    log "SUITE $TS HEAD=$(git rev-parse HEAD)"
    CUDA_VISIBLE_DEVICES='' $PY -m pytest tests -q -p no:cacheprovider -rf --durations=5 > "$E/yawaug_haa_${TS}_suite_full_cpu.log" 2>&1 || true
    tail -1 "$E/yawaug_haa_${TS}_suite_full_cpu.log" | tee -a "$LOG"; grep -E "^FAILED|^ERROR" "$E/yawaug_haa_${TS}_suite_full_cpu.log" | tee -a "$LOG" || true ;;
  dryrun)
    log "DRYRUN $TS HEAD=$(git rev-parse HEAD)"
    bash tools/exp06_haa_pipeline.sh "${1:-1}" yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot --dry-run 2>&1 | tee -a "$LOG" | grep -cE "^RUN|^JOB" ;;
  smoke)   # bounded room-frame readback on the card: 2 epochs stage-1-style fine-tune of yawaug on class_room + 4-sample eval (diagnostic, unfinalised)
    GPU=${1:-1}; log "SMOKE $TS GPU=$GPU HEAD=$(git rev-parse HEAD)"; gpu_empty "$GPU"
    S=ckpt/exp09/_smoke/$TS; mkdir -p "$S"
    CUDA_VISIBLE_DEVICES=$GPU $PY tools/exp06_smoke.py --entry exp06_haa_finetune --receipt "$S/ft_receipt.json" --run-type smoke --exploratory --provenance-out "$S/ft_provenance.json" --approved "$AP" --reviewed-commit "$(git rev-parse HEAD)" --alarm-seconds 600 --max-gb 40 -- --backbone simple --init "$YAWAUG_CKPT" --rooms class_room --save-dir "$S/ft" --epochs 2 --val-every 1 --batch-size 4 --val-batch-size 4 --seed 0 2>&1 | grep -vE "UserWarning|warnings.warn" | tee -a "$LOG"
    test -f "$S/ft/best.pth" || refuse "smoke fine-tune wrote no best.pth"
    CUDA_VISIBLE_DEVICES=$GPU $PY tools/exp06_smoke.py --entry exp06_haa_eval --receipt "$S/ev_receipt.json" --run-type smoke --exploratory --provenance-out "$S/ev_provenance.json" --approved "$AP" --reviewed-commit "$(git rev-parse HEAD)" --alarm-seconds 300 --max-gb 40 -- --backbone simple --checkpoint "$S/ft/best.pth" --rooms hallway --max-samples 4 --save-dir "$S/ev" --seed 0 2>&1 | grep -vE "UserWarning|warnings.warn" | tee -a "$LOG"
    $PY -c "import json,sys; [print(p, json.load(open(p)).get('outcome'), json.load(open(p)).get('exit_status')) for p in sys.argv[1:]]" "$S/ft_receipt.json" "$S/ev_receipt.json" | tee -a "$LOG"
    $PY -c "import json; a=json.load(open('$S/ev/args.json')); assert a.get('frame')=='room' and not a.get('heading'), a; print('room frame, no heading: ok')" | tee -a "$LOG"
    log "SMOKE OK (diagnostic)" ;;
  launch)
    GPU=${1:-1}; log "LAUNCH $TS GPU=$GPU HEAD=$(git rev-parse HEAD)"; clean_outside_worklog; gpu_empty "$GPU"
    [ ! -d "$OUT/yawaug" ] || refuse "$OUT/yawaug exists"
    df -h / | tail -1 | awk '{print "free on /:", $4}' | tee -a "$LOG"
    bash tools/exp06_haa_pipeline.sh "$GPU" yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot --dry-run 2>&1 | grep -E "EXP06_HAA_PIPELINE|^JOB|REFUS|refus" | tee -a "$LOG"
    PIDF=$E/yawaug_haa_${TS}_haa_gpu${GPU}.pid
    nohup setsid bash tools/exp06_haa_pipeline.sh "$GPU" yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot < /dev/null >> "$LOG" 2>&1 &
    echo $! > "$PIDF"; sleep 30
    kill -0 "$(cat "$PIDF")" 2>/dev/null && log "HAA queue launched: pid $(cat "$PIDF")" || { tail -20 "$LOG"; refuse "queue exited early"; } ;;
  summarize)
    log "SUMMARIZE $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    [ "$(find $OUT/yawaug -name completion.json | wc -l)" -eq 35 ] || refuse "yawaug has $(find $OUT/yawaug -name completion.json | wc -l)/35 completions (31 children + 4 jobs)"
    [ ! -e ckpt/exp09/stats.json ] && [ ! -e ckpt/exp09/summary.txt ] || refuse "outputs exist"
    CUDA_VISIBLE_DEVICES='' $PY tools/exp06_summarize_haa.py --experiment exp09 --exp09-root $OUT --approved "$AP" --approved-commit "$(git rev-parse HEAD)" --gate-g1 ckpt/exp06/gate_g1.json --json ckpt/exp09/stats.json --summary ckpt/exp09/summary.txt 2>&1 | grep -vE "UserWarning|warnings.warn" | tee -a "$LOG"
    test -f ckpt/exp09/stats.json || refuse "no stats.json"; log "SUMMARIZE OK $(sha256sum ckpt/exp09/stats.json | cut -c1-12)" ;;
  results)
    log "RESULTS $TS"; CUDA_VISIBLE_DEVICES='' $PY "$E/yawaug_haa_results_assets/make_results_md.py" --stats ckpt/exp09/stats.json 2>&1 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES='' $PY "$E/yawaug_haa_results_assets/make_results_html.py" --stats ckpt/exp09/stats.json 2>&1 | tee -a "$LOG"; log "RESULTS OK" ;;
  *) echo "steps: merge | refill | suite | dryrun [gpu] | smoke [gpu] | launch [gpu] | summarize | results"; exit 1 ;;
esac
