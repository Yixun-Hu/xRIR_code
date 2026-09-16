#!/bin/bash
# exp_06 launch day (GPU 1, after the peer's hand-over). Run ONE step per invocation; each step fails closed.
# Steps: gate | smoke | haa-smoke | probe | full. Logs: <record>/oriented_cyl_<YYYY-MM-DD_HH:MM:SS>_launchday_<step>.log (local time, SOP convention).
set -euo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
export PYTHONPATH=$R XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python; SHA=$(git rev-parse HEAD); GPU=${GPU:-1}
E=$R/worklog/worklog_yixun/exp_06_oriented_cyl_claude; APPROVED=worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json
step=${1:-help}; TS=$(date +%Y-%m-%d_%H:%M:%S); LOG=$E/oriented_cyl_${TS}_launchday_${step}.log
log() { echo "$*" | tee -a "$LOG"; }
case $step in
  gate)
    log "GATE $TS HEAD=$SHA GPU=$GPU"
    if git status --porcelain | grep -v '^.. worklog/' | grep -q .; then log "REFUSE: tree dirty outside worklog/"; git status --porcelain | grep -v '^.. worklog/' | tee -a "$LOG"; exit 2; fi
    if ! CUDA_VISIBLE_DEVICES='' $PY - "$R" "$SHA" "$APPROVED" <<'PY' 2>&1 | tee -a "$LOG"; then log "REFUSE: approvals do not admit HEAD (see above)"; exit 2; fi
import sys; repo, head, rel = sys.argv[1:4]
from tools import exp06_profiles as ap
a, i = ap.load_approved_digests(repo + '/' + rel, repo, head); f = ap.compute_code_digests(repo, head)
mism = [k for k in ap.TRAINING_KEYS if a['code'][k] != f[k]]; dev = ap.require(a, ap.TRAINING_KEYS)
print('approvals bound at', i['committed_at'][:12], '| TRAINING_KEYS mismatches', mism, '| require', dev)
if mism or dev: raise SystemExit('approvals do not admit HEAD')
PY
    if ! APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" 2>>"$LOG"); then log "REFUSE: GPU $GPU compute-process query failed"; exit 2; fi
    if [ -n "$APPS" ]; then log "REFUSE: GPU $GPU busy: $APPS"; exit 2; fi
    if ! MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i "$GPU" 2>>"$LOG"); then log "REFUSE: GPU $GPU memory query failed"; exit 2; fi
    log "GPU $GPU free: $MEM"
    # canonical preflight (HEAD == reviewed commit, clean tree, approvals, live launches, GPU) as the launcher's full mode will see it
    if ! $PY tools/exp06_finalize.py preflight --mode full --gpu "$GPU" --reviewed-commit "$SHA" --attempt-root ckpt/exp06/pretrain/xRIR_cylor_8_shot --attempt-root ckpt/exp06/_smoke --approved "$APPROVED" 2>&1 | tee -a "$LOG"; then log "REFUSE: canonical preflight failed"; exit 2; fi
    log "GATE OK" ;;
  smoke)   # rung 4 (a)(b) + fixture: launcher-owned lifecycle, finalised smoke completions (passed: true)
    bash tools/exp06_launch.sh smoke --gpu "$GPU" --reviewed-commit "$SHA" 2>&1 | tee -a "$LOG" ;;
  haa-smoke)  # rung 4 (c)(d): UNFINALISED diagnostics (plan §10a A4) — runner receipts + provenance only, never admissible
    FX=ckpt/exp06/_smoke/fixture_cylor.pth; [ -f "$FX" ] || $PY tools/exp06_smoke.py --make-fixture "$FX" 2>&1 | tee -a "$LOG"
    H1=ckpt/exp06/_smoke/h1_$TS; E1=ckpt/exp06/_smoke/e1_$TS; mkdir "$H1" "$E1"
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY tools/exp06_smoke.py --entry exp06_haa_finetune --receipt "$H1/receipt.json" --run-type smoke --provenance-out "$H1/provenance.json" --approved "$APPROVED" --reviewed-commit "$SHA" --alarm-seconds 300 --max-gb 3 -- --backbone cylindrical_oriented --init "$FX" --rooms class_room --heading-json-dir ckpt/exp06/heading --save-dir "$H1/run" --epochs 2 --val-every 1 --batch-size 4 --val-batch-size 4 --seed 0 2>&1 | tee -a "$LOG"
    $PY - "$H1/receipt.json" <<'PY'
import json, sys; r = json.load(open(sys.argv[1])); assert r.get('outcome') == 'ok' and r.get('exit_status') == 0, r; print('HAA finetune smoke receipt OK', r.get('wall_s'), r.get('peak_bytes'))
PY
    test -f "$H1/run/best.pth" || { log "REFUSE: fine-tune smoke wrote no best.pth"; exit 2; }
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY tools/exp06_smoke.py --entry exp06_haa_eval --receipt "$E1/receipt.json" --run-type smoke --provenance-out "$E1/provenance.json" --approved "$APPROVED" --reviewed-commit "$SHA" --alarm-seconds 300 --max-gb 3 -- --backbone cylindrical_oriented --checkpoint "$H1/run/best.pth" --heading-json-dir ckpt/exp06/heading --rooms hallway --max-samples 4 --save-dir "$E1/run" --seed 0 2>&1 | tee -a "$LOG"
    $PY - "$E1/receipt.json" <<'PY'
import json, sys; r = json.load(open(sys.argv[1])); assert r.get('outcome') == 'ok' and r.get('exit_status') == 0, r; print('HAA eval smoke receipt OK', r.get('wall_s'), r.get('peak_bytes'))
PY
    log "HAA SMOKES OK (diagnostic, unfinalised per A4)" ;;
  probe)   # rung 5: launcher-owned; receipt under the attempt root
    bash tools/exp06_launch.sh probe --gpu "$GPU" --reviewed-commit "$SHA" 2>&1 | tee -a "$LOG" ;;
  full)    # rung 7: the launcher (which supervises child + sink + finalizer) is itself detached and survives this shell
    PIDF=$E/oriented_cyl_${TS}_launchday_full.pid
    nohup setsid bash tools/exp06_launch.sh full --gpu "$GPU" --reviewed-commit "$SHA" < /dev/null >> "$LOG" 2>&1 &
    echo $! > "$PIDF"; sleep 20
    if kill -0 "$(cat "$PIDF")" 2>/dev/null; then log "FULL launched: launcher pid $(cat "$PIDF"), log $(basename "$LOG")"; grep -E "EXP06_LAUNCH|preflight|REFUSE|ABORT" "$LOG" | tail -5; else log "REFUSE: launcher exited early"; tail -20 "$LOG"; exit 2; fi ;;
  *) echo "steps: gate | smoke | haa-smoke | probe | full"; exit 1;;
esac
