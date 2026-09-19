#!/bin/bash
# exp_06 post-training sequence (plan v4 §6, §10a A6; written 2026-09-16 while the pretraining ran).
# Run ONE step per invocation; every step fails closed and logs to
#   <record>/oriented_cyl_<YYYY-MM-DD_HH:MM:SS>_posttrain_<step>.log   (local time, SOP convention)
# Steps, in order:
#   status    the pretraining finalised (launcher dead, completion.json admissible, final -> attempt)
#   merge     merge exp06-prelaunch (c867e24, A4/A5) into main  [needs PEER_ACK=1 after messaging the peer]
#   refill    approvals: code (4 changed keys), reused identities, artifacts.epoch_012 -> commit -> full suite green
#   receipt   legacy A/B hash receipt -> reused.legacy_receipt -> commit
#             (then, before g1: Codex verification of both approvals commits — review_prompts/codex_approvals_posttrain_prompt.md —
#              and plan §10a A1's integrative `full` review over 77ef292..HEAD — review_prompts/codex_code_full_prompt.md)
#   merge2    merge the post-review fix round exp06-fullfix [FULLFIX=<approved tip>, PEER_ACK=1]   (2026-09-18)
#   refill2   approvals: the six code keys the fix round changed -> commit -> full suite green
#   g1        gate G1 mirror probe on GPU $GPU -> artifacts.gate_g1 -> commit     (exit 3 fail / 4 inconclusive: stop, escalate) [PEER_HOLD=1]
#   haa       HAA pipeline queue on GPU $GPU (dry run, then detached)              [args: jobs; default = the full 6.2 set]
#   breceipt  arm B's reconstructed exp_01 training receipt (full-review F2; once)
#   sim       H3 evaluations: arm C (cyl_or) and arm B (exp_01 cyl) x seeds 42..46 (detached queue; needs EXP06_F2_LANDED=1 after the fullfix merge)
#   summarize summariser (HAA, H1/H1b/H2) + comparer (H3) -> ckpt/exp06/{stats.json,summary.txt,h3.json,h3.txt}
# Nothing here edits a training-closure file; the merge waits for `status` to pass because the running
# finalizer re-hashes the working tree of the TRAINING_KEYS closure against the reviewed blobs.
set -euo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd "$R"
export PYTHONPATH=$R XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python; GPU=${GPU:-1}
E=$R/worklog/worklog_yixun/exp_06_oriented_cyl_claude
AP_REL=worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json
ROOT=ckpt/exp06/pretrain/xRIR_cylor_8_shot; ATT=$ROOT/attempt_20260916T161415
PRELAUNCH=c867e245186c2e72eb481ad9e515e54098369f11
TRAILER="Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
step=${1:-help}; shift || true; TS=$(date +%Y-%m-%d_%H:%M:%S); LOG=$E/oriented_cyl_${TS}_posttrain_${step}.log
log() { echo "$*" | tee -a "$LOG"; }
refuse() { log "REFUSE: $*"; exit 2; }
clean_outside_worklog() { if git status --porcelain | grep -v '^.. worklog/' | grep -q .; then git status --porcelain | grep -v '^.. worklog/' | tee -a "$LOG"; refuse "tree dirty outside worklog/"; fi; }
gpu_empty() { local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU") || refuse "GPU $GPU query failed"; [ -z "$apps" ] || refuse "GPU $GPU busy: $apps"; log "GPU $GPU empty"; }
# approvals producer gate, as tools.exp06_approvals_api spells it (6.4 matrix); binds the committed bytes at HEAD
producer_ok() { CUDA_VISIBLE_DEVICES='' $PY - "$R" "$AP_REL" "$1" "$ROOT/final/epoch_012.pth" <<'PY' 2>&1 | tee -a "$LOG"
import sys; repo, rel, producer, ckpt = sys.argv[1:5]
from tools import exp06_approvals_api as api, provenance
head = provenance.git_state(repo)['HEAD']
rooms = ('class_room', 'complex_room', 'dampened_room', 'hallway')
headings = {r: repo + '/ckpt/exp06/heading/' + r + '.json' for r in rooms}
try:
    if hasattr(api, 'enforce_producer'):   # post-review round (full-review F3): equality of the identities about to run
        kw = {}
        if producer in ('mirror_probe', 'haa_children', 'sim_eval'): kw['checkpoint'] = repo + '/' + ckpt
        if producer == 'haa_children': kw['headings'] = headings
        if producer == 'mirror_probe': kw['headings'] = {'hallway': headings['hallway']}
        rec = api.enforce_producer(producer, repo, head, approved_path=repo + '/' + rel, **kw)
        dev = list(rec['deviations'])
        print('enforce_producer', producer, '| keys', rec['keys_checked'], '| approvals at', str(rec['approvals'].get('committed_at'))[:12], '| deviations', dev)
    else:
        approved, identity = api.load_approved_digests(repo + '/' + rel, repo=repo, commit=head)
        dev = list(api.require_producer(approved, producer))
        print('require_producer (pre-F3 code)', producer, '| approvals committed at', str(identity.get('committed_at'))[:12], '| deviations', dev)
except (ValueError, OSError) as error:
    raise SystemExit('approvals gate refused producer {}: {}'.format(producer, error))
if dev: raise SystemExit('approvals do not admit producer ' + producer)
PY
}
status_ok() {   # the pretraining is finalised and promoted
    [ -d "$ATT" ] || { ls -d ${ATT}_ABORTED_* 2>/dev/null | tee -a "$LOG"; refuse "attempt dir missing (aborted?)"; }
    if [ -f "$ATT/launch.pid" ] && kill -0 "$(cat "$ATT/launch.pid")" 2>/dev/null; then
        log "launcher $(cat "$ATT/launch.pid") still alive; trainer: $(tail -c 300 "$E"/oriented_cyl_20260916T161415_train_full.log | tr '\r' '\n' | grep 'Train Epoch' | tail -1)"
        log "epochs done: $(wc -l < "$ATT/history.jsonl")"; return 3; fi
    [ -f "$ATT/completion.json" ] || refuse "no completion.json (launcher dead): $(ls "$ATT" | tr '\n' ' ')"
    $PY - "$ATT" "$ROOT" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys, hashlib
att, root = sys.argv[1:3]
c = json.load(open(att + '/completion.json'))
assert c.get('run_type') == 'full' and c.get('admissible_arm') is True, {k: c.get(k) for k in ('run_type', 'admissible_arm', 'child_exit')}
assert c.get('epochs') == 12, c.get('epochs')
link = os.readlink(root + '/final'); assert link == os.path.basename(att), link
sha = hashlib.sha256(open(att + '/epoch_012.pth', 'rb').read()).hexdigest()
assert sha == c['artifacts']['epoch_012.pth'] == c['checkpoint']['sha256'], (sha, c['artifacts'].get('epoch_012.pth'))
print('FINALISED admissible_arm=True epochs=12 final ->', link, 'epoch_012 sha256', sha)
PY
}
case $step in
  status) log "STATUS $TS HEAD=$(git rev-parse HEAD)"; status_ok || exit $?; log "STATUS OK" ;;
  merge)
    log "MERGE $TS HEAD=$(git rev-parse HEAD)"; status_ok >/dev/null || refuse "pretraining not finalised"
    [ "$(git branch --show-current)" = main ] || refuse "not on main"
    [ "${PEER_ACK:-0}" = 1 ] || refuse "message the peer session (xrir-code-25) first, then rerun with PEER_ACK=1"
    clean_outside_worklog
    [ "$(git rev-parse exp06-prelaunch)" = "$PRELAUNCH" ] || refuse "exp06-prelaunch moved from $PRELAUNCH"
    git merge --no-ff exp06-prelaunch -m "exp06: merge the pre-launch round (A4/A5) after the pretraining finalised

Finalised HAA diagnostic run types haa_smoke_train/haa_smoke_eval, launcher smoke budgets
EXP06_SMOKE_MAX_GB/EXP06_SMOKE_ALARM_S (6 GB / 300 s), load_job_spec single-read snapshot,
allow-listed smoke artefacts must be regular files of the smoke tree. Reviewed by Codex
(oriented_cyl_codex_code_prelaunch_review.md, _close_review.md: approve for the post-training
merge). Merged only now because the confirmatory run bound the previous finalizer/launcher/smoke bytes.

$TRAILER" 2>&1 | tee -a "$LOG"
    $PY -m py_compile tools/exp06_*.py && bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh && git diff --check HEAD~1 HEAD | tee -a "$LOG"
    log "MERGE OK: $(git rev-parse HEAD)  (next: refill)" ;;
  refill)
    log "REFILL $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    git merge-base --is-ancestor "$PRELAUNCH" HEAD || refuse "the pre-launch round is not merged"
    status_ok >/dev/null || refuse "pretraining not finalised"
    $PY - "$R" "$AP_REL" "$ATT" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, hashlib
repo, rel, att = sys.argv[1:4]
from tools import exp06_profiles as ap, exp06_approvals_api as api, provenance
from tools import exp04_profiles, exp05_profiles
path = repo + '/' + rel; head = provenance.git_state(repo)['HEAD']
a = json.load(open(path)); before = json.dumps(a['code'], sort_keys=True)
fresh = ap.compute_code_digests(repo, head)
changed = sorted(k for k in a['code'] if a['code'][k] != fresh.get(k))
expected = {'finalize': '5ef318e8', 'summarize_haa': '2ad63b6d', 'smoke': '00e3b91b', 'launch_sh': 'c64b60df'}
assert set(changed) <= set(expected), ('unexpected code drift', changed)
for k, pre in expected.items():
    assert fresh[k].startswith(pre), (k, fresh[k][:12], 'expected prefix', pre)   # the Codex-verified digests at c867e24
a['code'] = {k: fresh[k] for k in a['code']}
# reused identities (verified 2026-09-16 from the five exp_04 control k=0 runs, all seeds identical)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
runs = [json.load(open(repo + '/ckpt/yaw_aug/eval/control_k8_seed%d_k0/eval_manifest.json' % s)) for s in range(42, 47)]
ev = {m['source_closures']['entrypoint']['sha256'] for m in runs}; wr = {m['source_closures']['writer']['sha256'] for m in runs}
assert len(ev) == 1 and len(wr) == 1, (ev, wr)
p4 = str(exp04_profiles.APPROVED_DIGESTS_PATH); _, id5 = exp05_profiles.load_approved_digests()
a['reused'].update(evaluator_exp04=ev.pop(), writer_exp04=wr.pop(), approved_digests_exp04=sha(p4),
                   approved_digests_exp05=id5['sha256'],
                   exp02_stats_sha256=sha(repo + '/ckpt/sim2real/stats.json'),
                   exp02_summary_sha256=sha(repo + '/ckpt/sim2real/summary.txt'))
c = json.load(open(att + '/completion.json'))
a['artifacts']['epoch_012'] = {'epoch': 12, 'path': att + '/epoch_012.pth', 'sha256': c['checkpoint']['sha256']}
assert sha(repo + '/' + att + '/epoch_012.pth') == c['checkpoint']['sha256']
api.validate(a)
open(path, 'w').write(json.dumps(a, sort_keys=True, indent=2) + '\n')
print('code changed:', changed); print('reused:', {k: (v if isinstance(v, dict) else v[:12]) for k, v in a['reused'].items()})
print('artifacts.epoch_012:', a['artifacts']['epoch_012'])
PY
    git add "$AP_REL" && git commit -q -m "exp06: approvals re-fill at the post-training merge (code x4, reused identities, artifacts.epoch_012)

code: finalize/summarize_haa/smoke/launch_sh recomputed at HEAD (the digests Codex verified at c867e24).
reused: exp_04 evaluator/writer closures of the five reused control k=0 runs, exp_04/exp_05
approvals identities, exp_02 canonical stats/summary. artifacts.epoch_012: the finalised
attempt_20260916T161415 checkpoint (completion.json admissible_arm true).

$TRAILER" && log "committed $(git rev-parse HEAD)"
    # CUDA hidden: a GPU parity test otherwise lands on the default card, which belongs to the peer's training (2026-09-17 incident)
    log "SUITE (must be all green now):"; SUITE=$(CUDA_VISIBLE_DEVICES='' $PY -m pytest tests -q -p no:cacheprovider 2>&1 | tail -1); log "$SUITE"
    echo "$SUITE" | grep -qE "failed|error" && refuse "suite not green after the re-fill"
    log "REFILL OK (next: receipt, then Codex verification of both commits)" ;;
  merge2)   # the post-review fix round (full-review F2-F5): branch exp06-fullfix, tip pinned in FULLFIX
    log "MERGE2 $TS HEAD=$(git rev-parse HEAD)"; [ "$(git branch --show-current)" = main ] || refuse "not on main"
    [ "${PEER_ACK:-0}" = 1 ] || refuse "message the peer session first, then rerun with PEER_ACK=1"
    clean_outside_worklog
    [ -n "${FULLFIX:-}" ] || refuse "set FULLFIX=<40-hex tip of exp06-fullfix approved by Codex>"
    git merge-base --is-ancestor "$FULLFIX" exp06-fullfix || refuse "$FULLFIX is not on exp06-fullfix"
    git merge --no-ff "$FULLFIX" -m "exp06: merge the post-review fix round (full-review F2-F5) at $FULLFIX

Training-evidence bindings validated by the sim launcher and re-validated by the comparer
(tools/exp06_train_evidence.py), a reconstructed exp_01 training receipt for arm B
(tools/exp06_legacy_train_receipt.py), producer-side approval enforcement
(exp06_approvals_api.enforce_producer wired into the mirror probe, sim launcher, HAA pipeline
prepare_job and the HAA finalizer), arm B restricted to its registered routes, and the §7
convergence policy around H3. Reviewed by Codex (oriented_cyl_codex_code_fullfix_review.md).

$TRAILER" 2>&1 | tee -a "$LOG"
    $PY -m py_compile tools/exp06_*.py && bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh && git diff --check HEAD~1 HEAD | tee -a "$LOG"
    log "MERGE2 OK: $(git rev-parse HEAD)  (next: refill2)" ;;
  refill2)  # approvals code re-fill after merge2: exactly the six keys the Coder reported, with the digests Codex verified
    log "REFILL2 $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    [ -n "${FULLFIX:-}" ] && git merge-base --is-ancestor "$FULLFIX" HEAD || refuse "the fix round is not merged (FULLFIX unset or not an ancestor)"
    $PY - "$R" "$AP_REL" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
repo, rel = sys.argv[1:3]
from tools import exp06_profiles as ap, exp06_approvals_api as api, provenance
path = repo + '/' + rel; head = provenance.git_state(repo)['HEAD']
a = json.load(open(path)); fresh = ap.compute_code_digests(repo, head)
changed = sorted(k for k in a['code'] if a['code'][k] != fresh.get(k))
expected = {'compare': 'ea97c6d7', 'eval_launch': '7314166e', 'finalize': '1edbaf2c',   # exp06-fullfix 31d2933 (fix cycle R1-R5)
            'haa_pipeline_sh': '2ad3e562', 'mirror_probe': 'ac503c0f', 'summarize_haa': '3f497fc4'}
assert set(changed) == set(expected), ('changed keys differ from the reviewed set', changed)
for k, pre in expected.items():
    assert fresh[k].startswith(pre), (k, fresh[k][:12], 'expected prefix', pre)
a['code'] = {k: fresh[k] for k in a['code']}
api.validate(a); open(path, 'w').write(json.dumps(a, sort_keys=True, indent=2) + '\n')
print('code changed:', changed); print({k: fresh[k][:16] for k in changed})
PY
    git add "$AP_REL" && git commit -q -m "exp06: approvals re-fill after the fix-round merge (code x6)

compare, eval_launch, finalize, haa_pipeline_sh, mirror_probe, summarize_haa recomputed at HEAD
(the digests the Coder reported and Codex verified on exp06-fullfix); 14 keys unchanged.

$TRAILER" && log "committed $(git rev-parse HEAD)"
    log "SUITE (must be all green now):"; SUITE=$(CUDA_VISIBLE_DEVICES='' $PY -m pytest tests -q -p no:cacheprovider 2>&1 | tail -1); log "$SUITE"
    echo "$SUITE" | grep -qE "failed|error" && refuse "suite not green after refill2"
    log "REFILL2 OK (next: Codex verification of the re-fill, then breceipt, g1 [PEER_HOLD=1], haa, sim [EXP06_F2_LANDED=1])" ;;
  merge3)   # a later reviewed tip of exp06-fullfix (e.g. the R3 fix cycle): FULLFIX=<approved tip>, MERGE_MSG=<one line>
    log "MERGE3 $TS HEAD=$(git rev-parse HEAD)"; [ "$(git branch --show-current)" = main ] || refuse "not on main"
    [ "${PEER_ACK:-0}" = 1 ] || refuse "message the peer session first, then rerun with PEER_ACK=1"
    clean_outside_worklog
    [ -n "${FULLFIX:-}" ] && [ -n "${MERGE_MSG:-}" ] || refuse "set FULLFIX=<40-hex approved tip> and MERGE_MSG=<subject>"
    git merge-base --is-ancestor "$FULLFIX" exp06-fullfix || refuse "$FULLFIX is not on exp06-fullfix"
    git merge --no-ff "$FULLFIX" -m "$MERGE_MSG

Reviewed by Codex (see the record's oriented_cyl_codex_code_*_review.md). Merged at $FULLFIX.

$TRAILER" 2>&1 | tee -a "$LOG"
    $PY -m py_compile tools/exp06_*.py && bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh && git diff --check HEAD~1 HEAD | tee -a "$LOG"
    log "MERGE3 OK: $(git rev-parse HEAD)  (next: refill3 with EXP06_REFILL_EXPECT)" ;;
  refill3)  # approvals code re-fill with an explicit expected set: EXP06_REFILL_EXPECT="key:prefix8,key:prefix8,..."
    log "REFILL3 $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    [ -n "${EXP06_REFILL_EXPECT:-}" ] || refuse "set EXP06_REFILL_EXPECT=key:prefix8,..."
    EXP06_REFILL_EXPECT="$EXP06_REFILL_EXPECT" $PY - "$R" "$AP_REL" <<'PY' 2>&1 | tee -a "$LOG"
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
    git add "$AP_REL" && git commit -q -m "exp06: approvals re-fill (${EXP06_REFILL_EXPECT//:*,/, } ) at $(git rev-parse --short HEAD)

Keys recomputed at HEAD after a reviewed merge of exp06-fullfix; the other keys unchanged.

$TRAILER" && log "committed $(git rev-parse HEAD)"
    log "REFILL3 OK (suite: run separately to a log file with -rf; then breceipt / sim)" ;;
  receipt)
    log "RECEIPT $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog; producer_ok legacy_receipt
    [ ! -e ckpt/exp06/legacy_receipt.json ] || refuse "ckpt/exp06/legacy_receipt.json exists"
    CUDA_VISIBLE_DEVICES='' $PY tools/exp06_summarize_haa.py --write-legacy-receipt ckpt/exp06/legacy_receipt.json --legacy-root ckpt/sim2real --approved "$AP_REL" --approved-commit "$(git rev-parse HEAD)" 2>&1 | tee -a "$LOG"
    $PY - "$R" "$AP_REL" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, hashlib
repo, rel = sys.argv[1:3]; from tools import exp06_approvals_api as api
path = repo + '/' + rel; a = json.load(open(path))
d = hashlib.sha256(open(repo + '/ckpt/exp06/legacy_receipt.json', 'rb').read()).hexdigest()
a['reused']['legacy_receipt'] = {'path': 'ckpt/exp06/legacy_receipt.json', 'sha256': d}
api.validate(a); open(path, 'w').write(json.dumps(a, sort_keys=True, indent=2) + '\n'); print('legacy_receipt', d)
PY
    git add "$AP_REL" && git commit -q -m "exp06: approve the legacy A/B hash receipt (reused.legacy_receipt)

$TRAILER" && log "committed $(git rev-parse HEAD)"
    log "RECEIPT OK (next: Codex verification of the two approvals commits, then g1)" ;;
  g1)
    log "G1 $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog; producer_ok mirror_probe
    [ ! -e ckpt/exp06/gate_g1.json ] || refuse "ckpt/exp06/gate_g1.json exists (exclusive create)"
    # The probe refuses if HEAD or the working-tree diff changes during its ~90-s run: no edits here, and the peer holds commits (PEER_HOLD=1 = pinged "G1 running").
    [ "${PEER_HOLD:-0}" = 1 ] || refuse "ping the peer 'G1 running, hold commits ~3 min' first, then rerun with PEER_HOLD=1"
    set +e
    CUDA_VISIBLE_DEVICES=$GPU $PY tools/exp06_mirror_probe.py --cylor-checkpoint "$ROOT/final/epoch_012.pth" --heading-json ckpt/exp06/heading/hallway.json --out ckpt/exp06/gate_g1.json --device cuda --approved "$AP_REL" --approved-commit "$(git rev-parse HEAD)" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}; set -e
    log "G1 exit $rc (0 pass / 3 fail / 4 inconclusive / 1 refusal)"
    [ "$rc" -eq 0 ] || { log "G1 did not pass: STOP, record, escalate to Yixun (plan §6.1, §11 fallbacks)"; exit "$rc"; }
    $PY - "$R" "$AP_REL" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, hashlib
repo, rel = sys.argv[1:3]; from tools import exp06_approvals_api as api
path = repo + '/' + rel; a = json.load(open(path))
g = json.load(open(repo + '/ckpt/exp06/gate_g1.json')); assert g['decision']['outcome'] == 'pass', g['decision']
a['artifacts']['gate_g1'] = hashlib.sha256(open(repo + '/ckpt/exp06/gate_g1.json', 'rb').read()).hexdigest()
api.validate(a); open(path, 'w').write(json.dumps(a, sort_keys=True, indent=2) + '\n'); print('gate_g1', a['artifacts']['gate_g1'], g['decision'].get('values'))
PY
    git add "$AP_REL" && git commit -q -m "exp06: approve the G1 gate artifact (pass)

$TRAILER" && log "committed $(git rev-parse HEAD)"
    log "G1 OK (next: haa)" ;;
  haa)
    JOBS=${*:-"cyl_or:0 cyl_or:1 cyl_or:2 control_hf:0 control_hf:1 control_hf:2 cyl_hf:0 cyl_hf:1 cyl_hf:2 zeroshot"}
    log "HAA $TS HEAD=$(git rev-parse HEAD) GPU=$GPU jobs=$JOBS"; clean_outside_worklog; producer_ok haa_children
    $PY -c "import json,sys; g=json.load(open('ckpt/exp06/gate_g1.json')); assert g['decision']['outcome']=='pass', g['decision']; print('G1 pass')" | tee -a "$LOG"
    gpu_empty
    export EXP06_HEADING_DIR=ckpt/exp06/heading EXP06_CYLOR_CKPT=$ROOT/final/epoch_012.pth
    log "DRY RUN:"; bash tools/exp06_haa_pipeline.sh "$GPU" $JOBS --dry-run 2>&1 | grep -E "EXP06_HAA_PIPELINE|^JOB|REFUS|refus" | tee -a "$LOG"
    PIDF=$E/oriented_cyl_${TS}_posttrain_haa_gpu${GPU}.pid
    nohup setsid bash tools/exp06_haa_pipeline.sh "$GPU" $JOBS < /dev/null >> "$LOG" 2>&1 &
    echo $! > "$PIDF"; sleep 30
    kill -0 "$(cat "$PIDF")" 2>/dev/null && log "HAA queue launched: pid $(cat "$PIDF"), log $(basename "$LOG")" || { tail -20 "$LOG"; refuse "HAA queue exited early"; } ;;
  breceipt)  # full-review F2: arm B's reconstructed training receipt over exp_01's retained artefacts (written once; launcher and comparer re-validate it)
    log "BRECEIPT $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog
    [ ! -e ckpt/exp06/legacy_train_receipt_cyl.json ] || refuse "ckpt/exp06/legacy_train_receipt_cyl.json exists (exclusive create)"
    CUDA_VISIBLE_DEVICES='' $PY tools/exp06_legacy_train_receipt.py --train-dir ckpt/xRIR_cyl_8_shot --checkpoint ckpt/xRIR_cyl_8_shot/epoch_12.pth --out ckpt/exp06/legacy_train_receipt_cyl.json 2>&1 | tee -a "$LOG"
    test -f ckpt/exp06/legacy_train_receipt_cyl.json || refuse "no receipt written"
    log "BRECEIPT OK (sha $(sha256sum ckpt/exp06/legacy_train_receipt_cyl.json | cut -c1-16)...)" ;;
  sim)
    log "SIM $TS HEAD=$(git rev-parse HEAD) GPU=$GPU"; clean_outside_worklog; producer_ok sim_eval; gpu_empty
    test -f ckpt/exp06/legacy_train_receipt_cyl.json || refuse "run breceipt first (arm B's training receipt)"
    # Codex full review 2026-09-17 F1: execute_run creates the run dir without parents → create them here (exclusive run dirs stay the launcher's).
    mkdir -p ckpt/exp06/sim_eval/cyl_or ckpt/exp06/sim_eval/cyl
    # F2: the comparer requires train_manifest + train_completion bindings on every exp_06-route run; until the post-review Coder round
    # lands (validated bindings for arm C, historical receipt for arm B) this step refuses rather than producing inadmissible runs.
    [ "${EXP06_F2_LANDED:-0}" = 1 ] || refuse "sim step blocked by full-review F2 (training-evidence bindings) until the Coder round is merged; then set EXP06_F2_LANDED=1 and add the bindings below"
    SHA=$(git rev-parse HEAD); Q=$E/oriented_cyl_${TS}_posttrain_sim_queue.sh
    { echo "#!/bin/bash"; echo "set -uo pipefail; cd $R; export PYTHONPATH=$R XRIR_DATA_PATH=$XRIR_DATA_PATH PYTHONDONTWRITEBYTECODE=1"
      for arm in "cyl_or cylindrical_oriented arm $ROOT/final/epoch_012.pth $ATT/completion.json $ATT/provenance.json" "cyl cylindrical baseline ckpt/xRIR_cyl_8_shot/epoch_12.pth ckpt/exp06/legacy_train_receipt_cyl.json ckpt/xRIR_cyl_8_shot/args.json"; do
        set -- $arm; name=$1; bb=$2; role=$3; ck=$4; tc=$5; tm=$6   # F2: train_completion / train_manifest bindings
        for s in 42 43 44 45 46; do
          m=ckpt/yaw_aug/reference_manifest_k8_seed$s.json; h=$($PY -c "import json,sys; from tools.reference_manifest import manifest_hash; print(manifest_hash(json.load(open(sys.argv[1]))))" "$m")
          echo "[ -f ckpt/exp06/sim_eval/$name/seed$s/completion.json ] && echo SKIP $name $s || $PY tools/exp06_eval_launch.py --backbone $bb --checkpoint $ck --checkpoint-role $role --checkpoint-epoch 12 --manifest $m --manifest-hash $h --out-dir ckpt/exp06/sim_eval/$name/seed$s --log-dir $E --data-root $XRIR_DATA_PATH --run-label ${name}_seed$s --reviewed-commit $SHA --num-shot 8 --conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --batch-size 16 --num-workers 6 --max-samples 0 --gl-seed $s --threads 4 --log-interval 10 --decomposition-batches 0 --gpu $GPU --bind-input train_completion=$tc --bind-input train_manifest=$tm --approved $AP_REL --approved-commit $SHA || echo FAILED $name $s"
        done; done; echo 'echo SIM_QUEUE_DONE'; } > "$Q"
    log "queue script $Q ($(grep -c exp06_eval_launch "$Q") evaluations; argv = exp_04's k=0 argv + role/epoch)"
    PIDF=$E/oriented_cyl_${TS}_posttrain_sim_gpu${GPU}.pid
    nohup setsid bash "$Q" < /dev/null >> "$LOG" 2>&1 &
    echo $! > "$PIDF"; sleep 20; kill -0 "$(cat "$PIDF")" 2>/dev/null && log "SIM queue launched: pid $(cat "$PIDF")" || { tail -20 "$LOG"; refuse "sim queue exited early"; } ;;
  summarize)
    log "SUMMARIZE $TS HEAD=$(git rev-parse HEAD)"; clean_outside_worklog; producer_ok summarize_haa; producer_ok compare
    SHA=$(git rev-parse HEAD)
    CUDA_VISIBLE_DEVICES='' $PY tools/exp06_summarize_haa.py --approved "$AP_REL" --approved-commit "$SHA" --gate-g1 ckpt/exp06/gate_g1.json --json ckpt/exp06/stats.json --summary ckpt/exp06/summary.txt 2>&1 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES='' $PY tools/exp06_compare.py --runs-a ckpt/yaw_aug/eval/control_k8_seed{42,43,44,45,46}_k0 --runs-b ckpt/exp06/sim_eval/cyl/seed{42,43,44,45,46} --runs-c ckpt/exp06/sim_eval/cyl_or/seed{42,43,44,45,46} --approved "$AP_REL" --approved-commit "$SHA" --json ckpt/exp06/h3.json --summary ckpt/exp06/h3.txt 2>&1 | tee -a "$LOG"
    log "SUMMARIZE OK" ;;
  *) echo "steps: status | merge | refill | receipt | merge2 | refill2 | merge3 | refill3 | g1 | haa [jobs] | breceipt | sim | summarize"; exit 1 ;;
esac
