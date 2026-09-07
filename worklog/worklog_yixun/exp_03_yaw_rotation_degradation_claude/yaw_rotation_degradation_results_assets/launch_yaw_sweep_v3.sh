#!/bin/bash
# exp_03 launcher v3 (Planner). Usage: launch_yaw_sweep_v3.sh gate|sweep|faulttest
# STATUS: NOT CERTIFIED FOR A CLEAN RE-RUN. The 2026-09-06 gate and sweep ran under launcher v2 with post-hoc binding; this
# script has only been exercised through refusal self-tests and stub fault tests (see the record). Known gaps: no isolated
# end-to-end gate -> decision -> binding -> sweep integration run has been executed; the decision command it prints uses the
# working-tree producer (acceptance recomputes with the producer pinned at the commit that took the historical decision, so a
# future decision must be pinned the same way and its retained copies created by hand); the evaluator_changed and
# checkpoint_changed fault injections force the abort branches directly rather than through the real guards; the
# rename-failure branch is untested. These are residual limitations recorded in the analysis; the launcher is not part of the
# experiment's evidence path.
# Clean workflow: `gate` runs the ten band-v3 measurements, binds their sidecars BEFORE any decision exists
# (bind_provenance.py --set gate: closure, environment, data identity, logs; gate inputs recorded only if present), then prints
# the decision command and the post-decision binding command (bind_provenance.py --set gate again adds the decision artefacts to
# the report; bindings are immutable, only that addition is accepted). `sweep` validates the decision through
# check_sweep_acceptance.gate_ok (with recomputation) before starting; a future sweep's acceptance must be run with
# --launcher-head <that HEAD> because the checker pins the historical 2026-09-06 sweep HEAD by default.
# `faulttest` exercises the abort paths with the retained stub evaluator eval_stub.py (EVALUATOR_OVERRIDE) and fault injection
# (FAULT_INJECT): evaluator exit 7, silent exit 0 (provenance failure), evaluator_changed, checkpoint_changed and missing_log;
# only logs created by this invocation are counted; nothing else is touched.
# Written after the Codex round-1 review of v2, revised after round 2 (findings 5-6) and round 3 (findings 2, 6). NOT used for the 2026-09-06 gate/sweep
# (those ran under v2); kept, self-tested, for re-launches.
# Guarantees: the reviewed *source closure* (12 files: the evaluator + its 11 repo-local imports, digest pinned below) asserted at preflight and
# before/after every run through bind_provenance.closure_record; the gate decision validated by check_sweep_acceptance.gate_ok
# (band v3, ten runs, bound sidecars, hash-bound decision) before a sweep; checkpoint digests asserted
# (sweep: against the gate sidecars; gate: against the digests taken at preflight); every input file required at preflight;
# provenance written atomically, read back, and any failure after the evaluator exits is routed through one abort handler that
# renames the run log *_ABORTED_<reason>_exit<rc>.log and records a non-zero code; gate mode runs the complete v3 measurement set
# (10 runs, per-run batch / TF32 / GL-seed / manifest) and reports GATE_MEASUREMENTS_DONE -- the pass decision is
# `tools/summarize_yaw.py --mode k0-gate` (command printed at the end); sweep mode refuses unless that decision is recorded with
# gate_pass true, on the seed-0 manifest, with a binding summary digest; output directories are reserved atomically before either
# GPU chain starts; the status log lives in the record folder; temp files are trapped; all scalars quoted, arguments as arrays.
set -u -o pipefail
REVIEWED_COMMIT=62c9107b4150e44c4ac410ff4cab359c1e71cc10
REVIEWED_EVAL_SHA=82a53bea3cd74b4805ea1680fb457fec27abf02e3d24a0997d21a5b2e7f21171
REVIEWED_CLOSURE_SHA=5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
ASSETS=/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets
REPO=/home/yixunhu/codespace/xRIR_code
cd "$REPO" || { echo "preflight: cannot cd to $REPO" >&2; exit 2; }
export PYTHONPATH="${PYTHONPATH:-}:$REPO"; export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
P="$HOME/miniconda3/envs/xRIR/bin/python"; [ -x "$P" ] || { echo "preflight: python not found at $P" >&2; exit 2; }
E="$REPO/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude"; [ -d "$E" ] && [ -w "$E" ] || { echo "preflight: record folder missing or not writable" >&2; exit 2; }
MAN=ckpt/yaw_rotation/reference_manifest.json;       HASH=47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d
MAN1=ckpt/yaw_rotation/reference_manifest_seed1.json; HASH1=6ca8164e5727d5f4db84569d5effa840b2cb6e6f77819a69bbb37bb30c6ab11e
MAN2=ckpt/yaw_rotation/reference_manifest_seed2.json; HASH2=757e5a966ee2d069a07accecbc43e46bd564ea7c788a28c219773df64e986234
case "${1:-}" in gate|sweep|faulttest) MODE=$1 ;; *) echo "usage: $0 gate|sweep|faulttest" >&2; exit 2 ;; esac
EVALUATOR_OVERRIDE="${EVALUATOR_OVERRIDE:-}"   # faulttest only: a stub script run instead of eval_yaw_rotation.py
FAULT_INJECT="${FAULT_INJECT:-}"               # faulttest only: evaluator_changed | checkpoint_changed | missing_log (forces that abort branch after the stub returns)
LAUNCHER_HEAD=$(git rev-parse HEAD) && [ -n "$LAUNCHER_HEAD" ] || { echo "preflight: git rev-parse failed" >&2; exit 2; }
TS0=$(date +%Y-%m-%d_%H:%M:%S)
STATUS_LOG="$E/yaw_rotation_degradation_${TS0}_${MODE}_launcher.log"
: > "$STATUS_LOG" || { echo "preflight: cannot write $STATUS_LOG" >&2; exit 2; }
CODES=$(mktemp) && [ -n "$CODES" ] || { echo "preflight: mktemp failed" >&2; exit 2; }
trap 'rm -f "$CODES"' EXIT
declare -A BB=( [control]=simple [cyl]=cylindrical [released]=simple )
declare -A CK=( [control]=ckpt/xRIR_simple_8_shot/epoch_12.pth [cyl]=ckpt/xRIR_cyl_8_shot/epoch_12.pth [released]=checkpoints/xRIR_unseen.pth )
declare -A APPROVED_CK_SHA=()
sha() { [ -r "$1" ] || return 1; local d; d=$(sha256sum "$1" | cut -d' ' -f1) && [ ${#d} -eq 64 ] || return 1; printf '%s\n' "$d"; }
say() { local line; line="$(date +%T) $*"; printf '%s\n' "$line" >> "$STATUS_LOG" || { echo "status log write failed: $line" >&2; exit 2; }; echo "$line"; }
die() { say "$*"; exit 2; }
check_evaluator() {  # the whole reviewed source closure, byte for byte, with no later commits
  local now; now=$(sha eval_yaw_rotation.py) || { say "EVALUATOR unreadable ($1)"; return 1; }
  [ "$now" = "$REVIEWED_EVAL_SHA" ] || { say "EVALUATOR MISMATCH ($1): $now != reviewed $REVIEWED_EVAL_SHA"; return 1; }
  "$P" - "$ASSETS" "$REVIEWED_CLOSURE_SHA" "$1" <<'PY' >> "$STATUS_LOG" 2>&1 || { say "SOURCE CLOSURE MISMATCH ($1): see $STATUS_LOG"; return 1; }
import os, sys
sys.path.insert(0, sys.argv[1]); import bind_provenance as bp
rec, digest = bp.closure_record(bp.source_closure(os.getcwd()), os.getcwd())
bad = [r["path"] for r in rec if r["reviewed_blob_sha256"] != r["working_tree_sha256"] or r["commits_after_reviewed"]]
if bad or digest != sys.argv[2]: raise SystemExit(f"closure check ({sys.argv[3]}): digest {digest[:12]} vs pinned {sys.argv[2][:12]}; differing files {bad}")
print(f"closure ok ({sys.argv[3]}): {len(rec)} files, digest {digest[:12]}")
PY
}
# ---- preflight: every input must exist before anything is reserved or started
for f in eval_yaw_rotation.py tools/summarize_yaw.py "$MAN" "$MAN1" "$MAN2" "${CK[control]}" "${CK[cyl]}" "${CK[released]}"; do [ -r "$f" ] || die "preflight: missing input $f"; done
for m in "$MAN:$HASH" "$MAN1:$HASH1" "$MAN2:$HASH2"; do
  "$P" -c 'import json,sys; from tools.reference_manifest import load_manifest, manifest_hash; p,h=sys.argv[1].rsplit(":",1); m=load_manifest(p); sys.exit(0 if manifest_hash(m)==h else 1)' "$m" || die "preflight: manifest hash mismatch for ${m%%:*}"
done
for role in control cyl released; do
  if [ "$MODE" = sweep ]; then
    side="ckpt/yaw_rotation/gate_${role}/provenance.json"; [ -r "$side" ] || die "preflight: missing $side (approved checkpoint digest)"
    APPROVED_CK_SHA[$role]=$("$P" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$side") && [ ${#APPROVED_CK_SHA[$role]} -eq 64 ] || die "preflight: cannot read the approved digest from $side"
  else
    APPROVED_CK_SHA[$role]=$(sha "${CK[$role]}") || die "preflight: cannot hash ${CK[$role]}"
  fi
done
check_evaluator preflight || exit 2
say "preflight ok: mode $MODE, launcher HEAD $LAUNCHER_HEAD, evaluator $REVIEWED_EVAL_SHA (reviewed $REVIEWED_COMMIT)"
# ---- one run; failures after the evaluator starts all go through abort_run
abort_run() {  # tag rc reason log -- every post-start failure comes through here; the *_ABORTED_* log is mandatory: a missing log or a failed rename is itself a failure (code 3)
  local tag=$1 rc=$2 reason=$3 log=$4
  if [ ! -f "$log" ]; then
    say "ABORT EVIDENCE MISSING: run log $log does not exist"; echo 3 >> "$CODES"
  elif ! mv "$log" "${log%.log}_ABORTED_${reason}_exit${rc}.log"; then
    say "ABORT EVIDENCE NOT PRESERVED: could not rename $log"; echo 3 >> "$CODES"
  fi
  say "FAILED $MODE $tag ($reason) exit $rc"; echo "$rc" >> "$CODES"; return "$rc"
}
run_one() {  # name gpu manifest hash tag batch log-interval extra-args...
  local name=$1 gpu=$2 man=$3 hash=$4 tag=$5 batch=$6 loginterval=$7; shift 7
  local out="ckpt/yaw_rotation/${MODE}_${tag}" ck="${CK[$name]}" ck_sha ts log cmd rc
  ts=$(date +%Y-%m-%d_%H:%M:%S); log="$E/yaw_rotation_degradation_${ts}_${MODE}_${tag}.log"
  ck_sha=$(sha "$ck") || { say "cannot hash $ck"; echo 1 >> "$CODES"; return 1; }
  [ "$ck_sha" = "${APPROVED_CK_SHA[$name]}" ] || { say "CHECKPOINT MISMATCH $tag: $ck_sha != approved ${APPROVED_CK_SHA[$name]}"; echo 1 >> "$CODES"; return 1; }
  check_evaluator "before $tag" || { echo 1 >> "$CODES"; return 1; }
  local evaluator=eval_yaw_rotation.py; [ "$MODE" = faulttest ] && [ -n "$EVALUATOR_OVERRIDE" ] && evaluator="$EVALUATOR_OVERRIDE"
  local -a args=("$evaluator" --backbone "${BB[$name]}" --checkpoint "$ck" --manifest "$man" --manifest-hash "$hash" --out-dir "$out" --batch-size "$batch" --num-workers 6 --threads 4 --log-interval "$loginterval" "$@")
  cmd="CUDA_VISIBLE_DEVICES=$gpu $(printf '%q ' "$P" "${args[@]}")"
  say "start $MODE $tag: $cmd"
  CUDA_VISIBLE_DEVICES="$gpu" "$P" "${args[@]}" > "$log" 2>&1; rc=$?
  [ "$rc" -eq 0 ] || { abort_run "$tag" "$rc" evaluator "$log"; return $?; }
  if [ "$MODE" = faulttest ] && [ -n "$FAULT_INJECT" ]; then   # force a post-evaluator abort branch without touching real files
    case "$FAULT_INJECT" in
      evaluator_changed)  abort_run "$tag" 1 evaluator_changed "$log"; return $? ;;
      checkpoint_changed) abort_run "$tag" 1 checkpoint_changed "$log"; return $? ;;
      missing_log)        rm -f "$log"; abort_run "$tag" 1 missing_log "$log"; return $? ;;
    esac
  fi
  check_evaluator "after $tag" || { abort_run "$tag" 1 evaluator_changed "$log"; return $?; }
  [ "$(sha "$ck")" = "$ck_sha" ] || { abort_run "$tag" 1 checkpoint_changed "$log"; return $?; }
  "$P" - "$out" "$ck" "$ck_sha" "$hash" "$LAUNCHER_HEAD" "$REVIEWED_COMMIT" "$REVIEWED_EVAL_SHA" "$cmd" "$log" <<'PY' >> "$STATUS_LOG" 2>&1
import hashlib, json, os, sys, datetime
out, ck, ck_sha, mh, head, rev_commit, rev_eval, cmd, log = sys.argv[1:10]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(f"{out}/metrics_yaw.json"))
if m["meta"]["manifest_hash"] != mh: raise SystemExit("manifest hash in metrics differs from the launcher's")
if h(ck) != ck_sha: raise SystemExit("checkpoint digest changed")
if h("eval_yaw_rotation.py") != rev_eval: raise SystemExit("evaluator digest changed")
rec = {"git_sha": head, "git_sha_note": "launcher HEAD at run time; the evaluator is bound by evaluator_sha256 to the reviewed commit; run bind_provenance.py for the source-closure binding",
       "reviewed_evaluator_commit": rev_commit, "evaluator_sha256": rev_eval, "checkpoint": ck, "checkpoint_sha256": ck_sha,
       "manifest_hash": mh, "per_sample_sha256": h(f"{out}/per_sample_yaw.json"), "metrics_sha256": h(f"{out}/metrics_yaw.json"),
       "command": cmd, "log": log, "written": datetime.datetime.now().astimezone().isoformat()}
tmp = f"{out}/provenance.json.tmp"
with open(tmp, "w") as f: json.dump(rec, f, indent=1); f.flush(); os.fsync(f.fileno())
os.replace(tmp, f"{out}/provenance.json")
if json.load(open(f"{out}/provenance.json")) != rec: raise SystemExit("provenance read-back differs")
print("provenance ok", out)
PY
  rc=$?; [ "$rc" -eq 0 ] || { abort_run "$tag" "$rc" provenance "$log"; return $?; }
  echo 0 >> "$CODES"; say "done $MODE $tag exit 0"; return 0
}
reserve() {  # atomically reserve every output directory or refuse the whole launch
  local -a made=(); local d
  for d in "$@"; do
    if mkdir "ckpt/yaw_rotation/${MODE}_${d}" 2>/dev/null; then made+=("ckpt/yaw_rotation/${MODE}_${d}"); else say "REFUSED: ckpt/yaw_rotation/${MODE}_${d} exists (or cannot be created); nothing started"; [ ${#made[@]} -gt 0 ] && rmdir "${made[@]}" 2>/dev/null; return 1; fi
  done
}
finish() {  # ra rb label
  local ra=$1 rb=$2 label=$3
  if [ "$ra" -eq 0 ] && [ "$rb" -eq 0 ] && [ -s "$CODES" ] && [ "$(sort -u "$CODES" | tr -d '\n')" = "0" ]; then say "${label} $(wc -l < "$CODES") runs ok"; return 0; fi
  say "${label%_*}_FAILED codes: $(tr '\n' ' ' < "$CODES")"; exit 1
}
K0=(--yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --decomposition-batches 0)
if [ "$MODE" = gate ]; then
  # the complete band-v3 measurement set (plan §12): 5 original + phase + tf32 + 2 cyl noise + cyl batch-1 shape
  reserve control released noise_seed1 phase_seed1 noise_cyl_seed1 noise_cyl_seed2 cyl noise_seed2 tf32 shape_cyl_b1 || exit 1
  ( run_one control 0 "$MAN" "$HASH" control 16 10 "${K0[@]}" && run_one released 0 "$MAN" "$HASH" released 16 10 "${K0[@]}" \
    && run_one control 0 "$MAN1" "$HASH1" noise_seed1 16 10 "${K0[@]}" && run_one control 0 "$MAN" "$HASH" phase_seed1 16 10 "${K0[@]}" --gl-seed 1 \
    && run_one cyl 0 "$MAN1" "$HASH1" noise_cyl_seed1 16 10 "${K0[@]}" && run_one cyl 0 "$MAN2" "$HASH2" noise_cyl_seed2 16 10 "${K0[@]}" ) & A=$!
  ( run_one cyl 1 "$MAN" "$HASH" cyl 16 10 "${K0[@]}" && run_one control 1 "$MAN2" "$HASH2" noise_seed2 16 10 "${K0[@]}" \
    && run_one control 1 "$MAN" "$HASH" tf32 16 10 "${K0[@]}" --tf32 && run_one cyl 1 "$MAN" "$HASH" shape_cyl_b1 1 100 "${K0[@]}" ) & B=$!
  wait "$A"; ra=$?; wait "$B"; rb=$?; finish "$ra" "$rb" GATE_MEASUREMENTS_DONE
  "$P" "$ASSETS/bind_provenance.py" --set gate >> "$STATUS_LOG" 2>&1 && say "gate sidecars bound pre-decision (bind_provenance.py --set gate: closure, environment, data identity, logs)" || { say "GATE_BINDING_FAILED (see $STATUS_LOG)"; exit 1; }
  say "NEXT (in this order): 1) take the decision with the command below (pin the producer commit you use; the acceptance checker pins its own), 2) copy the summary to the record folder as the retained decision file and the gate JSON to the assets folder, 3) run: $P $ASSETS/bind_provenance.py --set gate   # additive: adds the decision artefacts to the report; then 4) sweep"
  say "gate decision (not taken here): $P tools/summarize_yaw.py --mode k0-gate --band-rule v3 --runs ckpt/yaw_rotation/gate_control ckpt/yaw_rotation/gate_cyl ckpt/yaw_rotation/gate_released --manifest-hash $HASH --exp01-per-sample control=ckpt/xRIR_simple_8_shot/per_sample_unseen_epoch12.json cyl=ckpt/xRIR_cyl_8_shot/per_sample_unseen_epoch12.json --noise-runs ckpt/yaw_rotation/gate_noise_seed1 ckpt/yaw_rotation/gate_noise_seed2 --noise-runs-cyl ckpt/yaw_rotation/gate_noise_cyl_seed1 ckpt/yaw_rotation/gate_noise_cyl_seed2 --nuisance-runs ckpt/yaw_rotation/gate_phase_seed1 ckpt/yaw_rotation/gate_tf32 --shape-runs ckpt/yaw_rotation/gate_shape_cyl_b1 --released-baseline 0.0549,1.358,9.69 --json ckpt/yaw_rotation/gate_stats.json --summary ckpt/yaw_rotation/gate_summary.txt"
elif [ "$MODE" = faulttest ]; then
  [ -n "$EVALUATOR_OVERRIDE" ] && [ -r "$EVALUATOR_OVERRIDE" ] || die "faulttest needs EVALUATOR_OVERRIDE=<stub script> (retained: $ASSETS/eval_stub.py)"
  before=$(mktemp); ls "$E"/yaw_rotation_degradation_*_faulttest_*_ABORTED_*.log 2>/dev/null | sort > "$before"
  reserve stub_exit7 stub_noout stub_evchg stub_ckchg stub_nolog || exit 1
  run_one control 0 "$MAN" "$HASH" stub_exit7 16 10 --stub-exit 7; r1=$?
  run_one control 0 "$MAN" "$HASH" stub_noout 16 10 --stub-exit 0; r2=$?
  FAULT_INJECT=evaluator_changed  run_one control 0 "$MAN" "$HASH" stub_evchg 16 10 --stub-exit 0; r3=$?
  FAULT_INJECT=checkpoint_changed run_one control 0 "$MAN" "$HASH" stub_ckchg 16 10 --stub-exit 0; r4=$?
  FAULT_INJECT=missing_log        run_one control 0 "$MAN" "$HASH" stub_nolog 16 10 --stub-exit 0; r5=$?
  after=$(mktemp); ls "$E"/yaw_rotation_degradation_*_faulttest_*_ABORTED_*.log 2>/dev/null | sort > "$after"
  n_new=$(comm -13 "$before" "$after" | wc -l); rm -f "$before" "$after"
  say "faulttest: rc = $r1 (expect 7) $r2 (provenance, non-zero) $r3 (evaluator_changed) $r4 (checkpoint_changed) $r5 (missing_log); NEW ABORTED logs created by this invocation: $n_new (expect 4: the missing-log case has no log to preserve and records code 3); codes: $(tr '\n' ' ' < "$CODES")"
  rm -rf ckpt/yaw_rotation/faulttest_stub_exit7 ckpt/yaw_rotation/faulttest_stub_noout ckpt/yaw_rotation/faulttest_stub_evchg ckpt/yaw_rotation/faulttest_stub_ckchg ckpt/yaw_rotation/faulttest_stub_nolog
  if [ "$r1" -eq 7 ] && [ "$r2" -ne 0 ] && [ "$r3" -ne 0 ] && [ "$r4" -ne 0 ] && [ "$r5" -ne 0 ] && [ "$n_new" -eq 4 ] && grep -q '^3$' "$CODES"; then say "FAULTTEST_DONE"; exit 0; else say "FAULTTEST_FAILED"; exit 1; fi
else
  G=ckpt/yaw_rotation/gate_stats.json; [ -r "$G" ] || die "SWEEP_REFUSED: no gate decision at $G"
  "$P" - "$ASSETS" <<'PY' >> "$STATUS_LOG" 2>&1 || { say "SWEEP_REFUSED: gate validation failed (see $STATUS_LOG)"; exit 1; }
import sys; sys.path.insert(0, sys.argv[1]); import check_sweep_acceptance as c
f = c.full_gate_validation(recompute=True)   # the same checks as final acceptance: live environment/data identity, deep gate sidecars, retained report, recomputed decision
if f: raise SystemExit("gate validation failed: " + "; ".join(f))
print("gate ok: band v3, ten bound runs deep-checked, retained report consistent, decision recomputed with the pinned producer")
PY
  say "gate decision independently validated (check_sweep_acceptance.full_gate_validation, with recomputation): $G"
  reserve control released cyl || exit 1
  ( run_one control 0 "$MAN" "$HASH" control 16 10 && run_one released 0 "$MAN" "$HASH" released 16 10 ) & A=$!
  ( run_one cyl 1 "$MAN" "$HASH" cyl 16 10 ) & B=$!
  wait "$A"; ra=$?; wait "$B"; rb=$?; finish "$ra" "$rb" SWEEP_DONE
  say "next: bind_provenance.py, check_sweep_acceptance.py, then tools/summarize_yaw.py --mode full"
fi
