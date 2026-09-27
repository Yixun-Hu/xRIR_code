#!/bin/bash
# exp_10 yaw_pilot — end-game after all four GPU arms are complete: validate and bind the run
# evidence, canonical summary (4 arms), CPU-protocol summary, backend-sensitivity table, results
# Markdown + HTML, then an explicitly listed asset set with SHA256SUMS published atomically.
#
# Two blockers from the Codex tooling review shape this script:
#   finding 1 — the preflight now *validates and binds* the prescribed evidence for every arm
#     (validate_runs.py: full + probe meta, check_online bound and ok, probe controls ok, probe
#     and full parity for the K = 8 arms, the CPU record with its documented parity failure).
#     Its report is published as `validation.json`.
#   finding 2 — the asset set is built in a fresh staging directory from an EXPLICIT list (no
#     globs of neighbouring files), every expected file is verified present, nothing unexpected
#     is allowed in, SHA256SUMS is written and `sha256sum -c`-verified there, and only then does
#     `generated/` get replaced by a rename. Any error at any point leaves `generated/` and the
#     published Markdown / HTML exactly as they were — the rollback copies of all three are
#     kept until every one of the three publications has succeeded (round-4 review, finding 2),
#     and a *signal* rolls them back just like an error does (round-5 review, finding 1: SIGTERM
#     immediately before the final rename left new assets + new HTML + the old Markdown, and
#     the EXIT cleanup deleted both backups because the destinations existed).
#   round-6 review — every rollback copy is now completed and verified *while nothing has been
#     published yet* and marked complete only then, because `started` used to precede the
#     report copies: a `cp` that failed half-way left a truncated backup whose existence
#     restore() took for proof of completeness, so it deleted the intact published report and
#     installed the truncation (finding 1). And a rollback, once entered, always deals with
#     all three artefacts, with TERM / INT / HUP ignored for its whole duration, because
#     SIGTERM inside an explicit rollback used to leave old assets + new HTML + old Markdown
#     (finding 2).
#
# Every validation report is bound by the run directory it names, with no exception: the
# relocation alias this script used to pass for the CPU record accepted the GPU arm's
# reports (same basename, another execution) as the CPU protocol's evidence and reached
# `FINISH DONE` with them (round-4 review, finding 1). The CPU record's reports were
# regenerated in place and name `ckpt/exp10/cpu_protocol/released_k8_all` themselves.
#
# usage: exp10_finish.sh          (no arguments)
# env:   EXP10_REPO_ROOT, EXP10_EXPECT_N — overrides for the tests; the production defaults are
#        the repository path and 6337 queries, and both are logged.
set -euo pipefail
REPO=${EXP10_REPO_ROOT:-/home/yixunhu/codespace/xRIR_code}
EXPECT_N=${EXP10_EXPECT_N:-6337}
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate xRIR
export PYTHONPATH=${PYTHONPATH:-}:$(pwd); export CUDA_VISIBLE_DEVICES=""
E=worklog/worklog_yixun/exp_10_yaw_pilot_claude; A=$E/yaw_pilot_results_assets; G=$A/generated
LOG=$E/yaw_pilot_$(date +%Y-%m-%d_%H:%M:%S)_finish.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ARMS="released_k8 released_k1 control_k8 cyl_k8"
CPU=ckpt/exp10/cpu_protocol
STAGE=$A/generated.staging.$$
TMPOUT=$A/.finish_tmp.$$
PREV=$A/generated.previous.$$          # the verified copy of the asset set this run replaces
PREVT=$A/.finish_prev_assets.$$        # ... where that copy is made and verified first
PREVR=$A/.finish_prev.$$               # the verified copies of the Markdown / HTML it replaces
PREVRT=$PREVR.tmp                      # ... where those copies are made and verified first
DONE=.backup_complete                  # the marker that says one backup copy is whole
REPORTS="yaw_pilot_01_results.html yaw_pilot_results.md"

# ---- publication state, the rollback, and the traps that use them (round-5, finding 1) ------
# Publishing replaces three artefacts and cannot be one atomic step, so the state is tracked
# explicitly: `started` means the three may be a mixed generation and the backups are the only
# copies of the previous publication; `committed` means all three are published and the backups
# may go; `rolled_back` means restore() has already dealt with them. EVERY exit path — an
# unguarded error, a failed rename, SIGTERM / SIGINT / SIGHUP — rolls all three back while the
# state is `started`, and only `committed` — or `idle`, where nothing has been touched at all —
# lets the EXIT cleanup delete a backup that still holds bytes. What was there *before* this
# run is recorded before the first mutation, so the rollback can preserve original absence too
# (an artefact this run created is removed, not "restored" from a backup that never existed).
#
# A backup is restored from only when it is *known whole*: it is copied under a temporary name
# while the state is still `idle`, verified against the original (`cmp` for a report, a
# recursive `diff` for the asset directory), renamed into its backup slot and only then marked
# complete. restore() installs nothing that carries no marker, so a copy that failed half-way
# can never overwrite the intact artefact it was copied from (round-6 review, finding 1: the
# state went to `started` first, and the mere existence of a truncated backup was taken for
# proof of completeness). The markers are files under `$PREVR`, never inside `$PREV`, whose
# contents become the published asset set again the moment it is restored.
#
# And a rollback, once entered, runs to its end: TERM / INT / HUP are ignored for its whole
# duration — on every entry path, the explicit restore() after a failed publication as much as
# the trap-driven one — and the state becomes `rolled_back` only after the last of the three
# artefacts (round-6 review, finding 2: an explicit rollback marked itself done first and kept
# the handlers armed, so SIGTERM during it left three generations mixed). A signal delivered
# inside a rollback is therefore discarded, not queued, and the status reports what actually
# triggered the rollback: the failure's own code, or 128 + the signal that arrived first.
PUB_STATE=idle                         # idle | started | committed | rolled_back
PRE_G=0                                # did the asset directory exist before this run?
declare -A PRE_REPORT=()               # ... and each of the two reports?
KEEP_BACKUPS=0                         # restoration failed: the backups are the last copies
restore() {   # <what failed> — put all three artefacts back exactly as they were
  # Nothing here hides why an operation failed: whatever `rm` / `mv` print is the diagnosis
  # for the KEPT list below, so it is left on stderr rather than discarded.
  local what=$1 restored="" kept="" pre f
  trap '' HUP INT TERM                           # a rollback is never cut in half (finding 2)
  if [ -d "$PREV" ] && [ -f "$PREVR/generated$DONE" ]; then   # a whole copy of the old set
    rm -rf "$G" || true
    if [ ! -e "$G" ] && mv "$PREV" "$G"; then restored="$restored $G"
    else kept="$kept $PREV (the asset set this run replaced)"; fi
  elif [ -d "$PREV" ]; then                      # ... an unverified one is never installed
    kept="$kept $PREV (an INCOMPLETE copy of the asset set; $G was left exactly as it is)"
  elif [ "$PRE_G" = 0 ] && [ -d "$G" ]; then     # there was none: this run created it
    rm -rf "$G" || true
    if [ ! -e "$G" ]; then restored="$restored $G (removed: this run created it)"
    else kept="$kept $G (this run created it and it could not be removed)"; fi
  fi
  for f in $REPORTS; do
    pre=${PRE_REPORT[$f]:-0}
    if [ -f "$PREVR/$f" ] && [ -f "$PREVR/$f$DONE" ]; then   # a whole copy: put it back
      rm -f "$E/$f" || true
      if [ ! -e "$E/$f" ] && mv "$PREVR/$f" "$E/$f"; then restored="$restored $E/$f"
      else kept="$kept $PREVR/$f (the $f this run replaced)"; fi
    elif [ -f "$PREVR/$f" ]; then                # ... an unverified one is never installed
      kept="$kept $PREVR/$f (an INCOMPLETE copy of $f; $E/$f was left exactly as it is)"
    elif [ "$pre" = 0 ] && [ -f "$E/$f" ]; then  # absent before this run: absent after it
      rm -f "$E/$f" || true
      if [ ! -e "$E/$f" ]; then restored="$restored $E/$f (removed: this run created it)"
      else kept="$kept $E/$f (this run created it and it could not be removed)"; fi
    fi
  done
  PUB_STATE=rolled_back                          # all three are dealt with (finding 2)
  if [ -z "$kept" ]; then
    rm -rf "$PREVR" || true
    say "REFUSED: publishing $what failed; restored:$restored"
  else
    KEEP_BACKUPS=1
    say "REFUSED: publishing $what failed; restored:$restored; NOT RESTORED — these are KEPT for inspection by hand, each with what it holds:$kept"
  fi
}
cleanup() {
  rc=$?
  local left d
  trap - EXIT; trap '' HUP INT TERM              # no re-entry while things are put back
  if [ -d "$STAGE" ]; then rm -rf "$STAGE" || true; fi
  if [ -d "$TMPOUT" ]; then rm -rf "$TMPOUT" || true; fi
  # An unverified copy is never restored from, so it is never kept either (finding 1).
  rm -rf "$PREVT" "$PREVRT" 2>/dev/null || true
  # A death between the three publications (an unguarded error, a signal) must roll all three
  # back, and must never leave a backup behind as the only copy of the previous publication.
  if [ "$PUB_STATE" = started ]; then restore "the publication (it was interrupted)"; fi
  if [ "$PUB_STATE" = idle ] || [ "$PUB_STATE" = committed ]; then
    # `idle`: nothing was published, so every backup is a copy of an untouched artefact.
    # `committed`: all three are published. Either way the copies may go.
    rm -rf "$PREV" "$PREVR" 2>/dev/null || true
  elif [ "$KEEP_BACKUPS" = 1 ]; then
    left=""
    for d in "$PREV" "$PREVR"; do if [ -e "$d" ]; then left="$left $d"; fi; done
    if [ -n "$left" ]; then
      say "KEPT:$left — the rollback could not put everything back; the REFUSED line above says what each of these holds, so do not delete them by hand"
    else
      say "NOT RESTORED: the publication could not be put back; the REFUSED line above says what is where"
    fi
  else
    for d in "$PREV" "$PREVR"; do                # a successful rollback consumed these; only
      if [ -d "$d" ]; then                       # an empty staging directory may be deleted
        if [ -n "$(find "$d" -type f -print -quit 2>/dev/null)" ]; then
          KEEP_BACKUPS=1
          say "KEPT: $d still holds files, so it is not deleted"
        else rm -rf "$d" 2>/dev/null || true; fi
      fi
    done
  fi
  # The closing line may only claim the previous publication is back when it *is* back: the
  # round-4 review's finding 2 was a cleanup that said nothing had changed while three
  # generations were mixed, and a failed rollback is that same claim.
  if [ "$rc" -ne 0 ] && [ "$KEEP_BACKUPS" = 1 ]; then
    say "FINISH FAILED rc=$rc — the publication could NOT be put back in full; the KEPT paths above hold what was there before"
  elif [ "$rc" -ne 0 ]; then
    say "FINISH FAILED rc=$rc — $G, the Markdown report and the HTML page are the publication that was there before"
  fi
}
on_signal() {   # <name> <number> — roll back through the EXIT trap; never re-raise the signal
  trap '' HUP INT TERM
  case $PUB_STATE in                             # the message may only claim what is true
    started) say "FINISH INTERRUPTED: SIG$1 received; rolling the publication back" ;;
    committed) say "FINISH INTERRUPTED: SIG$1 received after the publication was complete" ;;
    rolled_back) say "FINISH INTERRUPTED: SIG$1 received; the rollback has already run" ;;
    *) say "FINISH INTERRUPTED: SIG$1 received; nothing has been published, so there is nothing to roll back" ;;
  esac
  exit $((128 + $2))
}
trap cleanup EXIT
trap 'on_signal TERM 15' TERM
trap 'on_signal INT 2' INT
trap 'on_signal HUP 1' HUP
mkdir -p "$STAGE" "$TMPOUT"
say "HEAD $(git rev-parse HEAD 2>/dev/null || echo unknown); repo $REPO; expect n=$EXPECT_N"

# ---- preflight: the evidence, validated and bound (finding 1) -------------------------------
python "$A/validate_runs.py" --root ckpt/exp10 --arms $ARMS --expect-n "$EXPECT_N" \
  --device cuda --cpu-run "$CPU/released_k8_all" \
  --json "$STAGE/validation.json" 2>&1 | tee -a "$LOG"
say "preflight passed: four GPU arms + the CPU record validated and bound"

# ---- canonical summaries, then the backend table -------------------------------------------
python tools/exp10_summarize.py --runs ckpt/exp10/released_k8_all ckpt/exp10/released_k1_all \
  ckpt/exp10/control_k8_all ckpt/exp10/cyl_k8_all --out ckpt/exp10/summary 2>&1 | tee -a "$LOG"
python tools/exp10_summarize.py --runs "$CPU/released_k8_all" --out "$CPU/summary" 2>&1 | tee -a "$LOG"
python "$A/make_backend_table.py" --gpu ckpt/exp10/summary/yaw_pilot_summary.json \
  --cpu "$CPU/summary/yaw_pilot_summary.json" --arm released_k8 \
  --out-md "$STAGE/backend_sensitivity_released_k8.md" \
  --out-json "$STAGE/backend_sensitivity_released_k8.json" 2>&1 | tee -a "$LOG"

# ---- the asset set, copied file by file (finding 2) ----------------------------------------
EXPECTED="validation.json backend_sensitivity_released_k8.md backend_sensitivity_released_k8.json"
PAR=""; ONL=""; PRB=""
for a in $ARMS; do
  mkdir -p "$STAGE/runs/$a"
  cp "ckpt/exp10/${a}_all/meta.json"                          "$STAGE/runs/$a/meta_all.json"
  cp "ckpt/exp10/${a}_all/check_online.json"                  "$STAGE/runs/$a/check_online_all.json"
  cp "ckpt/exp10/${a}_probe/meta.json"                        "$STAGE/runs/$a/meta_probe.json"
  cp "ckpt/exp10/${a}_probe/summary/yaw_pilot_summary.json"   "$STAGE/runs/$a/probe_summary.json"
  EXPECTED="$EXPECTED runs/$a/meta_all.json runs/$a/check_online_all.json runs/$a/meta_probe.json runs/$a/probe_summary.json"
  ONL="$ONL $a=ckpt/exp10/${a}_all/check_online.json"
  PRB="$PRB $a=ckpt/exp10/${a}_probe/summary/yaw_pilot_summary.json"
  if [ -f "ckpt/exp10/${a}_all/parity_exp03.json" ]; then   # released_k1 has no predecessor
    cp "ckpt/exp10/${a}_all/parity_exp03.json"    "$STAGE/runs/$a/parity_exp03_all.json"
    cp "ckpt/exp10/${a}_probe/parity_exp03.json"  "$STAGE/runs/$a/parity_exp03_probe.json"
    EXPECTED="$EXPECTED runs/$a/parity_exp03_all.json runs/$a/parity_exp03_probe.json"
    PAR="$PAR $a=ckpt/exp10/${a}_all/parity_exp03.json"
  fi
done
mkdir -p "$STAGE/cpu_protocol"
cp "$CPU/released_k8_all/meta.json"          "$STAGE/cpu_protocol/meta_all.json"
cp "$CPU/released_k8_all/check_online.json"  "$STAGE/cpu_protocol/check_online_all.json"
cp "$CPU/released_k8_all/parity_exp03.json"  "$STAGE/cpu_protocol/parity_exp03_all.json"
cp "$CPU/summary/yaw_pilot_summary.json"     "$STAGE/cpu_protocol/yaw_pilot_summary.json"
cp "$CPU/summary/yaw_pilot_tables.md"        "$STAGE/cpu_protocol/yaw_pilot_tables.md"
cp "$CPU/summary/yaw_pilot_gaps.csv"         "$STAGE/cpu_protocol/yaw_pilot_gaps.csv"
cp "$CPU/summary/yaw_pilot_gaps_released_k8.png" "$STAGE/cpu_protocol/yaw_pilot_gaps_released_k8_cpu.png"
cp "$CPU/summary/yaw_pilot_gaps_released_k8.pdf" "$STAGE/cpu_protocol/yaw_pilot_gaps_released_k8_cpu.pdf"
EXPECTED="$EXPECTED cpu_protocol/meta_all.json cpu_protocol/check_online_all.json cpu_protocol/parity_exp03_all.json"
EXPECTED="$EXPECTED cpu_protocol/yaw_pilot_summary.json cpu_protocol/yaw_pilot_tables.md cpu_protocol/yaw_pilot_gaps.csv"
EXPECTED="$EXPECTED cpu_protocol/yaw_pilot_gaps_released_k8_cpu.png cpu_protocol/yaw_pilot_gaps_released_k8_cpu.pdf"

# ---- the report: HTML first (it copies this summary's own outputs into the staging set) -----
HREF=$(python -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$G" "$E")
BACKEND="$STAGE/backend_sensitivity_released_k8.md"
CPUREC="$STAGE/cpu_protocol/yaw_pilot_tables.md"
python "$A/make_results_html.py" --summary ckpt/exp10/summary/yaw_pilot_summary.json \
  --out "$TMPOUT/yaw_pilot_01_results.html" --assets "$STAGE" --assets-href "$HREF" \
  --backend-table "$BACKEND" --cpu-record "$CPUREC" \
  --parity $PAR --check-online $ONL --probe $PRB 2>&1 | tee -a "$LOG"
python "$A/make_results_md.py" --summary ckpt/exp10/summary/yaw_pilot_summary.json \
  --out "$TMPOUT/yaw_pilot_results.md" --assets "$STAGE" --assets-href "$HREF" \
  --backend-table "$BACKEND" --cpu-record "$CPUREC" \
  --parity $PAR --check-online $ONL --probe $PRB 2>&1 | tee -a "$LOG"
EXPECTED="$EXPECTED yaw_pilot_summary.json yaw_pilot_tables.md yaw_pilot_gaps.csv"
EXPECTED="$EXPECTED yaw_pilot_gaps_all_arms.png yaw_pilot_gaps_all_arms.pdf"
for a in $ARMS; do EXPECTED="$EXPECTED yaw_pilot_gaps_${a}.png yaw_pilot_gaps_${a}.pdf"; done

# ---- the staged set is exactly the expected set, and it verifies --------------------------
MISSING=""
for f in $EXPECTED; do
  if [ ! -f "$STAGE/$f" ]; then MISSING="$MISSING $f"; fi
done
if [ -n "$MISSING" ]; then say "REFUSED: the staged asset set is incomplete:$MISSING"; exit 8; fi
EXTRA=""
while IFS= read -r f; do
  case " $EXPECTED " in
    *" $f "*) ;;
    *) EXTRA="$EXTRA $f" ;;
  esac
done < <(cd "$STAGE" && find . -type f ! -name SHA256SUMS -printf '%P\n' | sort)
if [ -n "$EXTRA" ]; then say "REFUSED: unexpected files in the staged asset set:$EXTRA"; exit 8; fi
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -printf '%P\0' | sort -z | xargs -0 sha256sum > SHA256SUMS )
( cd "$STAGE" && sha256sum -c --quiet SHA256SUMS ) 2>&1 | tee -a "$LOG"
say "staged $(wc -l < "$STAGE/SHA256SUMS") assets; SHA256SUMS verifies"

# ---- publish: all three artefacts, or none of them (finding 2) ----------------------------
# The previous assets used to be deleted as soon as the new ones were in place, before either
# report was published, so a failing final rename left new assets + new HTML + old Markdown
# (three generations mixed) while the cleanup claimed nothing had changed. Every backup is
# now kept until all three publications have succeeded — and the same rollback runs when the
# script is *terminated* here (round-5 review, finding 1), which is why the pre-run state is
# recorded before the first mutation: restore() must be able to tell an artefact it has to put
# back from one this run created, which has to be removed again.
if [ -d "$G" ]; then PRE_G=1; fi
for f in $REPORTS; do
  PRE_REPORT[$f]=0
  if [ -f "$E/$f" ]; then PRE_REPORT[$f]=1; fi
done
# The rollback copies, made and verified while the publication is still untouched (round-6
# review, finding 1). Each is copied under a temporary name, compared with the original it was
# copied from, renamed into its backup slot and only then marked complete — so a copy that
# failed or was interrupted half-way is never a backup restore() can find, and the artefact it
# was copied from has not been touched at all. Nothing below `PUB_STATE=started` has mutated
# the publication, which is why an `idle` exit may simply drop every copy.
mkdir -p "$PREVR" "$PREVRT"
for f in $REPORTS; do
  if [ "${PRE_REPORT[$f]}" = 0 ]; then continue; fi
  if ! cp -p "$E/$f" "$PREVRT/$f"; then
    say "REFUSED: the rollback copy of $f could not be made; nothing has been published"
    exit 12
  fi
  if ! cmp "$E/$f" "$PREVRT/$f" > "$TMPOUT/backup_check.txt" 2>&1; then
    say "REFUSED: the rollback copy of $f is not byte-identical to $E/$f ($(tr '\n' ' ' < "$TMPOUT/backup_check.txt")); nothing has been published"
    exit 12
  fi
  mv "$PREVRT/$f" "$PREVR/$f"
  : > "$PREVR/$f$DONE"
done
if [ "$PRE_G" = 1 ]; then
  rm -rf "$PREVT"
  if ! cp -a "$G" "$PREVT"; then
    say "REFUSED: the rollback copy of $G could not be made; nothing has been published"
    exit 12
  fi
  if ! diff -rq "$G" "$PREVT" > "$TMPOUT/backup_check.txt" 2>&1; then
    say "REFUSED: the rollback copy of $G does not match it file for file; nothing has been published"
    sed -n '1,10p' "$TMPOUT/backup_check.txt" | tee -a "$LOG"
    exit 12
  fi
  mv "$PREVT" "$PREV"
  : > "$PREVR/generated$DONE"
fi
rm -rf "$PREVRT"
say "rollback copies of what this run replaces: complete, verified and marked; publishing now"
PUB_STATE=started                       # from here on, any death rolls all three back
# `$G` is removed rather than renamed aside: a rename would itself be this run's first mutation
# of the publication, and every backup has to be complete before that happens. What restore()
# puts back is the verified copy in `$PREV`.
if [ -d "$G" ] && ! rm -rf "$G"; then restore "the asset set"; exit 9; fi
if ! mv "$STAGE" "$G"; then restore "the asset set"; exit 9; fi
if ! mv "$TMPOUT/yaw_pilot_01_results.html" "$E/yaw_pilot_01_results.html"; then
  restore "the HTML page"; exit 9
fi
if ! mv "$TMPOUT/yaw_pilot_results.md" "$E/yaw_pilot_results.md"; then
  restore "the Markdown report"; exit 9
fi
# All three are published: only now may the rollback copies go.
PUB_STATE=committed
rm -rf "$PREV" "$PREVR"
say "FINISH DONE: $(wc -l < "$G/SHA256SUMS") assets published in $G; report in $E"
