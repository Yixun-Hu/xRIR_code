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
#     kept until every one of the three publications has succeeded (round-4 review, finding 2).
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
PREV=$A/generated.previous.$$          # the asset set this run replaces
PREVR=$A/.finish_prev.$$               # the Markdown / HTML this run replaces
REPORTS="yaw_pilot_01_results.html yaw_pilot_results.md"
cleanup() {
  rc=$?
  if [ -d "$STAGE" ]; then rm -rf "$STAGE"; fi
  if [ -d "$TMPOUT" ]; then rm -rf "$TMPOUT"; fi
  # A death between the three publications (an unguarded error, a signal) must not leave a
  # backup behind as the only copy of the previous publication.
  if [ -d "$PREV" ]; then
    if [ ! -d "$G" ]; then mv "$PREV" "$G"; else rm -rf "$PREV"; fi
  fi
  if [ -d "$PREVR" ]; then
    for f in $REPORTS; do
      if [ -f "$PREVR/$f" ] && [ ! -f "$E/$f" ]; then mv "$PREVR/$f" "$E/$f"; fi
    done
    rm -rf "$PREVR"
  fi
  if [ "$rc" -ne 0 ]; then say "FINISH FAILED rc=$rc — $G, the Markdown report and the HTML page are the publication that was there before"; fi
}
trap cleanup EXIT
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
# now kept until all three publications have succeeded.
restore() {   # <what failed> — put all three artefacts back exactly as they were
  local what=$1 restored=""
  if [ -d "$PREV" ]; then
    rm -rf "$G"
    if mv "$PREV" "$G"; then restored="$restored $G"; fi
  elif [ -d "$G" ]; then
    rm -rf "$G"; restored="$restored $G (removed: this run created it)"
  fi
  for f in $REPORTS; do
    if [ -f "$PREVR/$f" ]; then
      rm -f "$E/$f"
      if mv "$PREVR/$f" "$E/$f"; then restored="$restored $E/$f"; fi
    elif [ -f "$E/$f" ]; then
      rm -f "$E/$f"; restored="$restored $E/$f (removed: this run created it)"
    fi
  done
  rm -rf "$PREVR"
  say "REFUSED: publishing $what failed; restored:$restored"
}
mkdir -p "$PREVR"
for f in $REPORTS; do
  if [ -f "$E/$f" ]; then cp -p "$E/$f" "$PREVR/$f"; fi
done
if [ -d "$G" ]; then mv "$G" "$PREV"; fi
if ! mv "$STAGE" "$G"; then restore "the asset set"; exit 9; fi
if ! mv "$TMPOUT/yaw_pilot_01_results.html" "$E/yaw_pilot_01_results.html"; then
  restore "the HTML page"; exit 9
fi
if ! mv "$TMPOUT/yaw_pilot_results.md" "$E/yaw_pilot_results.md"; then
  restore "the Markdown report"; exit 9
fi
# All three are published: only now may the rollback copies go.
if [ -d "$PREV" ]; then rm -rf "$PREV"; fi
rm -rf "$PREVR"
say "FINISH DONE: $(wc -l < "$G/SHA256SUMS") assets published in $G; report in $E"
