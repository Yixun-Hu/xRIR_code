#!/bin/bash
# exp_10 yaw_pilot — one arm end to end: probe (+controls) → check-online → parity vs exp_03 → gate → full run → check-online → parity (full).
# usage: exp10_arm_chain.sh <arm> <backbone> <checkpoint> <manifest> <manifest_hash> <num_shot> <device cpu|cuda> [gpu_index] [exp03_per_sample]
# Gate = comparator `ok` on the probe parity (all §7 criteria + A1 masks) AND probe controls ok (summariser controls table).
#
# The predecessor is NOT a free argument (Codex tooling review, finding 8): it comes from the
# arm table below, `none` is only released_k1's entry, and a 9th argument is accepted only if it
# repeats the table's path. Every comparator report is read back and required to say `ok: true`
# (finding 7) — the comparator exits 0 while reporting a failure, which is how a failed online
# check reached the full launch. Each stage's output directory must not exist and is reserved
# with `mkdir` before the evaluator starts (finding 9). The clean-tree check exempts only
# worklog narrative (`*.md`) and logs (`*.log`), and the three tools plus this script are hashed
# at the probe stage and re-verified before the full stage (finding 10). Those pins are this
# invocation's own: they are written read-only into the probe directory it reserved, so a
# concurrent chain that is refused cannot have replaced the baseline first (round-4 finding 6).
#
# env: EXP10_REPO_ROOT — repository root override for the tests (default: the repository path).
set -uo pipefail
SELF=$(readlink -f "$0")
ARM=$1; BB=$2; CK=$3; MAN=$4; HASH=$5; K=$6; DEV=$7; GPU=${8:-1}; EXP03_ARG=${9:-}
REPO=${EXP10_REPO_ROOT:-/home/yixunhu/codespace/xRIR_code}
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate xRIR
export PYTHONPATH=${PYTHONPATH:-}:$(pwd); export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
if [ "$DEV" = "cuda" ]; then export CUDA_VISIBLE_DEVICES=$GPU; else export CUDA_VISIBLE_DEVICES=""; fi
E=worklog/worklog_yixun/exp_10_yaw_pilot_claude
LOG=$E/yaw_pilot_$(date +%Y-%m-%d_%H:%M:%S)_${ARM}_chain_${DEV}.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
PINS=ckpt/exp10/${ARM}_probe/source_pins.sha256   # written inside the reserved probe dir
SOURCES="tools/exp10_yaw_pilot.py tools/exp10_compare.py tools/exp10_summarize.py $SELF"

# The tree may carry this experiment's narrative and its logs, nothing else: a dirty script or
# module — under worklog/ as much as anywhere else — is a different program from the reviewed
# one (finding 10).
DIRTY=$(git status --porcelain | awk '{p = substr($0, 4); if (p ~ /^worklog\/.*\.(md|log)$/) next; print}')
if [ -n "$DIRTY" ]; then say "REFUSED: tree dirty outside worklog narrative/logs:"; echo "$DIRTY" | tee -a "$LOG"; exit 3; fi

case "$ARM" in                                   # finding 8: the predecessor table
  released_k8) EXP03=ckpt/yaw_rotation/sweep_released/per_sample_yaw.json ;;
  control_k8)  EXP03=ckpt/yaw_rotation/sweep_control/per_sample_yaw.json ;;
  cyl_k8)      EXP03=ckpt/yaw_rotation/sweep_cyl/per_sample_yaw.json ;;
  released_k1) EXP03=none ;;                     # no exp_03 predecessor: controls only
  *) say "REFUSED: unknown arm $ARM (the record's arms are released_k8, released_k1, control_k8, cyl_k8)"; exit 3 ;;
esac
if [ -n "$EXP03_ARG" ] && [ "$EXP03_ARG" != "$EXP03" ]; then
  say "REFUSED: arm $ARM's exp_03 predecessor is $EXP03, not the $EXP03_ARG given"; exit 3
fi
say "HEAD $(git rev-parse HEAD) arm=$ARM backbone=$BB ckpt=$CK manifest=$MAN K=$K device=$DEV gpu=$GPU exp03=$EXP03"

require_ok() {  # <report json> <label> — the comparator exits 0 even when it reports ok=false
  local path=$1 label=$2 ok
  ok=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('ok') is True)" "$path" 2>/dev/null || echo unreadable)
  if [ "$ok" != "True" ]; then say "REFUSED: $label reports ok=$ok ($path)"; return 1; fi
  say "$label ok"
  return 0
}
run_stage() {  # <batches> <controls-flag>
  local B=$1 CF=$2 OUT=ckpt/exp10/${ARM}_$1
  if [ -e "$OUT" ]; then say "REFUSED: $OUT already exists (a partial or previous run)"; return 2; fi
  mkdir -p ckpt/exp10
  if ! mkdir "$OUT"; then say "REFUSED: could not reserve $OUT (a concurrent chain?)"; return 2; fi
  say "stage $B → $OUT (reserved)"
  if [ "$B" = "probe" ]; then
    # The source baseline belongs to THIS invocation, so it is written inside the directory
    # this invocation just reserved and made read-only (finding 6). It used to be a shared
    # per-arm file written before the reservation: a second chain replaced it and only then
    # refused the existing probe, and the first chain accepted the replacement as its own
    # baseline and ran the full stage with a changed evaluator.
    sha256sum $SOURCES > "$PINS" || { say "REFUSED: could not pin the sources in $PINS"; return 3; }
    chmod 0444 "$PINS"
    say "source pins ($PINS): $(awk '{printf "%s %s; ", substr($1,1,12), $2}' "$PINS")"
  fi
  python tools/exp10_yaw_pilot.py --arm "$ARM" --backbone "$BB" --checkpoint "$CK" --manifest "$MAN" --manifest-hash "$HASH" --num-shot "$K" \
    --ks 0,64,128,256,384 --batches "$B" $CF --device "$DEV" --threads 16 --out-dir "$OUT" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then say "stage $B FAILED rc=$rc"; return $rc; fi
  python tools/exp10_compare.py check-online "$OUT" --json "$OUT/check_online.json" 2>&1 | tee -a "$LOG" || { say "check-online FAILED to run"; return 4; }
  require_ok "$OUT/check_online.json" "check-online ($B)" || return 4
  if [ "$EXP03" != "none" ]; then
    python tools/exp10_compare.py parity-exp03 "$OUT" "$EXP03" --json "$OUT/parity_exp03.json" 2>&1 | tee -a "$LOG" || { say "parity FAILED to run"; return 5; }
    require_ok "$OUT/parity_exp03.json" "parity vs exp_03 ($B)" || return 5
  fi
  return 0
}

mkdir -p ckpt/exp10
run_stage probe --controls || exit $?
PROBE=ckpt/exp10/${ARM}_probe
python tools/exp10_summarize.py --runs "$PROBE" --out "$PROBE/summary" --n-boot 2000 2>&1 | tee -a "$LOG" || { say "probe summary FAILED"; exit 6; }
CTRL_OK=$(python -c "import json,sys; s=json.load(open('$PROBE/summary/yaw_pilot_summary.json')); a=s['arms'][0] if isinstance(s.get('arms'),list) else list(s['arms'].values())[0]; print(a['controls']['ok'])" 2>/dev/null || echo unknown)
if [ "$EXP03" != "none" ]; then PAR_OK=$(python -c "import json; print(json.load(open('$PROBE/parity_exp03.json'))['ok'])"); else PAR_OK=n/a; fi
say "GATE: controls_ok=$CTRL_OK parity_ok=$PAR_OK"
if [ "$CTRL_OK" != "True" ] || { [ "$EXP03" != "none" ] && [ "$PAR_OK" != "True" ]; }; then say "GATE FAILED — full run NOT launched"; exit 7; fi
# finding 10: the full stage has to run the same program the probe validated, and finding 6:
# the baseline it is checked against is the one this invocation wrote into its own probe dir.
if [ ! -f "$PINS" ]; then
  say "REFUSED: this invocation's source pins are missing ($PINS) — full run NOT launched"; exit 8
fi
if ! sha256sum -c --quiet "$PINS" 2>&1 | tee -a "$LOG"; then
  say "REFUSED: a pinned source changed since the probe (see $PINS) — full run NOT launched"; exit 8
fi
say "GATE PASSED — sources unchanged since the probe; launching full run"
run_stage all "" || exit $?
say "ARM DONE"
