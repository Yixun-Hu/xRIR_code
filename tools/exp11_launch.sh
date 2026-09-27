#!/usr/bin/env bash
# exp_11 launcher (plan v3 items 6 and 12). Certified invocation for a confirmatory run:
#   nohup setsid tools/exp11_launch.sh full --arm H --gpu 1 --reviewed-commit <sha> \
#       > <launcher.log> 2>&1 &
# Recovery after a crashed launcher (the child already exited):
#   tools/exp11_launch.sh finalize --arm H --gpu 1 --reviewed-commit <sha> \
#       --attempt <dir> --log <child.log> --child-exit <code>
# --dry-run prints every command and path (timestamps as <UTC>) and executes nothing.
#
# The child lifecycle -- exclusive directory, one drained pipe, the end marker, the exit
# receipt -- is tools/exp06_launch.sh's and is sourced here as a library; it is shared
# evidence, not an exp_11 decision, and forking it would fork the very bytes closed_log
# re-validates. Both files are bound by exp_11's `launch_sh` approvals key. What this
# script overrides is everything that names an experiment: the finalizer it calls
# (tools/exp11_finalize.py), the approvals, the attempt roots, the record and the arms.
#
# EXP11_LAUNCH_LIB=1 source tools/exp11_launch.sh defines the functions and returns, so
# the pipeline can reuse them without a mode.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
EXP06_LAUNCH_LIB=1 source tools/exp06_launch.sh
export PYTHONPATH="$PWD"

RECORD=worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude
APPROVED_DEFAULT="$RECORD/orientation_cue_fairness_results_assets/approved_digests.json"
SMOKE_DIR=ckpt/exp11/_smoke
PRETRAIN_ROOT="${EXP11_PRETRAIN_ROOT:-ckpt/exp11/pretrain}"
SIMPOR_ROOT="$PRETRAIN_ROOT/xRIR_simpor_8_shot"          # arm H
SIMPOR_YAW_ROOT="$PRETRAIN_ROOT/xRIR_simpor_yawaug_8_shot"   # arm I
HEADING_DIR="${EXP06_HEADING_DIR:-ckpt/exp06/heading}"
SMOKE_FLAGS="--epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --num-workers 4 --no-save"
SMOKE_ALARM_S="${EXP11_SMOKE_ALARM_S:-300}"
SMOKE_MAX_GB="${EXP11_SMOKE_MAX_GB:-6}"
# Plan section 7 reserves 36 h per pretraining; the child runs under that hard ceiling, so
# a stalled attempt fails and is aborted instead of holding the card indefinitely. The
# value goes straight to GNU timeout, where 0 means "no timeout", so an override is
# validated before anything starts and may only TIGHTEN the registered ceiling.
FULL_CEILING_MAX=129600
FULL_CEILING_S="${EXP11_FULL_CEILING_S-$FULL_CEILING_MAX}"

check_ceiling() {
    case "$FULL_CEILING_S" in
        ''|*[!0-9]*)
            echo "refusing: EXP11_FULL_CEILING_S must be a positive whole number of" \
                 "seconds, not '$FULL_CEILING_S'" >&2; return 1 ;;
    esac
    if [ "$FULL_CEILING_S" -le 0 ] || [ "$FULL_CEILING_S" -gt "$FULL_CEILING_MAX" ]; then
        echo "refusing: EXP11_FULL_CEILING_S $FULL_CEILING_S is not in (0," \
             "$FULL_CEILING_MAX]; an override may only tighten the 36 h ceiling" >&2
        return 1
    fi
}

# check_attempt <attempt> <root> <arm>: recovery may promote only this arm's own attempt.
# The basename is linked under the root --arm selects, so an attempt of the other arm
# would leave a dangling or wrong `final`; the directory AND the profile its recorded
# arguments select must both be the selected arm's.
CHECK_ATTEMPT_PY='
import json, sys
from pathlib import Path
from tools import exp11_recipe
attempt, root, arm = sys.argv[1:4]
expected = {"H": "H_RECIPE", "I": "I_RECIPE"}[arm]
try:
    directory, base = Path(attempt).resolve(), Path(root).resolve()
    if directory.parent != base:
        raise ValueError("attempt {} is not an attempt of the arm {} root {}".format(
            directory, arm, base))
    args = json.loads((directory / "args.json").read_text())
    profile = exp11_recipe.select_profile(args)
    if profile != expected:
        raise ValueError("attempt {} recorded the profile {}, not the {} of arm {}".format(
            directory, profile, expected, arm))
except (OSError, ValueError, KeyError) as error:
    raise SystemExit("refusing: " + str(error))
print("ATTEMPT ok {} arm={} profile={}".format(attempt, arm, expected))
'

check_attempt() {  # check_attempt <attempt>
    say "ATTEMPTCHECK $1 arm=$ARM root=$ARM_ROOT"
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -c "$CHECK_ATTEMPT_PY" "$1" "$ARM_ROOT" "$ARM"
}
ARM=""
ARM_BACKBONE=simple_oriented
ARM_YAW=0
ARM_SAVE_EVERY=500

usage() {
    echo "usage: $0 <smoke|probe|full|finalize> --arm <H|I> --gpu <g> --reviewed-commit <sha40>" >&2
    echo "       [--attempt-root <dir>] [--approved <json>] [--exploratory]" >&2
    echo "       [--attempt <dir> --log <path> --child-exit <n>] [--dry-run]" >&2
    exit 2
}

# arm_of <H|I>: the attempt root, augmentation and saving cadence of one pretraining arm.
# H is exp_01's recipe (save_every 500); I is exp_04's, whose constraints the trainer
# re-applies (--save-every 0 and no --resume).
arm_of() { case "$1" in
  H) ARM_ROOT="$SIMPOR_ROOT"; ARM_YAW=0; ARM_SAVE_EVERY=500;;
  I) ARM_ROOT="$SIMPOR_YAW_ROOT"; ARM_YAW=1; ARM_SAVE_EVERY=0;;
  *) echo "refusing: unknown arm $1 (H or I)" >&2; return 1;; esac
  ARM_BACKBONE=simple_oriented; }

# Every gate before a child starts, through exp_11's own finalizer: HEAD at the reviewed
# commit, a tree clean outside worklog/, no live launch.pid or child.pid under either
# attempt root or the smoke tree, exp_11's approvals, and -- for probe and full -- a card
# with no compute apps at all.
preflight() {
    local extra=()
    [ "${EXPLORATORY:-0}" -eq 0 ] || extra+=(--exploratory)
    [ "$MODE" != smoke ] || extra+=(--min-free-gb "$SMOKE_MAX_GB")
    run "$PYTHON" tools/exp11_finalize.py preflight --mode "$MODE" --gpu "$GPU" \
        --reviewed-commit "$COMMIT" --attempt-root "$SIMPOR_ROOT" \
        --attempt-root "$SIMPOR_YAW_ROOT" --attempt-root "$SMOKE_DIR" \
        --approved "$APPROVED" ${extra[@]+"${extra[@]}"}
}

finalize() {  # finalize <run dir> <log> <child-exit> <run-type> [receipt]
    local extra=()
    [ -z "${OWNER:-}" ] || extra+=(--owner-pid "$OWNER")
    [ $# -lt 5 ] || extra+=(--receipt "$5")
    run "$PYTHON" tools/exp11_finalize.py --run-dir "$1" --run-type "$4" --log "$2" \
        --child-exit "$3" ${extra[@]+"${extra[@]}"}
}

require_passed() {  # require_passed <run dir>: the next rung consumes this one's checkpoint
    if ! run "$PYTHON" tools/exp11_finalize.py passed --run-dir "$1" --run-type exp11_smoke; then
        say "STOP $1 did not pass; the rung that consumes its checkpoint is not started"
        exit 2
    fi
}

promote() {  # promote <attempt basename>: atomic, so `final` never points at nothing
    say "PROMOTE $ARM_ROOT/final -> $1"
    if [ "${DRY:-0}" -eq 0 ]; then
        ln -s -- "$1" "$ARM_ROOT/.final.$$"
        mv -Tf -- "$ARM_ROOT/.final.$$" "$ARM_ROOT/final"
    fi
}

# diagnostic <kind> <run dir> <log> <receipt> <entry argv...>: one enumerated diagnostic
# kind through the same lifecycle as a confirmatory child, finalized as exp11_smoke and
# never admissible as an arm.
diagnostic() {
    local kind="$1" dir="$2" log="$3" receipt="$4"
    shift 4
    local flags=(--kind "$kind" --receipt "$receipt" --provenance-out "$dir/provenance.json"
                 --approved "$APPROVED" --reviewed-commit "$COMMIT")
    [ "${EXPLORATORY:-0}" -eq 0 ] || flags+=(--exploratory)
    local cmd=("$PYTHON" tools/exp11_smoke.py "${flags[@]}" "$@")
    say "MKDIR $dir"
    say "SINK cat >> $log"
    say "PIDFILE $dir/launch.pid"
    say "RUN nohup setsid ${cmd[*]}"
    if [ "$DRY" -eq 1 ]; then
        say "MARKER EXP06_CHILD_EXIT <code> <iso> >> $log"
        finalize "$dir" "$log" '<code>' exp11_smoke "$receipt"
        return 0
    fi
    mkdir -p -- "$(dirname -- "$dir")" "$RECORD"
    mkdir -- "$dir"      # exclusive: a repeated stamp must not reuse a diagnostic run
    own_launch "$dir"
    : > "$log"
    run_child "$dir" "$log" "${cmd[@]}"
    close_child "$dir" "$log"
    if ! finalize "$dir" "$log" "$CHILD_STATUS" exp11_smoke "$receipt"; then
        abort "$dir" "$log" finalize_refused
        exit 2
    fi
    if [ "$CHILD_STATUS" -ne 0 ]; then
        abort "$dir" "$log" "child_failed_$CHILD_STATUS"
        exit "$CHILD_STATUS"
    fi
}

# The sourced library already resolved APPROVED to exp_06's record, so exp_11's default
# is assigned unconditionally; EXP11_APPROVED (or --approved) overrides it.
APPROVED="${EXP11_APPROVED:-$APPROVED_DEFAULT}"
COMMIT="${COMMIT:-}"
EXPLORATORY="${EXPLORATORY:-0}"
ARM_ROOT="${ARM_ROOT:-$SIMPOR_ROOT}"

if [ "${EXP11_LAUNCH_LIB:-0}" = 1 ]; then return 0; fi

MODE="${1:-}"
shift || true
case "$MODE" in smoke|probe|full|finalize) ;; *) usage ;; esac
GPU=""; COMMIT=""; ATTEMPT=""; LOG=""; CHILD_EXIT=""; DRY=0; ARM=H
while [ $# -gt 0 ]; do
    case "$1" in
        --arm) ARM="${2:-}"; shift 2 ;;
        --gpu) GPU="${2:-}"; shift 2 ;;
        --reviewed-commit) COMMIT="${2:-}"; shift 2 ;;
        --attempt) ATTEMPT="${2:-}"; shift 2 ;;
        --log) LOG="${2:-}"; shift 2 ;;
        --child-exit) CHILD_EXIT="${2:-}"; shift 2 ;;
        --approved) APPROVED="${2:-}"; shift 2 ;;
        --exploratory) EXPLORATORY=1; shift ;;
        --dry-run) DRY=1; shift ;;
        *) usage ;;
    esac
done
[ -n "$GPU" ] && [ -n "$COMMIT" ] || usage
arm_of "$ARM" || exit 2
check_ceiling || exit 2
[ "$MODE" != finalize ] || { [ -n "$ATTEMPT" ] && [ -n "$LOG" ] && [ -n "$CHILD_EXIT" ]; } || usage

if [ "$DRY" -eq 1 ]; then STAMP='<UTC>'; else STAMP="$(date -u +%Y%m%dT%H%M%S)"; fi
if [ "$MODE" = finalize ]; then OWNER=""        # recovery owns no live launch.pid
elif [ "$DRY" -eq 1 ]; then OWNER='<pid>'
else OWNER="$$"; fi

say "EXP11_LAUNCH mode=$MODE arm=$ARM gpu=$GPU commit=$COMMIT root=$ARM_ROOT stamp=$STAMP dry_run=$DRY"
say "ENV CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 OMP_NUM_THREADS=8 XRIR_DATA_PATH=$DATA_ROOT"
if [ "$DRY" -eq 0 ] && [ ! -d "$DATA_ROOT" ]; then
    echo "refusing: XRIR_DATA_PATH $DATA_ROOT is not a directory" >&2
    exit 2
fi
export XRIR_DATA_PATH="$DATA_ROOT"
CHILD_EXPLORATORY=()
[ "$EXPLORATORY" -eq 0 ] || CHILD_EXPLORATORY=(--exploratory)

case "$MODE" in
full)
    preflight
    attempt="$ARM_ROOT/attempt_$STAMP"
    log="$RECORD/orientation_cue_fairness_${STAMP}_train_full_${ARM}.log"
    say "MKDIR $attempt"
    say "SINK cat >> $log"
    say "PIDFILE $attempt/launch.pid"
    say "CEILING ${FULL_CEILING_S}s"
    child=(timeout --kill-after=60 "$FULL_CEILING_S"
           "$PYTHON" tools/exp11_train.py --backbone "$ARM_BACKBONE" --save-dir "$attempt"
           --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3
           --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12
           --seed 0 --tf32 --log-interval 50 --save-every "$ARM_SAVE_EVERY"
           --epoch-ckpt-every 1 --yaw-aug "$ARM_YAW" --yaw-aug-seed 0 --yaw-aug-width 512
           --run-type full --approved "$APPROVED" --reviewed-commit "$COMMIT")
    say "RUN nohup setsid ${child[*]}"
    if [ "$DRY" -eq 1 ]; then
        say "MARKER EXP06_CHILD_EXIT <code> <iso> >> $log"
        abort "$attempt" "$log" 'child_exit_<code>'
        finalize "$attempt" "$log" '<code>' exp11_train
        promote "attempt_$STAMP"
        exit 0
    fi
    mkdir -p -- "$ARM_ROOT"
    mkdir -- "$attempt"   # exclusive: a repeated stamp must not reuse an attempt
    own_launch "$attempt"
    mkdir -p -- "$RECORD"
    : > "$log"
    export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8
    run_child "$attempt" "$log" "${child[@]}"
    close_child "$attempt" "$log"
    status="$CHILD_STATUS"
    if [ "$status" -ne 0 ]; then
        abort "$attempt" "$log" "child_exit_$status"
        exit "$status"
    fi
    if ! finalize "$attempt" "$log" "$status" exp11_train; then
        abort "$attempt" "$log" finalize_refused
        exit 2
    fi
    promote "attempt_$STAMP"
    ;;
probe)
    preflight
    export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8
    diagnostic probe "$ARM_ROOT/probe_$STAMP" \
        "$RECORD/orientation_cue_fairness_${STAMP}_probe_${ARM}.log" \
        "$ARM_ROOT/probe_$STAMP.json" -- \
        --backbone "$ARM_BACKBONE" --save-dir "$ARM_ROOT/probe_$STAMP" \
        --epochs 1 --max-train-batches 200 --max-test-batches 20 --no-save --run-type probe \
        --batch-size 32 --accum-steps 2 --tf32 --num-workers 12 --decay-epochs 3 \
        --yaw-aug "$ARM_YAW" --save-every 0 --log-interval 50 \
        ${CHILD_EXPLORATORY[@]+"${CHILD_EXPLORATORY[@]}"}
    ;;
smoke)
    preflight
    export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8
    # (a) the oriented backbone on a bounded budget, through exp_11's training entry.
    diagnostic probe "$SMOKE_DIR/exp11_train_$STAMP" \
        "$RECORD/orientation_cue_fairness_${STAMP}_smoke_exp11_train.log" \
        "$SMOKE_DIR/receipt_exp11_train_$STAMP.json" \
        --alarm-seconds "$SMOKE_ALARM_S" --max-gb "$SMOKE_MAX_GB" -- \
        --backbone "$ARM_BACKBONE" --save-dir "$SMOKE_DIR/t0" $SMOKE_FLAGS \
        --save-every 0 --yaw-aug "$ARM_YAW" --run-type probe \
        ${CHILD_EXPLORATORY[@]+"${CHILD_EXPLORATORY[@]}"}
    # (b) the CPU fixtures the HAA smokes load; the exp_01 checkpoints cannot load into
    # the five-channel patch embedding, and the adapter needs its own keys.
    run "$PYTHON" tools/exp11_smoke.py --make-fixture "$SMOKE_DIR/fixture_simpor.pth" \
        --fixture-backbone simple_oriented
    run "$PYTHON" tools/exp11_smoke.py --make-fixture "$SMOKE_DIR/fixture_simpadapter.pth" \
        --fixture-backbone simple_adapter
    # (c) the HAA fine-tuning smoke in the heading frame, then (d) the evaluation smoke
    # that loads its best.pth -- gated on (c)'s own completion certifying it passed.
    h1="$SMOKE_DIR/haa_finetune_$STAMP"
    diagnostic haa_train_smoke "$h1" \
        "$RECORD/orientation_cue_fairness_${STAMP}_smoke_haa_finetune.log" \
        "$SMOKE_DIR/receipt_haa_finetune_$STAMP.json" \
        --alarm-seconds "$SMOKE_ALARM_S" --max-gb "$SMOKE_MAX_GB" -- \
        --backbone "$ARM_BACKBONE" --init "$SMOKE_DIR/fixture_simpor.pth" \
        --rooms class_room --heading-json-dir "$HEADING_DIR" --save-dir "$h1/run" \
        --epochs 2 --val-every 1 --batch-size 4 --val-batch-size 4 --seed 0
    require_passed "$h1"
    h2="$SMOKE_DIR/haa_eval_$STAMP"
    diagnostic haa_eval_smoke "$h2" \
        "$RECORD/orientation_cue_fairness_${STAMP}_smoke_haa_eval.log" \
        "$SMOKE_DIR/receipt_haa_eval_$STAMP.json" \
        --alarm-seconds "$SMOKE_ALARM_S" --max-gb "$SMOKE_MAX_GB" -- \
        --backbone "$ARM_BACKBONE" --checkpoint "$h1/run/best.pth" \
        --heading-json-dir "$HEADING_DIR" --rooms hallway --max-samples 4 \
        --save-dir "$h2/run" --seed 0
    # (e)/(f) the ADAPTER route of arms J and K. The oriented rungs above prove nothing
    # about it: a different backbone, the --adapter-heading-json-dir cue instead of the
    # frame, and the contextual loading mode (base weights first, the trained adapter
    # afterwards). Both rungs run here so J/K have a reviewed smoke before their queue.
    a1="$SMOKE_DIR/haa_adapter_finetune_$STAMP"
    diagnostic haa_train_smoke "$a1" \
        "$RECORD/orientation_cue_fairness_${STAMP}_smoke_haa_adapter_finetune.log" \
        "$SMOKE_DIR/receipt_haa_adapter_finetune_$STAMP.json" \
        --alarm-seconds "$SMOKE_ALARM_S" --max-gb "$SMOKE_MAX_GB" -- \
        --backbone simple_adapter --init "$SMOKE_DIR/fixture_simpadapter.pth" \
        --rooms class_room --adapter-heading-json-dir "$HEADING_DIR" \
        --save-dir "$a1/run" --epochs 2 --val-every 1 --batch-size 4 \
        --val-batch-size 4 --seed 0
    require_passed "$a1"
    a2="$SMOKE_DIR/haa_adapter_eval_$STAMP"
    diagnostic haa_eval_smoke "$a2" \
        "$RECORD/orientation_cue_fairness_${STAMP}_smoke_haa_adapter_eval.log" \
        "$SMOKE_DIR/receipt_haa_adapter_eval_$STAMP.json" \
        --alarm-seconds "$SMOKE_ALARM_S" --max-gb "$SMOKE_MAX_GB" -- \
        --backbone simple_adapter --checkpoint "$a1/run/best.pth" \
        --adapter-heading-json-dir "$HEADING_DIR" --rooms hallway --max-samples 4 \
        --save-dir "$a2/run" --seed 0
    ;;
finalize)
    preflight   # recovery is gated by the same reviewed commit, clean tree and pid checks
    check_attempt "$ATTEMPT" || exit 2
    [ "$DRY" -eq 0 ] || abort "$ATTEMPT" "$LOG" finalize_refused
    if ! finalize "$ATTEMPT" "$LOG" "$CHILD_EXIT" exp11_train; then
        abort "$ATTEMPT" "$LOG" finalize_refused
        exit 2
    fi
    promote "$(basename -- "$ATTEMPT")"
    ;;
esac
