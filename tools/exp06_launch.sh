#!/usr/bin/env bash
# exp_06 launcher (plan v4 sections 5 and 9). Certified invocation for the confirmatory run:
#   nohup setsid tools/exp06_launch.sh full --gpu 1 --reviewed-commit <sha> > <launcher.log> 2>&1 &
# Recovery after a crashed launcher (the child already exited):
#   tools/exp06_launch.sh finalize --gpu 1 --reviewed-commit <sha> \
#       --attempt <dir> --log <child.log> --child-exit <code>
# --dry-run prints every command and path (timestamps as <UTC>) and executes nothing.
# EXP06_LAUNCH_LIB=1 source tools/exp06_launch.sh defines the functions and returns, so the
# child lifecycle (run_child / close_child) is exercised by tests without a mode.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
PYTHON=/home/yixunhu/miniconda3/envs/xRIR/bin/python
DATA_ROOT="${XRIR_DATA_PATH:-/home/yixunhu/data_cache/AcousticRooms}"  # the registered mirror
RECORD=worklog/worklog_yixun/exp_06_oriented_cyl_claude
SMOKE_DIR=ckpt/exp06/_smoke
SMOKE_FLAGS="--epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --num-workers 4 --save-every 0 --no-save"

usage() {
    echo "usage: $0 <smoke|probe|full|finalize> --gpu <g> --reviewed-commit <sha40>" >&2
    echo "       [--attempt-root <dir>] [--attempt <dir> --log <path> --child-exit <n>] [--dry-run]" >&2
    exit 2
}

say() { printf '%s\n' "$*"; }
run() { say "RUN $*"; if [ "${DRY:-0}" -eq 0 ]; then "$@"; fi; }

preflight() {
    run "$PYTHON" tools/exp06_finalize.py preflight --mode "$MODE" --gpu "$GPU" \
        --reviewed-commit "$COMMIT" --attempt-root "$ATTEMPT_ROOT"
}

finalize() {  # finalize <attempt> <log> <child-exit> <run-type> [receipt]
    if [ $# -ge 5 ]; then
        run "$PYTHON" tools/exp06_finalize.py --run-dir "$1" --run-type "$4" --log "$2" \
            --child-exit "$3" --receipt "$5"
    else
        run "$PYTHON" tools/exp06_finalize.py --run-dir "$1" --run-type "$4" --log "$2" \
            --child-exit "$3"
    fi
}

promote() {  # promote <attempt basename>
    say "PROMOTE $ATTEMPT_ROOT/final -> $1"
    if [ "${DRY:-0}" -eq 0 ]; then
        ln -s -- "$1" "$ATTEMPT_ROOT/.final.$$"
        mv -Tf -- "$ATTEMPT_ROOT/.final.$$" "$ATTEMPT_ROOT/final"
    fi
}

abort() {  # abort <run dir> <log> <reason>: the SOP's _ABORTED_<reason> on both
    say "ABORT ${1}_ABORTED_$3"
    if [ "${DRY:-0}" -eq 0 ]; then
        [ ! -e "$1" ] || mv -- "$1" "${1}_ABORTED_$3"
        [ ! -e "$2" ] || mv -- "$2" "${2}_ABORTED_$3"
    fi
}

# diagnostic <run-type> <run dir> <log> <receipt> <smoke argv...>: a smoke or probe runs
# through the same lifecycle as the confirmatory child -- pid file, drained pipe, end
# marker, child_exit.json -- and is finalized as a never-admissible diagnostic.
diagnostic() {
    local kind="$1" dir="$2" log="$3" receipt="$4"
    shift 4
    local cmd=("$PYTHON" tools/exp06_smoke.py "$@")
    say "MKDIR $dir"
    say "SINK cat >> $log"
    say "PIDFILE $dir/launch.pid"
    say "RUN nohup setsid ${cmd[*]}"
    if [ "$DRY" -eq 1 ]; then
        say "MARKER EXP06_CHILD_EXIT <code> <iso> >> $log"
        finalize "$dir" "$log" '<code>' "$kind" "$receipt"
        return 0
    fi
    mkdir -p -- "$dir" "$RECORD"
    : > "$log"
    run_child "$dir" "$log" "${cmd[@]}"
    close_child "$dir" "$log"
    finalize "$dir" "$log" "$CHILD_STATUS" "$kind" "$receipt"
}

# run_child <attempt> <log> <command...>: every byte the child or any descendant writes
# goes through one pipe whose sink appends to the log. Waiting for the sink waits for EOF,
# which arrives only after every process holding the write end has exited -- so nothing can
# append after the end marker. Sets CHILD_PID and CHILD_STATUS.
run_child() {
    local attempt="$1" log="$2" pipe sink pid status=0 sink_status=0
    shift 2
    pipe="$attempt/child.pipe"
    rm -f -- "$pipe"
    mkfifo -m 600 -- "$pipe"
    cat >> "$log" < "$pipe" &
    sink=$!
    CHILD_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
    nohup setsid "$@" > "$pipe" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$attempt/launch.pid"
    wait "$pid" || status=$?
    wait "$sink" || sink_status=$?  # EOF: the child and every descendant closed the write end
    rm -f -- "$pipe"
    CHILD_PID="$pid"
    CHILD_STATUS="$status"
    # Blocker 4: a sink that died on a write error left an incomplete log. Nothing may be
    # published from it -- no end marker, no receipt, no completion.
    if [ "$sink_status" -ne 0 ]; then
        say "SINK_FAILED status=$sink_status log=$log"
        abort "$attempt" "$log" sink_failed
        exit 3
    fi
}

# close_child <attempt> <log>: append the end marker and bind exactly those bytes.
close_child() {
    say "MARKER EXP06_CHILD_EXIT $CHILD_STATUS <iso> >> $2"
    "$PYTHON" tools/exp06_finalize.py child-exit --run-dir "$1" --log "$2" \
        --child-pid "$CHILD_PID" --status "$CHILD_STATUS" --started-at "$CHILD_STARTED_AT"
}

if [ "${EXP06_LAUNCH_LIB:-0}" = 1 ]; then return 0; fi

MODE="${1:-}"
shift || true
case "$MODE" in smoke|probe|full|finalize) ;; *) usage ;; esac
GPU=""; COMMIT=""; ATTEMPT_ROOT=ckpt/exp06/pretrain/xRIR_cylor_8_shot
ATTEMPT=""; LOG=""; CHILD_EXIT=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU="${2:-}"; shift 2 ;;
        --reviewed-commit) COMMIT="${2:-}"; shift 2 ;;
        --attempt-root) ATTEMPT_ROOT="${2:-}"; shift 2 ;;
        --attempt) ATTEMPT="${2:-}"; shift 2 ;;
        --log) LOG="${2:-}"; shift 2 ;;
        --child-exit) CHILD_EXIT="${2:-}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) usage ;;
    esac
done
[ -n "$GPU" ] && [ -n "$COMMIT" ] || usage
[ "$MODE" != finalize ] || { [ -n "$ATTEMPT" ] && [ -n "$LOG" ] && [ -n "$CHILD_EXIT" ]; } || usage

if [ "$DRY" -eq 1 ]; then STAMP='<UTC>'; else STAMP="$(date -u +%Y%m%dT%H%M%S)"; fi

say "EXP06_LAUNCH mode=$MODE gpu=$GPU commit=$COMMIT root=$ATTEMPT_ROOT stamp=$STAMP dry_run=$DRY"
say "ENV CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 OMP_NUM_THREADS=8 XRIR_DATA_PATH=$DATA_ROOT"
if [ "$DRY" -eq 0 ] && [ ! -d "$DATA_ROOT" ]; then
    echo "refusing: XRIR_DATA_PATH $DATA_ROOT is not a directory" >&2
    exit 2
fi
export XRIR_DATA_PATH="$DATA_ROOT"

case "$MODE" in
full)
    preflight
    attempt="$ATTEMPT_ROOT/attempt_$STAMP"
    log="$RECORD/oriented_cyl_${STAMP}_train_full.log"
    say "MKDIR $attempt"
    say "SINK cat >> $log"
    say "PIDFILE $attempt/launch.pid"
    child=("$PYTHON" tools/exp06_train.py --backbone cylindrical_oriented --save-dir "$attempt"
           --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3
           --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12
           --seed 0 --tf32 --log-interval 50 --save-every 500 --epoch-ckpt-every 1
           --run-type full)
    say "RUN nohup setsid ${child[*]}"
    if [ "$DRY" -eq 1 ]; then
        say "MARKER EXP06_CHILD_EXIT <code> <iso> >> $log"
        abort "$attempt" "$log" '<reason>'
        finalize "$attempt" "$log" '<code>' full
        promote "attempt_$STAMP"
        exit 0
    fi
    mkdir -p -- "$ATTEMPT_ROOT"
    mkdir -- "$attempt"  # exclusive: a repeated stamp must not reuse an attempt
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
    if ! finalize "$attempt" "$log" "$status" full; then
        abort "$attempt" "$log" finalize_refused
        exit 2
    fi
    promote "attempt_$STAMP"
    ;;
probe)
    preflight
    # Bounded fit/timing probe (ladder rung 5): the full recipe for 200 batches, nothing saved.
    # The rung-4 ceilings (3 GB / 300 s) do not apply here -- this probe measures the real
    # 32 x 2 footprint on a card the preflight has just found empty.
    export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8
    diagnostic probe "$ATTEMPT_ROOT/probe_$STAMP" "$RECORD/oriented_cyl_${STAMP}_probe.log" \
        "$ATTEMPT_ROOT/probe_$STAMP.json" \
        --entry exp06_train --receipt "$ATTEMPT_ROOT/probe_$STAMP.json" \
        --alarm-seconds 2400 --max-gb 46 -- \
        --backbone cylindrical_oriented --save-dir "$ATTEMPT_ROOT/probe_$STAMP" \
        --epochs 1 --max-train-batches 200 --max-test-batches 20 --no-save --run-type probe \
        --batch-size 32 --accum-steps 2 --tf32 --num-workers 12
    ;;
smoke)
    preflight
    export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8
    # Plan section 9 (a): the trainer and the exp_06 entry on the same bounded argv, TF32 off.
    for entry in trainer exp06_train; do
        name="$entry"
        [ "$entry" = exp06_train ] && name=exp06_train_t0
        diagnostic smoke "$SMOKE_DIR/${name}_$STAMP" \
            "$RECORD/oriented_cyl_${STAMP}_smoke_${name}.log" \
            "$SMOKE_DIR/receipt_${name}_$STAMP.json" \
            --entry "$entry" --receipt "$SMOKE_DIR/receipt_${name}_$STAMP.json" \
            --alarm-seconds 300 --max-gb 3 -- \
            --backbone simple --save-dir "$SMOKE_DIR/t0" $SMOKE_FLAGS
    done
    # (b) the oriented backbone on the same budget.
    diagnostic smoke "$SMOKE_DIR/exp06_train_t1_$STAMP" \
        "$RECORD/oriented_cyl_${STAMP}_smoke_exp06_train_t1.log" \
        "$SMOKE_DIR/receipt_exp06_train_t1_$STAMP.json" \
        --entry exp06_train --receipt "$SMOKE_DIR/receipt_exp06_train_t1_$STAMP.json" \
        --alarm-seconds 300 --max-gb 3 -- \
        --backbone cylindrical_oriented --save-dir "$SMOKE_DIR/t1" $SMOKE_FLAGS
    # (c) the CPU fixture the round-2b HAA smokes load (no child, no log, no completion).
    run "$PYTHON" tools/exp06_smoke.py --make-fixture "$SMOKE_DIR/fixture_cylor.pth"
    ;;
finalize)
    preflight  # recovery is gated by the same reviewed commit, clean tree and live-pid checks
    [ "$DRY" -eq 0 ] || abort "$ATTEMPT" "$LOG" '<reason>'
    if ! finalize "$ATTEMPT" "$LOG" "$CHILD_EXIT" full; then
        abort "$ATTEMPT" "$LOG" '<reason>'
        exit 2
    fi
    promote "$(basename -- "$ATTEMPT")"
    ;;
esac
