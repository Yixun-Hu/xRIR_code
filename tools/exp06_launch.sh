#!/usr/bin/env bash
# exp_06 launcher (plan v4 sections 5 and 9). Certified invocation for the confirmatory run:
#   nohup setsid tools/exp06_launch.sh full --gpu 1 --reviewed-commit <sha> > <launcher.log> 2>&1 &
# Recovery after a crashed launcher (the child already exited):
#   tools/exp06_launch.sh finalize --gpu 1 --reviewed-commit <sha> \
#       --attempt <dir> --log <child.log> --child-exit <code>
# --dry-run prints every command and path (timestamps as <UTC>) and executes nothing.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
PYTHON=/home/yixunhu/miniconda3/envs/xRIR/bin/python
RECORD=worklog/worklog_yixun/exp_06_oriented_cyl_claude
SMOKE_DIR=ckpt/exp06/_smoke
SMOKE_FLAGS="--epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --num-workers 4 --save-every 0 --no-save"

usage() {
    echo "usage: $0 <smoke|probe|full|finalize> --gpu <g> --reviewed-commit <sha40>" >&2
    echo "       [--attempt-root <dir>] [--attempt <dir> --log <path> --child-exit <n>] [--dry-run]" >&2
    exit 2
}

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
say() { printf '%s\n' "$*"; }
run() { say "RUN $*"; if [ "$DRY" -eq 0 ]; then "$@"; fi; }

preflight() {
    run "$PYTHON" tools/exp06_finalize.py preflight --mode "$MODE" --gpu "$GPU" \
        --reviewed-commit "$COMMIT" --attempt-root "$ATTEMPT_ROOT"
}

finalize() {  # finalize <attempt> <log> <child-exit> <run-type>
    run "$PYTHON" tools/exp06_finalize.py --run-dir "$1" --run-type "$4" --log "$2" --child-exit "$3"
}

say "EXP06_LAUNCH mode=$MODE gpu=$GPU commit=$COMMIT root=$ATTEMPT_ROOT stamp=$STAMP dry_run=$DRY"
say "ENV CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 OMP_NUM_THREADS=8"

case "$MODE" in
full)
    preflight
    attempt="$ATTEMPT_ROOT/attempt_$STAMP"
    log="$RECORD/oriented_cyl_${STAMP}_train_full.log"
    say "MKDIR $attempt"
    say "TEE $log"
    say "PIDFILE $attempt/launch.pid"
    child=("$PYTHON" tools/exp06_train.py --backbone cylindrical_oriented --save-dir "$attempt"
           --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3
           --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12
           --seed 0 --tf32 --log-interval 50 --save-every 500 --epoch-ckpt-every 1
           --run-type full)
    say "RUN nohup setsid ${child[*]}"
    if [ "$DRY" -eq 1 ]; then
        say "MARKER EXP06_CHILD_EXIT <code> <iso> >> $log"
        say "ABORT ${attempt}_ABORTED_<reason>"
        finalize "$attempt" "$log" '<code>' full
        say "PROMOTE $ATTEMPT_ROOT/final -> attempt_$STAMP"
        exit 0
    fi
    mkdir -p -- "$ATTEMPT_ROOT"
    mkdir -- "$attempt"  # exclusive: a repeated stamp must not reuse an attempt
    mkdir -p -- "$RECORD"
    status=0
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 \
        nohup setsid "${child[@]}" > "$log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$attempt/launch.pid"
    tail -f --pid="$pid" "$log" &  # live view only; the child owns the file
    wait "$pid" || status=$?
    printf 'EXP06_CHILD_EXIT %s %s\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" >> "$log"
    if [ "$status" -ne 0 ]; then
        aborted="${attempt}_ABORTED_child_exit_$status"
        say "ABORT $aborted"
        mv -- "$attempt" "$aborted"
        exit "$status"
    fi
    if ! finalize "$attempt" "$log" "$status" full; then
        aborted="${attempt}_ABORTED_finalize_refused"
        say "ABORT $aborted"
        mv -- "$attempt" "$aborted"
        exit 2
    fi
    ln -s -- "attempt_$STAMP" "$ATTEMPT_ROOT/.final.$$"
    mv -Tf -- "$ATTEMPT_ROOT/.final.$$" "$ATTEMPT_ROOT/final"
    say "PROMOTE $ATTEMPT_ROOT/final -> attempt_$STAMP"
    ;;
probe)
    preflight
    # Bounded fit/timing probe (ladder rung 5): the full recipe for 200 batches, nothing saved.
    # The rung-4 ceilings (3 GB / 300 s) do not apply here -- this probe measures the real
    # 32 x 2 footprint on a card the preflight has just found empty.
    [ "$DRY" -eq 0 ] && mkdir -p -- "$ATTEMPT_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 \
    run "$PYTHON" tools/exp06_smoke.py --entry exp06_train \
        --receipt "$ATTEMPT_ROOT/probe_$STAMP.json" --alarm-seconds 2400 --max-gb 46 -- \
        --backbone cylindrical_oriented --save-dir "$ATTEMPT_ROOT/probe_$STAMP" \
        --epochs 1 --max-train-batches 200 --max-test-batches 20 --no-save --run-type probe \
        --batch-size 32 --accum-steps 2 --tf32 --num-workers 12
    ;;
smoke)
    preflight
    [ "$DRY" -eq 0 ] && mkdir -p -- "$SMOKE_DIR"
    # Plan section 9 (a): the trainer and the exp_06 entry on the same bounded argv, TF32 off.
    for entry in trainer exp06_train; do
        name="$entry"
        [ "$entry" = exp06_train ] && name=exp06_train_t0
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 \
        run "$PYTHON" tools/exp06_smoke.py --entry "$entry" \
            --receipt "$SMOKE_DIR/receipt_${name}_$STAMP.json" --alarm-seconds 300 --max-gb 3 -- \
            --backbone simple --save-dir "$SMOKE_DIR/t0" $SMOKE_FLAGS
    done
    # (b) the oriented backbone on the same budget.
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 \
    run "$PYTHON" tools/exp06_smoke.py --entry exp06_train \
        --receipt "$SMOKE_DIR/receipt_exp06_train_t1_$STAMP.json" --alarm-seconds 300 --max-gb 3 -- \
        --backbone cylindrical_oriented --save-dir "$SMOKE_DIR/t1" $SMOKE_FLAGS
    # (c) the CPU fixture the round-2b HAA smokes load.
    run "$PYTHON" tools/exp06_smoke.py --make-fixture "$SMOKE_DIR/fixture_cylor.pth"
    ;;
finalize)
    finalize "$ATTEMPT" "$LOG" "$CHILD_EXIT" full
    ;;
esac
