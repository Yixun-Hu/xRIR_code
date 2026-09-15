#!/usr/bin/env bash
# exp_06 HAA pipeline (plan v4 sections 6.2 and 6.4): one queue of jobs on one GPU.
#   tools/exp06_haa_pipeline.sh <gpu> <job>... [--dry-run]
#     job = <init>:<seed>  with init in {cyl_or, control_hf, cyl_hf},  or  zeroshot
# Every child runs in its own exclusive directory under
#   ckpt/exp06/sim2real/<init>/{seed<s>/{stage1,stage2_<room>,eval/<room>}, zeroshot/eval/<room>}
# and is finalized by tools/exp06_finalize.py as soon as it has exited; the job itself is
# finalized once every child of it is complete, from the --job-spec written before the
# first child starts. The live launcher's pid lives ONLY at the job root
# (seed<s>/launch.pid, zeroshot/launch.pid): the finalizer's verify_child refuses any live
# pid inside a child directory, and its --owner-pid exception covers only the directory
# being finalized. Each child gets child.pid, written by run_child.
#
# The child lifecycle -- exclusive directory, drained pipe, end marker, child_exit.json --
# is tools/exp06_launch.sh's, sourced here as a library; only the job-level finalization
# (--children/--expect/--job-spec) has no helper there and is defined below.
# EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh defines the functions and
# returns, so job_spec and child can be exercised without a queue.
# --dry-run prints every command and path (timestamps as <UTC>) and executes nothing.
set -euo pipefail
EXP06_LAUNCH_LIB=1 source "$(dirname -- "${BASH_SOURCE[0]}")/exp06_launch.sh"

OUT=ckpt/exp06/sim2real
ROOMS="class_room dampened_room hallway complex_room"
S1_ROOMS="class_room hallway complex_room"   # dampened excluded from stage 1, as in exp_02
S1="--epochs 1000 --val-every 10 --tf32"
S2="--epochs 200 --val-every 2 --tf32"
HEADING_DIR="${EXP06_HEADING_DIR:-ckpt/exp06/heading}"
HAA_ROOT="${HAA_XRIR_ROOT:-$HOME/data_cache/HAA_xrir}"
FRAME=heading                       # every exp_06 HAA arm runs in the heading frame
OWNER=""                            # set to this launcher's pid only at the job root
DRY=0

usage() {
    echo "usage: $0 <gpu> <job>... [--dry-run]   job = <init>:<seed> | zeroshot" >&2
    echo "       init in cyl_or | control_hf | cyl_hf" >&2
    exit 2
}

init_of() { case "$1" in
  cyl_or) echo "cylindrical_oriented ${EXP06_CYLOR_CKPT:-ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth}";;
  control_hf) echo "simple ckpt/xRIR_simple_8_shot/epoch_12.pth";;
  cyl_hf) echo "cylindrical ckpt/xRIR_cyl_8_shot/epoch_12.pth";;
  *) echo "refusing: unknown init $1" >&2; return 1;; esac; }

child_log() {  # child_log <init> <tag> <stage>
    echo "$RECORD/oriented_cyl_${STAMP}_haa_${1}_${2}_${3}.log"
}

# The finalizer's mandatory job declaration, written before the first child starts.
JOB_SPEC_PY='
import json, sys
from pathlib import Path
from tools import exp06_heading, provenance
path, init, checkpoint, seed, expect, backbone, frame, heading_dir = sys.argv[1:9]
rooms = sorted(["class_room", "dampened_room", "hallway", "complex_room"])
spec = {"init": init, "backbone": backbone, "frame": frame, "seed": int(seed),
        "init_sha256": provenance.sha256_file(checkpoint), "rooms": rooms, "expect": expect,
        "heading": {room: exp06_heading.read_heading_json(
            str(Path(heading_dir) / (room + ".json")))["k"] for room in rooms}}
Path(path).write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
'

job_spec() {  # job_spec <path> <init> <checkpoint> <seed> <expect> <backbone>
    say "JOBSPEC $1 init=$2 checkpoint=$3 seed=$4 expect=$5 backbone=$6 frame=$FRAME heading=$HEADING_DIR"
    if [ "$DRY" -eq 0 ]; then
        "$PYTHON" -c "$JOB_SPEC_PY" "$1" "$2" "$3" "$4" "$5" "$6" "$FRAME" "$HEADING_DIR"
    fi
}

# child <run-type> <dir> <log> <command...>: the launcher's lifecycle for one child.
# A completed child is skipped; a failed or refused one leaves _ABORTED_<reason> behind
# and never a completion.
child() {
    # A child directory never holds a live launcher pid, so it never claims the owner
    # exception: OWNER is shadowed here and the finalizer sees no --owner-pid.
    local kind="$1" dir="$2" log="$3" OWNER=""
    shift 3
    if [ -f "$dir/completion.json" ]; then say "SKIP $dir"; return 0; fi
    say "MKDIR $dir"
    say "SINK cat >> $log"
    say "PIDFILE $dir/child.pid"
    say "RUN nohup setsid $*"
    if [ "$DRY" -eq 1 ]; then
        say "MARKER EXP06_CHILD_EXIT <code> <iso> >> $log"
        finalize "$dir" "$log" '<code>' "$kind"
        return 0
    fi
    mkdir -p -- "$(dirname -- "$dir")" "$RECORD" || return 1
    mkdir -- "$dir" || { say "REFUSED $dir already exists"; return 1; }
    : > "$log"
    run_child "$dir" "$log" "$@"
    close_child "$dir" "$log" || { abort "$dir" "$log" child_exit_unclosed; return 1; }
    if [ "$CHILD_STATUS" -ne 0 ]; then
        abort "$dir" "$log" "child_exit_$CHILD_STATUS"
        return 1
    fi
    finalize "$dir" "$log" "$CHILD_STATUS" "$kind" || { abort "$dir" "$log" finalize_refused; return 1; }
}

# finalize_job <root> <log> <expect> <children...>: the one helper tools/exp06_launch.sh
# has no equivalent for -- its finalize() takes no --children/--expect/--job-spec.
finalize_job() {
    local root="$1" log="$2" expect="$3"
    shift 3
    say "MARKER EXP06_CHILD_EXIT 0 <iso> >> $log"
    if [ "$DRY" -eq 0 ]; then
        printf 'job %s expect %s children %s\n' "$root" "$expect" "$#" >> "$log"
        "$PYTHON" tools/exp06_finalize.py child-exit --run-dir "$root" --log "$log" \
            --child-pid "$$" --status 0 --started-at "$JOB_STARTED_AT" || return 1
    fi
    run "$PYTHON" tools/exp06_finalize.py --run-dir "$root" --run-type haa_job --log "$log" \
        --child-exit 0 --owner-pid "$OWNER" --children "$@" --expect "$expect" \
        --job-spec "$root/job_spec.json"
}

open_job() {  # open_job <root>: this launcher owns the job root, never a child
    say "MKDIR $1"
    say "PIDFILE $1/launch.pid"
    if [ "$DRY" -eq 0 ]; then
        mkdir -p -- "$1" "$RECORD"
        own_launch "$1"
    fi
}

if [ "${EXP06_PIPELINE_LIB:-0}" = 1 ]; then return 0; fi

[ $# -ge 1 ] || usage
GPU="$1"; shift
JOBS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        -*) usage ;;
        *) JOBS+=("$1"); shift ;;
    esac
done
[ ${#JOBS[@]} -gt 0 ] || usage

for job in ${JOBS[@]+"${JOBS[@]}"}; do   # refuse the whole queue before anything runs
    case "$job" in
        zeroshot) ;;
        *:*) init_of "${job%%:*}" > /dev/null || exit 2
             case "${job##*:}" in ''|*[!0-9]*)
                 echo "refusing: ${job##*:} is not a seed in $job" >&2; exit 2 ;; esac ;;
        *) echo "refusing: unknown job $job" >&2; exit 2 ;;
    esac
done

if [ "$DRY" -eq 1 ]; then STAMP='<UTC>'; OWNER='<pid>'; else STAMP="$(date -u +%Y%m%dT%H%M%S)"; OWNER="$$"; fi
say "EXP06_HAA_PIPELINE gpu=$GPU jobs=${JOBS[*]} stamp=$STAMP dry_run=$DRY heading=$HEADING_DIR out=$OUT"
say "ENV CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 OMP_NUM_THREADS=8 HAA_XRIR_ROOT=$HAA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 HAA_XRIR_ROOT="$HAA_ROOT"

run_zeroshot() {  # run_zeroshot <init>
    local name="$1" bb ck root children=() room evaluation
    set -- $(init_of "$name"); bb="$1"; ck="$2"
    root="$OUT/$name/zeroshot"
    say "JOB $name zeroshot backbone=$bb init=$ck expect=zeroshot"
    JOB_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
    open_job "$root"
    job_spec "$root/job_spec.json" "$name" "$ck" 0 zeroshot "$bb"
    for room in $ROOMS; do
        evaluation="$root/eval/$room"
        child haa_eval "$evaluation" "$(child_log "$name" zeroshot "eval_$room")" \
            "$PYTHON" tools/exp06_haa_eval.py --backbone "$bb" --checkpoint "$ck" \
            --rooms "$room" --heading-json-dir "$HEADING_DIR" --save-dir "$evaluation" \
            --seed 0 || return 1
        children+=("$evaluation")
    done
    finalize_job "$root" "$(child_log "$name" zeroshot job)" zeroshot "${children[@]}"
}

run_finetune() {  # run_finetune <init> <seed>
    local name="$1" seed="$2" bb ck root tag children=() room stage2 evaluation
    set -- $(init_of "$name"); bb="$1"; ck="$2"
    root="$OUT/$name/seed$seed"; tag="seed$seed"
    say "JOB $name $tag backbone=$bb init=$ck expect=finetune"
    JOB_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
    open_job "$root"
    job_spec "$root/job_spec.json" "$name" "$ck" "$seed" finetune "$bb"
    child haa_train "$root/stage1" "$(child_log "$name" "$tag" stage1)" \
        "$PYTHON" tools/exp06_haa_finetune.py --backbone "$bb" --init "$ck" \
        --rooms $S1_ROOMS --heading-json-dir "$HEADING_DIR" --save-dir "$root/stage1" \
        --seed "$seed" $S1 || return 1
    children+=("$root/stage1")
    for room in $ROOMS; do
        stage2="$root/stage2_$room"
        child haa_train "$stage2" "$(child_log "$name" "$tag" "stage2_$room")" \
            "$PYTHON" tools/exp06_haa_finetune.py --backbone "$bb" --init "$root/stage1/best.pth" \
            --rooms "$room" --heading-json-dir "$HEADING_DIR" --save-dir "$stage2" \
            --seed "$seed" $S2 || return 1
        evaluation="$root/eval/$room"
        child haa_eval "$evaluation" "$(child_log "$name" "$tag" "eval_$room")" \
            "$PYTHON" tools/exp06_haa_eval.py --backbone "$bb" --checkpoint "$stage2/best.pth" \
            --rooms "$room" --heading-json-dir "$HEADING_DIR" --save-dir "$evaluation" \
            --seed "$seed" || return 1
        children+=("$stage2" "$evaluation")
    done
    finalize_job "$root" "$(child_log "$name" "$tag" job)" finetune "${children[@]}"
}

for job in "${JOBS[@]}"; do
    if [ "$job" = zeroshot ]; then
        for name in cyl_or control_hf cyl_hf; do
            run_zeroshot "$name" || echo "FAILED zero-shot $name" >&2
        done
        continue
    fi
    run_finetune "${job%%:*}" "${job##*:}" || echo "FAILED job $job" >&2
done
say "QUEUE_DONE gpu=$GPU"
