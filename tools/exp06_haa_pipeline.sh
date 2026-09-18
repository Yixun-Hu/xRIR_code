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
# Round 2b finding 1: preparing a job is a gate, not an announcement. prepare_job opens the
# job root, verifies all four rooms' headings (decided, confirmatory and still hashing the
# selected HAA cache) while writing the declaration, and validates the written declaration
# with the finalizer's own load_job_spec. Every status is checked explicitly, because these
# functions are called through `||` from the queue, where bash suppresses errexit: no child
# of a job starts unless its whole preparation succeeded, and the queue counts failures and
# reports QUEUE_FAILED instead of QUEUE_DONE if any job failed.
#
# The child lifecycle -- exclusive directory, drained pipe, end marker, child_exit.json --
# is tools/exp06_launch.sh's, sourced here as a library; only the job-level finalization
# (--children/--expect/--job-spec) has no helper there and is defined below. A job is not a
# child of anything: it closes no log and writes no receipt (plan amendment A3).
# EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh defines the functions and
# returns, so job_spec, child, run_finetune and run_queue can be exercised without a queue.
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
OWNED_ROOT=0                        # 1 only when this attempt created the job root itself
DRY=0
INIT_BACKBONE=""                    # set by init_of; never word-split out of one string
INIT_CKPT=""

usage() {
    echo "usage: $0 <gpu> <job>... [--dry-run]   job = <init>:<seed> | zeroshot" >&2
    echo "       init in cyl_or | control_hf | cyl_hf" >&2
    exit 2
}

# init_of <name>: the backbone and checkpoint of one initialisation, as two variables.
# Nit 5: a checkpoint path may contain spaces, so it is never carried through word splitting.
init_of() { case "$1" in
  cyl_or) INIT_BACKBONE=cylindrical_oriented
          INIT_CKPT="${EXP06_CYLOR_CKPT:-ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth}";;
  control_hf) INIT_BACKBONE=simple; INIT_CKPT=ckpt/xRIR_simple_8_shot/epoch_12.pth;;
  cyl_hf) INIT_BACKBONE=cylindrical; INIT_CKPT=ckpt/xRIR_cyl_8_shot/epoch_12.pth;;
  *) echo "refusing: unknown init $1" >&2; return 1;; esac; }

child_log() {  # child_log <init> <tag> <stage>
    echo "$RECORD/oriented_cyl_${STAMP}_haa_${1}_${2}_${3}.log"
}

# The finalizer's mandatory job declaration, written before the first child starts. Every
# room's heading is verified here -- decided, confirmatory, and still hashing the cache
# files it was estimated from -- so a refused heading (a null roll) can never be declared.
JOB_SPEC_PY='
import json, sys
from pathlib import Path
from tools import exp06_haa_finetune as entry
from tools import provenance
path, init, checkpoint, seed, expect, backbone, frame, heading_dir, root = sys.argv[1:10]
rooms = sorted(["class_room", "dampened_room", "hallway", "complex_room"])
try:
    k_by_room, _ = entry.load_headings(heading_dir, rooms, root)
    undecided = [room for room in rooms if type(k_by_room.get(room)) is not int]
    if undecided:
        raise ValueError("no heading roll for " + ", ".join(undecided))
    digest = provenance.sha256_file(checkpoint)
except (OSError, ValueError) as error:
    raise SystemExit("refusing: " + str(error))
spec = {"init": init, "backbone": backbone, "frame": frame, "seed": int(seed),
        "init_sha256": digest, "rooms": rooms, "expect": expect,
        "heading": {room: k_by_room[room] for room in rooms}}
Path(path).write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
'

# The declaration is validated by the very code that will admit the job.
CHECK_SPEC_PY='
import sys
from tools import exp06_finalize
try:
    exp06_finalize.load_job_spec(sys.argv[1], sys.argv[2])
except (OSError, ValueError) as error:
    raise SystemExit("refusing: " + str(error))
'

job_spec() {  # job_spec <path> <init> <checkpoint> <seed> <expect> <backbone>
    say "JOBSPEC $1 init=$2 checkpoint=$3 seed=$4 expect=$5 backbone=$6 frame=$FRAME heading=$HEADING_DIR"
    [ "$DRY" -eq 1 ] || "$PYTHON" -c "$JOB_SPEC_PY" "$1" "$2" "$3" "$4" "$5" "$6" \
        "$FRAME" "$HEADING_DIR" "$HAA_ROOT"
}

check_spec() {  # check_spec <path> <expect>
    say "CHECKSPEC $1 expect=$2"
    [ "$DRY" -eq 1 ] || "$PYTHON" -c "$CHECK_SPEC_PY" "$1" "$2"
}

# Full-review F3: 6.4 says every producer refuses unless the recorded digests match the
# approved ones. This is that gate for the HAA children -- the eight code keys they run,
# recomputed at the queue HEAD and compared with the committed approvals, the init
# checkpoint (artifacts.epoch_012 for cyl_or, the registered exp_01 sha otherwise) and the
# four heading JSONs. It runs before the job root is acquired, so a refusal touches nothing.
APPROVALS_PY='
import sys
from pathlib import Path
from tools import exp04_profiles, provenance
from tools import exp06_approvals_api as api
init, checkpoint, heading_dir = sys.argv[1:4]
rooms = ["class_room", "complex_room", "dampened_room", "hallway"]
repo = str(Path(api.__file__).resolve().parents[1])
try:
    head = provenance.git_state(repo)["HEAD"]
    registered = {"control_hf": exp04_profiles.CONTROL, "cyl_hf": exp04_profiles.CYL}.get(init)
    if registered is not None:
        digest = provenance.sha256_file(checkpoint)
        if digest != registered["sha256"]:
            raise ValueError("the {} init {} hashes to {}, not the registered exp_01 {}".format(
                init, checkpoint, digest, registered["sha256"]))
    record = api.enforce_producer(
        "haa_children", repo, head, approved_path=api.approved_path_default(),
        checkpoint=None if registered is not None else checkpoint,
        headings={room: heading_dir + "/" + room + ".json" for room in rooms})
except (OSError, ValueError, KeyError) as error:
    raise SystemExit("refusing: " + str(error))
print("APPROVALS ok producer={} commit={} keys={}".format(
    record["producer"], str(record["approvals"].get("committed_at"))[:12],
    ",".join(record["keys_checked"])))
'

# approvals_ok <init> <checkpoint>: the producer gate, on the CPU and before any acquisition.
approvals_ok() {
    say "APPROVALS $1 checkpoint=$2 heading=$HEADING_DIR producer=haa_children"
    [ "$DRY" -eq 1 ] || CUDA_VISIBLE_DEVICES="" "$PYTHON" -c "$APPROVALS_PY" "$1" "$2" \
        "$HEADING_DIR"
}

# Nit 8: a refused preparation is a recovery state, not a terminal message. The receipt
# names the gate that refused, when, and the commit the queue was running.
PREPARE_FAILURE_PY='
import datetime, json, sys
from pathlib import Path
from tools import provenance
path, reason, root, aborted, owned = sys.argv[1:6]
Path(path).write_text(json.dumps({
    "schema_version": 2, "reason": reason, "job_root": root,
    "aborted_dir": aborted or None, "owned_root": owned == "1",
    "failed_at": datetime.datetime.now().astimezone().isoformat(),
    "git": provenance.git_state(str(Path(provenance.__file__).resolve().parents[1]))},
    sort_keys=True, indent=2) + "\n")
'

# prepare_failed <root> <reason>: round-3a finding 3. A refused preparation reports itself
# where it is entitled to write. Only an attempt that created the root itself (OWNED_ROOT)
# may put its receipt inside and rename it _ABORTED_; a root that pre-existed or was never
# acquired may belong to another or a finished job, so its failure is recorded beside it
# and nothing in it is touched. Always returns 1.
prepare_failed() {
    local root="$1" reason="$2" target
    say "REFUSED $reason $root"
    [ "$DRY" -eq 0 ] || return 1
    if [ "$OWNED_ROOT" -eq 1 ]; then
        target="${root}_ABORTED_prepare_${reason}"
        [ ! -e "$target" ] || target="${target}_$$"
        mkdir -p -- "$root" \
            && "$PYTHON" -c "$PREPARE_FAILURE_PY" "$root/preparation_failure.json" \
                 "$reason" "$root" "$target" 1 \
            || say "UNRECORDED preparation failure $root"
        if [ -e "$root" ] && mv -- "$root" "$target"; then say "ABORT $target"
        else say "UNABORTED $root"; fi
    else
        target="${root}_PREPARE_FAILED_$(date -u +%Y%m%dT%H%M%S).json"
        [ ! -e "$target" ] || target="${target%.json}_$$.json"
        mkdir -p -- "$(dirname -- "$root")" \
            && "$PYTHON" -c "$PREPARE_FAILURE_PY" "$target" "$reason" "$root" "" 0 \
            || say "UNRECORDED preparation failure $root"
        say "UNOWNED $root"
    fi
    return 1
}

# prepare_job <root> <init> <checkpoint> <seed> <expect> <backbone>: every gate that must
# pass before the first child of this job starts. A failure here is named and propagated.
prepare_job() {
    OWNED_ROOT=0                     # fail closed: ownership is only ever granted below
    approvals_ok "$2" "$3" || { prepare_failed "$1" approvals; return 1; }
    open_job "$1" || { prepare_failed "$1" open_job; return 1; }
    job_spec "$1/job_spec.json" "$2" "$3" "$4" "$5" "$6" \
        || { prepare_failed "$1" job_spec; return 1; }
    check_spec "$1/job_spec.json" "$5" || { prepare_failed "$1" check_spec; return 1; }
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
# Plan amendment A3: a job has no child process of its own. This shell orchestrates the
# nine (or four) children and is still running here, so it closes no log and writes no
# job-root child_exit.json -- a receipt there could only name itself, and the finalizer
# refuses one. The queue log is informational; $OWNER (= $$, the live launch.pid this
# launcher wrote at the job root) is the owner the job-level completion binds.
finalize_job() {
    local root="$1" log="$2" expect="$3"
    shift 3
    if [ "$DRY" -eq 0 ]; then
        printf 'job %s expect %s children %s\n' "$root" "$expect" "$#" >> "$log" || return 1
    fi
    run "$PYTHON" tools/exp06_finalize.py --run-dir "$root" --run-type haa_job --log "$log" \
        --child-exit 0 --owner-pid "$OWNER" --children "$@" --expect "$expect" \
        --job-spec "$root/job_spec.json"
}

# open_job <root>: this launcher owns the job root, never a child. The root is created
# exclusively, so OWNED_ROOT distinguishes a root this attempt made from one it resumed;
# an existing root is still adopted (a resumed queue skips its completed children).
# own_launch records this shell's $$ there whether the root is new or adopted, and that
# launch.pid is the owner the job completion binds: pre-merge finding 1 made it required
# evidence, so a job root without one is refused rather than certified ownerless.
open_job() {
    say "MKDIR $1"
    say "PIDFILE $1/launch.pid"
    OWNED_ROOT=0
    if [ "$DRY" -eq 1 ]; then OWNED_ROOT=1; return 0; fi
    mkdir -p -- "$(dirname -- "$1")" "$RECORD" || return 1
    if mkdir -- "$1" 2>/dev/null; then OWNED_ROOT=1
    elif [ ! -d "$1" ]; then return 1; fi
    own_launch "$1" || return 1
}

run_zeroshot() {  # run_zeroshot <init>
    local name="$1" bb ck root children=() room evaluation
    init_of "$name" || return 1
    bb="$INIT_BACKBONE"; ck="$INIT_CKPT"
    root="$OUT/$name/zeroshot"
    say "JOB $name zeroshot backbone=$bb init=$ck expect=zeroshot"
    prepare_job "$root" "$name" "$ck" 0 zeroshot "$bb" || return 1
    for room in $ROOMS; do
        evaluation="$root/eval/$room"
        child haa_eval "$evaluation" "$(child_log "$name" zeroshot "eval_$room")" \
            "$PYTHON" tools/exp06_haa_eval.py --backbone "$bb" --checkpoint "$ck" \
            --rooms "$room" --heading-json-dir "$HEADING_DIR" --save-dir "$evaluation" \
            --seed 0 --job-spec "$root/job_spec.json" || return 1
        children+=("$evaluation")
    done
    finalize_job "$root" "$(child_log "$name" zeroshot job)" zeroshot "${children[@]}"
}

run_finetune() {  # run_finetune <init> <seed>
    local name="$1" seed="$2" bb ck root tag children=() room stage2 evaluation
    init_of "$name" || return 1
    bb="$INIT_BACKBONE"; ck="$INIT_CKPT"
    root="$OUT/$name/seed$seed"; tag="seed$seed"
    say "JOB $name $tag backbone=$bb init=$ck expect=finetune"
    prepare_job "$root" "$name" "$ck" "$seed" finetune "$bb" || return 1
    child haa_train "$root/stage1" "$(child_log "$name" "$tag" stage1)" \
        "$PYTHON" tools/exp06_haa_finetune.py --backbone "$bb" --init "$ck" \
        --rooms $S1_ROOMS --heading-json-dir "$HEADING_DIR" --save-dir "$root/stage1" \
        --seed "$seed" --job-spec "$root/job_spec.json" $S1 || return 1
    children+=("$root/stage1")
    for room in $ROOMS; do
        stage2="$root/stage2_$room"
        child haa_train "$stage2" "$(child_log "$name" "$tag" "stage2_$room")" \
            "$PYTHON" tools/exp06_haa_finetune.py --backbone "$bb" --init "$root/stage1/best.pth" \
            --rooms "$room" --heading-json-dir "$HEADING_DIR" --save-dir "$stage2" \
            --seed "$seed" --job-spec "$root/job_spec.json" $S2 || return 1
        evaluation="$root/eval/$room"
        child haa_eval "$evaluation" "$(child_log "$name" "$tag" "eval_$room")" \
            "$PYTHON" tools/exp06_haa_eval.py --backbone "$bb" --checkpoint "$stage2/best.pth" \
            --rooms "$room" --heading-json-dir "$HEADING_DIR" --save-dir "$evaluation" \
            --seed "$seed" --job-spec "$root/job_spec.json" || return 1
        children+=("$stage2" "$evaluation")
    done
    finalize_job "$root" "$(child_log "$name" "$tag" job)" finetune "${children[@]}"
}

# run_queue <job>...: no failure is silent. Every job's status is counted, and the queue
# reports QUEUE_DONE only if every one of them succeeded.
run_queue() {
    local job name failed=0
    for job in "$@"; do
        if [ "$job" = zeroshot ]; then
            for name in cyl_or control_hf cyl_hf; do
                run_zeroshot "$name" \
                    || { echo "FAILED zero-shot $name" >&2; failed=$((failed + 1)); }
            done
            continue
        fi
        run_finetune "${job%%:*}" "${job#*:}" \
            || { echo "FAILED job $job" >&2; failed=$((failed + 1)); }
    done
    if [ "$failed" -ne 0 ]; then
        say "QUEUE_FAILED $failed"
        return 1
    fi
    say "QUEUE_DONE gpu=$GPU"
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
        *:*) init_of "${job%%:*}" || exit 2
             # Nit 5: exactly <init>:<integer seed>; everything after the first colon is
             # the seed, so cyl_or:anything:0 is refused rather than read as cyl_or:0.
             case "${job#*:}" in ''|*[!0-9]*)
                 echo "refusing: ${job#*:} is not a seed in $job" >&2; exit 2 ;; esac ;;
        *) echo "refusing: unknown job $job" >&2; exit 2 ;;
    esac
done

if [ "$DRY" -eq 1 ]; then STAMP='<UTC>'; OWNER='<pid>'; else STAMP="$(date -u +%Y%m%dT%H%M%S)"; OWNER="$$"; fi
say "EXP06_HAA_PIPELINE gpu=$GPU jobs=${JOBS[*]} stamp=$STAMP dry_run=$DRY heading=$HEADING_DIR out=$OUT"
say "ENV CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 OMP_NUM_THREADS=8 HAA_XRIR_ROOT=$HAA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 HAA_XRIR_ROOT="$HAA_ROOT"

run_queue ${JOBS[@]+"${JOBS[@]}"} || exit 1
