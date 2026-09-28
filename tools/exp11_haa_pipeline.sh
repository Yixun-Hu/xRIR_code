#!/usr/bin/env bash
# exp_11 HAA pipeline (plan v3 item 10): one queue of exp_11 jobs on one exclusive GPU.
#   tools/exp11_haa_pipeline.sh <gpu> <job>... [--dry-run]
#     job = <init>:<seed>  with init in {simple_or, simple_or_yaw, control_adapter,
#           yawaug_adapter}, or <init>:zeroshot for that arm's zero-shot set.
# Every child runs in its own exclusive directory under
#   ckpt/exp11/sim2real/<init>/{seed<s>/{stage1,stage2_<room>,eval/<room>}, zeroshot/eval/<room>}
# and is finalized by tools/exp11_finalize.py as soon as it has exited; the job itself is
# finalized once every child is complete, from the --job-spec written before the first
# child starts. The live launcher's pid lives ONLY at the job root.
#
# Arms and cues (plan v3 section 2.2):
#   simple_or       simple_oriented, exp_11's H pretraining, HEADING frame
#   simple_or_yaw   simple_oriented, exp_11's I pretraining, HEADING frame
#   control_adapter simple_adapter,  exp_01's control checkpoint, ROOM frame + adapter cue
#   yawaug_adapter  simple_adapter,  exp_04's aug checkpoint,     ROOM frame + adapter cue
# The adapter arms pass --adapter-heading-json-dir: the scene stays in the room frame and
# the shared loudspeaker heading is installed in the model, which is why their job spec
# declares adapter_heading/adapter_phi_deg and never `heading`.
#
# The child lifecycle is tools/exp11_launch.sh's (itself the pinned exp_06 library), so
# only the job-level finalization (--children/--expect/--job-spec) is defined here. A job
# is not a child of anything: it closes no log and writes no receipt (amendment A3).
# EXP11_PIPELINE_LIB=1 source tools/exp11_haa_pipeline.sh defines the functions and
# returns. --dry-run prints every command and path (timestamps as <UTC>) and runs nothing.
set -euo pipefail
EXP11_LAUNCH_LIB=1 source "$(dirname -- "${BASH_SOURCE[0]}")/exp11_launch.sh"

OUT="${EXP11_HAA_OUT:-ckpt/exp11/sim2real}"
ROOMS="class_room dampened_room hallway complex_room"
S1_ROOMS="class_room hallway complex_room"   # dampened excluded from stage 1, as in exp_02
S1="--epochs 1000 --val-every 10 --tf32"
S2="--epochs 200 --val-every 2 --tf32"
HAA_ROOT="${HAA_XRIR_ROOT:-$HOME/data_cache/HAA_xrir}"
FRAME=heading
CUE_FLAG=--heading-json-dir
OWNER=""
OWNED_ROOT=0
DRY=0
INIT_BACKBONE=""
INIT_CKPT=""
INIT_FRAME=heading

usage() {
    echo "usage: $0 <gpu> <job>... [--dry-run]   job = <init>:<seed> | <init>:zeroshot" >&2
    echo "       init in simple_or | simple_or_yaw | control_adapter | yawaug_adapter" >&2
    exit 2
}

# init_of <name>: backbone, checkpoint and frame of one initialisation, as variables.
# A checkpoint path may contain spaces, so it is never carried through word splitting, and
# the frame is assigned on every call so a mixed queue never inherits the previous job's.
init_of() { INIT_FRAME=heading; case "$1" in
  simple_or) INIT_BACKBONE=simple_oriented
      INIT_CKPT="${EXP11_SIMPOR_CKPT:-ckpt/exp11/pretrain/xRIR_simpor_8_shot/final/epoch_012.pth}";;
  simple_or_yaw) INIT_BACKBONE=simple_oriented
      INIT_CKPT="${EXP11_SIMPOR_YAW_CKPT:-ckpt/exp11/pretrain/xRIR_simpor_yawaug_8_shot/final/epoch_012.pth}";;
  control_adapter) INIT_BACKBONE=simple_adapter; INIT_FRAME=room
      INIT_CKPT="${EXP01_CONTROL_CKPT:-ckpt/xRIR_simple_8_shot/epoch_12.pth}";;
  yawaug_adapter) INIT_BACKBONE=simple_adapter; INIT_FRAME=room
      INIT_CKPT="${EXP09_YAWAUG_CKPT:-ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth}";;
  *) echo "refusing: unknown init $1" >&2; return 1;; esac; }

# cue_args: the flag a child of this job takes. Arms H/I rotate the scene into the heading
# frame; arms J/K stay in the room frame and install the shared cue in the model.
cue_args() {
    if [ "$FRAME" = heading ]; then CUE_FLAG=--heading-json-dir
    else CUE_FLAG=--adapter-heading-json-dir; fi
    CUE_ARGS=("$CUE_FLAG" "$HEADING_DIR")
}

child_log() {  # child_log <init> <tag> <stage>
    echo "$RECORD/orientation_cue_fairness_haa_${STAMP}_haa_${1}_${2}_${3}.log"
}

# The finalizer's mandatory job declaration, written before the first child starts. Every
# room's heading is verified here -- decided, confirmatory, and still hashing the cache it
# was estimated from -- so a refused heading can never be declared, and the adapter arms'
# shared heading is required to be one value across all four rooms.
JOB_SPEC_PY='
import json, sys
from pathlib import Path
from tools import exp11_haa_finetune as entry
from tools import provenance
path, init, checkpoint, seed, expect, backbone, frame, heading_dir, root = sys.argv[1:10]
rooms = sorted(["class_room", "dampened_room", "hallway", "complex_room"])
try:
    k_by_room, records = entry.legacy.load_headings(heading_dir, rooms, root)
    undecided = [room for room in rooms if type(k_by_room.get(room)) is not int]
    if undecided:
        raise ValueError("no heading roll for " + ", ".join(undecided))
    digest = provenance.sha256_file(checkpoint)
except (OSError, ValueError) as error:
    raise SystemExit("refusing: " + str(error))
spec = {"init": init, "backbone": backbone, "frame": frame, "seed": int(seed),
        "init_sha256": digest, "rooms": rooms, "expect": expect}
if frame == "heading":
    spec["heading"] = {room: k_by_room[room] for room in rooms}
else:                      # the adapter arms: one cue for every room, or no job at all
    declared = sorted({records[room]["phi_deg"] for room in rooms})
    if len(declared) != 1:
        raise SystemExit("refusing: the adapter needs one heading, not " + str(declared))
    spec["adapter_heading"] = {room: k_by_room[room] for room in rooms}
    spec["adapter_phi_deg"] = float(declared[0])
Path(path).write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
'

# The declaration is validated by the very code that will admit the job.
CHECK_SPEC_PY='
import sys
from tools import exp11_finalize
try:
    exp11_finalize.load_job_spec(sys.argv[1], sys.argv[2])
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

# The producer gate: exp_11's own approvals record, the HAA code keys recomputed at the
# queue HEAD, and the init checkpoint's identity. simple_or / simple_or_yaw are pinned by
# exp_11's artifact keys; control_adapter is exp_01's control (exp04_profiles.CONTROL) and
# yawaug_adapter is exp_04's approved checkpoints.aug, resolved through exp_06's finalizer
# so no unreviewed file can name either.
APPROVALS_PY='
import sys
from pathlib import Path
from tools import exp04_profiles, provenance
from tools import exp11_profiles as profiles
init, checkpoint, heading_dir = sys.argv[1:4]
rooms = ["class_room", "complex_room", "dampened_room", "hallway"]
repo = str(Path(profiles.__file__).resolve().parents[1])
producer = {"simple_or": "haa_simple_or", "simple_or_yaw": "haa_simple_or_yaw",
            "control_adapter": "haa_control_adapter",
            "yawaug_adapter": "haa_yawaug_adapter"}[init]
try:
    head = provenance.git_state(repo)["HEAD"]
    approved, receipt = profiles.load_approved_digests(
        profiles.APPROVED_DIGESTS_PATH, repo=repo, commit=head)
    profiles.require_producer(approved, producer)
    profiles.require(approved, profiles.PRODUCER_CODE_KEYS[producer], repo=repo, commit=head)
    # The heading records are approved exp_06 artefacts whichever frame reads them, so
    # every job is gated on all four, exactly as the exp_06 queue gates its own: they are
    # cheap to hash and the completeness rule stays as reviewed.
    from tools import exp06_approvals_api as api
    exp06_path = api.approved_path_default()
    if provenance.sha256_file(exp06_path) != approved["reused"]["approved_digests_exp06"]:
        raise ValueError("exp_06 approvals " + str(exp06_path) + " are not the approved bytes")
    exp06_approved, _ = api.load_approved_digests(exp06_path, repo=repo, commit=head)
    for room in rooms:
        pinned_heading = exp06_approved["artifacts"]["heading"][room]
        actual = provenance.sha256_file(heading_dir + "/" + room + ".json")
        if pinned_heading is None or actual != pinned_heading:
            raise ValueError("the {} heading record hashes to {}, not the approved {}".format(
                room, actual, pinned_heading))
    pinned = None
    if init == "control_adapter":
        pinned, source = exp04_profiles.CONTROL["sha256"], "the registered exp_01 control"
    elif init == "yawaug_adapter":
        # Arm K starts from the approved exp_04 checkpoints.aug, so the record that
        # resolves it must be the one exp_11 approved as reused.approved_digests_exp04 -
        # checked here, before it is read, and not only through the exp_06 pin.
        from tools import exp06_finalize as finalizer
        exp04_path = exp04_profiles.APPROVED_DIGESTS_PATH
        if provenance.sha256_file(exp04_path) != approved["reused"]["approved_digests_exp04"]:
            raise ValueError("exp_04 approvals " + str(exp04_path) + " are not the bytes "
                             "exp_11 approved as reused.approved_digests_exp04")
        aug = finalizer.exp04_aug_checkpoint(exp06_approved)
        pinned, source = aug["checkpoint"]["sha256"], "exp_04\x27s approved checkpoints.aug"
    else:
        key = {"simple_or": "simpor_epoch_012", "simple_or_yaw": "simpor_yaw_epoch_012"}[init]
        pinned, source = approved["artifacts"][key]["sha256"], "the approved artifacts." + key
    digest = provenance.sha256_file(checkpoint)
    if digest != pinned:
        raise ValueError("the {} init {} hashes to {}, not {} {}".format(
            init, checkpoint, digest, source, pinned))
except (OSError, ValueError, KeyError) as error:
    raise SystemExit("refusing: " + str(error))
print("APPROVALS ok producer={} commit={} keys={}".format(
    producer, str(receipt.get("committed_at"))[:12],
    ",".join(profiles.PRODUCER_CODE_KEYS[producer])))
'

approvals_ok() {  # approvals_ok <init> <checkpoint>: on the CPU, before any acquisition
    say "APPROVALS $1 checkpoint=$2 heading=$HEADING_DIR"
    [ "$DRY" -eq 1 ] || CUDA_VISIBLE_DEVICES="" "$PYTHON" -c "$APPROVALS_PY" "$1" "$2" \
        "$HEADING_DIR"
}

# An EXPLICIT exclusive-card census: the pipeline does not inherit the launcher's
# preflight, so every job checks for itself that nvidia-smi reports no compute apps on
# this card before it starts a child.
CENSUS_PY='
import sys
from tools import exp06_finalize as finalizer
try:
    apps = finalizer.gpu_compute_apps(int(sys.argv[1]))
except (OSError, ValueError) as error:
    raise SystemExit("refusing: " + str(error))
if apps:
    raise SystemExit("refusing: GPU {} is busy with compute apps {}".format(sys.argv[1], apps))
print("CENSUS gpu={} compute_apps=none".format(sys.argv[1]))
'

card_is_exclusive() {
    say "CENSUS gpu=$GPU"
    [ "$DRY" -eq 1 ] || CUDA_VISIBLE_DEVICES="" "$PYTHON" -c "$CENSUS_PY" "$GPU"
}

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

# prepare_failed <root> <reason>: a refused preparation reports itself where it is
# entitled to write. Only an attempt that created the root itself may rename it.
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

open_job() {  # this launcher owns the job root, never a child
    say "MKDIR $1"
    say "PIDFILE $1/launch.pid"
    OWNED_ROOT=0
    if [ "$DRY" -eq 1 ]; then OWNED_ROOT=1; return 0; fi
    mkdir -p -- "$(dirname -- "$1")" "$RECORD" || return 1
    if mkdir -- "$1" 2>/dev/null; then OWNED_ROOT=1
    elif [ ! -d "$1" ]; then return 1; fi
    own_launch "$1" || return 1
}

# prepare_job <root> <init> <checkpoint> <seed> <expect> <backbone>: every gate that must
# pass before the first child of this job starts, the card census included.
prepare_job() {
    OWNED_ROOT=0                     # fail closed: ownership is only ever granted below
    card_is_exclusive || { prepare_failed "$1" census; return 1; }
    approvals_ok "$2" "$3" || { prepare_failed "$1" approvals; return 1; }
    open_job "$1" || { prepare_failed "$1" open_job; return 1; }
    job_spec "$1/job_spec.json" "$2" "$3" "$4" "$5" "$6" \
        || { prepare_failed "$1" job_spec; return 1; }
    check_spec "$1/job_spec.json" "$5" || { prepare_failed "$1" check_spec; return 1; }
}

# child <run-type> <dir> <log> <command...>: the launcher's lifecycle for one child.
child() {
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
    finalize "$dir" "$log" "$CHILD_STATUS" "$kind" \
        || { abort "$dir" "$log" finalize_refused; return 1; }
}

# finalize_job <root> <log> <expect> <children...>: a job has no child process of its own,
# so it closes no log and writes no job-root child_exit.json (amendment A3).
finalize_job() {
    local root="$1" log="$2" expect="$3"
    shift 3
    if [ "$DRY" -eq 0 ]; then
        printf 'job %s expect %s children %s\n' "$root" "$expect" "$#" >> "$log" || return 1
    fi
    run "$PYTHON" tools/exp11_finalize.py --run-dir "$root" --run-type exp11_haa_job \
        --log "$log" --child-exit 0 --owner-pid "$OWNER" --children "$@" --expect "$expect" \
        --job-spec "$root/job_spec.json"
}

run_zeroshot() {  # run_zeroshot <init>
    local name="$1" bb ck root children=() room evaluation FRAME CUE_ARGS CUE_FLAG
    init_of "$name" || return 1
    bb="$INIT_BACKBONE"; ck="$INIT_CKPT"; FRAME="$INIT_FRAME"; cue_args
    root="$OUT/$name/zeroshot"
    say "JOB $name zeroshot backbone=$bb init=$ck expect=zeroshot frame=$FRAME"
    prepare_job "$root" "$name" "$ck" 0 zeroshot "$bb" || return 1
    for room in $ROOMS; do
        evaluation="$root/eval/$room"
        child exp11_haa_eval "$evaluation" "$(child_log "$name" zeroshot "eval_$room")" \
            "$PYTHON" tools/exp11_haa_eval.py --backbone "$bb" --checkpoint "$ck" \
            --rooms "$room" "${CUE_ARGS[@]}" --save-dir "$evaluation" \
            --seed 0 --job-spec "$root/job_spec.json" || return 1
        children+=("$evaluation")
    done
    finalize_job "$root" "$(child_log "$name" zeroshot job)" zeroshot "${children[@]}"
}

run_finetune() {  # run_finetune <init> <seed>
    local name="$1" seed="$2" bb ck root tag children=() room stage2 evaluation FRAME \
          CUE_ARGS CUE_FLAG
    init_of "$name" || return 1
    bb="$INIT_BACKBONE"; ck="$INIT_CKPT"; FRAME="$INIT_FRAME"; cue_args
    root="$OUT/$name/seed$seed"; tag="seed$seed"
    say "JOB $name $tag backbone=$bb init=$ck expect=finetune frame=$FRAME"
    prepare_job "$root" "$name" "$ck" "$seed" finetune "$bb" || return 1
    child exp11_haa_finetune "$root/stage1" "$(child_log "$name" "$tag" stage1)" \
        "$PYTHON" tools/exp11_haa_finetune.py --backbone "$bb" --init "$ck" \
        --rooms $S1_ROOMS "${CUE_ARGS[@]}" --save-dir "$root/stage1" \
        --seed "$seed" --job-spec "$root/job_spec.json" $S1 || return 1
    children+=("$root/stage1")
    for room in $ROOMS; do
        stage2="$root/stage2_$room"
        child exp11_haa_finetune "$stage2" "$(child_log "$name" "$tag" "stage2_$room")" \
            "$PYTHON" tools/exp11_haa_finetune.py --backbone "$bb" \
            --init "$root/stage1/best.pth" --rooms "$room" "${CUE_ARGS[@]}" \
            --save-dir "$stage2" --seed "$seed" \
            --job-spec "$root/job_spec.json" $S2 || return 1
        evaluation="$root/eval/$room"
        child exp11_haa_eval "$evaluation" "$(child_log "$name" "$tag" "eval_$room")" \
            "$PYTHON" tools/exp11_haa_eval.py --backbone "$bb" \
            --checkpoint "$stage2/best.pth" --rooms "$room" "${CUE_ARGS[@]}" \
            --save-dir "$evaluation" --seed "$seed" \
            --job-spec "$root/job_spec.json" || return 1
        children+=("$stage2" "$evaluation")
    done
    finalize_job "$root" "$(child_log "$name" "$tag" job)" finetune "${children[@]}"
}

# run_queue <job>...: no failure is silent; QUEUE_DONE only if every job succeeded.
run_queue() {
    local job failed=0
    for job in "$@"; do
        if [ "${job#*:}" = zeroshot ]; then
            run_zeroshot "${job%%:*}" \
                || { echo "FAILED zero-shot ${job%%:*}" >&2; failed=$((failed + 1)); }
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

if [ "${EXP11_PIPELINE_LIB:-0}" = 1 ]; then return 0; fi

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
        *:*) init_of "${job%%:*}" || exit 2
             # Exactly <init>:<integer seed> or <init>:zeroshot; everything after the
             # first colon is the seed, so simple_or:anything:0 is refused.
             case "${job#*:}" in zeroshot) ;; ''|*[!0-9]*)
                 echo "refusing: ${job#*:} is not a seed in $job" >&2; exit 2 ;; esac ;;
        *) echo "refusing: unknown job $job" >&2; exit 2 ;;
    esac
done

if [ "$DRY" -eq 1 ]; then STAMP='<UTC>'; OWNER='<pid>'; else STAMP="$(date -u +%Y%m%dT%H%M%S)"; OWNER="$$"; fi
say "EXP11_HAA_PIPELINE gpu=$GPU jobs=${JOBS[*]} stamp=$STAMP dry_run=$DRY heading=$HEADING_DIR out=$OUT"
say "ENV CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 OMP_NUM_THREADS=8 HAA_XRIR_ROOT=$HAA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=0 OMP_NUM_THREADS=8 HAA_XRIR_ROOT="$HAA_ROOT"

run_queue ${JOBS[@]+"${JOBS[@]}"} || exit 1
