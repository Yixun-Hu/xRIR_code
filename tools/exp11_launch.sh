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
# The pretraining roots. EXP11_PRETRAIN_ROOT exists for the dry-run tests alone: it moves
# the attempt roots AND the roots preflight scans for live pid files, so a real launch may
# never take it. It is honoured only with EXP11_TEST_ROOTS=1 *and* --dry-run, and is
# refused outright otherwise (see apply_root_override below).
PRETRAIN_ROOT=ckpt/exp11/pretrain
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

# The value is validated as a BOUNDED DECIMAL STRING before any arithmetic: bash's
# integer comparisons error out above 2**63-1, which left the `if` false and the launch
# running with an overflowing value handed to GNU timeout. Six digits is the whole
# contract (129600 has six), so the numeric comparison below can never be out of range;
# leading zeros are refused with everything else that is not a plain decimal.
check_ceiling() {
    local why="must be a plain decimal of at most 6 digits in (0, $FULL_CEILING_MAX]"
    case "$FULL_CEILING_S" in
        ''|*[!0-9]*|0*|???????*)
            echo "refusing: EXP11_FULL_CEILING_S '$FULL_CEILING_S' $why" >&2; return 1 ;;
    esac
    if [ "$FULL_CEILING_S" -le 0 ] || [ "$FULL_CEILING_S" -gt "$FULL_CEILING_MAX" ]; then
        echo "refusing: EXP11_FULL_CEILING_S $FULL_CEILING_S $why;" \
             "an override may only tighten the 36 h ceiling" >&2
        return 1
    fi
}

# apply_root_override: EXP11_PRETRAIN_ROOT is a dry-run test affordance and nothing else.
apply_root_override() {
    [ -n "${EXP11_PRETRAIN_ROOT:-}" ] || return 0
    if [ "${EXP11_TEST_ROOTS:-0}" != 1 ] || [ "$DRY" -ne 1 ]; then
        echo "refusing: EXP11_PRETRAIN_ROOT moves the attempt roots and the roots" \
             "preflight scans, so it is honoured only with EXP11_TEST_ROOTS=1 and" \
             "--dry-run" >&2
        return 1
    fi
    PRETRAIN_ROOT="$EXP11_PRETRAIN_ROOT"
    SIMPOR_ROOT="$PRETRAIN_ROOT/xRIR_simpor_8_shot"
    SIMPOR_YAW_ROOT="$PRETRAIN_ROOT/xRIR_simpor_yawaug_8_shot"
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
    # The path that is VALIDATED must be the path that is finalized and promoted, or an
    # alias (an existing `final`, or a link from outside) would be validated here and
    # promoted by its own basename - a self-referencing or dangling link under the root.
    directory, base = Path(attempt).resolve(), Path(root).resolve()
    if directory.parent != base:
        raise ValueError("attempt {} is not an attempt of the arm {} root {}".format(
            directory, arm, base))
    if not directory.name.startswith("attempt_"):
        raise ValueError("{} is not an attempt_<UTC> directory".format(directory))
    args = json.loads((directory / "args.json").read_text())
    profile = exp11_recipe.select_profile(args)
    if profile != expected:
        raise ValueError("attempt {} recorded the profile {}, not the {} of arm {}".format(
            directory, profile, expected, arm))
except (OSError, ValueError, KeyError) as error:
    raise SystemExit("refusing: " + str(error))
print("ATTEMPT ok {} arm={} profile={}".format(attempt, arm, expected))
print("ATTEMPT canonical {}".format(directory))
'

# check_attempt <attempt>: sets CANONICAL_ATTEMPT to the resolved directory, which is
# what finalization and promotion then use.
check_attempt() {
    local output
    say "ATTEMPTCHECK $1 arm=$ARM root=$ARM_ROOT"
    output="$(CUDA_VISIBLE_DEVICES="" "$PYTHON" -c "$CHECK_ATTEMPT_PY" "$1" "$ARM_ROOT" \
              "$ARM")" || return 1
    printf '%s\n' "$output"
    CANONICAL_ATTEMPT="${output##*ATTEMPT canonical }"
    case "$CANONICAL_ATTEMPT" in
        "$ARM_ROOT"/attempt_*|/*) ;;
        *) echo "refusing: no canonical attempt was resolved for $1" >&2; return 1 ;;
    esac
}
ARM=""
ARM_BACKBONE=simple_oriented
ARM_YAW=0
ARM_SAVE_EVERY=500

usage() {
    echo "usage: $0 <smoke|probe|full|finalize> --arm <H|I> --gpu <g> --reviewed-commit <sha40>" >&2
    echo "       [--attempt-root <dir>] [--approved <json>] [--exploratory]" >&2
    echo "       [--attempt <dir> --log <path> --child-exit <n>] [--dry-run]" >&2
    echo "       finalize also takes [--replace-final] [--break-lock]" >&2
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

# --- publishing one arm is serialised -----------------------------------------------
# Close review 2: recovery had no ownership at all, so two invocations could interleave
# -- one finalising while the other renamed the shared attempt out from under it -- and
# report success over a dangling `final`. Every path that validates, finalises or
# promotes an attempt of an arm takes this lock first. `mkdir` is the atomic primitive:
# it succeeds for exactly one caller. A held lock REFUSES immediately, it never waits,
# and a stale one is evidence of an interrupted publication, so it is cleared only by an
# explicit --break-lock and never on a timer.
LOCK_DIR=""
LOCK_HELD=0
# The GENERATION of the lock this invocation holds. A lock directory is not identity
# enough: it can be retired and re-created between two statements of one shell, so a
# process-local flag would let an earlier owner remove its successor's lock. The owner
# file carries this nonce, and cleanup acts only where the nonce is still ours.
LOCK_NONCE=""
INTERRUPTED=0
# A lock whose owner file has not landed yet is an acquisition in progress, not a stale
# lock; only one older than this may be broken.
LOCK_GRACE_S="${EXP11_LOCK_GRACE_S:-30}"
BREAKING_DIR=""
BREAKING_HELD=0
LOCK_NONCE_CANDIDATE=""

# Every reader of an owner file ends in `|| true`: an absent or unreadable owner is
# explicit NON-ownership, and `sed`'s failure must never propagate under `set -e` --
# that is what pre-empted the signal handler's own exit.
lock_nonce_of() {  # the generation recorded in one lock directory, or nothing
    sed -n 's/^pid [0-9][0-9]* nonce \([^ ][^ ]*\).*/\1/p' "$1/owner" 2>/dev/null || true
}

lock_pid_of() {
    sed -n 's/^pid \([0-9][0-9]*\).*/\1/p' "$1/owner" 2>/dev/null || true
}

lock_owner_line() {
    cat -- "$1/owner" 2>/dev/null | tr -d '\n' || true
}

lock_inode_of() {
    stat -c %i -- "$1" 2>/dev/null || true
}

# Older than the initialisation grace? `mkdir` publishes the pathname before the owner
# file lands, so a lock without an owner is an acquisition in progress until it is old
# enough to be an abandoned one.
lock_older_than_grace() {
    local age
    age="$(( $(date +%s) - $(stat -c %Y -- "$1" 2>/dev/null || echo 0) ))"
    [ "$age" -ge "$LOCK_GRACE_S" ]
}

lock_is_live() {  # a pid that is gone does not hold a lock; a live one is never overridden
    local pid
    pid="$(lock_pid_of "$1")"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# A test-only ordering hook: wait (bounded) for a file to appear. It takes a PATH, never
# a command, so nothing here evaluates anything the environment supplies.
lock_barrier() { lock_wait_for "${EXP11_LOCK_BARRIER:-}"; }

# The same hook at the one point inside acquisition the schedules need to interleave at:
# after the owner file has landed and before self-verification.
lock_acquire_barrier() { lock_wait_for "${EXP11_ACQUIRE_BARRIER:-}"; }

lock_wait_for() {
    local file="$1" waited=0
    [ -n "$file" ] || return 0
    while [ ! -e "$file" ] && [ "$waited" -lt 100 ]; do sleep 0.1; waited=$((waited + 1)); done
}

release_lock() {
    release_breaking
    [ "$LOCK_HELD" -eq 1 ] || return 0
    LOCK_HELD=0
    local now=""
    if [ -f "$LOCK_DIR/owner" ]; then
        now="$(lock_nonce_of "$LOCK_DIR")"
    else
        say "UNLOCK SKIPPED $LOCK_DIR has no owner: this invocation owns nothing there"
        return 0
    fi
    if [ -n "$LOCK_NONCE" ] && [ "$now" != "$LOCK_NONCE" ]; then
        say "UNLOCK SKIPPED $LOCK_DIR holds the generation ${now:-none}, not $LOCK_NONCE"
        return 0
    fi
    rm -f -- "$LOCK_DIR/owner"
    rmdir -- "$LOCK_DIR" 2>/dev/null || true
    say "UNLOCK $LOCK_DIR"
}

# The BREAKING META-LOCK. Retirement exposes the lock pathname for as long as it takes
# to re-create it, and nothing may acquire it in that window -- which is what let a
# breaker and an ordinary caller both end up holding. `take_lock` refuses while this
# exists, and a second breaker refuses on it rather than racing the first.
take_breaking() {
    BREAKING_DIR="$LOCK_DIR.breaking"
    if ! mkdir -- "$BREAKING_DIR" 2>/dev/null; then
        echo "refusing: a break of $LOCK_DIR is already in progress" \
             "($(lock_owner_line "$BREAKING_DIR"))" >&2
        return 1
    fi
    BREAKING_HELD=1
    printf 'pid %s nonce %s arm %s at %s\n' "$$" "$LOCK_NONCE_CANDIDATE" "$ARM" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BREAKING_DIR/owner"
}

release_breaking() {
    [ "$BREAKING_HELD" -eq 1 ] || return 0
    BREAKING_HELD=0
    local now=""
    [ ! -f "$BREAKING_DIR/owner" ] || now="$(lock_nonce_of "$BREAKING_DIR")"
    if [ -n "$now" ] && [ "$now" != "$LOCK_NONCE_CANDIDATE" ]; then
        say "UNBREAKING SKIPPED $BREAKING_DIR is not this invocation's"
        return 0
    fi
    rm -f -- "$BREAKING_DIR/owner"
    rmdir -- "$BREAKING_DIR" 2>/dev/null || true
}

# A signal handler must END this invocation. A cleanup-only handler returns, and bash
# resumes -- which would let finalization and promotion continue with the lock released.
on_signal() {  # on_signal <NAME> <number>
    INTERRUPTED=1
    say "SIGNAL $1"
    release_lock
    exit $((128 + $2))
}

# Checked before every protected step, so nothing runs after a handler has fired even if
# some future caller swallows the exit.
not_interrupted() {
    [ "$INTERRUPTED" -eq 0 ] && return 0
    echo "refusing: a signal handler has fired; this invocation does not continue" >&2
    return 1
}

# break_lock <observed-nonce>: retire the SPECIFIC stale lock this invocation observed,
# under the breaking meta-lock so nothing can acquire the exposed pathname meanwhile.
#
# The two schedules the close review demonstrated are closed as follows. (1) An acquirer
# paused between `mkdir` and its owner write leaves an owner-less lock: that is an
# acquisition in progress until it is older than the grace, so the breaker refuses rather
# than retiring it. (2) A breaker whose observation is out of date fails the generation
# comparison, and because the meta-lock is held for the whole break, no third caller can
# slip into the exposed pathname even in the defensive restore branch.
break_lock() {
    local observed="$1" retired="$LOCK_DIR.broken.${1:-empty}" now
    [ -d "$LOCK_DIR" ] || return 0
    take_breaking || return 1
    if lock_is_live "$LOCK_DIR"; then
        echo "refusing: --break-lock will not override the live owner of $LOCK_DIR" \
             "($(lock_owner_line "$LOCK_DIR"))" >&2
        return 1
    fi
    if [ ! -f "$LOCK_DIR/owner" ]; then
        if ! lock_older_than_grace "$LOCK_DIR"; then
            echo "refusing: $LOCK_DIR has no owner file yet and is younger than the" \
                 "${LOCK_GRACE_S}s grace: that is an acquisition in progress, not a" \
                 "stale lock" >&2
            return 1
        fi
    fi
    now="$(lock_nonce_of "$LOCK_DIR")"
    if [ "$now" != "$observed" ]; then
        echo "refusing: $LOCK_DIR now holds the generation ${now:-none}, not the" \
             "${observed:-none} this invocation observed; another publication has" \
             "taken it" >&2
        return 1
    fi
    lock_barrier
    if ! mv -T -- "$LOCK_DIR" "$retired" 2>/dev/null; then
        echo "refusing: could not retire the stale lock $LOCK_DIR as $retired;" \
             "another --break-lock reached it first" >&2
        return 1
    fi
    # Defensive: with the meta-lock held nothing else can have replaced the directory
    # between the comparison and the rename, so this branch should be unreachable.
    now="$(lock_nonce_of "$retired")"
    if [ "$now" != "$observed" ]; then
        mv -T -- "$retired" "$LOCK_DIR" 2>/dev/null \
            || say "UNRESTORED $retired could not be put back as $LOCK_DIR"
        echo "refusing: the lock changed generation while it was being retired" >&2
        return 1
    fi
    say "BREAKLOCK $LOCK_DIR retired as $retired owner=$(lock_owner_line "$retired")"
}

take_lock() {  # take_lock <mode>
    LOCK_DIR="$ARM_ROOT/.publish.lock"
    say "LOCK $LOCK_DIR"
    # A dry run touches nothing outside the test roots, so the lock is real exactly
    # where the tests exercise it and announced everywhere else.
    if [ "$DRY" -eq 1 ] && [ "${EXP11_TEST_ROOTS:-0}" != 1 ]; then return 0; fi
    mkdir -p -- "$ARM_ROOT" || return 1
    # The generation is chosen BEFORE anything is created, so the meta-lock and the lock
    # itself carry the same invocation identity.
    LOCK_NONCE_CANDIDATE="$$-$(date -u +%s%N)-${RANDOM}${RANDOM}"
    if [ "$BREAK_LOCK" -eq 1 ] && [ -d "$LOCK_DIR" ]; then
        break_lock "$(lock_nonce_of "$LOCK_DIR")" || { release_breaking; return 1; }
    fi
    if [ -d "$LOCK_DIR.breaking" ] && [ "$BREAKING_HELD" -eq 0 ]; then
        echo "refusing: a break of $LOCK_DIR is in progress; the lock pathname is not" \
             "available while it is being retired" >&2
        return 1
    fi
    if ! mkdir -- "$LOCK_DIR" 2>/dev/null; then
        local live=no
        lock_is_live "$LOCK_DIR" && live=yes
        echo "refusing: another publication holds the lock $LOCK_DIR (owner:" \
             "$(lock_owner_line "$LOCK_DIR"), live=$live)." \
             "Publishing one arm is serialised; a stale lock is cleared only with" \
             "--break-lock, never automatically" >&2
        return 1
    fi
    local inode
    inode="$(lock_inode_of "$LOCK_DIR")"
    LOCK_NONCE="$LOCK_NONCE_CANDIDATE"
    LOCK_HELD=1
    trap release_lock EXIT
    trap 'on_signal INT 2' INT
    trap 'on_signal TERM 15' TERM
    # The owner file lands atomically inside the directory this invocation created: a
    # write into a retired directory fails here rather than appearing in somebody else's.
    if ! { printf 'pid %s nonce %s mode %s arm %s at %s\n' "$$" "$LOCK_NONCE" "$1" \
               "$ARM" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/.owner.$$" \
           && mv -- "$LOCK_DIR/.owner.$$" "$LOCK_DIR/owner"; }; then
        LOCK_HELD=0
        echo "refusing: could not record ownership of $LOCK_DIR; it was retired while" \
             "this invocation was acquiring it" >&2
        return 1
    fi
    lock_acquire_barrier
    # SELF-VERIFICATION. Between `mkdir` and now the directory may have been retired and
    # re-created by somebody else; this invocation then owns nothing and must remove
    # nothing. Same inode, our own generation, and no break in progress.
    if [ "$(lock_inode_of "$LOCK_DIR")" != "$inode" ] \
       || [ "$(lock_nonce_of "$LOCK_DIR")" != "$LOCK_NONCE" ] \
       || { [ -d "$LOCK_DIR.breaking" ] && [ "$BREAKING_HELD" -eq 0 ]; }; then
        LOCK_HELD=0
        echo "refusing: $LOCK_DIR is no longer the lock this invocation created" \
             "(inode or generation changed, or a break began); nothing was removed" >&2
        return 1
    fi
    release_breaking
}

# published_target: what `final` resolves to under this arm root, or nothing.
published_target() {
    [ -L "$ARM_ROOT/final" ] || return 1
    readlink -f -- "$ARM_ROOT/final"
}

# already_published <canonical>: this attempt is the published one and it completed.
already_published() {
    local target
    target="$(published_target)" || return 1
    [ "$target" = "$1" ] && [ -f "$1/completion.json" ]
}

# check_final_conflict <canonical>: a DIFFERENT existing `final` is never replaced by a
# recovery unless the operator asks for it, and the replaced target is logged first.
check_final_conflict() {
    local target
    target="$(published_target)" || return 0
    [ "$target" != "$1" ] || return 0
    if [ "$REPLACE_FINAL" -eq 0 ]; then
        echo "refusing: $ARM_ROOT/final already publishes $target, not $1; pass" \
             "--replace-final to replace it" >&2
        return 1
    fi
    say "REPLACING $ARM_ROOT/final -> $(basename -- "$target") with $(basename -- "$1")"
}

# abort_recovery <canonical> <log>: the abort path of a REFUSED recovery. A malformed
# invocation does not establish that a finished run failed, so an attempt that is
# already published or already completed is left exactly as it is -- directory,
# completion and `final` untouched -- and the finalizer's own cause stands on stderr.
# Full mode keeps aborting unconditionally: it launched the attempt it is aborting.
abort_recovery() {
    local canonical="$1" log="$2" why=""
    [ -f "$canonical/completion.json" ] && why="it has completed"
    [ "$(published_target || true)" != "$canonical" ] || why="it is published as final"
    if [ -n "$why" ]; then
        say "PRESERVED $canonical ($why); the refusal is this invocation's, not the run's"
        return 0
    fi
    abort "$canonical" "$log" finalize_refused
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
CANONICAL_ATTEMPT="${CANONICAL_ATTEMPT:-}"
BREAK_LOCK="${BREAK_LOCK:-0}"
INTERRUPTED="${INTERRUPTED:-0}"
LOCK_NONCE_CANDIDATE="${LOCK_NONCE_CANDIDATE:-}"
REPLACE_FINAL="${REPLACE_FINAL:-0}"
DRY="${DRY:-0}"
ARM="${ARM:-H}"

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
        --break-lock) BREAK_LOCK=1; shift ;;
        --replace-final) REPLACE_FINAL=1; shift ;;
        --dry-run) DRY=1; shift ;;
        *) usage ;;
    esac
done
[ -n "$GPU" ] && [ -n "$COMMIT" ] || usage
apply_root_override || exit 2
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
    take_lock full || exit 2   # publishing this arm is serialised with any recovery
    if [ "$DRY" -eq 1 ]; then
        say "MARKER EXP06_CHILD_EXIT <code> <iso> >> $log"
        abort "$attempt" "$log" 'child_exit_<code>'
        finalize "$attempt" "$log" '<code>' exp11_train
        promote "attempt_$STAMP"
        release_lock
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
    not_interrupted || exit 2
    if ! finalize "$attempt" "$log" "$status" exp11_train; then
        abort "$attempt" "$log" finalize_refused
        exit 2
    fi
    not_interrupted || exit 2
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
    take_lock finalize || exit 2   # alias resolution, validation and publication
    check_attempt "$ATTEMPT" || exit 2
    # Everything below uses the canonical directory, never the path as it was typed.
    if already_published "$CANONICAL_ATTEMPT"; then
        say "SKIP $CANONICAL_ATTEMPT is already published as final and has completed"
        say "ALREADY PUBLISHED $ARM_ROOT/final -> $(basename -- "$CANONICAL_ATTEMPT")"
        exit 0
    fi
    check_final_conflict "$CANONICAL_ATTEMPT" || exit 2
    [ "$DRY" -eq 0 ] || abort_recovery "$CANONICAL_ATTEMPT" "$LOG"
    not_interrupted || exit 2
    if ! finalize "$CANONICAL_ATTEMPT" "$LOG" "$CHILD_EXIT" exp11_train; then
        abort_recovery "$CANONICAL_ATTEMPT" "$LOG"
        exit 2
    fi
    not_interrupted || exit 2
    promote "$(basename -- "$CANONICAL_ATTEMPT")"
    ;;
esac
