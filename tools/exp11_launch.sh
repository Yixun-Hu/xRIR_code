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
# Publishing one arm is serialised by an `flock` on a per-arm lock FILE. The lock is held
# by a holder process leased to this launcher's life, never on a descriptor of the
# launcher's own: **it vanishes within ~1 s of the launcher's exit or death and is never
# inherited by the trainer or the log sink**. There is no cleanup code, no owner record
# and no stale-lock concept -- nothing to clear by hand and nothing to get wrong (plan
# section 11 amendment A1, close review 6).
#
# Whether anything of an arm is RUNNING is decided from its pid files, and one module
# reads those: tools/exp11_pidrecord.py, which answers `record <pid>` or `norecord` and
# is checked against two known files before every scan and every resolution. Anything
# else it says or does is "unknown", and unknown is not an answer: **a broken reader
# closes the arm to every automated decision; it never opens it** (close review 11).
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

# apply_reader_override: EXP11_PYTHON lets a regression break the pid reader the way a
# broken environment would -- a missing interpreter, an unimportable module -- so the
# fail-closed paths can be exercised against this file rather than a copy of it. Like
# EXP11_PRETRAIN_ROOT it is a dry-run test affordance and is refused outright otherwise:
# a real launch decides liveness with the interpreter the pinned library names, and no
# invocation may swap the interpreter that also finalizes, publishes and trains.
apply_reader_override() {
    [ -n "${EXP11_PYTHON:-}" ] || return 0
    if [ "${EXP11_TEST_ROOTS:-0}" != 1 ] || [ "$DRY" -ne 1 ]; then
        echo "refusing: EXP11_PYTHON replaces the interpreter that reads every pid file," \
             "so it is honoured only with EXP11_TEST_ROOTS=1 and --dry-run" >&2
        return 1
    fi
    PYTHON="$EXP11_PYTHON"
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
    echo "       finalize also takes [--replace-final]" >&2
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

# --- publishing one arm is serialised, by the kernel --------------------------------
# Plan section 11 amendment A1, close review 6. Publishing one arm admits exactly one
# invocation at a time: recovery from alias resolution through promotion, and a full run
# from before it launches its child through promotion, so a recovery is refused while a
# training run still owns the arm.
#
# The lock is an `flock` on a per-arm FILE, and it is NOT held by this shell: every
# descriptor this shell has open is inherited by the background log sink and by the
# detached training child, and an orphaned sink would keep an arm locked for good. It is
# held by a HOLDER PROCESS whose life is leased to this launcher's -- it exits as soon as
# its parent changes -- so **the lock vanishes within a second of this launcher's exit or
# death**, on every path including a signal and SIGKILL, and is never inherited by the
# trainer or the log sink. The attempt is non-blocking: it refuses at once,
# never waits.
LOCK_FILE=""
HOLDER_PID=""

hold_arm_lock() {  # hold_arm_lock <mode>
    LOCK_FILE="$ARM_ROOT/.publish.lock"
    say "LOCK $LOCK_FILE"
    # A dry run touches nothing outside the test roots, so the lock is real exactly
    # where the tests exercise it and announced everywhere else.
    if [ "$DRY" -eq 1 ] && [ "${EXP11_TEST_ROOTS:-0}" != 1 ]; then return 0; fi
    mkdir -p -- "$ARM_ROOT" || return 1
    local verdict="" pid="" out=""
    # `exec` inside the substitution makes the holder a direct child of this shell, so
    # its lease is this shell's pid. The read end is closed the moment the verdict is in,
    # so no descriptor of ours reaches the sink or the child either.
    exec {out}< <(exec "$PYTHON" tools/exp11_lock_holder.py "$LOCK_FILE" $$)
    read -r -t 30 verdict pid <&"$out" || true
    exec {out}<&-
    if [ "$verdict" = refused ]; then
        echo "refusing: $LOCK_FILE is held by another invocation publishing this arm;" \
             "one publication per arm at a time. Its holder ends with that invocation," \
             "so there is nothing to clear by hand" >&2
        return 1
    fi
    if [ "$verdict" != acquired ] || [ -z "$pid" ]; then
        # Not someone else's lock: the holder never reported one. Say which it is.
        echo "refusing: the lock holder for $LOCK_FILE did not start (no verdict);" \
             "nothing was published" >&2
        return 1
    fi
    HOLDER_PID="$pid"
    trap 'on_signal INT 2' INT
    trap 'on_signal TERM 15' TERM
    trap 'drop_arm_lock' EXIT
    say "LOCKED $LOCK_FILE mode=$1"
}

# drop_arm_lock: end our own holder, so the ordinary path releases at once instead of
# waiting out the lease. Every other path -- a signal, a crash, a kill -9 -- is covered
# by the lease itself, which is why this needs no error handling of its own. A recorded
# pid is not a licence to signal it: if that holder is already gone the number belongs to
# whoever the kernel gave it to next, so nothing is sent unless the process is still this
# shell's child.
drop_arm_lock() {
    [ -n "$HOLDER_PID" ] || return 0
    local parent=""
    parent="$(sed -n 's/^.*) [A-Za-z] \([0-9][0-9]*\).*/\1/p' \
              "/proc/$HOLDER_PID/stat" 2>/dev/null || true)"
    [ "$parent" = "$$" ] && { kill -TERM "$HOLDER_PID" 2>/dev/null || true; }
    wait "$HOLDER_PID" 2>/dev/null || true
    HOLDER_PID=""
}

# --- what the lock protects is a quiet arm -------------------------------------------
# The lock says "one publication at a time"; it does not say "no trainer is running".
# That is this scan's job, and it has to run UNDER the lock: a scan taken before
# acquisition can go stale in the window between, which is exactly how a recovery could
# promote an older attempt while a newer trainer of the same arm kept writing
# (close review 7, blocker 1). `kill -0` asks the kernel whether a pid still exists; it
# is a probe and sends no signal.
UNRESOLVED_GRACE_S="${EXP11_UNRESOLVED_GRACE_S:-120}"
SCAN_REASON=""
NEWLINE=$'\n'   # a real newline: a verdict that is two lines is not a verdict

# THREE answers, never two (close review 11). The grammar lives in
# tools/exp11_pidrecord.py and nowhere else -- this shell does NOT read pid files, since
# Bash command substitution drops NUL bytes (close review 10, blocker 1). But a reader
# that cannot run -- no interpreter, an unimportable module, a crash -- used to be
# indistinguishable from "this file holds no record": both were exit 1. Read as "no
# record", a broken reader makes every pid file in the arm look dead, and a resolution
# then retires the attempt of a registered, running trainer. So the reader now answers
# `record <pid>` or `norecord` on stdout with exit 0, and everything else is UNKNOWN.
#
# pid_record <file>: 0 and the pid on stdout | 1 no record | 2 the reader failed.
#
# The verdict is checked as RAW BYTES, in a file, before anything of it passes through a
# shell expansion: command substitution deletes trailing newlines and NUL bytes, so
# `norecord\n\n` and `norecord\0\n` -- neither of which is the protocol -- arrived here
# as clean verdicts and retired a live trainer's attempt (close review 12, blocker 2).
# Only the digits of an accepted pid cross back, and digits survive intact.
READER_UNKNOWN=2

# --- three states for every path this launcher asks about (close review 14) ----------
# `[ -f x ]` answers "no" for a file that is not there AND for one nobody was allowed to
# look at, and every lifecycle guard that used it turned the second into the first: a
# marker behind a mode-000 door read as no marker, a completion receipt behind one read
# as no receipt, a `final` through an inaccessible directory read as nothing published.
#
# probe_path <path>: 0 it is there (PROBE_MODE holds its %f mode) | 1 it is NOT there |
# 2 nobody could say. An absence is a CLAIM, made only about a directory we could look
# into; a name that exists as a link we cannot follow is not an absence at all.
PROBE_MODE=""
probe_path() {
    local parent=""
    PROBE_MODE=""
    if PROBE_MODE="$(stat -L -c %f -- "$1" 2>/dev/null)"; then
        case "$PROBE_MODE" in ''|*[!0-9a-fA-F]*) PROBE_MODE=""; return "$READER_UNKNOWN" ;; esac
        return 0
    fi
    PROBE_MODE=""
    # The name may still BE there, as a symlink that dangles or loops or whose target we
    # may not reach. That is not "there is nothing here".
    stat -c %f -- "$1" >/dev/null 2>&1 && return "$READER_UNKNOWN"
    parent="$(dirname -- "$1")"
    # ... and an absence may only be claimed about a directory we could enter.
    stat -L -c %f -- "$parent/." >/dev/null 2>&1 || return "$READER_UNKNOWN"
    [ -x "$parent" ] || return "$READER_UNKNOWN"
    return 1
}

# file_present <path>: 0 a regular file is there | 1 no regular file is there (absent, or
# the name is something else) | 2 nobody could say.
file_present() {
    local status=0
    probe_path "$1" || status=$?
    [ "$status" -ne "$READER_UNKNOWN" ] || return "$READER_UNKNOWN"
    [ "$status" -eq 0 ] || return 1
    [ $(( 0x$PROBE_MODE & 0xF000 )) -eq $(( 0x8000 )) ] || return 1
}

# probe_link_target <path>: 0 and the canonical target on stdout | 1 there is no such
# link | 2 it is there and cannot be resolved, or nobody could look. `readlink -f`
# answered "nothing" for both of the last two.
probe_link_target() {
    local mode="" parent="" target=""
    if ! mode="$(stat -c %f -- "$1" 2>/dev/null)"; then       # the LINK, not its target
        parent="$(dirname -- "$1")"
        stat -L -c %f -- "$parent/." >/dev/null 2>&1 || return "$READER_UNKNOWN"
        [ -x "$parent" ] || return "$READER_UNKNOWN"
        return 1
    fi
    case "$mode" in ''|*[!0-9a-fA-F]*) return "$READER_UNKNOWN" ;; esac
    [ $(( 0x$mode & 0xF000 )) -eq $(( 0xA000 )) ] || return 1   # not a symlink at all
    target="$(realpath -e -- "$1" 2>/dev/null)" || return "$READER_UNKNOWN"
    printf '%s\n' "$target"
}

pid_record() {
    local out="" pid="" kind="" size="" status=0
    # What the path IS, decided before anything opens it: a directory, FIFO, socket or
    # device is definitely not a pid record, and opening a FIFO would wait for a writer
    # forever. A stat that FAILS decides nothing here -- the module tells a confirmed
    # absence from a path it was not allowed to inspect (close review 12, blocker 1),
    # which is why this is no longer `[ -f ]`: that reads an unreadable parent as "gone".
    # `%f` is the mode in hex, not `%F`'s phrase: that phrase is translated, so matching
    # it would classify an ordinary file as "not a record" under any other locale.
    if kind="$(stat -L -c %f -- "$1" 2>/dev/null)"; then
        case "$kind" in ''|*[!0-9a-fA-F]*) return "$READER_UNKNOWN" ;; esac
        if [ $(( 0x$kind & 0xF000 )) -ne $(( 0x8000 )) ]; then
            return 1                          # definite: not a regular file, unopened
        fi
    fi
    out="$(mktemp "${TMPDIR:-/tmp}/exp11_verdict.XXXXXX")" || return "$READER_UNKNOWN"
    "$PYTHON" -m tools.exp11_pidrecord "$1" > "$out" 2>/dev/null || status=$?
    if [ "$status" -ne 0 ]; then rm -f -- "$out"; return "$READER_UNKNOWN"; fi
    # A verdict is eighteen bytes at most ('record ' + ten digits + a newline), so the
    # size answers first and no rogue reader's output is ever read, compared or -- worst
    # of all -- materialised as a shell variable (close review 13, nonblocking).
    size="$(stat -c %s -- "$out" 2>/dev/null)" || { rm -f -- "$out"; return "$READER_UNKNOWN"; }
    case "$size" in ''|*[!0-9]*) rm -f -- "$out"; return "$READER_UNKNOWN" ;; esac
    if [ "$size" -gt 64 ]; then rm -f -- "$out"; return "$READER_UNKNOWN"; fi
    if cmp -s "$out" <(printf 'norecord\n'); then rm -f -- "$out"; return 1; fi
    # Digits only -- a candidate, not yet a verdict: the bytes must then match the whole
    # line exactly, which is what refuses padding, NULs, CR and trailing junk.
    pid="$(head -c 64 -- "$out" | tr -cd '0-9')"
    case "$pid" in ''|*[!0-9]*) rm -f -- "$out"; return "$READER_UNKNOWN" ;; esac
    [ "${#pid}" -le 10 ] || { rm -f -- "$out"; return "$READER_UNKNOWN"; }
    if cmp -s "$out" <(printf 'record %s\n' "$pid"); then
        rm -f -- "$out"
        printf '%s\n' "$pid"
        return 0
    fi
    rm -f -- "$out"
    return "$READER_UNKNOWN"
}

# pid_alive <file>: 0 the recorded pid is alive | 1 it is not | 2 nobody can say.
pid_alive() {
    local pid="" status=0
    pid="$(pid_record "$1")" || status=$?
    [ "$status" -ne "$READER_UNKNOWN" ] || return "$READER_UNKNOWN"
    [ "$status" -eq 0 ] || return 1
    kill -0 "$pid" 2>/dev/null
}

# registration_complete <attempt>: child.pid holds one pid, by the grammar above. The
# shell creates that file by redirection before the pid reaches it, so existence alone
# proves nothing; what makes a launch *registered* is a number to look for. Whether that
# number is still alive is a separate question, asked by pid_alive of the same bytes.
# 0 complete | 1 not complete | 2 unknown -- the unknown is never folded into "not".
registration_complete() {
    pid_record "$1/child.pid" >/dev/null
}

# check_pid_reader: ask the reader two questions whose answers are known, before any
# decision is taken. A reader stuck on one verdict never fails -- "record 1" for
# everything makes every pid file a registration, "norecord" for everything makes every
# trainer dead -- so only a probe catches it. The probes go through the same $PYTHON and
# the same invocation as every real read.
# probe_reader <file> <expected bytes> <where the answer goes>: 0 when the reader exited
# 0 and printed EXACTLY those bytes. Byte for byte, like every real read.
probe_reader() {
    "$PYTHON" -m tools.exp11_pidrecord "$1" > "$3" 2>/dev/null || return 1
    cmp -s "$3" "$2"
}

check_pid_reader() {
    local dir="" bad=""
    dir="$(mktemp -d "${TMPDIR:-/tmp}/exp11_pidreader.XXXXXX")" || {
        echo "refusing: cannot create a probe directory for the pid reader" >&2
        return 1; }
    printf '1\n' > "$dir/one"
    : > "$dir/empty"
    printf 'record 1\n' > "$dir/want_one"
    printf 'norecord\n' > "$dir/want_empty"
    probe_reader "$dir/one" "$dir/want_one" "$dir/got_one" || bad="a file holding 1"
    probe_reader "$dir/empty" "$dir/want_empty" "$dir/got_empty" ||
        bad="${bad:+$bad and }an empty file"
    rm -rf -- "$dir"
    [ -z "$bad" ] && return 0
    echo "refusing: pid reader unhealthy: $PYTHON -m tools.exp11_pidrecord did not" \
         "answer byte for byte on $bad -- 'record 1' and 'norecord', one line each and" \
         "nothing more, no padding and no NULs. Nothing of this arm can be decided by a" \
         "reader that cannot say what a pid file holds" >&2
    return 1
}

# probe_live <pid file> <what is alive>: 0 nothing found, 1 alive, 2 unknown. The last
# two set SCAN_REASON and must be propagated unchanged -- an unknown liveness is not a
# quiet arm, it is no answer at all.
probe_live() {
    local status=0
    pid_alive "$1" || status=$?
    case "$status" in
        0) SCAN_REASON="$2"; return 1 ;;
        "$READER_UNKNOWN")
            SCAN_REASON="liveness unknown: pid reader failed on $1. Nothing is scanned,"\
" retired or published on an answer nobody gave"
            return "$READER_UNKNOWN" ;;
    esac
    return 0
}

# list_attempts <arm root> <listing file>: 0 only when the root really was ENUMERATED,
# the listing then holding its attempt_* entries NUL-separated. A directory with mode
# 0300 is searchable and writable but NOT listable: every known pathname under it still
# works -- both lock files open, `stat` answers, the holder starts -- while a glob over
# it silently expands to nothing. That read as "this arm has no attempts", and a
# resolution retired the attempt of a registered, running trainer (close review 13). A
# successful stat of a name we already knew is never evidence that we saw the directory.
list_attempts() {
    local root="$1" out="$2" canonical=""
    # The DIRECTORY the root denotes, not the name it was given by: `-d`, `-r` and `-x`
    # all follow a symlink, but `find -P` does not descend one handed to it as a
    # starting point -- it returned an empty listing and success for an arm with a live
    # trainer in it (close review 14, blocker 1). `realpath -e` is checked: a root that
    # cannot be resolved is not an empty arm.
    canonical="$(realpath -e -- "$root" 2>/dev/null)" || return 1
    [ -d "$canonical" ] || return 1
    [ -r "$canonical" ] && [ -x "$canonical" ] || return 1   # listing needs read AND search
    find "$canonical" -mindepth 1 -maxdepth 1 -name 'attempt_*' -print0 > "$out" 2>/dev/null
}

# scan_arm <arm root> [<attempt being resolved>]: 0 when every attempt of the arm is
# finished and accounted for, 1 when one is not (SCAN_REASON says which), 2 when the
# reader failed and no answer exists. A broken reader closes the arm to every automated
# decision; it never opens it (close review 11).
scan_arm() {
    local root="$1" resolving="${2:-}" attempt="" name="" here="" listing="" mode=""
    local -a attempts=()
    SCAN_REASON=""
    # The attempt being resolved is excluded by canonical identity, never by spelling:
    # the caller's path and this glob's may name the same directory differently.
    [ -z "$resolving" ] || resolving="$(realpath -- "$resolving" 2>/dev/null || printf '%s' "$resolving")"
    listing="$(mktemp "${TMPDIR:-/tmp}/exp11_attempts.XXXXXX")" || {
        SCAN_REASON="liveness unknown: cannot create a listing file for $root"
        return "$READER_UNKNOWN"; }
    if ! list_attempts "$root" "$listing"; then
        rm -f -- "$listing"
        SCAN_REASON="liveness unknown: the arm root cannot be enumerated ($root); an"\
" attempt nobody can see is not an attempt that is not there, and nothing of this arm is"\
" scanned, retired or published on that"
        return "$READER_UNKNOWN"
    fi
    # -d '' is the NUL delimiter: an attempt name may hold anything but a NUL.
    while IFS= read -r -d '' attempt; do attempts+=("$attempt"); done < "$listing"
    rm -f -- "$listing"
    for attempt in ${attempts[@]+"${attempts[@]}"}; do
        # `-d` answers "no" both for a name that is not a directory and for one nobody
        # was allowed to look at; only a stat that SUCCEEDED classifies (close review 13).
        mode="$(stat -L -c %f -- "$attempt" 2>/dev/null)" || {
            SCAN_REASON="liveness unknown: $attempt is in this arm and cannot be"\
" inspected, so what it holds cannot be known"
            return "$READER_UNKNOWN"; }
        case "$mode" in ''|*[!0-9a-fA-F]*)
            SCAN_REASON="liveness unknown: $attempt has no readable file mode"
            return "$READER_UNKNOWN" ;;
        esac
        [ $(( 0x$mode & 0xF000 )) -eq $(( 0x4000 )) ] || continue   # not a directory
        name="$(basename -- "$attempt")"
        here="$(realpath -- "$attempt" 2>/dev/null || printf '%s' "$attempt")"
        probe_live "$attempt/launch.pid" "$name has a live launcher (launch.pid)" || return $?
        probe_live "$attempt/child.pid" "$name has a live trainer (child.pid)" || return $?
        probe_live "$attempt/train.pid" "$name has a live trainer (train.pid)" || return $?
        [ "$here" != "$resolving" ] || continue
        # `child.pid` is the WRAPPER's (GNU timeout), not the trainer's, so a complete
        # and dead one accounts for nothing: only train.pid names the trainer and only
        # train.exit says it finished (close review 10, blocker 2). Each of the three is
        # a three-state question: a marker we could not look at closes the arm, because
        # the alternative is closing our eyes and calling it quiet (close review 14).
        local marker=0 recorded=0 finished=0
        file_present "$attempt/launching" || marker=$?
        file_present "$attempt/train.pid" || recorded=$?
        file_present "$attempt/train.exit" || finished=$?
        if [ "$marker" -eq "$READER_UNKNOWN" ] || [ "$recorded" -eq "$READER_UNKNOWN" ] ||
           [ "$finished" -eq "$READER_UNKNOWN" ]; then
            SCAN_REASON="liveness unknown: $name has a launching marker, a train.pid or a"\
" train.exit that cannot be inspected, so whether a trainer of it is accounted for cannot"\
" be known"
            return "$READER_UNKNOWN"
        fi
        if [ "$marker" -eq 0 ] &&
           { [ "$recorded" -ne 0 ] || [ "$finished" -ne 0 ]; }; then
            SCAN_REASON="$name has an unresolved launch: a launching marker and no"\
" finished trainer (train.pid and train.exit), so a trainer of it may be running or may"\
" have died unrecorded -- child.pid names the timeout wrapper, not the trainer."\
" Resolve it"\
" with --resolve-unregistered $attempt once nothing of the arm is alive and the marker"\
" is older than ${UNRESOLVED_GRACE_S}s"
            return 1; fi
    done
    return 0
}

require_quiet_arm() {  # require_quiet_arm [<attempt being resolved>]
    local status=0
    say "SCAN $ARM_ROOT"
    if [ "$DRY" -eq 1 ] && [ "${EXP11_TEST_ROOTS:-0}" != 1 ]; then return 0; fi
    check_pid_reader || return "$READER_UNKNOWN"   # before anything is looked at
    scan_arm "$ARM_ROOT" "${1:-}" || status=$?
    [ "$status" -eq 0 ] && return 0
    echo "refusing: $SCAN_REASON" >&2
    return "$status"
}

# --- the intent marker: the half-second the frozen lifecycle cannot cover -----------
# `run_child` forks the detached trainer and writes child.pid only afterwards, and the
# trainer needs a moment to write its own. A launcher that dies inside that window would
# leave a running trainer nothing names. So the intent to launch is declared BEFORE the
# fork and withdrawn only once the TRAINER is accounted for -- a complete child.pid and a
# train.exit. `child.pid` holds the timeout wrapper's pid, so its death says nothing about
# the trainer (close review 10). A marker left behind means "a trainer of this attempt may
# be running, or died without saying so", and the arm stays closed until an operator
# resolves it: a trainer that crashes before registering fails closed, by design.
mark_launching() {  # mark_launching <attempt>
    printf 'launcher %s\nat %s\narm %s\n' \
        "$$" "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "$ARM" > "$1/launching"
}

clear_launching() {  # clear_launching <attempt>: only once the TRAINER is accounted for
    # This runs after run_child returns, so the wrapper and the sink are gone. A complete
    # child.pid proves the wrapper was recorded; train.exit proves the trainer itself
    # finished. A trainer that crashed before registering leaves the marker standing and
    # the attempt unresolved -- fail closed, by design: the operator answers it.
    # `|| return 0` covers BOTH "no complete child.pid" and "the reader could not say":
    # a marker is withdrawn on an answer, never on the absence of one (close review 11).
    # And not at all while the arm it belongs to cannot even be enumerated: the marker is
    # protection, and protection is not withdrawn on an understanding we do not have
    # (close review 13).
    local listing=""
    listing="$(mktemp "${TMPDIR:-/tmp}/exp11_attempts.XXXXXX")" || return 0
    if ! list_attempts "$(dirname -- "$1")" "$listing"; then rm -f -- "$listing"; return 0; fi
    rm -f -- "$listing"
    registration_complete "$1" || return 0
    file_present "$1/train.exit" || return 0   # absent OR unknown: the marker stays
    rm -f -- "$1/launching"
}

# resolve_unregistered <attempt>: an operator's explicit answer to a marker nothing can
# explain any more. It can only ever abort: an attempt with no child.pid has no exit
# receipt and can never be published.
resolve_unregistered() {
    local attempt="" root="" name="" now=0 stamp=0 age=0
    # This is the one decision that deliberately ignores its own target's marker -- the
    # tombstone is what makes that safe -- so the liveness answers are all that stand
    # between a running trainer and a retired directory. Ask the reader for them only
    # once it has proved it can answer (close review 11).
    check_pid_reader || return "$READER_UNKNOWN"
    # Identity is the canonical directory, never the spelling: an operator's path may be
    # absolute where the root is relative, or reach the arm through a symlink.
    [ -d "$1" ] || { echo "refusing: $1 is not a directory" >&2; return 1; }
    attempt="$(realpath -- "$1")" || return 1
    root="$(realpath -- "$ARM_ROOT")" || return 1
    name="$(basename -- "$attempt")"
    # The lock and the scan are this arm's, so the resolution reaches no further.
    if [ "$(dirname -- "$attempt")" != "$root" ]; then
        echo "refusing: $attempt is not an attempt of the selected arm ($root); this" \
             "invocation locked and scanned that arm and no other" >&2
        return 1
    fi
    case "$name" in
        attempt_*) case "${name#attempt_}" in ''|*[!0-9T]*)
            echo "refusing: $name is not an attempt directory of this arm" >&2
            return 1 ;; esac ;;
        *) echo "refusing: $name is not an attempt directory of this arm" >&2
           return 1 ;;
    esac
    # A directory we cannot look INTO says nothing about what is in it -- least of all
    # that there is no marker and no live pid file. Not being able to look is never an
    # answer (close review 12, blocker 1).
    if ! stat -L -c %f -- "$attempt/." >/dev/null 2>&1; then
        echo "refusing: liveness unknown: $attempt cannot be inspected, so neither its" \
             "launching marker nor its pid files nor its receipts can be read; nothing" \
             "of this arm is retired on that" >&2
        return "$READER_UNKNOWN"
    fi
    local marker=0
    file_present "$attempt/launching" || marker=$?
    if [ "$marker" -eq "$READER_UNKNOWN" ]; then
        echo "refusing: liveness unknown: $attempt/launching cannot be inspected; a" \
             "marker nobody can look at is not a marker that is not there" >&2
        return "$READER_UNKNOWN"
    fi
    [ "$marker" -eq 0 ] || {
        echo "refusing: $attempt has no launching marker; nothing is unresolved" >&2
        return 1; }
    # NOT "child.pid is complete": that file holds the `timeout` wrapper's pid, and a
    # wrapper that died with its trainer unregistered is precisely the state the scan
    # now calls unresolved (close review 10, blocker 2). Refusing on it would leave the
    # arm shut for good -- closed by the scan, unanswerable by the operator. What may
    # never be retired this way is a run that actually finished: a completion receipt,
    # or the published `final`. Everything else is decided by the quiet scan below --
    # nothing of the arm alive -- and by the marker's age.
    local receipt=0 published=0 target=""
    file_present "$attempt/completion.json" || receipt=$?
    if [ "$receipt" -eq "$READER_UNKNOWN" ]; then
        echo "refusing: liveness unknown: $attempt/completion.json cannot be inspected," \
             "so whether this run finished cannot be known and nothing of it is retired" >&2
        return "$READER_UNKNOWN"
    fi
    if [ "$receipt" -eq 0 ]; then
        echo "refusing: $attempt has a completion receipt; it is a finished run, not an" \
             "unresolved launch" >&2
        return 1
    fi
    target="$(probe_link_target "$ARM_ROOT/final")" || published=$?
    if [ "$published" -eq "$READER_UNKNOWN" ]; then
        echo "refusing: liveness unknown: $ARM_ROOT/final cannot be resolved, so whether" \
             "this attempt is the published one cannot be known" >&2
        return "$READER_UNKNOWN"
    fi
    if [ "$published" -eq 0 ] && [ "$target" = "$attempt" ]; then
        echo "refusing: $attempt is published as $ARM_ROOT/final; a published attempt is" \
             "never retired as an unresolved launch" >&2
        return 1
    fi
    # The REGISTRATION lock, distinct from the publication lock this invocation already
    # holds: a full-mode launcher holds the publication lock for its child's whole run,
    # so a trainer could never take that one. This one is held only across a
    # registration and across this resolution -- exactly the pair that must not
    # interleave, because the trainer's checks and its write straddle the rename below
    # (close review 9, blocker 2). The recovery and full-mode scans do not take it: an
    # in-flight registration is already covered there by the launching marker or by a
    # complete live child.pid, and a recovery that had to queue behind every trainer
    # would be a new way to stall an arm.
    local registration="$root/.registration.lock" reg_fd=""
    : >> "$registration" 2>/dev/null || true
    exec {reg_fd}>>"$registration" || {
        echo "refusing: cannot open the registration lock $registration" >&2; return 1; }
    if ! flock -n "$reg_fd"; then
        exec {reg_fd}>&-
        echo "refusing: a registration in progress holds $registration; a trainer is" \
             "between its checks and its write, so nothing of this arm may be retired" >&2
        return 1
    fi
    local scanned=0
    scan_arm "$ARM_ROOT" "$attempt" || scanned=$?
    if [ "$scanned" -ne 0 ]; then
        exec {reg_fd}>&-
        echo "refusing: $SCAN_REASON" >&2
        return "$scanned"          # 2 = the reader failed: this resolution decides nothing
    fi
    now="$(date -u +%s)"
    stamp="$(date -u -r "$attempt/launching" +%s 2>/dev/null || printf '%s' "$now")"
    age=$((now - stamp))
    if [ "$age" -lt "$UNRESOLVED_GRACE_S" ]; then
        exec {reg_fd}>&-
        echo "refusing: $attempt/launching is ${age}s old and a trainer that has not" \
             "registered yet is still possible; wait until it is ${UNRESOLVED_GRACE_S}s" >&2
        return 1
    fi
    say "TOMBSTONE $root/$name.resolved"
    say "ABORT $attempt -> ${attempt}_ABORTED_unregistered"
    if [ "$DRY" -eq 0 ]; then
        # Written BEFORE the rename, and never removed by anything here: a trainer that
        # wakes after this can no longer find its directory, and if it somehow can, this
        # is what tells it the launch was answered. The scan ignores it -- it is a
        # record, not a claim (close review 8, blocker 2).
        # Exactly the conditions this resolution checked -- nothing else is asserted.
        # `train.exit` is NOT among them: a stale marker beside a finished trainer is
        # resolvable too (close review 12, wording).
        printf 'resolved-by %s\nat %s\nreason %s\n' "$$" \
            "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" \
            "unresolved launch, retired under this arm's publication and registration locks: a launching marker older than ${UNRESOLVED_GRACE_S}s, no recorded pid of this arm alive, no completion receipt, and not the published final. A trainer that never registered cannot be seen at all; this tombstone is what refuses it." \
            > "$root/$name.resolved"
        mv -- "$attempt" "${attempt}_ABORTED_unregistered"
        # The marker has been answered; leaving it would close the arm for good, since
        # the scan reads every attempt_* directory, retired ones included.
        rm -f -- "${attempt}_ABORTED_unregistered/launching"
    fi
    exec {reg_fd}>&-   # the whole critical section is done: scan, tombstone, rename
    # Precisely what was established, and only that: both locks were held, no recorded
    # pid of the arm answered a liveness probe, the attempt has no completion receipt and
    # is not the published `final`, and its marker was older than the grace. A trainer
    # that never registered cannot be seen at all -- which is why the tombstone above is
    # written first and never removed (close review 11 and 12, wording).
    say "RESOLVED $attempt: marker older than ${UNRESOLVED_GRACE_S}s, no recorded pid of" \
        "this arm alive, no completion receipt, not the published final. It can never be" \
        "published, and the tombstone refuses any trainer of it that is still to come"
}

# A deterministic pause, so a regression can hold this invocation open between two
# launchers. It exists only inside the test roots and costs a real launch nothing.
scan_barrier() {
    local file="${EXP11_SCAN_BARRIER:-}"
    [ -n "$file" ] && [ "${EXP11_TEST_ROOTS:-0}" = 1 ] || return 0
    : > "$file.at"
    while [ -e "$file" ]; do sleep 0.05; done
}

# A signal handler must END this invocation. A cleanup-only handler returns, and bash
# resumes -- which would let finalization and promotion continue after the interruption.
# The lock needs no releasing here: the EXIT trap ends the holder, and if anything went
# wrong with that the lease ends it within a second anyway.
on_signal() {  # on_signal <NAME> <number>
    INTERRUPTED=1
    say "SIGNAL $1"
    exit $((128 + $2))
}

# Checked before every protected step, so nothing runs after a handler has fired even if
# some future caller swallows the exit.
not_interrupted() {
    [ "$INTERRUPTED" -eq 0 ] && return 0
    echo "refusing: a signal handler has fired; this invocation does not continue" >&2
    return 1
}

# published_target: what `final` resolves to under this arm root | 1 there is no
# `final` | 2 there is one and nobody can say what it publishes. `readlink -f` answered
# "nothing" for the last two alike (close review 14).
published_target() {
    probe_link_target "$ARM_ROOT/final"
}

# already_published <canonical>: this attempt is the published one and it completed.
# 0 yes | 1 no | 2 unknown -- and unknown is never "no".
already_published() {
    local target="" status=0
    target="$(published_target)" || status=$?
    [ "$status" -ne "$READER_UNKNOWN" ] || return "$READER_UNKNOWN"
    [ "$status" -eq 0 ] || return 1
    [ "$target" = "$1" ] || return 1
    file_present "$1/completion.json"
}

# check_final_conflict <canonical>: a DIFFERENT existing `final` is never replaced by a
# recovery unless the operator asks for it, and the replaced target is logged first.
check_final_conflict() {
    local target="" status=0
    target="$(published_target)" || status=$?
    if [ "$status" -eq "$READER_UNKNOWN" ]; then
        echo "refusing: $ARM_ROOT/final exists and cannot be resolved, so what it" \
             "publishes is unknown; nothing replaces a publication nobody can read" >&2
        return "$READER_UNKNOWN"
    fi
    [ "$status" -eq 0 ] || return 0
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
    local canonical="$1" log="$2" why="" target="" receipt=0 status=0
    file_present "$canonical/completion.json" || receipt=$?
    target="$(published_target)" || status=$?
    if [ "$receipt" -eq "$READER_UNKNOWN" ] || [ "$status" -eq "$READER_UNKNOWN" ]; then
        echo "refusing: whether $canonical completed or is published cannot be" \
             "inspected; an attempt nobody can read is not an attempt to abort" >&2
        return "$READER_UNKNOWN"
    fi
    [ "$receipt" -ne 0 ] || why="it has completed"
    [ "$status" -ne 0 ] || [ "$target" != "$canonical" ] || why="it is published as final"
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
INTERRUPTED="${INTERRUPTED:-0}"
REPLACE_FINAL="${REPLACE_FINAL:-0}"
RESOLVE_UNREGISTERED="${RESOLVE_UNREGISTERED:-}"
DRY="${DRY:-0}"
ARM="${ARM:-H}"

if [ "${EXP11_LAUNCH_LIB:-0}" = 1 ]; then return 0; fi

MODE="${1:-}"
shift || true
case "$MODE" in smoke|probe|full|finalize) ;; *) usage ;; esac
GPU=""; COMMIT=""; ATTEMPT=""; LOG=""; CHILD_EXIT=""; DRY=0; ARM=H
RESOLVE_UNREGISTERED=""
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
        --replace-final) REPLACE_FINAL=1; shift ;;
        --resolve-unregistered) RESOLVE_UNREGISTERED="${2:-}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) usage ;;
    esac
done
[ -n "$GPU" ] && [ -n "$COMMIT" ] || usage
apply_root_override || exit 2
apply_reader_override || exit 2
arm_of "$ARM" || exit 2
check_ceiling || exit 2
[ "$MODE" != finalize ] || [ -n "$RESOLVE_UNREGISTERED" ] \
    || { [ -n "$ATTEMPT" ] && [ -n "$LOG" ] && [ -n "$CHILD_EXIT" ]; } || usage

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
    scan_barrier
    # From before the child launch through promotion: a recovery on this arm is refused
    # while this run owns it. The arm-wide scan runs under the lock, never before it.
    hold_arm_lock full || exit 2
    require_quiet_arm || exit 2
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
    mark_launching "$attempt"   # before the fork: the child may exist before child.pid
    run_child "$attempt" "$log" "${child[@]}"
    clear_launching "$attempt"  # only if the trainer left a train.exit
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
    scan_barrier
    hold_arm_lock finalize || exit 2   # alias resolution, validation and publication
    if [ -n "$RESOLVE_UNREGISTERED" ]; then
        resolve_unregistered "$RESOLVE_UNREGISTERED" || exit 2
        exit 0
    fi
    require_quiet_arm || exit 2        # arm-wide, under the lock: nothing may be running
    preflight   # recovery is gated by the same reviewed commit, clean tree and pid checks
    check_attempt "$ATTEMPT" || exit 2
    # Everything below uses the canonical directory, never the path as it was typed.
    PUBLISHED=0
    already_published "$CANONICAL_ATTEMPT" || PUBLISHED=$?
    if [ "$PUBLISHED" -eq "$READER_UNKNOWN" ]; then
        echo "refusing: whether $CANONICAL_ATTEMPT is already published cannot be" \
             "inspected; this recovery decides nothing" >&2
        exit 2
    fi
    if [ "$PUBLISHED" -eq 0 ]; then
        say "SKIP $CANONICAL_ATTEMPT is already published as final and has completed"
        say "ALREADY PUBLISHED $ARM_ROOT/final -> $(basename -- "$CANONICAL_ATTEMPT")"
        exit 0
    fi
    check_final_conflict "$CANONICAL_ATTEMPT" || exit 2
    if [ "$DRY" -eq 1 ]; then abort_recovery "$CANONICAL_ATTEMPT" "$LOG" || exit 2; fi
    not_interrupted || exit 2
    if ! finalize "$CANONICAL_ATTEMPT" "$LOG" "$CHILD_EXIT" exp11_train; then
        abort_recovery "$CANONICAL_ATTEMPT" "$LOG"
        exit 2
    fi
    not_interrupted || exit 2
    promote "$(basename -- "$CANONICAL_ATTEMPT")"
    ;;
esac
