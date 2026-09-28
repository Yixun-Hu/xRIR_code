# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 14**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `f81a9f7`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 22:2x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close13_review.md` via `coder_prompts/round2_fix14_opus_prompt.md`

**Signals:** the only processes signalled are helpers these tests started — `sleep` stubs
and `flock` holders — through their own `Popen` handles. `kill -0` remains a probe.

## Commits

| SHA | Subject |
|---|---|
| `f0f56df` | exp11 fix14 (RED): an arm root nobody can list reads as an arm with nothing in it |
| `bcd981c` | exp11 fix14 (GREEN): an arm is quiet only once it has been enumerated |
| `b9e5877` | exp11 fix14 (RED): the verdict read is unbounded |
| `a2b006a` | exp11 fix14 (GREEN): the size answers before anything reads the verdict |

## The blocker [P1] — an arm is quiet only once somebody has looked at it

A directory with mode 0300 is searchable and writable but **not listable**. Every known
pathname under it still works — both lock files open, `stat` answers, the holder process
starts, the marker reads — while the glob `"$root"/attempt_*` silently expands to
nothing. The unmatched pattern then failed `-d`, the loop ran zero times, and `scan_arm`
returned "quiet". A resolution tombstoned and renamed the attempt of a registered,
running trainer. RED `f0f56df` reproduces it: `exit 0`, `RESOLVED`, tombstone, rename,
with the trainer's `train.pid` alive throughout.

Enumeration is now something that has to **succeed**:

```bash
list_attempts() {
    local root="$1" out="$2"
    [ -d "$root" ] || return 1
    [ -r "$root" ] && [ -x "$root" ] || return 1     # listing needs read AND search
    find "$root" -mindepth 1 -maxdepth 1 -name 'attempt_*' -print0 > "$out" 2>/dev/null
}
```

`scan_arm` walks that listing (NUL-separated, so an attempt name may hold anything but a
NUL) instead of a glob whose emptiness was indistinguishable from an empty arm:

* the listing fails → **unknown**, `SCAN_REASON` = "liveness unknown: the arm root cannot
  be enumerated (`<root>`); an attempt nobody can see is not an attempt that is not
  there", exit 2, nothing written;
* the listing succeeds and holds no `attempt_*` → that really is an answer, and the arm
  is quiet;
* inside the loop, `[ -d "$attempt" ] || continue` is gone: `-d` says "no" both for a
  name that is not a directory and for one nobody was allowed to look at. The type now
  comes from a `stat` that must succeed (mode bits, as in `pid_record`), and a `stat`
  that fails is unknown.

`require_quiet_arm` and `resolve_unregistered` propagate that status as they already did
for a failed reader, so both modes' scans and every resolution refuse with exit 2.
`clear_launching` withdraws a marker only while the arm it belongs to can still be
enumerated: the marker is protection, and protection is not withdrawn on an
understanding we do not have.

A successful `stat` of a name we already knew is never evidence that we saw the
directory — that is the whole shape of this defect, and it is stated in the header of
`list_attempts`.

## Nonblocking — the verdict read is bounded

A verdict is eighteen bytes at most (`record ` + ten digits + a newline). The shell now
refuses anything larger on the file's **size**, before `cmp` and before any expansion,
and the digit extraction reads a bounded prefix (`head -c 64 | tr -cd '0-9'`), so a rogue
reader's megabytes are never read, compared, or materialised as a shell variable. The
comparison still demands exact bytes, so nothing over 64 bytes could have matched anyway.
RED `b9e5877` pins the bound and drives 2 MiB of `record 111…` through the resolution
path (unknown, nothing written).

## The regressions

All non-root (they skip when the euid is 0, since root ignores the bits they turn off),
all with the real sourced library, the real locks and a real live registered trainer.

| test | what it establishes |
|---|---|
| `test_an_arm_root_that_cannot_be_listed_is_never_quiet` (0300 and 0100 × full and finalize) | the scan each mode runs refuses with exit 2 and "cannot be enumerated"; no `PROMOTE` |
| `test_a_resolution_under_an_unlistable_root_retires_nothing` (0300, 0100) | exit 2, no `RESOLVED`, no `TOMBSTONE`, and afterwards: the attempt, its marker and its `train.pid` (still the live stub's) all intact, no `.resolved`, no `_ABORTED_unregistered` |
| `test_an_unlistable_root_keeps_the_marker` | `clear_launching` with a complete `child.pid` **and** a `train.exit` keeps the marker while the arm cannot be enumerated |
| `test_an_empty_readable_root_is_quiet` | the other side: a listing that succeeded and found nothing leaves the arm quiet (the fix may not simply refuse everything) |
| `test_an_entry_the_scan_cannot_stat_is_unknown` | the listing succeeds, one entry is a symlink into a `chmod 000` directory: unknown, not "not a directory" |
| `test_a_flood_of_verdict_bytes_is_unknown_and_never_materialised` | the bound is in the launcher, and 2 MiB of `record 111…` on `train.pid` is unknown with nothing written |

The lock files are created before the root's mode is changed, so the publication lock
still acquires exactly as the reviewer observed — the refusal comes from the scan, not
from a lock that could not be taken.

## Line counts

| commit | production | tests |
|---|---:|---:|
| `f0f56df` (RED) | — | +122 |
| `bcd981c` (GREEN) | `exp11_launch.sh` +48/−3 | — |
| `b9e5877` (RED) | — | +21 |
| `a2b006a` (GREEN) | `exp11_launch.sh` +8/−2 | — |
| range `f81a9f7..HEAD` | **+56 / −5 = 61 changed** | +143 |

Every commit is under 200 changed lines; no exception this cycle.

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`
(`git diff 9f98bbb..HEAD` reports nothing for either). No `model/*.py`, no `tools/exp06_*`
file, no `tools/exp10_*`, no exp_03-pinned file was touched. `bash -n` passes on both
shells, `py_compile` on every exp_11 module, and `git diff --check f81a9f7..HEAD` is
silent.

## A note on the previous cycle's recorded suite

The fix-13 suite log was produced at `5593d69`, that cycle's last production commit; the
log file itself was committed at `bc0b363` (a record-only commit above it, as were
`f81a9f7`'s). The production bytes the suite ran against are `5593d69`'s.

## Digests

Taken at `a2b006a`, the cycle's last production commit. This cycle touched only
`tools/exp11_launch.sh`, so **exactly one** key moves.

**exp_06** — one key still differs from the record re-filled at `9f98bbb`, and it is
unchanged since fix 13 (no Python closure moved this cycle):

| key | approved at `9f98bbb` | fix 13 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `40626dd26bde…` | `40626dd26bde…` |

The other 19 exp_06 keys are identical to the approved record.

**exp_11** — all eight keys, against fix 13's tip `f81a9f7`:

| key | digest | vs `f81a9f7` |
|---|---|---|
| `train` | `0f518d3f79cb6b992d0ff6057888f93c0bae0d9520eeb8c0522d1ab40d11b9c2` | unchanged |
| `finalize` | `256349e7dfa55880ee37eb6c97192ade560af1aff322941962e0b861c860be73` | unchanged |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `3792093141e393d3fb134a41971eaa7ad980311add80463f737b9d8fa51d4afc` | **moved** |
| `smoke` | `9940aadc525cbb01ceb5925b38056657630145f2998690c8d077399a1608ccbf` | unchanged |
| `summarize_haa` | `40626dd26bdeac80f1cbfc01d243e43d5ab0017d59bb9129c11b86771ef6495d` | unchanged |

Only `launch_sh`, through its own shell bytes — as the prompt predicted. exp_11's
approvals record remains all-null and every producer refuses today.
