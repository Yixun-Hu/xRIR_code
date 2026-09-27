# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 6**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `86b1a46`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 09:2x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close5_review.md` and the Planner's design change, plan §11 amendment **A1**, via `coder_prompts/round2_fix6_opus_prompt.md`

**Signals:** no process outside this cycle's own test helpers was signalled. The mutual-exclusion
tests start a lock-holding subprocess and end it with `Popen.terminate()` on that handle —
a process the test itself created — which is what the prompt permits.

## Commits

| SHA | Subject |
|---|---|
| `811de75` | exp11 fix6 (RED): the per-arm lock must be the kernel's, not a hand-rolled directory |
| `f80574a` | exp11 fix6 (GREEN): the per-arm lock is an flock on a file, held by the kernel |

## The change — the hand-rolled lock is gone; the kernel holds the lock

Five cycles of `mkdir`-lock repair each closed the interleaving the reviewer had
demonstrated and exposed the next. The Planner's amendment replaces the mechanism rather
than repairing it again, and that is what this cycle does.

`hold_arm_lock <mode>` opens the per-arm lock **file** `<arm-root>/.publish.lock` on a
descriptor this process owns (`exec {LOCK_FD}>>"$LOCK_FILE"`) and takes `flock -n` on it
for the **whole** protected section:

* **recovery** — alias resolution → attempt validation → finalization → promotion;
* **full** — from **before the child launch** through promotion, so a recovery on that arm
  is refused while a training run still owns it.

`-n` refuses immediately, never waits, and says what a reader needs to know: the lock is
held by another invocation and *the kernel releases it when that process ends, so there is
nothing to clear by hand*. That is the whole ownership story — **a lock vanishes with its
holder** — now stated as an invariant in the launcher header. There is no cleanup code, no
owner record, no generation, and no stale-lock concept left to get wrong, on any exit path
including a signal.

**Deleted:** `take_lock`, `release_lock`, `break_lock`, `take_breaking`,
`release_breaking`, the `owner` file, the nonce/generation, `LOCK_GRACE_S`, the
`.breaking` meta-lock, `--break-lock` (and its usage line), `EXP11_LOCK_BARRIER`,
`EXP11_ACQUIRE_BARRIER`, and every helper that parsed an owner file
(`lock_nonce_of`, `lock_pid_of`, `lock_owner_line`, `lock_inode_of`,
`lock_older_than_grace`, `lock_is_live`, `lock_barrier`, `lock_acquire_barrier`,
`lock_wait_for`). `on_signal` keeps `INTERRUPTED=1`, the `SIGNAL <name>` line and
`exit 128+n`; it no longer releases anything because it no longer needs to.

**Kept unchanged:** canonical attempt resolution (`check_attempt`), ownership-preserving
`abort_recovery`, idempotent `already_published`, `check_final_conflict` with
`--replace-final`, `check_ceiling`, `EXP11_PRETRAIN_ROOT` honoured only with
`EXP11_TEST_ROOTS=1` **and** `--dry-run`, the terminating INT/TERM handlers and the
`not_interrupted` guards before finalization and promotion in both modes.

**Dry run** announces `LOCK <file>` and takes no lock, except under `EXP11_TEST_ROOTS=1`
where the real `flock` is taken so the tests exercise the real primitive.

## Tests

* **Mutual exclusion** — a helper subprocess (`exec 9>file; flock -n 9; …`, with the
  keep-alive child's copy of the descriptor closed so terminating the helper really drops
  the lock) holds the arm lock while a recovery and a full-mode dry run each refuse
  immediately with the held-lock message, printing neither `PROMOTE` nor `nohup setsid`.
* **Release on exit** — after the helper ends, the same recovery succeeds and promotes.
* **A refusal changes nothing** — attempt directory, completion bytes, the `final` link
  and the lock file are byte-identical before and after, with no `_ABORTED_` sibling.
* **No trace left** — the launcher source contains none of the deleted names and its usage
  no longer mentions `break-lock`.
* **Dry-run** announces the lock and creates no lock file under the real roots.
* The **ordinary recovery behaviours** (preservation of a published or completed attempt,
  idempotency, the `final` conflict with and without `--replace-final`, the orphan abort)
  and the **registered-handler replays** (`TERM` → 143, `INT` → 130, no continuation) all
  still pass, now with no lock bookkeeping in them at all.

## Line counts (this cycle, honestly)

| commit | production | tests | record |
|---|---|---|---|
| RED | — | −267 removed, +101 added | — |
| GREEN | `tools/exp11_launch.sh` **44/−235** (191 lines *smaller*) | +**net** as above | 2 golden lines |
| range `86b1a46..HEAD` | 44/−235 launcher | 96/−386 tests | 412 record lines (prompt 11, copied review 397, plan 4, review prompt 3) |

No production commit in this cycle approaches 200 changed lines; the launcher shrank.

## Report corrections carried over

| item | corrected figure |
|---|---|
| `519b48f` (fix 5 GREEN) alone | **149 insertions / −31 = 180 changed lines**, i.e. **under** 200 — fix 5's report wrongly recorded it as an exception |
| `4f72ac2..519b48f` (fix 5 RED + GREEN) | 306/−27 combined |
| `c8b2ff0` remaining 333 insertions | **331 record lines + 2 golden lines** |
