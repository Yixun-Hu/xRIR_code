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
| `dbb420a` | exp11 bookkeeping: this report (record only) |
| `04c703e` | exp11 fix6 (RED): full mode must take the arm lock once, not twice |
| `531db17` | exp11 fix6 (GREEN): one acquisition per invocation, held to promotion |
| `76d82f3` | exp11 fix6 (RED): taking the lock must not write to the file it locks |
| `94efbc2` | exp11 fix6 (GREEN): acquisition opens the lock file, it does not write it |

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
* **One acquisition** — a test-roots full dry run exits 0, never prints the held-lock
  refusal, and emits exactly one `LOCK` and one `LOCKED` line (`04c703e`).
* **Acquisition writes nothing** — bytes placed in the lock file survive a successful
  recovery that takes and releases it (`76d82f3`).
* The **ordinary recovery behaviours** (preservation of a published or completed attempt,
  idempotency, the `final` conflict with and without `--replace-final`, the orphan abort)
  and the **registered-handler replays** (`TERM` → 143, `INT` → 130, no continuation) all
  still pass, now with no lock bookkeeping in them at all.

## A defect the cycle's own review caught: the lock was taken twice

Moving `hold_arm_lock full` to before the child launch left the **original** call in
place, so full mode acquired the arm lock twice. `flock(2)` treats two descriptors on the
same file as independent **even inside one process** — `exec {LOCK_FD}>>` allocates a new
descriptor and therefore a new open file description each time — so the second acquisition
refused the launcher its own lock and a real `full` run would have exited 2 before the
child started.

Nothing in the suite caught it: a dry run outside the test roots announces the lock and
takes none (both calls return early, the only visible trace being a duplicated `LOCK` line
in the two full-mode goldens, which had been regenerated *with* the duplicate), and the
one test-roots full-mode test holds the lock from a helper and expects a refusal, which it
got for the wrong reason.

`04c703e` (RED) asserts the property directly — a test-roots full dry run exits 0, prints
no "another invocation" refusal and emits exactly one `LOCK` and one `LOCKED` line — and
failed with the self-refusal. `531db17` (GREEN) deletes the leftover acquisition and drops
the duplicated golden line. **One acquisition per invocation, one descriptor, held from
before the child launch through promotion.**

## And one more, found by re-reading the acquisition: it truncated the file it locks

`hold_arm_lock` opened with `: > "$LOCK_FILE"` as a "can I create this?" check before the
`flock` attempt. `>` **truncates**, so an invocation that was about to be refused still
wrote inside the holder's critical section — and the check was redundant, because
`exec {LOCK_FD}>>"$LOCK_FILE"` creates the file itself and reports its own failure.

`76d82f3` (RED) puts bytes in the lock file, runs a recovery that succeeds, and asserts
the bytes survive; it failed with `b'' == b'not the launcher\n'`. `94efbc2` (GREEN)
deletes the truncation. **Acquisition now only opens and locks; it writes nothing.**

## Line counts (this cycle, honestly)

| commit | production | tests | record |
|---|---|---|---|
| RED | — | −267 removed, +101 added | — |
| GREEN | `tools/exp11_launch.sh` **44/−235** (191 lines *smaller*) | +**net** as above | 2 golden lines |
| RED 2 (`04c703e`) | — | +16 | — |
| GREEN 2 (`531db17`) | `tools/exp11_launch.sh` −1 | — | −2 golden lines |
| RED 3 (`76d82f3`) | — | +15 | — |
| GREEN 3 (`94efbc2`) | `tools/exp11_launch.sh` 3/−3 | — | — |
| range `86b1a46..HEAD` | `tools/exp11_launch.sh` **43/−235** | `tests/test_exp11_shell.py` **124/−381** | 509/−2 record (prompt 11, copied review 397, plan 4, review prompt 3, this report; −2 golden) |

No production commit in this cycle approaches 200 changed lines; the launcher shrank.

## Digests

Taken at `94efbc2`, the cycle's last production commit.

**exp_06** — exactly **one** key moved against the record re-filled at `9f98bbb`; the other
nineteen are unchanged:

| key | approved at 9f98bbb | now |
|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `deff6d49d8e9…` |

**exp_11** — all eight keys:

| key | digest |
|---|---|
| `train` | `d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571` |
| `finalize` | `3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e` |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` |
| `launch_sh` | `3d432c1de510e54ab1c48f19f264829bf53f907935563756b2a73f8d47cf9a80` |
| `smoke` | `62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e` |
| `summarize_haa` | `deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a` |

**Exactly one key moved against the fix-5 tip `b3de57f`: `launch_sh`**
(`fbc01245…` → `3d432c1d…`), which is the whole of this cycle's production change; the other
seven are unchanged. Values for the eventual reviewed merge — exp_11's approvals record
stays all-null and every exp_11 producer refuses today.

## Test results

* **Targeted set** — all thirteen `tests/test_exp11_*.py` plus
  `tests/test_exp06_summarize_haa.py` and `tests/test_exp09_sim_eval_closures.py`:
  **353 passed** in 457 s (`tests/test_exp11_shell.py` alone: **54**). Fix 5 reported 369
  with 70 shell tests; the arithmetic is 369 − 70 + 54 = 353, i.e. the drop is exactly the
  sixteen net hand-rolled-lock tests that no longer describe anything the launcher does.
  The pinned exp_06/exp_09 and exp_11 phase-1 oracles still reproduce their payloads and
  rendered summaries byte for byte.
* `bash -n` on both shells, `py_compile` on every exp_11 module and test, and
  `git diff --check 86b1a46..HEAD`: clean.
* **Full CPU suite** — FULL_SUITE_PLACEHOLDER

## Report corrections carried over

| item | corrected figure |
|---|---|
| `519b48f` (fix 5 GREEN) alone | **149 insertions / −31 = 180 changed lines**, i.e. **under** 200 — fix 5's report wrongly recorded it as an exception |
| `4f72ac2..519b48f` (fix 5 RED + GREEN) | 306/−27 combined |
| `c8b2ff0` remaining 333 insertions | **331 record lines + 2 golden lines** |
