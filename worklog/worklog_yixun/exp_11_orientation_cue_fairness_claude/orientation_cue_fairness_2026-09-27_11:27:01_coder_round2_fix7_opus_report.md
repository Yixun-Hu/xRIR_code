# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 7**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `94efbc2`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 11:2x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close6_review.md` via `coder_prompts/round2_fix7_opus_prompt.md`

**Signals:** no process outside this cycle's own helpers was signalled. The holder tests end
their own `Popen` handles; the lifecycle regressions end nothing — they release an orphaned
sink by opening and closing its FIFO, which is what the reviewer did.

## Commits

| SHA | Subject |
|---|---|
| `1dbf556` | exp11 fix7 (RED): no orphan of the launcher may keep an arm locked |
| `1794086` | exp11 fix7 (GREEN): a holder process owns the lock, leased to the launcher's life |
| `cdd6379` | exp11 fix7 (RED): the lock holder belongs in the launcher's approved closure |
| `eae0bec` | exp11 fix7 (GREEN): launch_sh binds the lock holder |
| `3db254a` | exp11 fix7: restore the behaviour A1 still requires |
| `4da8b55` | exp11 fix7 (RED): a holder that never started is not another invocation |
| `e351fec` | exp11 fix7 (GREEN): only `refused` means another invocation holds the arm |
| `091b3b9` | exp11 fix7 (RED): the launcher must signal nothing but its own holder |
| `63c1bed` | exp11 fix7 (GREEN): release signals the holder only while it is still our child |

## Blocker 1 [P1] — the lock was inherited, so an orphan could keep an arm locked

`exec {LOCK_FD}>>"$LOCK_FILE"` gave the launcher shell the lock on one of **its own**
descriptors, and the frozen lifecycle hands every open descriptor to the background log
sink (`tools/exp06_launch.sh`: `cat >> "$log" < "$pipe" &`) and to the detached trainer
(`nohup setsid "$@" > "$pipe" 2>&1 &`). Interrupted after the sink starts, the launcher
exits 143 and the sink is left blocked opening a FIFO whose writer will never arrive —
holding the arm's lock forever, with no trainer running. The header's "released when the
launcher exits" promise was simply false: release came when the **last inherited
descriptor** closed.

**The lock is no longer the launcher's to hold.** `tools/exp11_lock_holder.py` (new) is a
process whose only job is to hold it:

* opens the lock file `O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC` (never truncates, writes no
  bytes) and sets `FD_CLOEXEC` explicitly; it execs nothing, so the lock cannot escape;
* `fcntl.flock(LOCK_EX | LOCK_NB)`: on failure prints `refused` and exits 1; on success
  prints `acquired <pid>` and then does nothing but watch its parent;
* **the lease** — `while os.getppid() == <launcher pid>: sleep 0.5`. When the launcher
  exits or dies for *any* reason (normal exit, signal, `SIGKILL`) the holder is reparented,
  sees the changed parent and ends, so the kernel drops the lock within ~1 s;
* `SIGTERM`/`SIGINT` end it immediately, which is how the ordinary path releases.

`hold_arm_lock` starts it as `exec {out}< <(exec "$PYTHON" tools/exp11_lock_holder.py …
$$)` — the inner `exec` matters: without it bash interposes a subshell and the holder's
parent is not the launcher, so the lease would fire at once. The launcher reads the single
verdict line with a bounded `read -t 30`, **closes the read end immediately**, and keeps
only `HOLDER_PID`. From that moment the launcher owns no descriptor connected to the lock,
so nothing the lifecycle spawns can inherit one. `drop_arm_lock` (the new `EXIT` trap, and
the end of every protected section) terminates the launcher's own holder so release is
immediate on the normal path; every other path is covered by the lease itself.

Header and usage now say what is true: *the lock is held by a holder process leased to this
launcher's life; it vanishes within ~1 s of the launcher's exit or death and is never
inherited by the trainer or the log sink.*

### The regression, at the reviewer's exact boundary

`test_no_orphan_of_the_launcher_keeps_the_arm_locked` replays the frozen `run_child`
prologue line for line — and **asserts those two lines are still byte-identical to
`tools/exp06_launch.sh`**, so the replay cannot drift from production — then evaluates the
registered TERM body. It is parameterised over the reviewer's two cases:

* `[False]` — the sink has started, the child has not: the launcher exits 143 and the arm
  must be free;
* `[True]` — an ordinary detached child is running: the launcher exits 143 and the arm must
  be free *while that child is still alive*.

Both failed before the fix with `an orphan of the launcher still holds the arm lock`, and
both pass now. The launcher's output goes to files rather than pipes, because a captured
pipe only reaches EOF once the orphans end — the very thing under test.

`tests/test_exp11_lock_holder.py` (new, 7 tests) covers the holder itself: the
`acquired <pid>` / `refused` protocol and its exit codes, the ppid lease (a stand-in
launcher exits without releasing anything; the lock frees), `SIGTERM` from its parent, that
it writes no bytes into the file it locks, that the descriptor is close-on-exec, and that a
malformed call is refused.

### What the lock deliberately no longer outlives

The lock now ends with the *launcher*, not with the trainer it detached. That is the
behaviour the review asked for — an orphaned sink must never retain it — and it means a
trainer whose launcher died no longer holds the arm. What still guards that case is the
check that was always meant to: `preflight` refuses a launch while a live pid file exists
under either arm root. The lock serialises **publication**; the pid file is what says a run
is in progress.

## Blocker 2 [P2] — the behaviour A1 still requires is back

Six behaviours went out with the hand-rolled lock's own tests although none of them is
about the lock mechanism. Restored with fixtures that take the real lock through the real
holder: incomplete-orphan recovery aborts; same-target recovery is idempotent; a different
existing `final` is a conflict, and `--replace-final` resolves it; the TERM handler
terminates (143) and the INT handler terminates (130) rather than resuming; the
`not_interrupted` guard refuses. Plus one the new design earns:
`test_the_handler_releases_the_arm_it_was_publishing` — after the handler has ended the
launcher, the arm is free again.

## The lock holder is part of the launcher's approved closure

`tools/exp11_profiles.py` now binds `launch_sh` to
`('tools/exp11_launch.sh', 'tools/exp06_launch.sh', 'tools/exp11_lock_holder.py')`
(`cdd6379` RED → `eae0bec` GREEN). Without it the one file that decides mutual exclusion
could change without moving any approved digest.

## Line counts

| commit | production | tests |
|---|---:|---:|
| `1dbf556` (RED) | — | +219 |
| `1794086` (GREEN) | +105 / −22 = **127 changed** | +6 / −1 |
| `cdd6379` (RED) | — | +4 / −2 |
| `eae0bec` (GREEN) | +2 / −1 = 3 changed | — |
| `3db254a` | — | +92 |
| `4da8b55` (RED) | — | +17 |
| `e351fec` (GREEN) | +7 / −1 | — |
| `091b3b9` (RED) | — | +21 |
| `63c1bed` (GREEN) | +9 / −3 | — |
| range `94efbc2..HEAD` | +120 / −23 = **143 changed** | +357 / −3 |

Every production commit is well under 200 changed lines.

## Two more things this cycle's own re-reading caught

* **A refusal has to name the real obstacle.** Every non-`acquired` verdict, EOF included,
  was reported as "held by another invocation publishing this arm". A lock path that
  cannot be opened at all is a different failure, and sending a reader to look for a
  publication that is not happening is the kind of message that costs an afternoon. Only
  `refused` now says that; anything else says the holder did not start and that nothing
  was published.
* **A recorded pid is not a licence to signal it.** `HOLDER_PID` comes out of the holder's
  own announcement; if that holder has already died, the number belongs to whoever the
  kernel gave it to next. `drop_arm_lock` now reads `/proc/<pid>/stat` and sends `SIGTERM`
  only while the process is still this shell's child. The regression starts a bystander
  and requires it to survive a release that names its pid — it did not, before the fix.

## Report corrections carried over (fix 6's table was wrong)

Close review 6 recomputed the per-commit numbers; fix 6's report understated them.

| commit | production | tests |
|---|---:|---:|
| `811de75` | — | +92 / −386 |
| `f80574a` | +44 / −235 = **279 changed lines — an EXCEPTION to the 200-line bound** | +7 / −3 |
| `04c703e` | — | +17 |
| `531db17` | −1 | — |
| `76d82f3` | — | +16 |
| `94efbc2` | +2 / −2 | — |
| entire committed range | +43 / −235 | +124 / −381 |

`f80574a` changes **279 production lines**. Fix 6's claim that "no production commit in
this cycle approaches 200 changed lines" was wrong: the launcher *shrank* by 191 lines, but
the bound counts insertions **plus** deletions and this commit exceeds it. Recorded here as
the exception it is. Committed record changes in that range total **+509/−2**.

The earlier carried-over corrections stand: `519b48f` alone is +149/−31 = 180 changed
lines (under 200); `4f72ac2..519b48f` is +306/−27; `c8b2ff0`'s remaining 333 insertions are
331 record lines plus 2 golden lines.

## Digests

Taken at `63c1bed`, the cycle's last production commit.

**exp_06** — exactly **one** key differs from the record re-filled at `9f98bbb`, the only
one exp_11 is allowed to move:

| key | approved at 9f98bbb | fix 6 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `deff6d49d8e9…` | `2443b9312798…` |

The other nineteen exp_06 keys are unchanged. `tools/exp06_summarize_haa.py` imports
`tools/exp11_profiles.py`, which this cycle edited, so its value moved again within the
same single allowed key.

**exp_11** — all eight keys, and which moved against fix 6's tip `94efbc2`:

| key | digest | vs `94efbc2` |
|---|---|---|
| `train` | `fb955253b97b88be807c6ec98349e4eb3597fe85fbeba2ec5397d30d30f167ad` | **moved** |
| `finalize` | `08c4d41a25bdc731f23c01b7dd02e904055586908ed541aeccbf2fa54e406717` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `6dc160f805af0991371e754ffd400ac4a2235b310db76b1d7c6a3ef426f0868d` | **moved** |
| `smoke` | `7dea6b3661f639c5ed86755b9119b7fa5310729fa54b8e84be338efc91292938` | **moved** |
| `summarize_haa` | `2443b9312798e286f1dfa4dabd721040689bdbdf914c26a9112e9c9cb2d96bcc` | **moved** |

**Five keys moved, and why.** `launch_sh` moved for both of this cycle's reasons: the
launcher changed and the holder joined its file list. The other four moved for one reason
only — `tools/exp11_profiles.py` is in the import closure of `tools.exp11_train`,
`tools.exp11_finalize`, `tools.exp11_smoke` and `tools.exp06_summarize_haa`, so binding the
holder into `CODE_SPECS` moved every key that reads the approvals definition. That is the
blast radius of changing the approvals definition itself, and it is honest: a producer that
enforces a different requirement matrix is a different producer. No exp_11 producer's
*behaviour* changed. `haa_finetune`, `haa_eval` and `haa_pipeline_sh` do not import the
profiles module and are untouched.

exp_11's approvals record remains all-null and every exp_11 producer refuses today; these
values are for the eventual reviewed merge.

## Test results

* **Targeted set** — all fourteen `tests/test_exp11_*.py` (the new
  `tests/test_exp11_lock_holder.py` included) plus `tests/test_exp06_summarize_haa.py` and
  `tests/test_exp09_sim_eval_closures.py`: **371 passed** in 507 s at the final tree `63c1bed`
  (`tests/test_exp11_shell.py` alone: **65**, `tests/test_exp11_lock_holder.py`: **7**).
  Fix 6 ended at 353; the restored behaviours, the lifecycle regressions, the holder unit
  tests and the two late fixes account for the 18.
* `bash -n` on both shells, `py_compile` on every changed module and test, and
  `git diff --check 94efbc2..HEAD`: clean. `tools/exp06_launch.sh` is byte-identical to
  `94efbc2`, as are every other frozen file.
* **Full CPU suite** — run detached at `814fbec` (the same tree as the last code
  commit `63c1bed`) to
  `orientation_cue_fairness_2026-09-27_12:05_suite_full_cpu_814fbec.log` in this record.
  FIX7_SUITE_RESULT
  (An earlier detached run at `ad50998` was stopped when the two late fixes landed; its
  log is the truncated `…_11:45_suite_full_cpu_ad50998.log` and it stands for nothing.)
