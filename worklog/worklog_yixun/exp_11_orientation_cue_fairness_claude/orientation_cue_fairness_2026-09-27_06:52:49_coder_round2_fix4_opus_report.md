# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 4**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `48b7bd4`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU, **no process was ever signalled**
**Started:** 2026-09-27 06:4x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close3_review.md` (request changes, two defects in the new lock handling) and `coder_prompts/round2_fix4_opus_prompt.md`

## Commits

| SHA | Subject |
|---|---|
| `fa12f00` | exp11 fix4 (RED): signal handlers resume; locks have no generation |
| `16b1eff` | exp11 fix4 (GREEN): terminating signal handlers and lock generations |

Both defects are in `tools/exp11_launch.sh`; every ordinary recovery behaviour the
previous cycle added verified independently and is untouched here.

## Blocker 1 [P2] — signal handlers cleaned up but did not terminate

`trap release_lock EXIT INT TERM` installed a *cleanup-only* handler. Bash resumes after
such a handler returns, so a TERM arriving while the synchronous finalizer ran would
release the lock and then let promotion continue **unlocked**.

**Fix.** `on_signal <NAME> <number>` sets `INTERRUPTED=1`, logs `SIGNAL <name>`, performs
the ownership-safe release and **exits `128 + n`**. `INT` and `TERM` are registered to it
(`trap 'on_signal INT 2' INT`, `trap 'on_signal TERM 15' TERM`); the `EXIT` trap keeps the
ownership-safe cleanup. Belt and braces, `not_interrupted` guards **both protected steps
in both modes** — before finalization and before promotion, in `full` and in `finalize` —
so nothing downstream can run after a handler has fired even if some future caller were
to swallow the exit.

**Regression, without signalling anything.** The test reads the *registered* handler body
back out of `trap -p TERM`, `eval`s that body, and asserts execution never returns:
no `CONTINUED_AFTER_HANDLER`, no `PROMOTE`, exit **143**, and `SIGNAL TERM` in the log.
The same for `INT` (exit **130**). A third test drives `not_interrupted` directly in both
states.

## Blocker 2 [P2] — locks had no generation, and breaking was not serialised

`release_lock` trusted a process-local flag, so an earlier owner's exit removed its
*successor's* lock; and `break_lock` retired whatever it found rather than the lock it had
observed, so two breakers that both saw the same stale owner could each delete the other's
fresh lock.

**Fix — the lock carries a generation.** `take_lock` writes `pid <pid> nonce <nonce> …`
into the owner file, where the nonce is unique to the invocation. `release_lock` removes
the directory **only while that nonce is still the one recorded there**; otherwise it logs
`UNLOCK SKIPPED … holds the generation X, not Y` and leaves it alone. A process can no
longer delete a lock it does not own, whatever its local flag says.

**Fix — breaking is serialised on the observed generation.** `break_lock` now takes the
generation the caller *observed* and:

1. refuses outright if the recorded owner pid is **live** (`--break-lock` is for an
   interrupted publication, never for overriding a running one);
2. refuses if the lock's generation has changed since the observation;
3. atomically retires that specific lock with `mv -T <lock> <lock>.broken.<observed>`.
   This is the serialisation point: the rename fails for a second breaker once the first
   has moved the directory, and because the target is named for the *observed* generation,
   a repeat of the same observation collides with the already-retired directory instead of
   succeeding twice;
4. re-reads the retired directory's owner and, if what moved was **not** what was
   observed, puts it back (`mv -T` in reverse) and refuses — so a late breaker can never
   publish over a live generation. A put-back that itself fails is logged `UNRESTORED`
   rather than passed over.

`take_lock` reads the generation once and hands it to `break_lock`, so the observation and
the action are explicitly connected.

**Test hook.** `EXP11_LOCK_BARRIER` names a **path**, never a command: `lock_barrier`
waits (bounded, 10 s) for that file to appear at a defined point inside `break_lock`. It
lets the tests interleave two breakers deterministically without the launcher evaluating
anything the environment supplies.

**Regressions, with the real functions.** A successor generation survives the previous
owner's exit; a breaker whose observed generation is gone refuses and leaves the current
lock byte-for-byte intact; `--break-lock` refuses a live owner pid (using the test's own
pid) and leaves the lock; and two genuinely concurrent breakers — one paused at the
barrier in a subprocess, one racing past it — leave **exactly one** holder with the lock
still present.

## Process — commit-size exception

`c8b2ff0` (fix cycle 3's GREEN commit) is **524 insertions**, of which the launcher and
its tests are **205**; the remainder is the copied Codex review carried in the same
commit. Fix cycle 3's report claimed "every commit in this cycle is under 200 changed
lines" — that is wrong, and `c8b2ff0` is recorded here as an exception alongside round 2's
ten. This cycle's GREEN commit is **224 insertions / 9 deletions** across the launcher and
its tests, so it is an exception too: the lock rewrite is one coherent unit (generation,
release, break and the handlers reference one another) and splitting it would leave a
state in which the lock is half-generational.
