# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 5**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `4f72ac2`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU, **no process was ever signalled**
**Started:** 2026-09-27 08:0x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close4_review.md` (request changes, two lock defects) and `coder_prompts/round2_fix5_opus_prompt.md`

## Commits

| SHA | Subject |
|---|---|
| `ad9317e` | exp11 fix5 (RED): the acquisition race, and a cleanup path that bypasses the exit |
| `519b48f` | exp11 fix5 (GREEN): a breaking meta-lock, an acquisition grace and self-verification |

Both defects are in `tools/exp11_launch.sh`. **The prompt's design was implemented as
specified** — breaking meta-lock, initialisation grace, self-verification, rename-based
retirement re-verified under the meta-lock — rather than an equivalent of my own.

## Blocker 1 [P2] — the acquisition race

`mkdir` makes the lock pathname visible **before** the owner file exists. A breaker that
accepted an empty observed generation with no live pid retired a directory whose acquirer
was still initialising; the acquirer then wrote its owner into the breaker's directory and
both held the lock, and the first exit removed the shared pathname so a third caller could
acquire. The second schedule reached `UNRESTORED` with two live holders.

**(a) Breaking meta-lock.** `break_lock` first takes `<lock>.breaking` (atomic `mkdir`,
owner file with the same nonce discipline, released on every exit path via
`release_breaking`, which `release_lock` and therefore the `EXIT` trap call). It is held
for the **whole** break. `take_lock` refuses immediately while it exists, so the pathname
exposed by retirement is never acquirable, and a second breaker refuses on the meta-lock
instead of racing the first.

**(b) Initialisation grace.** A lock with **no owner file** is an acquisition in progress,
not a stale lock: `break_lock` refuses it unless the directory's mtime is older than
`LOCK_GRACE_S` (30 s, overridable for tests). That closes schedule 1 at its root — the
empty generation is no longer breakable while its acquirer is alive.

**(c) Self-verification.** `take_lock` records the directory's **inode** after `mkdir`,
writes the owner **atomically inside that directory** (`.owner.$$` then `mv` to `owner`,
so a write into a retired directory fails closed rather than landing in somebody else's),
and then re-checks three things: the same inode, its **own** generation in the owner file,
and no `<lock>.breaking`. On any mismatch it **refuses and removes nothing** — it never
owned a surviving lock. That closes schedule 2 from the acquirer's side.

**(d) Retirement.** Still rename-based on the observed generation
(`mv -T <lock> <lock>.broken.<observed>`), re-verified after the move under the meta-lock.
With (a) nothing can replace the directory in between, so the restore branch is documented
as unreachable in normal operation and kept as defensive code with its `UNRESTORED` line.

## Blocker 2 [P2] — a cleanup path could pre-empt the handler's exit

`now="$(lock_nonce_of "$LOCK_DIR")"` propagated `sed`'s failure when the owner file was
absent, so under `set -e` the handler exited 2 before `UNLOCK SKIPPED` and before its
`exit 128+n`.

Every owner-file reader (`lock_nonce_of`, `lock_pid_of`, `lock_owner_line`,
`lock_inode_of`) now ends in `|| true`, and `release_lock` tests for the file first:
an absent or unreadable owner is **explicit non-ownership**, logged
`UNLOCK SKIPPED … has no owner: this invocation owns nothing there`, and the lock is left
alone. No cleanup path can bypass the handler's unconditional `exit $((128 + n))`.

## Regressions (deterministic file barriers, never a signal)

`EXP11_ACQUIRE_BARRIER` joins `EXP11_LOCK_BARRIER`; both take a **path**, never a command,
and wait boundedly (10 s). The acquisition hook sits at the one point the schedules need:
after the owner file has landed and before self-verification.

* **Schedule 1** — an acquirer paused between `mkdir` and self-verification while a
  breaker runs: **exactly one holder**, an ordinary third caller is refused *while the
  holder still holds it*, and the lock is gone only after the holder's own exit.
* **Schedule 2** — a breaker that observed an empty generation while the acquirer
  completes: the breaker refuses, the acquirer remains the sole holder, and `UNRESTORED`
  never appears.
* **Grace** — a young owner-less lock refuses the breaker ("acquisition in progress");
  an old owner-less one is still breakable.
* **Meta-lock** — nothing acquires the pathname while `.breaking` exists, and a second
  breaker refuses on it.
* **Self-verification** — a lock retired under an acquirer leaves the successor's lock
  intact; the acquirer refuses and removes nothing.
* **Missing owner** — the registered `TERM` handler replayed with the owner file deleted:
  `SIGNAL TERM`, `UNLOCK SKIPPED … no owner`, **exit 143**, lock preserved, no
  `CONTINUED`. And `release_lock` alone in the same state returns cleanly and preserves
  the lock.

## Report corrections carried over

| item | corrected figure |
|---|---|
| `16b1eff` (fix 4 GREEN) | **96 insertions / −9**, launcher only — the 224 figure was RED+GREEN combined |
| `fa12f00` (fix 4 RED) | 128 test insertions |
| `a2cb3b6` | **437 record-only lines** (prompt 9, report 92, copied review 333, review prompt 3) — recorded here as a record-only commit |
| `c8b2ff0` (fix 3 GREEN) | **205 changed lines** in the launcher and its tests (116/−2 + 75/−12), not "insertions"; the commit's other 332 lines are the copied review and prompts |

This cycle's GREEN commit is **306 insertions / −27** across `tools/exp11_launch.sh` and
`tests/test_exp11_shell.py`, so it exceeds 200 changed lines and is recorded as an
exception: the meta-lock, the grace, the atomic owner write and the self-verification are
one interlocking mechanism, and any split would leave a state in which the lock is
partly protected.

## Digests

Taken at `b3de57f`; the last commit that touches production code is `519b48f` (the commit
after it is this record), so these values stand for the cycle.

**exp_06** — exactly **one** key moved against the record re-filled at `9f98bbb`; the
other nineteen are unchanged:

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
| `launch_sh` | `fbc012454f0d7962d1c58fd778baa670d1340e56cab71131ba769e95f214d4c8` |
| `smoke` | `62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e` |
| `summarize_haa` | `deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a` |

**Exactly one key moved against the fix-4 tip `4f72ac2`: `launch_sh`**
(`3cff85cc…` → `fbc01245…`), which is the whole of this cycle's production change; the
other seven are unchanged. Values for the eventual reviewed merge — exp_11's record stays
all-null and every exp_11 producer refuses today.

## Tests

* **Targeted set** — all `tests/test_exp11_*.py` plus `tests/test_exp06_summarize_haa.py`
  and `tests/test_exp09_sim_eval_closures.py`: **369 passed** in 478 s
  (`tests/test_exp11_shell.py` alone: 70). The pinned exp_06/exp_09 and exp_11 phase-1
  oracles still reproduce their payloads and rendered summaries byte for byte.
* `bash -n` on both shells, `py_compile` on every exp_11 module and test, and
  `git diff --check` over `4f72ac2..HEAD`: clean.

## Full CPU suite

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest tests -q -rf` on the frozen
tree (clean outside `worklog/`):

```
1 failed, 3699 passed, 55 skipped, 307 warnings in 3224.26s (0:53:44)
FAILED tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes
```

The single failure is the **expected exp_06 approvals re-fill** — `code.summarize_haa`
has moved and the record is stale until a reviewer re-fills it at the reviewed merge,
exactly as round 1 did with `9f98bbb`. It is the mechanism that detects the one key this
round is allowed to move, not a regression.

Progression across this round's six runs: 3 614 → 3 651 → 3 669 → 3 683 → 3 690 →
**3 699** passing, with 2 → 1 → 1 → 1 → 1 → 1 failures — all of them that same
approvals-drift detector after the first run's load-average SIGTERM flake, which has not
recurred.

## Status

Both lock defects are closed by the design the prompt specified, and the four
commit-size figures are corrected. Ready for re-review at this tip.
