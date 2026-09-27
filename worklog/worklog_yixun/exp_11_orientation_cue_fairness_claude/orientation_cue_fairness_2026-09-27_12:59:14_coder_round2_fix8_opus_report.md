# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 8**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `e19bb44`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 12:5x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close7_review.md` via `coder_prompts/round2_fix8_opus_prompt.md`

**Signals:** nothing outside this cycle's own helpers was signalled. The stubs are ended
through their own `Popen` handles or left to exit on their own; `kill -0` appears in the
launcher and in one test helper as a **probe** — signal 0 delivers nothing and only asks
the kernel whether a pid still exists.

## Commits

| SHA | Subject |
|---|---|
| `edd3ca9` | exp11 fix8 (RED): the liveness scan must run under the lock, arm-wide |
| `56e7493` | exp11 fix8 (GREEN): the arm-wide liveness scan runs under the lock |
| `dfa41e8` | exp11 fix8 (RED): a launch nothing recorded must still block its arm |
| `1f4fbf3` | exp11 fix8 (GREEN): the launcher declares its intent before the fork |
| `46f45ee` | exp11 fix8: the resolution answers the marker, and says so on disk |
| `9a0d953` | exp11 fix8 (RED): the trainer must name itself before anything can fail |
| `23ca66b` | exp11 fix8 (GREEN): the trainer writes train.pid first and train.exit last |
| `d255d28` | exp11 fix8: hygiene — no repo writes from tests, logs exempt, counts corrected |

## What the two blockers had in common

Fix 7 divided the problem as *"the lock says one publication at a time; the live-pid
preflight says a run is in progress."* Close review 7 accepted the lock and broke the
other half twice: the liveness answer was **taken at the wrong moment** (blocker 1) and
**asked of the wrong set of facts** (blocker 2). Both are now answered under the lock,
over the whole arm.

## Blocker 1 [P1] — the scan now runs UNDER the lock, arm-wide

`preflight` ran *before* `hold_arm_lock` in both modes, so its answer could go stale in
the window between: a recovery could pass its scan, pause, and promote an older attempt
after another launcher had started a trainer of the same arm and died. And
`tools/exp11_finalize.py` only ever looked at the directory being recovered, so an
independently valid older attempt passed on its own merits.

Both modes now do, in this order: **acquire the lock → scan the arm → everything else.**

```
hold_arm_lock <mode> || exit 2
require_quiet_arm   || exit 2
preflight
```

`scan_arm` walks every `attempt_*` directory of the arm root and refuses on a live
`launch.pid`, a live `child.pid`, a live `train.pid`, or a `launching` marker with no
`child.pid`. Under the lock the answer cannot go stale, because no other launcher can
start a trainer of this arm while we hold it — full mode takes the same lock before its
child exists.

**Regressions.** One is direct: a live `train.pid` in `attempt_…111111` must refuse a
recovery of `attempt_…000000` (it promoted, before). The other is the reviewer's
interleaving, forced open deterministically: `EXP11_SCAN_BARRIER` (honoured **only**
under `EXP11_TEST_ROOTS=1`) pauses the recovery *before* it takes the lock; while it
waits, the test creates a trainer of another attempt whose launcher is already gone; the
recovery resumes, takes the lock, scans under it, and refuses. A scan taken before the
pause would have seen a quiet arm — which is why the RED run promoted.

## Blocker 2 [P1] — the window the frozen lifecycle cannot close

`run_child` forks the detached trainer and writes `child.pid` **afterwards**. A launcher
that dies in between leaves a dead `launch.pid`, no `child.pid`, and a live trainer that
no scan can see. `tools/exp06_launch.sh` is frozen, so exp_11 closes the window from both
sides instead:

**(a) The launcher declares its intent.** `mark_launching` writes `<attempt>/launching`
(launcher pid, ISO timestamp, arm) immediately before `run_child`; `clear_launching`
removes it once `child.pid` exists. A marker with no `child.pid` means *a trainer of this
attempt may be running and nothing has recorded it*, so the scan closes the **whole arm**
and names the attempt.

An operator answers it explicitly:
`finalize --arm H --gpu g --reviewed-commit <sha> --resolve-unregistered <attempt>`.
That path can only ever **abort**: an attempt with no `child.pid` has no exit receipt, so
it can never be published. It refuses unless the attempt really is unresolved (a marker,
no `child.pid`), nothing of the arm is alive, and the marker is older than
`UNRESOLVED_GRACE_S` (120 s, overridable for the tests) — and it removes the marker it
answered, because the scan reads retired directories too and a marker left behind would
close the arm for good. That last one was a real bug the test found.

**(b) The trainer names itself.** `tools/exp11_train.py` writes `<save-dir>/train.pid`
(pid, start time) as its **first act after parsing** — before admission, before the data
inventories, before anything else that can fail — and appends `train.exit <code>` on
every path it controls. `main()` is now that registration wrapper; the steps themselves
moved unchanged into `run()`. The marker in (a) covers only the sub-second before this
write; from that moment the trainer is visible to the scan on its own.

**Regressions at the reviewer's exact boundaries.** The replay of `run_child`'s prologue
(asserting the two spawn lines are still byte-identical to the frozen library) with a stub
that registers the way the trainer does:

* **TERM before anything recorded it** — the stub holds its registration behind a gate the
  test controls, so the arm is inspected in the true worst case: launcher dead, no
  `child.pid`, no `train.pid` yet. Recovery of another attempt refuses on the marker
  (it promoted, before).
* **TERM after PID recording** — `child.pid` written, trainer alive: refuses on the live
  trainer.
* Then the full resolution path: refused while the stub lives, refused inside the grace,
  aborted to `*_ABORTED_unregistered` once both are settled, and the arm is open again —
  the same recovery that was refused now promotes.

## Hygiene

* `tests/test_exp11_lock_holder.py` no longer writes `tools/.cloexec_probe.lock` into the
  repository; the probe uses `tmp_path`.
* The record's `.gitattributes` exempts `*_suite_full_cpu*.log` from the whitespace check
  (pytest's own trailing spaces in a progress line), alongside the existing rule for the
  verbatim Codex reviews. `git diff --check e19bb44..HEAD` is clean over the whole range.
* Fix 7's three per-commit counts corrected in that report: `4da8b55` tests **+16**,
  `091b3b9` tests **+20**, `63c1bed` production **+10/−3**.

## Line counts

| commit | production | tests | record |
|---|---:|---:|---:|
| `edd3ca9` (RED) | — | +78 | — |
| `56e7493` (GREEN) | `exp11_launch.sh` +67/−3 | — | 2 golden lines |
| `dfa41e8` (RED) | — | +78 | — |
| `1f4fbf3` (GREEN) | `exp11_launch.sh` +56/−1 | — | — |
| `46f45ee` | `exp11_launch.sh` +6/−1 | +89 | — |
| `9a0d953` (RED) | — | +46 | — |
| `23ca66b` (GREEN) | `exp11_train.py` +45/−2 | — | — |
| `d255d28` | — | +3/−6 | +338/−3 (copied review 313) |
| range `e19bb44..HEAD` | **+173 / −6 = 179 changed** | +294/−6 | +342/−5 |

Every commit is under 200 changed lines; no exception to record this cycle.

## Digests

Taken at `d255d28`, the cycle's tip (its last production commit is `23ca66b`; `d255d28`
touches only tests and the record).

**exp_06** — exactly **one** key differs from the record re-filled at `9f98bbb`, the only
one exp_11 is allowed to move:

| key | approved at 9f98bbb | fix 7 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `2443b9312798…` | `a5db73dc4453…` |

The other nineteen exp_06 keys are unchanged.

**exp_11** — all eight keys, and which moved against fix 7's tip `e19bb44`:

| key | digest | vs `e19bb44` |
|---|---|---|
| `train` | `79f37491d8c967e899545f5ca62bc0e5ab6987a133eb9d3d9db2ec569198102d` | **moved** |
| `finalize` | `d3b641c98536fcf734ac6d1010410d169b2404ca804ac4472c4832cbb7fcd42b` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `2be8b61f2dc714ccfc5b244a54d36c27049c5c919014cfc650a860a248ad011f` | **moved** |
| `smoke` | `db4b2d25fbb51d2cb85f091a429f7d589d41a467103d70a0812d01f27e0f2620` | **moved** |
| `summarize_haa` | `a5db73dc4453ce82f94fa197dabaf1e93f6d9e148db70da30c8b453b4a12894c` | **moved** |

**Five keys moved, for exactly two file changes.** `launch_sh` moved through its own shell
bytes. The other four moved through `tools/exp11_train.py`, which is in the source closure
of `train`, `finalize`, `smoke` and `summarize_haa` — verified directly with
`exp11_profiles._paths_of`. `haa_finetune`, `haa_eval` and `haa_pipeline_sh` contain
neither file and are untouched. The same five moved in fix 7, then through
`exp11_profiles.py`; the set is a property of the closure graph, not of these changes.

exp_11's approvals record remains all-null and every exp_11 producer refuses today.

## Test results

* **Targeted set** — all fourteen `tests/test_exp11_*.py` plus
  `tests/test_exp06_summarize_haa.py` and `tests/test_exp09_sim_eval_closures.py`:
  **384 passed** in 556 s
  (`tests/test_exp11_shell.py` alone: **72**, `tests/test_exp11_train.py`: **15**,
  `tests/test_exp11_lock_holder.py`: **7**). Fix 7 ended at 371; the 13 new passes are
  this cycle's regressions. (Corrected in fix 9 from close review 8: the shell file
  collects **72**, not 78.)
* `bash -n` on both shells, `py_compile` on every changed module and test, and
  `git diff --check e19bb44..HEAD`: clean. `tools/exp06_launch.sh` and
  `tools/exp06_finalize.py` are byte-identical to `e19bb44`.
* **Full CPU suite** — run detached at `5fcb17c` (the same tree as the last code commit
  `d255d28`; `5fcb17c` adds only this report) to
  `orientation_cue_fairness_2026-09-27_13:15_suite_full_cpu_5fcb17c.log` in this record.
  **1 failed, 3714 passed, 55 skipped** in 3474 s (57:53). The one
  failure is the standing, expected
  `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`:
  `code.summarize_haa` has moved and the record is re-filled at the reviewed merge, as
  round 1 did with `9f98bbb`, not on this branch. Fix 7 was 3701 passed; the 13 new
  passes are this cycle's regressions. Nothing else regressed.
