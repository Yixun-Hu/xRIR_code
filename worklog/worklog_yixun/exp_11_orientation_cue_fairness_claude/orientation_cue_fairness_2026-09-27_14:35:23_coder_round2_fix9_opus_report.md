# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 9**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `7abeb4d`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 14:3x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close8_review.md` via `coder_prompts/round2_fix9_opus_prompt.md`

**Signals:** nothing outside this cycle's own helpers was signalled; stubs are ended
through their own handles or left to exit. `kill -0` remains a probe, not a signal.

## Commits

| SHA | Subject |
|---|---|
| `ff610ff` | exp11 fix9 (RED): a child.pid is a registration only when it holds a pid |
| `3c32cb2` | exp11 fix9 (GREEN): registration_complete is the test, not the file's existence |
| `d268f58` | exp11 fix9 (RED): the resolution must stay inside the arm it locked and scanned |
| `bc4ad3d` | exp11 fix9 (GREEN): the resolution is confined to the arm, by canonical identity |
| `c530e32` | exp11 fix9 (RED): a delayed trainer must fail closed, not train on regardless |
| `e4bd6a3` | exp11 fix9 (GREEN): the trainer fails closed; the resolution needs a tombstone |
| `71ac3c9` | exp11 fix9 (GREEN): the resolution writes its answer before it renames |
| `b49d90b` | exp11 fix9: the whole unregistered path, end to end |
| `652c463` | exp11 bookkeeping: fix-8 count correction and close review 8 (record only) |

## What the three blockers had in common

Fix 8 built the unregistered-launch path on three implicit assumptions, and close
review 8 falsified all three: that a `child.pid` **file** is a registration, that a
marker's **age** says something about the trainer, and that a path **string** names an
attempt of this arm. Each is now an explicit, checked property.

## Blocker 1 [P1] — a registration is a pid, not a file

The shell creates `child.pid` by redirection *before* the pid reaches it, so a crash in
that instant leaves an empty file (or, from a replayed handler, one holding
`SIGNAL TERM`). Both `scan_arm` and `clear_launching` treated the file's existence as the
registration: the marker was withdrawn and the arm read as quiet while the trainer ran.

`registration_complete <attempt>` is now the test — `child.pid` exists **and** its whole
content is `^[0-9]+$`. Whether that pid is still alive is a separate question, `pid_alive`'s.
The marker stays unresolved until registration is complete, `clear_launching` withdraws it
only then, and `resolve_unregistered` calls an attempt "already registered" only then.

Regressions: an empty `child.pid`, one holding `SIGNAL TERM`, a bare newline and a
non-numeric string all keep the arm blocked with the marker present; a complete
`child.pid` naming a long-dead pid resolves the launch and the recovery promotes.

## Blocker 2 [P1] — a delayed trainer fails closed; age proves nothing

A trainer that has not woken yet is not bound by the marker's age. After an operator
resolved the launch, it could still start and run inside a retired attempt.

**`register_trainer` refuses with exit 3, writing nothing, when:**

1. **the save-dir is gone** — the launcher creates the attempt, the trainer never does.
   `run()` no longer calls `os.makedirs` either (both calls are now existence checks), so
   a refused trainer cannot resurrect what a resolution retired;
2. **a tombstone `<attempt>.resolved` stands beside it** — that launch was answered;
3. **the attempt carries no launch record at all** — from the fork onward an ordinary
   launch always has `launching` (until `clear_launching`) or a complete `child.pid`, so
   neither means neither.

`--no-save` runs are diagnostics no scan reads: they write no sidecars and only the
tombstone can refuse them. `SystemExit(None)` is recorded as `train.exit 0`, not 1.

**The tombstone** is written by `resolve_unregistered` under the lock, after the quiet
scan and the grace, and **before** the rename — a renamed directory tells a waking
trainer nothing it can read. Nothing in the launcher ever removes it and the scan ignores
it: it is a record, not a claim.

**The end-to-end regression** runs the real sourced lifecycle with a stub trainer that
imports and calls the real `register_trainer`, held behind a gate. The launcher dies
before `child.pid`; nothing names the trainer; the recovery is refused on the marker; the
operator ages the marker past the grace and resolves the launch; the tombstone is written
and the attempt retired; only then is the gate opened. The trainer refuses, writes no
`train.pid`, recreates nothing, and the next recovery of the arm promotes.
**Verified to fail against `tools/exp11_train.py` as it stood at `e4bd6a3~1`** (the stub
slept through instead of refusing), and to pass after.

## Blocker 3 [P1] — the resolution stays inside the arm it locked

`resolve_unregistered` neither canonicalised nor confined its target: holding H's lock
and having scanned H, it retired an **I** attempt whose trainer was alive. It also
excluded the target from the quiet scan by string, so the same directory spelled two ways
was two directories.

It now `realpath`s both the arm root and the target, requires the target's canonical
parent to **equal** that root and its basename to match `attempt_<digits/T>`, and
`scan_arm` excludes the target by canonical identity rather than by the spelling it was
handed. `args.json` is not required — a launch that died before the trainer wrote one is
the ordinary case, and demanding it would refuse exactly the launches this path exists
for.

Regressions: an I attempt passed with `--arm H` is refused with the directory, its marker
and the arm untouched; an absolute target inside a *relative* arm root is accepted and
retired; `attempt`, `attempt_../escape`, `final` and an already-retired
`*_ABORTED_unregistered` name are all refused.

## Report correction

Fix 8's report said the shell file collects **78** tests; it collects **72**. Corrected
in that report (`652c463`).

## Line counts

| commit | production | tests |
|---|---:|---:|
| `ff610ff` (RED) | — | +48 |
| `3c32cb2` (GREEN) | `exp11_launch.sh` +19/−5 | — |
| `d268f58` (RED) | — | +66 |
| `bc4ad3d` (GREEN) | `exp11_launch.sh` +26/−4 | — |
| `c530e32` (RED) | — | +79 |
| `e4bd6a3` (GREEN) | `exp11_train.py` +60/−11 | +37/−4 |
| `71ac3c9` (GREEN) | `exp11_launch.sh` +7 | — |
| `b49d90b` | — | +56 |
| range `7abeb4d..HEAD` | **+112 / −20 = 132 changed** | +286/−4 |

Every commit is well under 200 changed lines; no exception this cycle.
(Counts corrected in fix 10 from close review 9; `71ac3c9` was the last
production commit of the cycle.)

## Digests

Taken at `652c463`, the cycle's tip (its last production commit is `71ac3c9`).

**exp_06** — exactly **one** key differs from the record re-filled at `9f98bbb`:

| key | approved at 9f98bbb | fix 8 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `a5db73dc4453…` | `a1c2b1c0dfc1…` |

**exp_11** — all eight keys, and which moved against fix 8's tip `7abeb4d`:

| key | digest | vs `7abeb4d` |
|---|---|---|
| `train` | `726b515cf9f42b4eed4f06c137f938b22d59fe8cae6e48de5f0bb64dc887be19` | **moved** |
| `finalize` | `928a1d64a0db18494ad574fcfd29a61c9b2e238894c872ac9677193c517c4ac7` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `9178e7062ee891965f4fd8bead4d59ac3598556f316815520d21fc3e888f978c` | **moved** |
| `smoke` | `99b4d570bc1dba86281f6c228936389b52042e92a502bd9fe2ca13b722c39862` | **moved** |
| `summarize_haa` | `a1c2b1c0dfc1277b8d3f3375251b313df1d98bfaf0ec67a5c21434eb088cdbc0` | **moved** |

The same five as fix 8, for the same two files: `launch_sh` through its own shell bytes,
and `train`, `finalize`, `smoke`, `summarize_haa` because `tools/exp11_train.py` is in all
four source closures. `haa_finetune`, `haa_eval` and `haa_pipeline_sh` contain neither and
are untouched. exp_11's approvals record remains all-null and every producer refuses today.

## Test results

* **Targeted set** — all fourteen `tests/test_exp11_*.py` plus
  `tests/test_exp06_summarize_haa.py` and `tests/test_exp09_sim_eval_closures.py`:
  **411 passed** in 561 s
  (`tests/test_exp11_shell.py` alone: **90**, `tests/test_exp11_train.py`: **24**,
  `tests/test_exp11_lock_holder.py`: **7**). Fix 8 was 384; the 27 new passes are this
  cycle's regressions.
* `bash -n` on both shells, `py_compile` on every changed module and test, and
  `git diff --check 7abeb4d..HEAD`: clean. `tools/exp06_launch.sh` and
  `tools/exp06_finalize.py` are byte-identical to `7abeb4d`.
* **Full CPU suite** — run detached at `616fb6a` (the same tree as the last code
  commit `b49d90b`) to
  `orientation_cue_fairness_2026-09-27_14:50_suite_full_cpu_616fb6a.log` in this record.
  **1 failed, 3741 passed, 55 skipped** in 3638 s (1:00:38). The one
  failure is the standing, expected
  `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`:
  `code.summarize_haa` has moved and the record is re-filled at the reviewed merge, as
  round 1 did with `9f98bbb`, not on this branch. Fix 8 was 3714 passed; the 27 new
  passes are this cycle's regressions. Nothing else regressed.
