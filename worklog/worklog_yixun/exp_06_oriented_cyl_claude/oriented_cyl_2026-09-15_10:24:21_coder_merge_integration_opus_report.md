**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

Merge-integration fix for exp_06 "oriented_cyl" (tests only). Worktree
`/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`, from tip `3a9e86a`
(= training gate `a313253` + merged round-2b tip `3df20b8` + main). CPU only
(`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`). `/home/yixunhu/codespace/xRIR_code_wt2b`
was not read; no process I did not start was signalled; nothing under `ckpt/` was written;
this report is the only file written under the main tree.

## Commits (3a9e86a..786161a)

| SHA | subject | changed lines |
|---|---|---|
| `a2e2550` | the pipeline fixtures close a child that has really exited | +8/-2 = 10 |
| `d82c31b` | the approvals test derives absence from the checkout, not from a round | +14/-1 = 15 |
| `786161a` | pin the one merge conflict the fixtures must not hide | +32/-0 = 32 |

Both trailer lines present on all three (`git log -1 --format=%B \| grep -c` = 2 each).
`git diff main...HEAD --diff-filter=M --name-only` is empty and
`git diff 3a9e86a..HEAD --name-only` is exactly `tests/test_exp06_haa_pipeline.py` and
`tests/test_exp06_profiles.py`: **no tool module and no product file was touched**, so
`tools/exp06_finalize.py`, the wrappers, the exp_03 pinned closure and the exp_04/05
digests are byte-identical to `3a9e86a`.

## Red → green

Baseline at `3a9e86a` (full command below):
`1 failed, 845 passed, 5 skipped, 11 errors in 934.28s`.

* 11 errors, all in `tests/test_exp06_haa_pipeline.py`, from the module fixture
  `finetune_seed` → `make_train_child` → `exp06_finalize.finalize(save, 'haa_train', …)`:
  `ValueError: child_exit.json records child_pid <pid>, a process that is still alive`.
  Cause exactly as diagnosed: round-2b's `close_child` helper (commit `2fce9ef`) closed
  every child with `--child-pid str(os.getpid())`, the live test process, while the
  training-gate fix (commit `56dc945`, finding 3) made `child_exit_receipt` refuse a
  receipt whose own `child_pid` is alive.
  **Fix (`a2e2550`, fixture only):** `close_child` records `DEAD_PID`, the
  `pid_max`-based helper `tests/test_exp06_finalize.py` already defines and uses in
  `seal()`. Imported, not duplicated. The tests that intentionally use a live pid to
  assert refusal are untouched — `test_a_live_pid_in_a_child_refuses_the_job`,
  `test_a_receipt_naming_a_live_child_is_refused`
  (`tests/test_exp06_finalize_gate.py`), the `launch.pid` owner-exception cases in
  `tests/test_exp06_finalize.py` and `tests/test_exp06_launch.py` — and so is the job
  root's own live `launch.pid` with `--owner-pid`, which is the documented exception.
  After it: `tests/test_exp06_haa_pipeline.py` → `33 passed` (was `22 passed, 11 errors`).

* 1 failure, `tests/test_exp06_profiles.py::test_every_present_key_gets_a_digest_and_absent_modules_are_noted`
  — **not** a pid issue. Its last line asserted
  `'haa_finetune' in skipped and 'haa_pipeline_sh' in skipped`, i.e. that those modules are
  "not in this checkout yet". That was true when the training branch wrote it (`3154c3b`)
  and stopped being true the moment round 2b merged; round 2b never touched
  `tools/exp06_profiles.py` or its test, so there is no contract conflict here, only a
  stale name pin. `tools/exp06_profiles.py` behaves correctly: `present_keys()` is derived
  from the checkout and the round-2b keys are now legitimately digested (they stay
  **unapproved** — all-null in the template — as intended).
  **Fix (`d82c31b`, test only):** the assertion now re-derives the absent set from the
  working tree (`skipped == absent`, `absent` computed from `CODE_SPECS` file paths
  independently of the module's own constant) and pins that the merged round's keys
  (`haa_finetune`, `haa_eval`, `haa_pipeline_sh`) are digested. It cannot go stale when
  round 3 lands. After it: `tests/test_exp06_profiles.py` → `20 passed, 1 skipped`.

Full suite after those two commits: **857 passed, 5 skipped, 0 failed, 0 errors** (18:16).

## The one real merge conflict — NOT a fixture issue, product decision required

`tools/exp06_haa_pipeline.sh::finalize_job` (round 2b, commit `140c3c9`) closes the **job
root's** log with

```sh
"$PYTHON" tools/exp06_finalize.py child-exit --run-dir "$root" --log "$log" \
    --child-pid "$$" --status 0 --started-at "$JOB_STARTED_AT" || return 1
run "$PYTHON" tools/exp06_finalize.py --run-dir "$root" --run-type haa_job --log "$log" \
    --child-exit 0 --owner-pid "$OWNER" --children "$@" ...
```

`$$` is the pipeline shell itself, and that same shell then runs the `haa_job` finalizer, so
the job root's receipt always names a process that is alive. `tools/exp06_finalize.py::finalize`
calls `child_exit_receipt(run_dir, …)` for **every** run type, and the gate fix `56dc945`
put a hard `_require(not _alive(receipt['child_pid']), …)` there with an explicit
"No owner exception here: only the launcher's `launch.pid` may be alive at finalisation."
`refuse_live_launch` does forgive the matching `<root>/launch.pid` for `--owner-pid`;
`child_exit_receipt` does not. Verified empirically (`--runxfail`):

```
ValueError: child_exit.json records child_pid 3327444, a process that is still alive
  tools/exp06_finalize.py:161
```

So **every HAA job the real pipeline finalizes will be refused at the job level** on this
merged branch. The nine children per seed are fine — their pids genuinely are dead by the
time the launcher finalizes them, which is why `a2e2550` is an honest fixture fix — but the
job root is not a fixture artefact: the round-2b fixture's `os.getpid()` there was
*faithful* to `$$`.

Per the brief I did not touch product code. Instead `786161a` records the conflict in the
suite as a **strict** xfail,
`tests/test_exp06_haa_pipeline.py::test_the_job_receipt_the_pipeline_really_writes_is_admissible`:
it rewrites the module fixture's job-root receipt with a live pid (what `$$` records),
asserts the job is admitted, and restores the fixture in `finally`. It xfails today with
the message above and will turn into a **failure** the moment either side is fixed, forcing
the marker's removal rather than letting the resolution pass unnoticed.

Which side gives is the Planner's call, not the Coder's. The two candidates:

1. `child_exit_receipt` gains the same owner exception `refuse_live_launch` already has
   (`--owner-pid` may equal `receipt['child_pid']`) — narrow, but it weakens the finding-3
   guarantee for `full`/`haa_train`/`haa_eval` unless it is restricted to `haa_job`.
2. `finalize_job` stops writing a self-naming receipt — e.g. the job root's completion
   evidence derives its `child_exit` binding from the children's receipts and the job log
   marker instead of a `child-exit` receipt of its own. This keeps the gate fix intact but
   changes `haa_job`'s evidence contract and its dry-run golden output
   (`job_finalize` in `tests/test_exp06_haa_pipeline.py`, which pins `--owner-pid <pid>`).

Nothing else in either wrapper writes a live pid: `tools/exp06_launch.sh:143` passes the
real, already-exited `$CHILD_PID`.

## Validation commands

| command | outcome |
|---|---|
| `python -m pytest tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider` (at `3a9e86a`) | `1 failed, 845 passed, 5 skipped, 11 errors in 934.28s` |
| `python -m pytest tests/test_exp06_haa_pipeline.py -q` (at `3a9e86a`) | `22 passed, 11 errors in 46.09s` |
| `python -m pytest tests/test_exp06_haa_pipeline.py -q` (at `a2e2550`) | `33 passed in 108.65s` |
| `python -m pytest tests/test_exp06_profiles.py -q` (at `3a9e86a`) | `1 failed, 19 passed, 1 skipped in 39.39s` |
| `python -m pytest tests/test_exp06_profiles.py -q` (at `d82c31b`) | `20 passed, 1 skipped in 38.39s` |
| full suite (at `d82c31b`) | **`857 passed, 5 skipped in 1096.17s`** |
| `python -m pytest tests/test_exp06_haa_pipeline.py -q -rxX` (at `786161a`) | `33 passed, 1 xfailed in 114.19s` |
| **full suite (at `786161a`), exit 0** | **`857 passed, 5 skipped, 1 xfailed in 1106.42s (0:18:26)`** |
| `python -m py_compile` on both touched test files | clean |
| `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh` | clean (not modified; checked only) |
| `git diff --check` | clean |
| `git status --porcelain` | empty |

Full-suite command, run from the worktree with `PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt`,
`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`:

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests/test_exp06_*.py \
    tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider
```

## Deviations from the brief

* The brief anticipated only fixture pid issues. Two failures were not that: the
  `test_exp06_profiles.py` stale name pin, which I fixed because the fix is entirely inside
  a test and strengthens rather than weakens its contract; and the `finalize_job` job-root
  receipt, which I did **not** fix — it needs product code, so it is reported above and
  pinned as a strict xfail instead of being left as a red suite or silently papered over.
  If the Planner would rather see it red, reverting `786161a` restores that state
  (`786161a` adds only the xfail test).
* No other deviation. No `tools/` file, no shell wrapper, no `tests/test_provenance.py`
  and no `tests/test_exp03_record_tools.py` was modified.
