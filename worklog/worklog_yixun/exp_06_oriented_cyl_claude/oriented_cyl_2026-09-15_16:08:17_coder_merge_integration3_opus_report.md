**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 merge-integration fix 3 — pre-merge review finding 1 (ownerless job)

Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`, from `a40f81b`.
CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), `PYTHONPATH` / `XRIR_DATA_PATH`
/ `PYTHONDONTWRITEBYTECODE` / `NUMBA_CACHE_DIR` as the round-2a rules require. No process
was signalled; no `pkill`/`pgrep`; `/home/yixunhu/codespace/xRIR_code_wt2b` was not read.
Only the four permitted files were touched.

## Commits

| SHA | subject | changed lines (ins/del) |
| --- | --- | --- |
| `17cdb61` | exp06: a job root must record the owner it binds (red) | 85 (82/3) — tests only |
| `5091f6c` | exp06: a job binds the owner its root records, always (finding 1) | 24 (19/5) |

Both end with the two trailer lines (`Co-Authored-By: Claude Opus 5`, `Claude-Session:`).
No amend/rebase/reset/push. Files changed vs `a40f81b`: `tools/exp06_finalize.py`,
`tools/exp06_haa_pipeline.sh`, `tests/test_exp06_finalize.py`,
`tests/test_exp06_haa_pipeline.py` (101 insertions, 8 deletions).

## The fix

`tools/exp06_finalize.py::haa_job_completion` now requires the job-root owner
unconditionally:

```python
owner = refuse_live_launch(run_dir, owner_pid)
_require(owner is not None, '{} records no launch.pid: a job binds the owner that ran '
         'it'.format(run_dir))
_require(owner_pid is None or owner == owner_pid, ...)   # unchanged
```

* A missing `launch.pid` is refused by name whether or not `--owner-pid` was supplied, so
  the reviewer's `owner_pid: null` acceptance is gone. Nothing is written on refusal.
* A malformed or non-positive `launch.pid` was already refused inside `_pid_of`/`_alive`
  (`unreadable <path>: …` / `<n> is not a pid`); that path is now pinned by regressions
  rather than left uncovered.
* Positivity is guaranteed by `_alive`'s `type(pid) is int and pid > 0` check, which every
  `_pid_of` call passes through, so no second numeric check was added.
* Equality with a supplied `--owner-pid` stays mandatory; the live-launcher exception
  (a live job-root `launch.pid` is admissible only when it equals `--owner-pid`) and
  recovery finalisation with a recorded **dead** launcher are unchanged — the latter is
  now the path every positive job fixture exercises.
* The `haa_job` paragraph of the module docstring and the `haa_job_completion` docstring
  state the new contract.

## `tools/exp06_haa_pipeline.sh::open_job` — verified, not changed in behaviour

The review states that "`open_job()` itself creates no `launch.pid`". **That is not what the
script does at `a40f81b`**: `open_job` calls `own_launch "$1"` (sourced from
`tools/exp06_launch.sh`), which writes the pipeline shell's `$$` to `<root>/launch.pid`,
for a new root and for one adopted from an interrupted queue alike; the dry-run goldens
already carry `PIDFILE <root>/launch.pid` (`tests/test_exp06_haa_pipeline.py`
`finetune_job` / `zeroshot_job`), and `finalize_job` passes that same `$$` as
`--owner-pid`. I verified it directly before writing anything:

```
$ bash -c 'EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh; DRY=0; RECORD=...; \
           open_job "$1"; echo "shell_pid=$$"' bash <tmp>/job
MKDIR <tmp>/job ; PIDFILE <tmp>/job/launch.pid ; shell_pid=3721022
$ cat <tmp>/job/launch.pid   ->   3721022
```

What was genuinely missing was **coverage**: no test asserted the non-dry behaviour, which
is what let the state look creatable. So the shell change is documentation only (a comment
on `open_job` recording that the pid it writes is the owner the completion now requires),
the goldens are unchanged because the announced lines were already correct, and the gap is
closed by a new test that runs `open_job` for real, twice, and compares the file with the
shell's own `$$` (`test_open_job_records_the_owner_pid_at_the_job_root`). Deviation from
the prompt's expectation, reported rather than silently "fixed": no functional edit to
`open_job` was needed.

## Fixtures and regressions

* `tests/test_exp06_finalize.py::open_job` (the fixture helper, not the shell function)
  now writes `launch.pid` with `DEAD_PID` (`/proc/sys/kernel/pid_max`), so every positive
  and every other-cause job fixture carries a dead recorded owner, exactly as a recovery
  finalisation would. Tests that model the live launcher keep writing `os.getpid()` and
  passing `--owner-pid os.getpid()` (`test_the_job_owner_may_finalize_once_every_child_is_dead`,
  `test_a_live_child_is_never_certified_by_a_job`, the whole
  `tests/test_exp06_haa_pipeline.py::finetune_seed` family, which already wrote its own).
* `test_a_declared_job_owner_must_be_the_launch_pid_at_the_job_root[False]` now *unlinks*
  the pid file (it previously relied on the fixture writing none) and still refuses with
  `owner`.
* New `test_a_job_whose_root_records_no_owner_is_never_admissible`: 8 cases —
  {missing, malformed} × {`--owner-pid` omitted, supplied} × {finetune, zeroshot} —
  through `finalize()`; each refuses by name and writes no `completion.json`.
* New `test_the_cli_binds_the_job_owner_or_refuses_the_job`: for finetune and zeroshot,
  four refusals through the real parser and `main()` (malformed/missing × option
  omitted/supplied, exit status 2 with `launch.pid` on stderr), then a positive control
  that returns 0 and whose `completion.json` records `owner_pid == DEAD_PID`.

## Validation

| command | outcome |
| --- | --- |
| `pytest tests/test_exp06_finalize.py tests/test_exp06_finalize_haa_consumer.py -q` (before any edit) | 293 passed (baseline) |
| `pytest tests/test_exp06_finalize.py -q -k "owner or launch_pid or job_owner or cli_binds"` (red commit) | **6 failed**, 8 passed — the 4 `missing` cases and both CLI cases; the CLI failure is `main(...) == 0` with the owner file removed, i.e. the reviewer's reproduction |
| `python -m py_compile tools/exp06_finalize.py tests/test_exp06_finalize.py tests/test_exp06_haa_pipeline.py` | ok |
| `bash -n tools/exp06_haa_pipeline.sh` | ok |
| `git diff --check` | clean (both commits) |
| `pytest tests/test_exp06_finalize.py tests/test_exp06_finalize_haa_consumer.py tests/test_exp06_haa_pipeline.py -q` (after the fix) | 340 passed |
| full suite `pytest tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider -rxXs` | **875 passed, 5 skipped, 0 failed, 0 xfail** in 1035 s, `EXIT_STATUS=0` |

The 4 `malformed` cases and the CLI's malformed pair were green before the fix (that path
already refused inside `_pid_of`); they are recorded as regressions, not as red evidence.
Full-suite skips are the three pre-existing CUDA-only forwards, the record asset that lives
in the main tree, and exp_03's two-GPU acceptance check — the same five as the branch's
previous full run (864 passed there; +11 here is exactly the new cases).

## Discrepancies / deviations

1. The review's premise that `open_job` writes no `launch.pid` is incorrect at `a40f81b`
   (evidence above). The binding gap was in the finalizer and in the fixtures only; the
   shell keeps its behaviour and gains a comment plus a real regression test.
2. No numeric-positivity check was added to `haa_job_completion` because `_pid_of` →
   `_alive` already refuses `0`, negatives and non-integers before the value can be
   recorded. The regressions pin this.
