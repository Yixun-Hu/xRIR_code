**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

Merge-integration FIX 2 for exp_06 "oriented_cyl": plan amendment **A3** (HAA job completion
unit). Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`, from tip
`786161a`. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`).
`/home/yixunhu/codespace/xRIR_code_wt2b` was not read; no process I did not start was
signalled; nothing under `ckpt/` was written; this report is the only file written under the
main tree. Exactly four files were touched:

    tools/exp06_finalize.py  tools/exp06_haa_pipeline.sh
    tests/test_exp06_finalize.py  tests/test_exp06_haa_pipeline.py

`git diff main...HEAD --diff-filter=M --name-only` is empty, so no file that exists on `main`
was modified (the exp_03 pinned closure, `tests/test_provenance.py` and
`tests/test_exp03_record_tools.py` included).

## Commits (786161a..HEAD)

| SHA | subject | changed lines |
|---|---|---|
| `da5eb93` | require the A3 job contract of the finalizer (red) | +93/-33 = 126 |
| `c57bca2` | a job binds its spec, its owner and its children, not a receipt (A3) | +42/-3 = 45 |
| `6429e1c` | the queue's job finalisation writes no receipt (red) | +26/-6 = 32 |
| `a40f81b` | the pipeline's job finalisation closes no log of its own (A3) | +8/-7 = 15 |

Both trailer lines on all four (`grep -c` on each message = 2). Every commit is a test-only
red or the implementation that turns it green; none exceeds 200 changed lines.

## What A3 changed

**`tools/exp06_finalize.py` (`c57bca2`, job branch only).** `finalize` now dispatches
`haa_job` to a new `haa_job_completion` **before** any child-exit evidence is touched, so
`closed_log` and `child_exit_receipt` are never called for a job. The job branch:

1. refuses a job root that holds a `child_exit.json`, cause `job roots carry no child exit
   receipt (A3)`;
2. calls `refuse_live_launch(run_dir, owner_pid)` exactly as before — the live launcher at
   the job root is still admissible only under the `--owner-pid` exception — and then
   **binds** that owner: `owner_pid is None or owner == owner_pid`, so a declared owner that
   is not the `launch.pid` the job root actually holds (or a missing one) is refused;
3. records the queue log through the new `job_log(log)` as `{path, sha256}` — informational,
   with no marker requirement, no receipt comparison and no stale-log re-hash; `log=None`
   records `null` (the CLI still requires `--log`, and the pipeline passes it);
4. keeps `child_exit == 0` required and recorded, and then delegates unchanged to
   `haa_job_evidence` → `verify_child` / `check_job_spec` / `job_lineage`, so every expected
   child's completion is still re-validated by re-running its own role validator, its log and
   receipt rehashed, and its live-pid contract enforced with **no** owner exception.

The evidence that every writer had exited therefore still exists in full — it lives in the
nine children, where the processes actually were. Job completions gain `owner_pid` and lose
`child_exit_time` and `child_exit_receipt`.

Every other run type is byte-for-byte as it was: the only other edit in `finalize` is
dropping the now-unreachable `haa_job_evidence` arm of the evidence conditional.

**`tools/exp06_haa_pipeline.sh` (`a40f81b`, `finalize_job` only).** The job-root
`child-exit` call and the job-root `say "MARKER …"` line are gone; `finalize_job` appends its
one queue line to the job log (`|| return 1`, so a failed write is not silent) and hands the
root, that log, `--owner-pid "$OWNER"` (= `$$`, the `launch.pid` `open_job` wrote at the job
root) and the children to the finalizer. `JOB_STARTED_AT` existed only to stamp the receipt
that is no longer written, so its two assignments in `run_zeroshot`/`run_finetune` go with it
— the only lines outside `finalize_job` this commit touches.

## Tests

* `tests/test_exp06_haa_pipeline.py`: the strict xfail
  `test_the_job_receipt_the_pipeline_really_writes_is_admissible` is **gone**, replaced by the
  passing `test_the_job_the_pipeline_really_writes_is_admissible` — the module fixture now
  leaves a job root exactly as the pipeline does (queue log, live `launch.pid` = the test
  process, no receipt), and the job of nine dead children finalizes with
  `admissible_arm: true` under `--owner-pid os.getpid()`, binding the queue log by hash and
  carrying no `child_exit_receipt`/`child_exit_time`. New `test_a_receipt_at_the_job_root_is_refused`
  and `test_a_real_job_finalisation_leaves_no_receipt_at_the_job_root` (library mode, DRY=0,
  `run` stubbed: the job log gets the queue line only, no `EXP06_CHILD_EXIT`, no
  `child_exit.json`, and the finalizer argv is the A3 one). The two dry-run goldens lost the
  job-root marker line.
* `tests/test_exp06_finalize.py`: the fixture helper `open_job` replaces `seal` at the job
  root (`write_job` and the bare-claims test). New: a receipt at the job root is refused by
  name; the queue log is bound by path+hash with no marker; a declared `--owner-pid` must be
  the job root's `launch.pid` (parametrised over a mismatching and an absent one).
  `test_a_live_child_is_never_certified_by_a_job` keeps both of its cases and is now faithful
  to production: without an owner the live root launcher refuses, and with the owner declared
  the live pid **inside** the child refuses.
* Children's contracts untouched: `test_a_receipt_naming_a_live_child_is_refused`,
  `test_the_receipt_and_the_child_pid_sidecar_must_agree`,
  `test_a_live_pid_in_a_child_refuses_the_job`, `test_a_child_whose_log_was_never_closed_is_refused`
  and the whole `tests/test_exp06_finalize_gate.py` pass unchanged.

## Validation

| command | outcome |
|---|---|
| `pytest tests/test_exp06_finalize.py -q -k job` (at `da5eb93`, red) | `58 failed, 1 passed, 217 deselected in 144.34s` |
| `pytest tests/test_exp06_haa_pipeline.py tests/test_exp06_finalize_haa_consumer.py -q` (at `da5eb93`, red) | `15 failed, 37 passed in 113.36s` |
| `pytest tests/test_exp06_finalize.py -q -k job` (at `c57bca2`) | `59 passed, 217 deselected in 139.16s` |
| `pytest tests/test_exp06_haa_pipeline.py tests/test_exp06_finalize_haa_consumer.py tests/test_exp06_finalize_gate.py -q` (at `c57bca2`) | `124 passed in 207.53s` |
| `pytest tests/test_exp06_haa_pipeline.py -q -k "golden or zeroshot_job or real_job"` (at `6429e1c`, red) | `3 failed, 33 deselected in 3.23s` |
| `pytest tests/test_exp06_haa_pipeline.py -q` (at `a40f81b`) | `36 passed in 114.21s` |
| **full suite (at `a40f81b`), exit 0** | **`864 passed, 5 skipped, 91 warnings in 1093.80s (0:18:13)` — 0 failed, 0 errors, 0 xfail** |
| `python -m py_compile tools/exp06_finalize.py tests/test_exp06_finalize.py tests/test_exp06_haa_pipeline.py` | clean |
| `bash -n tools/exp06_haa_pipeline.sh tools/exp06_launch.sh` | clean |
| `bash tools/exp06_haa_pipeline.sh 7 cyl_or:0 --dry-run` | job ends at the `haa_job` finalisation, no job-root marker |
| `git diff 786161a..HEAD --check` / `git status --porcelain` | clean / empty |
| `git diff main...HEAD --diff-filter=M --name-only` | empty |

Full-suite command, from the worktree with `PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt`,
`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`:

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests/test_exp06_*.py \
    tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider -rxX
```

The count reconciles exactly with the baseline `786161a` (`857 passed, 5 skipped, 1 xfailed`):
857 + 1 (the xfail is now a passing test) + 6 new tests = **864**, and `grep -rn xfail tests/`
now finds nothing. `-rxX` printed no xfail/xpass section.

## Deviations, judgement calls and notes for the Reviewer

1. **Strict owner binding.** A3 says the job completion binds "the owner `launch.pid` (must
   equal `--owner-pid` when given)". I implemented the strict reading: when `--owner-pid` is
   given, the job root must hold a `launch.pid` **equal** to it — an absent one is refused
   too. This is what production always satisfies (`open_job` → `own_launch` writes it before
   the first child). It required one existing test,
   `test_a_live_child_is_never_certified_by_a_job`, to be rewritten from a two-value loop into
   two explicit cases; both still assert `alive` and the test is now closer to production
   (the no-owner case is refused by the live root launcher, the owner case by the live pid
   inside the child).
2. **`child_exit` kept for `haa_job`.** A job has no child, but `--child-exit` is still a
   required CLI argument (leaving the parser untouched keeps every other run type
   byte-for-byte) and the pipeline passes `0`. The job branch keeps requiring it to be `0`
   and records it, with a job-specific message. Say so if you would rather it were dropped
   from the record.
3. **`JOB_STARTED_AT` removed.** Its only reader was the job-root `child-exit` call. Leaving
   two `date -u` assignments no one reads would have been dead state, so `a40f81b` removes
   them — two lines outside `finalize_job`, and the only part of the shell change not inside
   that function.
4. **Job completion schema changed** (`haa_job` only): `owner_pid` added; `child_exit_time`
   and `child_exit_receipt` removed; `log` is `{path, sha256}` of the queue log (or `null` if
   a caller passes `log=None`; the CLI cannot). Any round-3 consumer of a *job-level*
   `completion.json` that reads `child_exit_time`/`child_exit_receipt` must be updated when
   round 3 merges — I could not check those modules, since they live only in the worktree I
   was told not to read. Child completions are unchanged, and it is the children that carry
   the exit evidence.
5. **`finalize`'s one-line docstring** still reads "Verify one child's evidence for its run
   type" — it said that before this change too, and the job path is documented at
   `haa_job_completion` and in the module docstring. Not changed, because a docstring-only
   edit has no red test to precede it.
6. **No stale-log re-hash for a job.** The final `provenance.sha256_file(log) == log_digest`
   check stays for every child run type. For a job the queue log is explicitly informational,
   so it is hashed once and recorded; re-checking it would re-introduce a proof obligation A3
   removes (and the pipeline's own queue line is written before the finalizer starts).
7. **Operational note.** My first detached full-suite run was killed by `SIGTERM` seconds
   after launch (`EXIT_STATUS=143`, empty output); relaunching with `setsid --fork` survived.
   The run shared the host with another session's pytest, which is why 1093 s here matches the
   pre-change baseline despite the extra tests.
