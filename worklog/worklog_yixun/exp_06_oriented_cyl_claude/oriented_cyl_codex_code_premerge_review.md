**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Scope and verdict

Reviewed `exp06-window` at **a40f81b256c92f0ddbd2fce08cac0c9815717064**, including both merges, the integration tests, and A3. Read the required SOP, plan and amendments, prior reviews, experiment records, composed tooling, Coder prompts and reports. Rounds 3a/3b remain outside this review.

**Verdict: request changes. One blocking finding.**

## Findings

1. **Blocker — A job can be certified without its required owner.**  
   **Location:** [tools/exp06_finalize.py:1557](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1557), with the fixture gap at [tests/test_exp06_finalize.py:1014](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_finalize.py:1014).

   When `launch.pid` is absent, `refuse_live_launch()` returns `None`. Omitting `--owner-pid` then satisfies `owner_pid is None or owner == owner_pid`, allowing an admissible job with **`owner_pid: null`**.

   I reproduced this through both `finalize()` and the actual CLI parser/`main()` using retained valid child artifacts. Every child validator ran; only filesystem changes and completion publication were represented in memory. The CLI returned:

   ```text
   EXP06_FINALIZE_OK ... "admissible_arm": true ... "run_type": "haa_job"
   CLI_REPRO: exit=0; owner_pid=null
   ```

   A3 requires the completion to bind the job’s owner `launch.pid`. Removing the job receipt while allowing this ownerless path leaves that required binding absent. The new explicit-owner tests miss it; `open_job()` itself creates no `launch.pid`, so positive tests currently accept this invalid state.

   **Minimal fix:** Require a valid, positive PID from the job-root `launch.pid` unconditionally. Preserve matching-owner handling for a live launcher and recovery with a recorded dead launcher. Keep equality mandatory when `--owner-pid` is supplied. Update valid fixtures and add missing-owner refusal regressions with the CLI option both omitted and supplied, covering finetune and zero-shot jobs.

## A3 assessment

Apart from finding 1, the change preserves the required checks:

- Children undergo fresh role validation, artifact comparison, log/receipt verification, job-spec checks, and checkpoint-lineage checks.
- Live receipt PIDs remain refused for **all five child run types**: `full`, `smoke`, `probe`, `haa_train`, and `haa_eval`. The job owner exception does not extend into children.
- A job-root `child_exit.json` is refused.
- `finalize_job` passes owner, spec and children, writes no job receipt or marker, and propagates finalizer and log-write failures.

**Additional Coder questions:** Retaining integer `child_exit == 0` is acceptable as a redundant caller-status check; it is not child-exit evidence and need not be ignored. Removing `child_exit_time`/`child_exit_receipt` from jobs follows A3.

The eventual round-3 consumer must reconcile the actual schema: `owner_pid`, `job_spec: {path, sha256}`, and `children: {relative_child_name: sha256}`. I did not review the child branch; this reconciliation is not an additional blocker here.

## Verification

All execution used the prescribed Python and CPU environment. No files were modified, no GPU work started, and no processes were signalled.

### Merge integrity and pins

- `git status --porcelain=v1`: **empty**.
- `git diff main...exp06-window --diff-filter=M --name-only`: **empty**.
- `git diff main...exp06-window --check`: **clean**.
- Checked against `main` at `a498554e087eeea33358829bfba946617a8bdb8b`.
- Reconstructed both merge results from parent/base content in memory: **zero mismatches**. The two files changed on both sides of `3f6bb56` contained disjoint edits; `3a9e86a` introduced no overlapping content changes.
- Conflict-marker scan of added experiment code/tests: **none**.
- **41 pinned/composed files:** byte-identical to `main`, zero mismatches.
- Recomputed closure digests:
  - exp_03, **12 files**: `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`
  - Pinned trainer, **11 files**: `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`
- In-memory compilation of **13 changed Python files** passed.
- `bash -n tools/exp06_haa_pipeline.sh tools/exp06_launch.sh`: passed.

### Tests and adversarial checks

Attempted the full suite:

```bash
/home/yixunhu/miniconda3/envs/xRIR/bin/python -m pytest \
  tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py \
  -q -p no:cacheprovider -rxXs
```

It failed **before collection** because the sandbox provides no writable temporary directory.

The read-only replay used `pytest.main()` with `--capture=sys -p no:cacheprovider -rsxX`, excluding cases requiring temporary-file fixtures:

**258 passed, 5 skipped, 606 deselected, 25 warnings in 92.94 seconds.**

Import adaptations reused an existing Matplotlib cache and disabled Numba JIT. PID queries used `/proc` instead of signal-zero calls. This is a qualified subset replay, not an independent full-suite pass.

Further verification:

- **23 in-memory adversarial cases:** 20 expected refusals, two valid-owner positive controls, and the ownerless acceptance reported above.
- Separate CLI reproduction confirmed finding 1.
- Three non-dry shell probes confirmed success/failure propagation and refusal before finalization when the log cannot be opened; logs used an anonymous pipe.
- Encoder symmetry, heading-boundary, factory-state and recipe tests passed in the subset.
- Independent Griffin–Lim comparison: unseeded output and Torch RNG advancement were bit-identical to the pinned implementation; Python/NumPy RNG states were unchanged. Seeded same-query output was exact and different-query output differed.
- No remaining `xfail` markers. The five skips comprise three CUDA forwards, one unavailable worktree record copy, and one exp_03 two-GPU acceptance check. I separately verified the main-tree record copy matches the template. These skips do not conceal the former A3 failure; finding 1 is missing coverage.

### Interrupted Coder run

The retained A3 `full_suite.log` contains `Terminated` and **`EXIT_STATUS=143`**, corroborating the interruption. It does not identify the signal sender.

The subsequent `full_suite2.log` records **864 passed, 5 skipped, 91 warnings in 1093.80 seconds**, followed by **`EXIT_STATUS=0`**. This corroborates the report’s completed replacement verification. No stale tracked artifact or skipped replacement verification was found.

## Required before closure

**Exact blocking list: finding 1 only — require the job-root owner PID even when `--owner-pid` is omitted, and add the corresponding regressions.**

No additional regression was found in the previously approved pretraining path. The branch at `a40f81b` should receive this fix and close review before merging.
