**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Verdict: approve for merge

Reviewed every changed line in **a40f81b..5091f6c**, at HEAD `5091f6cc9c599f8157267026509793e491a941e9`, against the required SOP, approved plan/amendments, reviews, notebooks, prior experiment records, composed tooling, and this round’s Coder prompt/report.

**The sole pre-merge blocker is resolved. No new blocking findings.**

## Findings and disposition

1. **Previous severity: blocker — ownerless job certification. Status: resolved.**  
   **Location:** [tools/exp06_finalize.py:1566](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1566).

   `haa_job_completion()` now unconditionally requires the owner returned from the job-root `launch.pid`. The existing `_pid_of()` → `_alive()` path rejects nonnumeric and nonpositive PIDs. Equality remains mandatory whenever `--owner-pid` is supplied.

   Replaying the actual script entry through `runpy.run_path(..., run_name='__main__')`, including its parser and exit handling, refused ownerless finetune and zero-shot jobs:

   ```text
   EXP06_FINALIZE_REFUSED ... records no launch.pid: a job binds the owner that ran it
   exit=2; published=0
   ```

   **Additional fix required:** none.

   **Correction to the preceding review:** the Coder is correct. At `a40f81b`, the shell’s `open_job()` already called `own_launch()`, which writes `$$` to the job-root `launch.pid`. The defect was in finalizer enforcement and fixture coverage. The shell change in this round is comments only.

## Adversarial verification

The following matrix passed for **both finetune and zero-shot**, using retained valid child artifacts and the real validators:

| Job-root `launch.pid` | `--owner-pid` | Result |
|---|---|---|
| Missing | Omitted or supplied | Refused, exit 2 |
| Empty, `0`, `-1`, `abc` | Omitted or supplied | Refused, exit 2 |
| Live unrelated process | Omitted or unequal | Refused, exit 2 |
| Live unrelated process | Equal | Accepted, matching the specified declaration-based exception |
| Dead PID `4194304` | Omitted or equal | Accepted; correct `owner_pid` recorded |
| Dead PID `4194304` | Unequal | Refused, exit 2 |

**32 matrix cases: 26 refusals, 6 admissions.** Repeating every admitted case produced identical serialized completion content.

Filesystem changes—including removal of the retained job-root receipt and completion, PID substitutions, and publication—were represented in memory. Child artifacts and their validation remained real. PID liveness used `/proc` instead of signal-zero calls.

## Verification record

### Tests and CLI

- Attempted the prescribed full suite:

  ```bash
  /home/yixunhu/miniconda3/envs/xRIR/bin/python -m pytest \
    tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py \
    -q -p no:cacheprovider -rxXs
  ```

  **Blocked before collection:** no writable temporary directory.

- Read-only subset via `pytest.main()`, `--capture=sys -p no:cacheprovider -rsxX`:  
  **258 passed, 5 skipped, 617 deselected, 25 warnings; 84.77 seconds.**
- Replayed the **10 new Python regression cases** directly with retained children and in-memory owner/publication fixtures: **10 passed**.
- Actual script-entry ownerless reproductions: **2 refused**, no publication.
- Actual pipeline dry-run:

  ```bash
  bash tools/exp06_haa_pipeline.sh '' cyl_or:0 zeroshot --dry-run
  ```

  **Exit 0:** four owner-file announcements and four correctly formed job-finalizer commands. Dry-run creates no roots; certification attempts for all four absent roots were refused. The new filesystem-writing `open_job` regression could not run independently in this sandbox.

Import adaptations reused an existing Matplotlib cache and disabled Numba JIT. Execution otherwise used the prescribed CPU environment. The Coder reports **875 passed, 5 skipped, zero xfail**; I did not independently reproduce that full-suite result.

### Scope, merges, and pins

- `git diff a40f81b..HEAD -- tools/exp06_finalize.py`: only module documentation and `haa_job_completion()` changed. **The other 76 functions, including every non-job dispatch path and shared PID helper, are byte-identical.**
- Reconstructed merges `3f6bb56` and `3a9e86a` from parent/base content in memory: **zero content mismatches**.
- Current `main`: `bf6027eb4249d7fb41b3eabed82332037a2e32a2`. No overlapping changed paths for the pending merge.
- **41 pinned/composed source files**, plus the original HAA pipeline and two pinned test files: byte-identical to `main`.
- Recomputed closure digests:
  - exp_03, 12 files: `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`
  - Trainer, 11 files: `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`
- Three changed Python files compiled in memory; `bash -n` passed.
- Whitespace checks clean; worktree clean; modified-file list versus `main` empty; no conflict markers or `xfail` markers.

No files were modified, GPU work started, or processes signalled.

**Exact blocking list before this round closes: empty (`[]`).**
