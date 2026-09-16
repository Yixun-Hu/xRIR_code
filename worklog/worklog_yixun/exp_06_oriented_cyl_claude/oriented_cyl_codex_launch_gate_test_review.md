**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Verdict: request changes

The populated-record checks work in this checkout and reject the required tampering cases. One explicit requirement remains unmet: graceful skipping outside Git.

## Findings

1. **Should-fix — the non-Git skip is unreachable during module import.**  
   **Location:** [tests/test_exp06_profiles.py:231](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_profiles.py:231), interacting with [line 13](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_profiles.py:13).

   The new binding test calls `tracked_at()` before deciding to skip, but the module first executes the unguarded, existing `HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], ...)`. Outside a Git checkout, collection fails before the helper or record fixture can run.

   **Reproduction:** An in-memory pytest collector executed the unchanged module source with its location under non-Git `/tmp`. Result: **no tests collected, one collection error, pytest exit 2**; `git rev-parse HEAD` exited **128** at line 13. Testing `tracked_at()` after importing the module misses this failure.

   **Why it matters:** The round explicitly requires graceful handling outside Git. Currently even the template and record-schema tests cannot collect there.

   **Minimal fix:** Guard or defer HEAD resolution, and skip Git-dependent binding/digest checks when Git identity is unavailable. Preserve the independent template/schema checks. Add a regression exercising actual module collection outside Git, then replay the required suite. Keep production code and approval bytes unchanged.

## Verification performed

### Scope and hygiene

- Read the required SOP, plan/reviews, notebook including the role change, prior-experiment records, composed tooling, Coder prompt and report.
- Read every changed line in `git diff 4dd37ec^..4dd37ec`.
- Confirmed parent **`3975afe`**, with only worklog changes between `53307b6` and that parent.
- Commit contains **63 insertions and 8 deletions**, solely in `tests/test_exp06_profiles.py`; both required authorship/session trailers are present.
- `git diff --check 3975afe..4dd37ec`: clean.
- `git diff 3975afe..4dd37ec -- . ':!tests/test_exp06_profiles.py'`: empty.
- `git diff main...exp06-window --diff-filter=M --name-only`: empty.
- The corresponding `exp06-launch-gate` comparison lists only the reviewed test file.
- **41 protected source paths** match working-tree, HEAD, base, checked `main`, and pre-merge `ec7e649` bytes. Independently recomputed both pins:
  - exp_03, **12 files**: `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`
  - Original trainer, **11 files**: `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`

### Record checks and adversarial probes

Template null assertions are unchanged. All three replacement checks pass against the actual record:

- Schema and key sets agree, including nested sections.
- Approval bytes bind to HEAD; SHA-256 is `eba8ad3ac7713bb5deb1365fdb07d43a0ddc321bf8483dedafeff32c53bcbd9e`.
- All **15** filled code digests match recomputation; all **8** training keys are filled; `require(...) == []`.
- Digest computation preserves Python, NumPy and Torch RNG states.

In-memory negative controls produced:

| Probe | Result |
|---|---|
| Alter each filled digest | **15/15 rejected** |
| Null each training digest | **8/8 rejected** |
| Claim a digest for each absent module | **4/4 rejected** |
| Add section or nested keys | **6/6 rejected** |
| Bind current record to `4d7a6cb` or `fc1ee2a` | Both rejected |
| Add whitespace to approval bytes | Binding rejected |
| Missing record | **3 skipped**, clean exit |

The one-directional rule correctly allows an unfilled, present non-training key and later module availability without demanding immediate approval.

### Test replay

Used the specified interpreter/environment, CPU only, `PYTHONDONTWRITEBYTECODE=1`, `--capture=sys`, and `-p no:cacheprovider`. In-memory cache accommodations and write/signal guards preserved read-only execution; Numba JIT was disabled.

- Profiles selection with  
  `-k 'not malformed and not approvals_can_be and not approvals_that and not binding_needs'`:  
  **11 passed, 12 deselected**, 42.45 s.
- Encoder/factory/heading/recipe/train/eval/HAA modules plus the exp_03 pin test, excluding temporary-filesystem fixtures:  
  **211 passed, 3 CUDA-skipped, 65 deselected**, 65.94 s.
- Re-executed the deleted parent test in memory: expected assertion failure; first mismatch **byte 332**, `b'"' != b'n'`.
- Required suite command with `--collect-only`: **905 collected**, **882 unique**. The profiles module contributes 46 cases because it appears explicitly and through the glob.
- Coder report records **901 passed / 4 skipped**. Collection counts corroborate that total; the full write-requiring suite was **not independently replayed**.
- In-memory compilation passed. Final HEAD remains `4dd37ec`, with a clean worktree. No files were modified, GPU work started, or processes signaled.

## Exact blocking list

**[1]** Fix the non-Git collection/skip path, add its regression, and obtain a green required-suite replay before this round closes.
