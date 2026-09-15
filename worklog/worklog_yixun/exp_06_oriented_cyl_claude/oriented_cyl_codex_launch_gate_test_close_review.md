**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Verdict: approve — launch gate closes at the merge of this branch

Reviewed commit **`4ea1b2b`**, range **`4dd37ec..4ea1b2b`**, on `exp06-launch-gate`. The previous should-fix is resolved. **No new blocking findings.**

## Findings and disposition

1. **Prior should-fix 1 — resolved.**  
   [tests/test_exp06_profiles.py:31](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_profiles.py:31) defers and memoises Git identity resolution. The module-scoped `head` fixture skips when identity is unavailable; all seven checkout-dependent tests reach that fixture. Import no longer executes `git rev-parse`. No further fix required.

2. **Record-existence guard — acceptable deviation.**  
   [Line 338](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_profiles.py:338) preserves the existing optional-record contract. The regression requires the record-schema test to pass when its asset exists. Both present and absent cases passed the adapted replay.

3. **Temporary-repository tests — acceptable deviation.**  
   [Line 187](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_profiles.py:187) constructs an independent repository, so these tests need Git’s executable but do not need the surrounding checkout’s identity. Passing outside a checkout is appropriate; the new `OSError` handling covers unavailable Git.

**Negative-control clarification:** `@needs_git` records names; the fixture performs skipping. Removing only the decorator still skips successfully. The Coder’s documented control also replaces the fixture dependency with an unguarded Git call. That combined control failed as intended.

## Verification performed

Read the required SOP, plan v4 and amendments, three plan reviews, Fable round-1 review, notebook including the role change, prior-experiment records, composed tooling, and this round’s prompt/report. Read every changed line.

### Scope and integrity

- `git diff --numstat 4dd37ec..4ea1b2b`: **111 insertions, 23 deletions**, only `tests/test_exp06_profiles.py`.
- `git diff --check 4dd37ec..4ea1b2b`: clean.
- `git diff main...exp06-launch-gate --diff-filter=M --name-only`: only that test file.
- `git diff main...exp06-window --diff-filter=M --name-only`: empty.
- Both required commit trailers are present.
- **42 protected paths**, including the HAA pipeline shell, match working-tree, HEAD, base, checked `main` (`8cc4fd7`), and pre-merge `ec7e649` bytes.
- Recomputed pins:
  - exp_03, **12 files**: `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`
  - Trainer, **11 files**: `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`
- Production profiles, template, and approval bytes are unchanged. Approval SHA-256 remains `eba8ad3ac7713bb5deb1365fdb07d43a0ddc321bf8483dedafeff32c53bcbd9e`.
- Endpoint comparison `main..HEAD` additionally shows an unrelated exp_07 notebook difference from main advancing; this branch introduces no non-test change.
- In-memory compilation passed; final worktree is clean at `4ea1b2b`.

### Tests and adversarial checks

Used the specified interpreter/environment, CPU only.

- Module import with subprocess calls forbidden: **passed**.
- Successful lookup, exit 128, missing executable, and permission failure: each cached correctly across three calls, with **one subprocess invocation**.
- Real child pytest processes running in `/tmp`, using an in-memory module collector:
  - Fixed source: **4 passed, 7 skipped, 13 deselected**, exit **0**.
  - Parent source: collection failure, exit **2**.
  - Decorator-only removal: exit **0**, fixture still skips.
  - Unguarded-Git control: **1 failed, 4 passed, 6 skipped**, exit **1**.
- Regression assertions replayed with scratch I/O in memory and real child pytest outcomes:
  - Record present: **10 passed, 14 skipped**; regression accepted.
  - Record absent: **9 passed, 15 skipped**; regression accepted.
  - Unguarded Git, missing required template result, and missing required Git-skip result: **all rejected**.
  - Six temporary-repository cases were explicitly sandbox-skipped in these adapted runs.
- Read-only selection of profiles, encoder, factory, heading, recipe, trainer, evaluation, HAA, and the exp_03 pin test: **222 passed, 3 CUDA-skipped, 78 deselected**, 89.70 seconds.
- Required suite with `--collect-only --capture=sys -p no:cacheprovider`: **907 collected, 883 unique**, including 48 profiles cases because the module appears twice. This corroborates the Coder’s reported **903 passed / 4 skipped** total.

**Replay limitation:** The literal pytest invocation failed before collection because the sandbox has no writable temporary directory. Consequently, the copy-to-disk regression and full suite were not independently replayed unchanged. The full-suite result is the Coder’s reported **903 passed / 4 skipped, exit 0**. Read-only numerical checks disabled Numba JIT and used cache accommodations with write/signal guards.

No files were modified, GPU work started, or processes signaled.

## Exact blocking list

**None.**
