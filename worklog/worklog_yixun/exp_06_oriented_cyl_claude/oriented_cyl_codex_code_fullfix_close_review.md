# Codex close review — exp_06 fix cycle R1–R5 (exp06-fullfix ac05dd9..31d2933)

- Reviewer: OpenAI Codex `gpt-6-astra`, `reasoning effort: xhigh` (CLI override), sandbox read-only, worktree `/home/yixunhu/codespace/xRIR_code_wt` branch `exp06-fullfix` at 31d2933 (main ddfcb66)
- Prompt: `review_prompts/codex_code_review_briefing.md` + `review_prompts/codex_code_fullfix_close_prompt.md`
- Run: 2026-09-18 10:46:08 → 10:57:26 local, log `oriented_cyl_2026-09-18_10:46:08_codex_code_fullfix_close_review.log`, Codex session 01a0b4fb-62e3-71b2-b667-a60d8bc692ae

## Final message

**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-18.

**Verdict: request changes.** Reviewed `exp06-fullfix`, range `ac05dd9..31d2933`, against main `ddfcb66`. R1, R2, R4 and R5 are closed. R3’s artifact-membership checks are fixed, but its producer-closure validation remains incomplete. Merging, re-filling the six digests and obtaining a green suite would therefore not yet make this snapshot approved for confirmatory launch.

1. **R3 — blocker, remaining — A receipt with no substantive producer closure is still admitted.**  
   [tools/exp06_legacy_train_receipt.py:236](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_legacy_train_receipt.py:236)

   `verify()` requires `source_closure` to be a dictionary, then checks only `strict is True` and a falsey `drift`. It does not validate the producer module, commit, file records, closure digest, or consistency of the recorded working-tree and reviewed-blob hashes.

   **Reproduction:** I built a receipt from the real, retained 18-artifact CYL training directory using the writer’s default strict path. Replacing only its source closure with:

   ```json
   {"strict": true}
   ```

   remained accepted by the registered-baseline evidence check. I also passed this receipt through `launcher.build_fields()`, persisted those exact fields in an in-memory evaluation directory, and obtained acceptance from `exp06_compare.admit_run()` as arm B.

   Separate probes were also accepted with an empty closure file list, an incorrect closure digest, a differing working-tree hash, or a missing reviewed-blob hash while `drift` remained empty. All retained training artifacts and the registered CYL checkpoint remained real and unchanged.

   This leaves the requested “no closure” and dirty-producer refusals incomplete: the reader trusts the closure’s summary flags even when its evidence is absent or contradictory. The receipt fixture at [tests/test_exp06_legacy_train_receipt.py:44](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_legacy_train_receipt.py:44) currently treats an empty file list and placeholder digest as acceptable producer evidence.

   **Minimal fix:** Validate the expected producer identity, recorded commit, nonempty closure membership and digest; validate reviewed-blob identities against that recorded commit; derive drift from the file records and reject missing or differing hashes. Preserve historical-commit validity and the allowance for dirty worklog files. Add reader and launcher/comparer regressions for the strict-only closure, empty membership, corrupted digest and contradictory file hashes, while retaining acceptance of the real clean receipt.

**Close-review results**

| Item | Independently verified result |
|---|---|
| **R1** | The API derives required-key closures from its own `CODE_SOURCES`; `exp06_profiles.py` is unchanged. Actual older-commit approvals were refused with `code.haa_pipeline_sh` / `tools/exp06_haa_pipeline.sh` and `code.finalize` / `tools/exp06_finalize.py` named. In-memory changed-byte and missing-reviewed-blob probes with matching HEAD approvals were refused. Exploratory mode retained the deviations; clean source was admitted. |
| **R2** | The shared registration rule admits the real CYL receipt. A valid receipt from the real control training directory was refused by `launcher.main()` before `execute_run`: **zero execution calls and no output directory**. |
| **R3, fixed portions** | Writer and reader share `receipt_names()`. The two-file receipt was refused, naming all 16 omitted artifacts. Explicit drift, `strict: false` and absent/null closure were refused. The real clean **18-artifact** receipt verified, including args/history/registration checks. All 18 retained artifacts passed through the supplied hashing binder. |
| **R4** | Both arms’ actual launcher-built fields were admitted with explicit approval arguments and real repository closures: entrypoint **17 files**, inherited writer **15**, exp_06 writer **25**. Corrupting `writer_exp06.sha256` was refused for both arms with `closure digest writer_exp06`. |
| **R5** | Both `headings=None` and `headings={}` were refused for `haa_children`, naming all four missing rooms. Exploratory mode retained those deviations; complete headings were admitted. |

The R1 rule also admitted the **actual main tree with dirty worklog files and no changes outside worklog**. The R3 membership rule accepted the real training directory despite additional evaluation files and nonmatching checkpoint filenames. The writer’s default path recorded `strict: true` and a real three-file producer closure.

**Verification performed**

- Read the SOP, approved plan and amendments, three Codex plan reviews, Fable review, notebook including the role change, relevant exp_02/exp_03 records, composed tooling, previous R1–R5 review, fix prompt/report and post-training runbook. Read every changed line in `git diff ac05dd9..31d2933`.
- `git diff --check ac05dd9..31d2933` and `git diff --check main..HEAD`: **passed**.
- In-memory `compile()` of all **9 changed Python files**: **passed**. Used this instead of writing bytecode.
- `bash -n` on `tools/exp06_launch.sh`, `tools/exp06_haa_pipeline.sh` and `posttrain_runbook.sh`: **passed**.
- Protected-path `git diff main..HEAD --name-only`: **empty**, including exp_03’s 12 files, the trainer’s 11-file import closure, `sim_to_real`, exp_04/exp_05 tooling, shared comparison/results/provenance tools, `exp06_profiles.py`, approvals template and record copy. The modified-file list from `git diff main...HEAD --diff-filter=M --name-only` contains only earlier exp_06 files.
- Exp_03’s closure remains `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`.
- `git rev-list --reverse ac05dd9..31d2933` with per-commit `git show --numstat`: **9 commits**, total changed-line counts **21, 7, 124, 57, 52, 59, 86, 66, 98**. Maximum **124**, all within 200.
- Read-only Python heredocs exercised the approval, receipt and launcher/comparer probes above. Synthetic receipts and evaluation outputs existed only in memory; repository admission predicates remained active.

The CPU test replay used the specified Python/environment, CUDA disabled, thread limits of four, `PYTHONDONTWRITEBYTECODE=1`, and:

```python
pytest.main(
    sorted(glob.glob("tests/test_exp06*.py")) + [
        "tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged",
        "-q", "--capture=sys", "-p", "no:cacheprovider",
    ],
    plugins=[ReadOnlySubset()],
)
```

`ReadOnlySubset` deselected tests requiring temporary-file or FD-capture fixtures. Result: **425 passed, 6 skipped, 1 expected approvals-drift failure, 866 deselected**, in **138.01 seconds**. The passing count includes the exp_03 pin regression. Coverage included symmetry, factory/parameter identity, trainer-state bit-exactness and statistical/RNG checks.

The sandbox adaptations disabled Numba JIT and made Matplotlib read its existing cache. I did **not** independently replay the complete writable-fixture suite. The Coder’s **3080 passed / 50 skipped / 1 expected failure** full-suite result and **1290 passed / 6 skipped / 1 expected failure** exp_06 result remain reported results. The approvals-drift failure is an actual failing test pending re-fill, not an `xfail`.

**Digest verification at `31d2933`**

Recomputed `exp06_profiles.compute_code_digests(repo, 'HEAD')` and cross-checked the API’s 19 code keys. Exactly these six differ from committed approvals; the other **14 keys are unchanged**:

| Key | Recomputed SHA-256 |
|---|---|
| `compare` | `ea97c6d72a74e4e7df49a0c042d753cb86c89f8989555841b6b8a8cdc41419e6` |
| `eval_launch` | `7314166eeb58a66c5096c0e67fdc1c9a1bfec7fc40d6c57cb6e1ad8fd756cd4f` |
| `mirror_probe` | `ac503c0f265576c478c45dff4c3376cfa029d47c434fcd11db3e5c7182309597` |
| `summarize_haa` | `3f497fc4eb71efdff624b07a71822e8a6bf000df96580f9ecfb77e0b440267e6` |
| `finalize` | `1edbaf2caaffc6ba4e65d6a7fc6bda119f7e4597523f0072bb323c0af7af6b18` |
| `haa_pipeline_sh` | `2ad3e562c1e322d3a035e2aed3bb28d55c940d20e0e9516d88c242a7df0055a7` |

The drift guard fails for exactly these six. The latest runbook’s `refill2` prefixes match them.

**Exact blocking list before this round closes: R3’s remaining producer-closure validation, as finding 1 above.** R1, R2, R4 and R5 need no further changes from this review. After R3 is fixed and reviewed, recompute approvals at the resulting merge tip, update the runbook’s expected digests as necessary, record a green CPU suite, and follow `breceipt → g1 → haa → sim` with the existing gates.

No files were modified or saved, no GPU work was started, and no processes were signalled.
tokens used
288,895
**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-18.

**Verdict: request changes.** Reviewed `exp06-fullfix`, range `ac05dd9..31d2933`, against main `ddfcb66`. R1, R2, R4 and R5 are closed. R3’s artifact-membership checks are fixed, but its producer-closure validation remains incomplete. Merging, re-filling the six digests and obtaining a green suite would therefore not yet make this snapshot approved for confirmatory launch.

1. **R3 — blocker, remaining — A receipt with no substantive producer closure is still admitted.**  
   [tools/exp06_legacy_train_receipt.py:236](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_legacy_train_receipt.py:236)

   `verify()` requires `source_closure` to be a dictionary, then checks only `strict is True` and a falsey `drift`. It does not validate the producer module, commit, file records, closure digest, or consistency of the recorded working-tree and reviewed-blob hashes.

   **Reproduction:** I built a receipt from the real, retained 18-artifact CYL training directory using the writer’s default strict path. Replacing only its source closure with:

   ```json
   {"strict": true}
   ```

   remained accepted by the registered-baseline evidence check. I also passed this receipt through `launcher.build_fields()`, persisted those exact fields in an in-memory evaluation directory, and obtained acceptance from `exp06_compare.admit_run()` as arm B.

   Separate probes were also accepted with an empty closure file list, an incorrect closure digest, a differing working-tree hash, or a missing reviewed-blob hash while `drift` remained empty. All retained training artifacts and the registered CYL checkpoint remained real and unchanged.

   This leaves the requested “no closure” and dirty-producer refusals incomplete: the reader trusts the closure’s summary flags even when its evidence is absent or contradictory. The receipt fixture at [tests/test_exp06_legacy_train_receipt.py:44](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_legacy_train_receipt.py:44) currently treats an empty file list and placeholder digest as acceptable producer evidence.

   **Minimal fix:** Validate the expected producer identity, recorded commit, nonempty closure membership and digest; validate reviewed-blob identities against that recorded commit; derive drift from the file records and reject missing or differing hashes. Preserve historical-commit validity and the allowance for dirty worklog files. Add reader and launcher/comparer regressions for the strict-only closure, empty membership, corrupted digest and contradictory file hashes, while retaining acceptance of the real clean receipt.

**Close-review results**

| Item | Independently verified result |
|---|---|
| **R1** | The API derives required-key closures from its own `CODE_SOURCES`; `exp06_profiles.py` is unchanged. Actual older-commit approvals were refused with `code.haa_pipeline_sh` / `tools/exp06_haa_pipeline.sh` and `code.finalize` / `tools/exp06_finalize.py` named. In-memory changed-byte and missing-reviewed-blob probes with matching HEAD approvals were refused. Exploratory mode retained the deviations; clean source was admitted. |
| **R2** | The shared registration rule admits the real CYL receipt. A valid receipt from the real control training directory was refused by `launcher.main()` before `execute_run`: **zero execution calls and no output directory**. |
| **R3, fixed portions** | Writer and reader share `receipt_names()`. The two-file receipt was refused, naming all 16 omitted artifacts. Explicit drift, `strict: false` and absent/null closure were refused. The real clean **18-artifact** receipt verified, including args/history/registration checks. All 18 retained artifacts passed through the supplied hashing binder. |
| **R4** | Both arms’ actual launcher-built fields were admitted with explicit approval arguments and real repository closures: entrypoint **17 files**, inherited writer **15**, exp_06 writer **25**. Corrupting `writer_exp06.sha256` was refused for both arms with `closure digest writer_exp06`. |
| **R5** | Both `headings=None` and `headings={}` were refused for `haa_children`, naming all four missing rooms. Exploratory mode retained those deviations; complete headings were admitted. |

The R1 rule also admitted the **actual main tree with dirty worklog files and no changes outside worklog**. The R3 membership rule accepted the real training directory despite additional evaluation files and nonmatching checkpoint filenames. The writer’s default path recorded `strict: true` and a real three-file producer closure.

**Verification performed**

- Read the SOP, approved plan and amendments, three Codex plan reviews, Fable review, notebook including the role change, relevant exp_02/exp_03 records, composed tooling, previous R1–R5 review, fix prompt/report and post-training runbook. Read every changed line in `git diff ac05dd9..31d2933`.
- `git diff --check ac05dd9..31d2933` and `git diff --check main..HEAD`: **passed**.
- In-memory `compile()` of all **9 changed Python files**: **passed**. Used this instead of writing bytecode.
- `bash -n` on `tools/exp06_launch.sh`, `tools/exp06_haa_pipeline.sh` and `posttrain_runbook.sh`: **passed**.
- Protected-path `git diff main..HEAD --name-only`: **empty**, including exp_03’s 12 files, the trainer’s 11-file import closure, `sim_to_real`, exp_04/exp_05 tooling, shared comparison/results/provenance tools, `exp06_profiles.py`, approvals template and record copy. The modified-file list from `git diff main...HEAD --diff-filter=M --name-only` contains only earlier exp_06 files.
- Exp_03’s closure remains `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`.
- `git rev-list --reverse ac05dd9..31d2933` with per-commit `git show --numstat`: **9 commits**, total changed-line counts **21, 7, 124, 57, 52, 59, 86, 66, 98**. Maximum **124**, all within 200.
- Read-only Python heredocs exercised the approval, receipt and launcher/comparer probes above. Synthetic receipts and evaluation outputs existed only in memory; repository admission predicates remained active.

The CPU test replay used the specified Python/environment, CUDA disabled, thread limits of four, `PYTHONDONTWRITEBYTECODE=1`, and:

```python
pytest.main(
    sorted(glob.glob("tests/test_exp06*.py")) + [
        "tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged",
        "-q", "--capture=sys", "-p", "no:cacheprovider",
    ],
    plugins=[ReadOnlySubset()],
)
```

`ReadOnlySubset` deselected tests requiring temporary-file or FD-capture fixtures. Result: **425 passed, 6 skipped, 1 expected approvals-drift failure, 866 deselected**, in **138.01 seconds**. The passing count includes the exp_03 pin regression. Coverage included symmetry, factory/parameter identity, trainer-state bit-exactness and statistical/RNG checks.

The sandbox adaptations disabled Numba JIT and made Matplotlib read its existing cache. I did **not** independently replay the complete writable-fixture suite. The Coder’s **3080 passed / 50 skipped / 1 expected failure** full-suite result and **1290 passed / 6 skipped / 1 expected failure** exp_06 result remain reported results. The approvals-drift failure is an actual failing test pending re-fill, not an `xfail`.

**Digest verification at `31d2933`**

Recomputed `exp06_profiles.compute_code_digests(repo, 'HEAD')` and cross-checked the API’s 19 code keys. Exactly these six differ from committed approvals; the other **14 keys are unchanged**:

| Key | Recomputed SHA-256 |
|---|---|
| `compare` | `ea97c6d72a74e4e7df49a0c042d753cb86c89f8989555841b6b8a8cdc41419e6` |
| `eval_launch` | `7314166eeb58a66c5096c0e67fdc1c9a1bfec7fc40d6c57cb6e1ad8fd756cd4f` |
| `mirror_probe` | `ac503c0f265576c478c45dff4c3376cfa029d47c434fcd11db3e5c7182309597` |
| `summarize_haa` | `3f497fc4eb71efdff624b07a71822e8a6bf000df96580f9ecfb77e0b440267e6` |
| `finalize` | `1edbaf2caaffc6ba4e65d6a7fc6bda119f7e4597523f0072bb323c0af7af6b18` |
| `haa_pipeline_sh` | `2ad3e562c1e322d3a035e2aed3bb28d55c940d20e0e9516d88c242a7df0055a7` |

The drift guard fails for exactly these six. The latest runbook’s `refill2` prefixes match them.

**Exact blocking list before this round closes: R3’s remaining producer-closure validation, as finding 1 above.** R1, R2, R4 and R5 need no further changes from this review. After R3 is fixed and reviewed, recompute approvals at the resulting merge tip, update the runbook’s expected digests as necessary, record a green CPU suite, and follow `breceipt → g1 → haa → sim` with the existing gates.

No files were modified or saved, no GPU work was started, and no processes were signalled.

