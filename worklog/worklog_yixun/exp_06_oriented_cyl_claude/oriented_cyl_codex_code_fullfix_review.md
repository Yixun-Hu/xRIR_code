# Codex review — exp_06 fix round (full-review F2–F5) on exp06-fullfix ac05dd9

- Reviewer: OpenAI Codex `gpt-6-astra`, `reasoning effort: xhigh` (CLI override), sandbox read-only, worktree `/home/yixunhu/codespace/xRIR_code_wt` branch `exp06-fullfix` at ac05dd9 (main ddfcb66)
- Prompt: `review_prompts/codex_code_review_briefing.md` + `review_prompts/codex_code_fullfix_prompt.md`
- Run: 2026-09-18 00:17:52 → 00:31:57 local, log `oriented_cyl_2026-09-18_00:17:52_codex_code_fullfix_review.log`, Codex session 01a0b2bc-3421-7130-b91e-27364b93c123

## Final message

**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-18.

Reviewed `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-fullfix`, HEAD `ac05dd9096034ed3af337e58dcdcc9f358ed8a05`, against `ddfcb66`. Verification completed at **04:30 UTC**. No files modified, GPU work started, or processes signaled.

**Verdict: request changes.** F4 and F5 are closed. F2 and F3 have substantial improvements but retain reproducible admission gaps. Re-filling approvals and recording a green suite are insufficient to authorize the confirmatory launch until findings **R1–R3** below are fixed.

1. **R1 — blocker — Producer approvals authenticate committed blobs without requiring the executing source bytes to match.**  
   [tools/exp06_approvals_api.py:309](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_approvals_api.py:309)

   `compute_code_digests(repo, commit, keys)` returns digests derived from **reviewed Git blobs**. Its closure records also contain working-tree hashes, but `enforce_producer` discards those records and never checks their equality.

   **Evidence:** With the actual committed approvals and real checkpoint/headings, `enforce_producer('haa_children', repo, 'ddfcb66', …)` returned `admissibility='confirmatory'`, `deviations=[]`, although the returned closure records identified changed working-tree bytes in `tools/exp06_haa_pipeline.sh` and `tools/exp06_finalize.py`. A separate in-memory probe supplied otherwise matching HEAD approvals and simulated a changed working-tree hash for `exp06_mirror_probe.py`; the mirror gate likewise returned **confirmatory with no deviations**. The comparison logic itself was not mocked.

   This leaves the F3 promise—approval of the code about to execute—unenforced. The mirror probe’s subsequent source capture records this difference but does not reject it.

   **Minimal fix:** Inspect each required key’s closure records, reject missing reviewed blobs and working-tree/reviewed-blob mismatches, and name the key and affected paths. Preserve those deviations in exploratory mode. Add regressions for both an older approval commit and dirty source at HEAD.

2. **R2 — blocker — The baseline launcher does not enforce arm B’s registered checkpoint hash.**  
   [tools/exp06_eval_launch.py:125](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_eval_launch.py:125)

   `training_evidence()` calls the shared checker without `registered_sha256`. Consequently, the receipt verifier’s registered-checkpoint comparison is skipped. The approvals gate also intentionally receives no checkpoint for baseline runs.

   **Evidence:** I constructed, entirely in memory, a valid receipt from the real **SimpleViT control** training directory. The launcher’s baseline evidence check accepted its checkpoint, `651e3a37…`. Calling the comparer’s shared rule with the registered CYL hash correctly refused it, naming the expected `8ba344ad…`.

   Thus launcher and comparer do not enforce the same admission rule, contrary to the Coder report’s §4(b). After approvals are re-filled, this mismatch can reach execution and create an unusable run before the comparer rejects it.

   **Minimal fix:** Pass `exp04_profiles.CYL['sha256']` for baseline evidence at launch, preferably through a shared role-to-registration rule. Test that the wrong registered arm is refused **before `execute_run`**, with no output created.

3. **R3 — blocker — Receipt verification accepts incomplete retained evidence and a dirty producer receipt.**  
   [tools/exp06_legacy_train_receipt.py:211](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_legacy_train_receipt.py:211)

   `verify()` checks the receipt’s self-declared enumeration and hashes its listed files. It does not require the writer’s complete membership, independently revalidate the args/history contract, or reject `source_closure.drift`.

   **Evidence:** Starting from a real, clean, **18-artifact** cylindrical receipt, I retained only `args.json` and `epoch_12.pth` and recomputed `files_sha256`. Both launcher validation and the comparer’s registered-CYL evidence rule accepted this **two-file receipt**. Both also accepted an otherwise intact receipt declaring producer-source drift.

   Because this receipt is not independently approved by hash, its self-declared enumeration cannot establish complete retained evidence. The dirty-receipt acceptance also contradicts `--allow-dirty`’s “never confirmatory” contract.

   **Minimal fix:** Derive and check the expected artifact membership during verification, including required files, retained epoch checkpoints and present optional artifacts; revalidate args/history/registered-arm consistency; reject diagnostic or dirty producer evidence. Continue routing artifact hashes through the comparer’s supplied binder. Add reader-side regressions for omitted artifacts and dirty receipts.

4. **R4 — should-fix before H3 — The integration test does not admit the manifest built by the launcher.**  
   [tests/test_exp06_sim_evidence.py:126](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_sim_evidence.py:126)

   The test calls `launcher.build_fields()`, then independently creates another manifest through `write_run()`. It compares only `mutable_inputs` before admitting that second manifest. A launcher/comparer incompatibility elsewhere can therefore remain hidden.

   The latest runbook also explicitly supplies `--approved` and `--approved-commit`; the test helper currently exercises their defaults.

   **Minimal fix:** Persist the actual launcher-built fields, construct matching synthetic outputs/completion around those fields, and pass that directory through `admit_run` for both arms. Include the current runbook’s explicit approval arguments. This is the remaining test-composition gap in F2(d).

5. **R5 — should-fix; preferably include in this pre-launch round — `HEADING_COMPLETE` can be bypassed by omitting `headings`.**  
   [tools/exp06_approvals_api.py:325](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_approvals_api.py:325)

   Coverage enforcement is inside `if headings is not None`. With matching approvals, `haa_children` rejects `headings={}` but accepts `headings=None` as confirmatory.

   The current pipeline supplies all four headings, so this does not block its present invocation. It does contradict the shared API’s documented mandatory-coverage rule.

   **Minimal fix:** Treat omitted headings as an empty mapping for producers in `HEADING_COMPLETE`, and test both cases. Including this now avoids another shared-source digest change after runs exist.

The **arm B design is sound in principle**: `train_completion` can be a reconstructed receipt, `train_manifest` can be exp_01’s enumerated `args.json`, and the registered checkpoint plus independently validated retained evidence can provide authority without a new approvals-schema key. The implementation needs R2 and R3 to meet that design.

The receipt writer does **not** call `enforce_producer`. Its gates are the reviewed exp_01 registration, checkpoint placement/name, args/history/backbone/epoch checks, and—by default—its source closure matching HEAD. Its publication uses exclusive creation. The two new modules are covered by the launcher and comparer closures.

I verified the following:

- **Briefing and source review:** Read the SOP, relevant plan sections and changelogs, three plan reviews, Fable review, prior integrative F1–F5 review, notebook and role change, round prompt/report, exp_02/exp_03 records, composed tooling, and every changed line in `ddfcb66..ac05dd9`.
- **Scope/static checks:** `git diff main..HEAD --stat` confirms **16 files, +2092/−53**. `git diff --check main..HEAD` passed. In-memory `compile()` passed for all **15 changed Python files**. `bash -n` passed for the changed HAA pipeline and the current runbook. Final worktree status remained clean.
- **Pins:** Both requested protected-path diffs were empty. The modified-file list contains only earlier exp_06 files. Exp_03’s **12-file** closure remains `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`. The trainer’s **11-file import closure** is unchanged against main. The approvals template and record copy are unchanged.
- **Pin regression:**  
  `python -m pytest tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged -q --capture=sys -p no:cacheprovider`  
  Result: **1 passed**.
- **Exp_06 tests:** Two adapted `pytest.main(..., ['-q', '--capture=sys', '-p', 'no:cacheprovider'])` batches covered all `tests/test_exp06*.py`, excluding tests requiring temporary-file or FD-capture fixtures. Combined result: **423 passed, 6 skipped, 1 expected approvals-drift failure, 853 deselected**. The earlier approvals-only run was **41 passed, 4 deselected**, overlapping this total.
- **Sandbox limitation:** Ordinary pytest with `TMPDIR=<worktree>/.tmp` failed before collection with **“No usable temporary directory found.”** `/tmp` and the other fallback locations were also unavailable. Adapted runs used `NUMBA_DISABLE_JIT=1` and an in-memory Matplotlib cache-discovery shim to read the existing cache. They did not replace repository admission predicates. The writable-fixture tests, including the complete new integration/finalization tests, and the full suite were **not independently replayed**. The Coder’s **3066 passed / 50 skipped / 1 expected failure** remains a reported result, not my verification.
- **F2, arm C:** The real full-run completion/provenance/checkpoint passed. Nine in-memory corruptions covering run type, admissibility, diagnostic/exploratory status, epoch count, artifact/checkpoint hashes, provenance binding and reviewed commit were refused. Missing training bindings and evaluation-record masquerading were refused for both roles with **zero `execute_run` calls**.
- **F3 positive coverage:** Wrong well-formed code, checkpoint and heading digests were refused with `code.encoder`, `artifacts.epoch_012` and `artifacts.heading.hallway` named. Exploratory mode retained multiple deviations and labeled the receipt diagnostic. Both HAA child approval checks accepted their actual closures and refused altered closure hashes with the correct key named. The producer map and four wiring locations match the intended responsibilities; R1 remains the executing-byte gap.
- **F4:** The real `ckpt/yaw_aug/eval/cyl_k8_seed42_k0` was refused as B for route `exp04`. All five real control runs, seeds **42–46**, remained admitted as A. Exp_06 requires explicit role/epoch evidence; exp_05 establishes epoch through its registration.
- **F5:** Identical arms produced **“not converged”** for both contrasts and metrics, recording attempts at **200 and 800 draws**. A separate failed-then-successful probe made exactly one retry at **20→80 draws**. Python and NumPy global RNG states were unchanged. `tools/paired_compare.py` is untouched.
- **Architecture regression:** The passing CPU subset includes parameter-delta, active-yaw, joint heading-cancellation, mirror-discrimination, factory identity and trainer-state bit-exactness checks. GPU checks remained skipped.

Recomputing `exp06_profiles.compute_code_digests(wt, 'HEAD')` gave exactly the requested six changed keys; the other fourteen matched:

| Key | Recomputed SHA-256 |
|---|---|
| `compare` | `1de772a0690fb01b8b045c25a1b144f32973f995e36204a4b1bd9ac3f5202e93` |
| `eval_launch` | `e1d685870818df3b2a3fe6aa60ee90db16b75a7f961f4015d46ab540dfa5f067` |
| `finalize` | `1edbaf2caaffc6ba4e65d6a7fc6bda119f7e4597523f0072bb323c0af7af6b18` |
| `haa_pipeline_sh` | `2ad3e562c1e322d3a035e2aed3bb28d55c940d20e0e9516d88c242a7df0055a7` |
| `mirror_probe` | `df41d6b6903125bbd35fb377dc44b160f9023fd761e60a66167bf0c8931cd91f` |
| `summarize_haa` | `4538e1129d6f5e6da3037f1e6586f899a08d4c1848d1d138548d9929bb10e595` |

The drift-guard failure corresponds to exactly these six keys.

**Process assessment, reporting-only:** The three F2 commits contain **516/361/343 changed lines**, including tests; `fd00c2b` also exceeds 200 at **221**. Their non-test totals are **278/193/27/89** respectively. The SOP says “generally <200 changed lines of code”; the round prompt imposes the stricter total-line limit. The split remains reviewable, and I found no unrelated or protected-file changes hidden in it. Record the exception accurately; history rewriting is not necessary for this review.

The latest runbook I read already contains F1’s parent-directory creation, `refill2`, `breceipt`, both training bindings, explicit approval arguments, and the `EXP06_F2_LANDED` gate. These are present preparations, not evidence of a successful launch.

**Exact blocking findings the Coder must fix before this round closes: R1, R2 and R3.** Fix and regression-test them before confirmatory G1/HAA/sim execution, then recompute approvals at the actual resulting merge tip and record a green suite. The digest values above describe `ac05dd9`; they must be recomputed after fixes. Address R4 before H3 reporting, preferably before launch; bundle R5 into the present source-fix round. F4, F5 and the commit-size exception introduce no additional launch blockers.
tokens used
427,610
**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-18.

Reviewed `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-fullfix`, HEAD `ac05dd9096034ed3af337e58dcdcc9f358ed8a05`, against `ddfcb66`. Verification completed at **04:30 UTC**. No files modified, GPU work started, or processes signaled.

**Verdict: request changes.** F4 and F5 are closed. F2 and F3 have substantial improvements but retain reproducible admission gaps. Re-filling approvals and recording a green suite are insufficient to authorize the confirmatory launch until findings **R1–R3** below are fixed.

1. **R1 — blocker — Producer approvals authenticate committed blobs without requiring the executing source bytes to match.**  
   [tools/exp06_approvals_api.py:309](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_approvals_api.py:309)

   `compute_code_digests(repo, commit, keys)` returns digests derived from **reviewed Git blobs**. Its closure records also contain working-tree hashes, but `enforce_producer` discards those records and never checks their equality.

   **Evidence:** With the actual committed approvals and real checkpoint/headings, `enforce_producer('haa_children', repo, 'ddfcb66', …)` returned `admissibility='confirmatory'`, `deviations=[]`, although the returned closure records identified changed working-tree bytes in `tools/exp06_haa_pipeline.sh` and `tools/exp06_finalize.py`. A separate in-memory probe supplied otherwise matching HEAD approvals and simulated a changed working-tree hash for `exp06_mirror_probe.py`; the mirror gate likewise returned **confirmatory with no deviations**. The comparison logic itself was not mocked.

   This leaves the F3 promise—approval of the code about to execute—unenforced. The mirror probe’s subsequent source capture records this difference but does not reject it.

   **Minimal fix:** Inspect each required key’s closure records, reject missing reviewed blobs and working-tree/reviewed-blob mismatches, and name the key and affected paths. Preserve those deviations in exploratory mode. Add regressions for both an older approval commit and dirty source at HEAD.

2. **R2 — blocker — The baseline launcher does not enforce arm B’s registered checkpoint hash.**  
   [tools/exp06_eval_launch.py:125](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_eval_launch.py:125)

   `training_evidence()` calls the shared checker without `registered_sha256`. Consequently, the receipt verifier’s registered-checkpoint comparison is skipped. The approvals gate also intentionally receives no checkpoint for baseline runs.

   **Evidence:** I constructed, entirely in memory, a valid receipt from the real **SimpleViT control** training directory. The launcher’s baseline evidence check accepted its checkpoint, `651e3a37…`. Calling the comparer’s shared rule with the registered CYL hash correctly refused it, naming the expected `8ba344ad…`.

   Thus launcher and comparer do not enforce the same admission rule, contrary to the Coder report’s §4(b). After approvals are re-filled, this mismatch can reach execution and create an unusable run before the comparer rejects it.

   **Minimal fix:** Pass `exp04_profiles.CYL['sha256']` for baseline evidence at launch, preferably through a shared role-to-registration rule. Test that the wrong registered arm is refused **before `execute_run`**, with no output created.

3. **R3 — blocker — Receipt verification accepts incomplete retained evidence and a dirty producer receipt.**  
   [tools/exp06_legacy_train_receipt.py:211](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_legacy_train_receipt.py:211)

   `verify()` checks the receipt’s self-declared enumeration and hashes its listed files. It does not require the writer’s complete membership, independently revalidate the args/history contract, or reject `source_closure.drift`.

   **Evidence:** Starting from a real, clean, **18-artifact** cylindrical receipt, I retained only `args.json` and `epoch_12.pth` and recomputed `files_sha256`. Both launcher validation and the comparer’s registered-CYL evidence rule accepted this **two-file receipt**. Both also accepted an otherwise intact receipt declaring producer-source drift.

   Because this receipt is not independently approved by hash, its self-declared enumeration cannot establish complete retained evidence. The dirty-receipt acceptance also contradicts `--allow-dirty`’s “never confirmatory” contract.

   **Minimal fix:** Derive and check the expected artifact membership during verification, including required files, retained epoch checkpoints and present optional artifacts; revalidate args/history/registered-arm consistency; reject diagnostic or dirty producer evidence. Continue routing artifact hashes through the comparer’s supplied binder. Add reader-side regressions for omitted artifacts and dirty receipts.

4. **R4 — should-fix before H3 — The integration test does not admit the manifest built by the launcher.**  
   [tests/test_exp06_sim_evidence.py:126](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_sim_evidence.py:126)

   The test calls `launcher.build_fields()`, then independently creates another manifest through `write_run()`. It compares only `mutable_inputs` before admitting that second manifest. A launcher/comparer incompatibility elsewhere can therefore remain hidden.

   The latest runbook also explicitly supplies `--approved` and `--approved-commit`; the test helper currently exercises their defaults.

   **Minimal fix:** Persist the actual launcher-built fields, construct matching synthetic outputs/completion around those fields, and pass that directory through `admit_run` for both arms. Include the current runbook’s explicit approval arguments. This is the remaining test-composition gap in F2(d).

5. **R5 — should-fix; preferably include in this pre-launch round — `HEADING_COMPLETE` can be bypassed by omitting `headings`.**  
   [tools/exp06_approvals_api.py:325](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_approvals_api.py:325)

   Coverage enforcement is inside `if headings is not None`. With matching approvals, `haa_children` rejects `headings={}` but accepts `headings=None` as confirmatory.

   The current pipeline supplies all four headings, so this does not block its present invocation. It does contradict the shared API’s documented mandatory-coverage rule.

   **Minimal fix:** Treat omitted headings as an empty mapping for producers in `HEADING_COMPLETE`, and test both cases. Including this now avoids another shared-source digest change after runs exist.

The **arm B design is sound in principle**: `train_completion` can be a reconstructed receipt, `train_manifest` can be exp_01’s enumerated `args.json`, and the registered checkpoint plus independently validated retained evidence can provide authority without a new approvals-schema key. The implementation needs R2 and R3 to meet that design.

The receipt writer does **not** call `enforce_producer`. Its gates are the reviewed exp_01 registration, checkpoint placement/name, args/history/backbone/epoch checks, and—by default—its source closure matching HEAD. Its publication uses exclusive creation. The two new modules are covered by the launcher and comparer closures.

I verified the following:

- **Briefing and source review:** Read the SOP, relevant plan sections and changelogs, three plan reviews, Fable review, prior integrative F1–F5 review, notebook and role change, round prompt/report, exp_02/exp_03 records, composed tooling, and every changed line in `ddfcb66..ac05dd9`.
- **Scope/static checks:** `git diff main..HEAD --stat` confirms **16 files, +2092/−53**. `git diff --check main..HEAD` passed. In-memory `compile()` passed for all **15 changed Python files**. `bash -n` passed for the changed HAA pipeline and the current runbook. Final worktree status remained clean.
- **Pins:** Both requested protected-path diffs were empty. The modified-file list contains only earlier exp_06 files. Exp_03’s **12-file** closure remains `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`. The trainer’s **11-file import closure** is unchanged against main. The approvals template and record copy are unchanged.
- **Pin regression:**  
  `python -m pytest tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged -q --capture=sys -p no:cacheprovider`  
  Result: **1 passed**.
- **Exp_06 tests:** Two adapted `pytest.main(..., ['-q', '--capture=sys', '-p', 'no:cacheprovider'])` batches covered all `tests/test_exp06*.py`, excluding tests requiring temporary-file or FD-capture fixtures. Combined result: **423 passed, 6 skipped, 1 expected approvals-drift failure, 853 deselected**. The earlier approvals-only run was **41 passed, 4 deselected**, overlapping this total.
- **Sandbox limitation:** Ordinary pytest with `TMPDIR=<worktree>/.tmp` failed before collection with **“No usable temporary directory found.”** `/tmp` and the other fallback locations were also unavailable. Adapted runs used `NUMBA_DISABLE_JIT=1` and an in-memory Matplotlib cache-discovery shim to read the existing cache. They did not replace repository admission predicates. The writable-fixture tests, including the complete new integration/finalization tests, and the full suite were **not independently replayed**. The Coder’s **3066 passed / 50 skipped / 1 expected failure** remains a reported result, not my verification.
- **F2, arm C:** The real full-run completion/provenance/checkpoint passed. Nine in-memory corruptions covering run type, admissibility, diagnostic/exploratory status, epoch count, artifact/checkpoint hashes, provenance binding and reviewed commit were refused. Missing training bindings and evaluation-record masquerading were refused for both roles with **zero `execute_run` calls**.
- **F3 positive coverage:** Wrong well-formed code, checkpoint and heading digests were refused with `code.encoder`, `artifacts.epoch_012` and `artifacts.heading.hallway` named. Exploratory mode retained multiple deviations and labeled the receipt diagnostic. Both HAA child approval checks accepted their actual closures and refused altered closure hashes with the correct key named. The producer map and four wiring locations match the intended responsibilities; R1 remains the executing-byte gap.
- **F4:** The real `ckpt/yaw_aug/eval/cyl_k8_seed42_k0` was refused as B for route `exp04`. All five real control runs, seeds **42–46**, remained admitted as A. Exp_06 requires explicit role/epoch evidence; exp_05 establishes epoch through its registration.
- **F5:** Identical arms produced **“not converged”** for both contrasts and metrics, recording attempts at **200 and 800 draws**. A separate failed-then-successful probe made exactly one retry at **20→80 draws**. Python and NumPy global RNG states were unchanged. `tools/paired_compare.py` is untouched.
- **Architecture regression:** The passing CPU subset includes parameter-delta, active-yaw, joint heading-cancellation, mirror-discrimination, factory identity and trainer-state bit-exactness checks. GPU checks remained skipped.

Recomputing `exp06_profiles.compute_code_digests(wt, 'HEAD')` gave exactly the requested six changed keys; the other fourteen matched:

| Key | Recomputed SHA-256 |
|---|---|
| `compare` | `1de772a0690fb01b8b045c25a1b144f32973f995e36204a4b1bd9ac3f5202e93` |
| `eval_launch` | `e1d685870818df3b2a3fe6aa60ee90db16b75a7f961f4015d46ab540dfa5f067` |
| `finalize` | `1edbaf2caaffc6ba4e65d6a7fc6bda119f7e4597523f0072bb323c0af7af6b18` |
| `haa_pipeline_sh` | `2ad3e562c1e322d3a035e2aed3bb28d55c940d20e0e9516d88c242a7df0055a7` |
| `mirror_probe` | `df41d6b6903125bbd35fb377dc44b160f9023fd761e60a66167bf0c8931cd91f` |
| `summarize_haa` | `4538e1129d6f5e6da3037f1e6586f899a08d4c1848d1d138548d9929bb10e595` |

The drift-guard failure corresponds to exactly these six keys.

**Process assessment, reporting-only:** The three F2 commits contain **516/361/343 changed lines**, including tests; `fd00c2b` also exceeds 200 at **221**. Their non-test totals are **278/193/27/89** respectively. The SOP says “generally <200 changed lines of code”; the round prompt imposes the stricter total-line limit. The split remains reviewable, and I found no unrelated or protected-file changes hidden in it. Record the exception accurately; history rewriting is not necessary for this review.

The latest runbook I read already contains F1’s parent-directory creation, `refill2`, `breceipt`, both training bindings, explicit approval arguments, and the `EXP06_F2_LANDED` gate. These are present preparations, not evidence of a successful launch.

**Exact blocking findings the Coder must fix before this round closes: R1, R2 and R3.** Fix and regression-test them before confirmatory G1/HAA/sim execution, then recompute approvals at the actual resulting merge tip and record a green suite. The digest values above describe `ac05dd9`; they must be recomputed after fixes. Address R4 before H3 reporting, preferably before launch; bundle R5 into the present source-fix round. F4, F5 and the commit-size exception introduce no additional launch blockers.

