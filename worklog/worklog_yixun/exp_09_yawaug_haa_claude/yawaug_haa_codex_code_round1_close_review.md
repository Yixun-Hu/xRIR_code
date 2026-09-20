# Codex close review — exp_09 round-1 fix cycle (exp09-yawaug 0cecf8e..9e41c65)

- Reviewer: OpenAI Codex `gpt-6-astra`, `reasoning effort: xhigh` (CLI override), sandbox read-only, worktree at 9e41c65
- Prompt: exp_06 briefing + `review_prompts/codex_code_round1_close_prompt.md`; run 2026-09-20 00:49:27 → 01:01:35, log `yawaug_haa_2026-09-20_00:49:27_codex_code_round1_close_review.log`, Codex session 01a0bd25-d3a0-70d1-8241-2879d3e50cc7

## Final message

**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-20.

**Verdict: approve with changes.** F1–F3 are closed, and the exp_09 source changes are approved for merge. Launch readiness has one additional outstanding condition: the current environment now produces a **third exp_07 approval-guard failure**, reproduced on `main`, beyond the two permitted by this briefing.

Reviewed branch `exp09-yawaug`, HEAD `9e41c657a2a2e236f42e2aba26291f6db48c4ce0`, including every changed line in `0cecf8e..9e41c657`. No files were saved or modified; no GPU work or process signalling was performed.

1. **F1 — previous blocker: closed.**  
   Locations: [tools/exp06_finalize.py:1170](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1170), [tests/test_exp09_sim_eval_closures.py:51](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp09_sim_eval_closures.py:51).

   `git diff main -- tools/exp06_approvals_api.py` is empty. The resolver now lives in the finalizer, which is absent from all four `eval`, `eval_launch`, `compare`, and `mirror_probe` closures.

   All ten real simulated-evaluation manifests passed `check_route` and source revalidation against both the current approvals and an in-memory proposed re-fill containing exactly the three changed keys. The new regression covers closure membership, digest preservation, and the retained manifests’ source identities. No further fix is required.

2. **F2 — previous should-fix before launch: closed.**  
   Location: [tools/exp06_finalize.py:1203](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1203).

   The loader’s returned identity is compared with `reused.exp04_approved_digests_sha256` before `checkpoints.aug` is consumed.

   I replayed the replacement-between-reads attack using the real loader with substituted, schema-valid bytes and a corresponding simulated Git binding. It was refused at the **“as parsed”** identity check. The unchanged-record control succeeded, and the real checkpoint hashes to:

   `f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299`

   The replacement regression and matching-identity control exist in [tests/test_exp09_aug_checkpoint.py:122](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp09_aug_checkpoint.py:122). No further fix is required.

3. **F3 — previous should-fix before summarising: closed.**  
   Locations: [tools/exp06_summarize_haa.py:1149](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1149) and [tools/exp06_summarize_haa.py:1378](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1378).

   The default exp_06 analysis and CLI JSON omit `experiment`; exp_09 includes it. Against the fixed pre-round base `3b7d829`, my synthetic comparison produced identical ordered keys, **63,771 bytes of serialised statistics**, and **10,683 bytes of rendered summary**. Default CLI JSON also matched byte for byte with publication I/O replaced in memory.

   The default CLI keys remain `json`, `sha256`, `summary_sha256`, `H1`, `H1b`. Base-schema and CLI regressions are present. No further fix is required.

4. **F4 — should-fix before launch: the current failure set exceeds the expressly permitted baseline.**  
   Locations: [tests/test_exp07_profiles.py:253](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp07_profiles.py:253), [exp_07 approved_digests.json:7](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/approved_digests.json:7).

   Besides `seen_simple-simple-0` and `seen_cyl-cylindrical-0`, **`seen_aug-simple-1` now fails**: `ckpt/exp07/seen_aug/final/epoch_012.pth` exists, while its runtime approval remains null.

   I reproduced all three exp_07 failures in the main checkout at `32e389b853ad48784ca9d855364c04a1252a236b`. The relevant test, profile, and approval file are unchanged between the reviewed branch and `main`. This is a shared experiment-state issue, not an exp_09 code regression, but the requested “only two exp_07 guards failing” acceptance condition is currently unmet.

   **Minimal resolution:** the Planner/exp_07 owner must resolve the approval state, or obtain an explicit change to the accepted baseline failures, then record the CPU-suite result. This does not call for an exp_09 source edit.

Recomputing `exp06_profiles.compute_code_digests(wt, 'HEAD')` across all twenty keys found **exactly three moved keys**:

| Key | Digest at reviewed HEAD |
|---|---|
| `finalize` | `ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093` |
| `haa_pipeline_sh` | `4a401ee3d80f2e3c77ecf68fd538e0108d6c649c92680c1b0373824626ad8c08` |
| `summarize_haa` | `3fc1e82d374c8e5fb8130d5223b7d744168b0462dabc1d22017423b4a1ad3376` |

The other seventeen keys are unchanged, including `eval`, `eval_launch`, `compare`, `mirror_probe`, `haa_finetune`, and `haa_eval`. `summarize_haa` is the only additional approval key whose closure imports the finalizer.

**Independent verification performed:**

- Read the review briefings, approved plans and relevant experiment records, prior findings, fix prompt, Coder report, and complete fix-cycle diff.
- Ran `git diff --check 0cecf8e..HEAD`, `bash -n tools/exp06_haa_pipeline.sh`, and in-memory compilation of all eight changed Python files: passed.
- Checked all nineteen branch commits: each is at most 200 added-plus-deleted lines; maximum **186**. Every commit carries both required trailers.
- Checked forbidden/pinned files against `main`: unchanged. The exp_03 exact twelve-file closure regression passed; all eleven files in the training import closure are byte-identical to `main`.
- Confirmed the modified-file list contains only four existing exp_06 test files and `exp06_finalize.py`, `exp06_haa_pipeline.sh`, and `exp06_summarize_haa.py`; the two exp_09 regression modules are additions.
- Replayed all ten simulated-evaluation routes and source checks under current and proposed approvals: **10/10 passed in each mode**, with 690 source-file checks per mode.
- Ran `verify_job(..., sensitivity=False, inputs={})` using the reviewed modules against the recorded main-tree paths: **12/12 historical HAA jobs and 93/93 children admitted**. Current approvals remained unchanged; historical approval identities were preserved. Wrong-child-closure and unapproved-historical-closure probes were refused.
- Compared base-versus-tip exp_06 dry-run queues byte for byte. Yawaug seed, zero-shot, complete-queue and mixed-frame goldens passed: **31 children and four jobs**, correct exp_09 paths, room frame, and no child heading argument.
- Replayed frame/init refusals: room specs or children carrying headings, heading specs without headings, wrong per-sample frame, wrong pipeline checkpoint, wrong reused approval identity, and wrong summary-job init were rejected. Executing `JOB_SPEC_PY` in memory produced a heading-free room spec with the actual approved checkpoint digest.
- Rechecked E1 strict zero/margin boundaries, independent harm and non-inferiority conclusions, and void/unconverged suppression. E2 retained eleven cells and `0.05/11`; exp_09 analysis retained A/B/C/E, eleven E3 cells and 44 side-split cells. Missing yawaug was refused. Python, NumPy and Torch global RNG states were unchanged.

All Python checks used the specified interpreter, CPU-only environment and thread limits. Read-only accommodations disabled cache writes, used memory-backed temporary approval copies, and replaced signal-based liveness queries with `/proc` reads.

The pytest replays, invoked through `pytest.main` with `-q -s -p no:cacheprovider`, were:

| Selection | Independent result |
|---|---|
| `test_exp09_sim_eval_closures.py` plus the exp_03 closure regression | **4 passed** |
| Read-only-compatible cases from approvals, finalizer, HAA consumer, pipeline, summariser and aug-checkpoint modules | **94 passed, 458 deselected** |
| exp_06 drift guard plus exp_07 runtime-approval parametrisation | **4 failed**: expected three-key exp_06 drift, plus all three exp_07 guards |
| exp_07 runtime-approval parametrisation in the main checkout | **3 failed**, identically |

The initial unadapted test attempt encountered sandbox cache/temp-write failures. I did not replay the full write-dependent suite.

**Coder-reported results, distinguished from my replay:** the full suite at `db4121e` reported **3 failed, 3263 passed, 50 skipped**; the final-tip four-module rerun reported **494 passed**. Separate reported counts were summariser 98, aug checkpoint 8, simulated closures 3, and approvals API 50. My current guard replay establishes the additional exp_07 failure described in F4; it does not reproduce the report’s earlier failure set.

**Exact blocking list:** no remaining Coder implementation fixes for F1–F3. **F4 is the sole additional outstanding launch-readiness finding**, owned by the Planner/exp_07 coordination.

After merging the reviewed tip, the Planner must still recompute and re-fill exactly the three moved keys at the merge tip, record an acceptable CPU-suite result after resolving F4, and complete the bounded two-epoch class_room yawaug fine-tune plus four-sample hallway evaluation. Both exploratory receipts must succeed, with readback confirming room frame, null headings and correct checkpoint lineage. Once those conditions pass, the reviewed implementation is suitable for the exp_09 HAA launch and exp_09 summariser.
tokens used
332,548
**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-20.

**Verdict: approve with changes.** F1–F3 are closed, and the exp_09 source changes are approved for merge. Launch readiness has one additional outstanding condition: the current environment now produces a **third exp_07 approval-guard failure**, reproduced on `main`, beyond the two permitted by this briefing.

Reviewed branch `exp09-yawaug`, HEAD `9e41c657a2a2e236f42e2aba26291f6db48c4ce0`, including every changed line in `0cecf8e..9e41c657`. No files were saved or modified; no GPU work or process signalling was performed.

1. **F1 — previous blocker: closed.**  
   Locations: [tools/exp06_finalize.py:1170](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1170), [tests/test_exp09_sim_eval_closures.py:51](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp09_sim_eval_closures.py:51).

   `git diff main -- tools/exp06_approvals_api.py` is empty. The resolver now lives in the finalizer, which is absent from all four `eval`, `eval_launch`, `compare`, and `mirror_probe` closures.

   All ten real simulated-evaluation manifests passed `check_route` and source revalidation against both the current approvals and an in-memory proposed re-fill containing exactly the three changed keys. The new regression covers closure membership, digest preservation, and the retained manifests’ source identities. No further fix is required.

2. **F2 — previous should-fix before launch: closed.**  
   Location: [tools/exp06_finalize.py:1203](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1203).

   The loader’s returned identity is compared with `reused.exp04_approved_digests_sha256` before `checkpoints.aug` is consumed.

   I replayed the replacement-between-reads attack using the real loader with substituted, schema-valid bytes and a corresponding simulated Git binding. It was refused at the **“as parsed”** identity check. The unchanged-record control succeeded, and the real checkpoint hashes to:

   `f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299`

   The replacement regression and matching-identity control exist in [tests/test_exp09_aug_checkpoint.py:122](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp09_aug_checkpoint.py:122). No further fix is required.

3. **F3 — previous should-fix before summarising: closed.**  
   Locations: [tools/exp06_summarize_haa.py:1149](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1149) and [tools/exp06_summarize_haa.py:1378](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1378).

   The default exp_06 analysis and CLI JSON omit `experiment`; exp_09 includes it. Against the fixed pre-round base `3b7d829`, my synthetic comparison produced identical ordered keys, **63,771 bytes of serialised statistics**, and **10,683 bytes of rendered summary**. Default CLI JSON also matched byte for byte with publication I/O replaced in memory.

   The default CLI keys remain `json`, `sha256`, `summary_sha256`, `H1`, `H1b`. Base-schema and CLI regressions are present. No further fix is required.

4. **F4 — should-fix before launch: the current failure set exceeds the expressly permitted baseline.**  
   Locations: [tests/test_exp07_profiles.py:253](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp07_profiles.py:253), [exp_07 approved_digests.json:7](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/approved_digests.json:7).

   Besides `seen_simple-simple-0` and `seen_cyl-cylindrical-0`, **`seen_aug-simple-1` now fails**: `ckpt/exp07/seen_aug/final/epoch_012.pth` exists, while its runtime approval remains null.

   I reproduced all three exp_07 failures in the main checkout at `32e389b853ad48784ca9d855364c04a1252a236b`. The relevant test, profile, and approval file are unchanged between the reviewed branch and `main`. This is a shared experiment-state issue, not an exp_09 code regression, but the requested “only two exp_07 guards failing” acceptance condition is currently unmet.

   **Minimal resolution:** the Planner/exp_07 owner must resolve the approval state, or obtain an explicit change to the accepted baseline failures, then record the CPU-suite result. This does not call for an exp_09 source edit.

Recomputing `exp06_profiles.compute_code_digests(wt, 'HEAD')` across all twenty keys found **exactly three moved keys**:

| Key | Digest at reviewed HEAD |
|---|---|
| `finalize` | `ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093` |
| `haa_pipeline_sh` | `4a401ee3d80f2e3c77ecf68fd538e0108d6c649c92680c1b0373824626ad8c08` |
| `summarize_haa` | `3fc1e82d374c8e5fb8130d5223b7d744168b0462dabc1d22017423b4a1ad3376` |

The other seventeen keys are unchanged, including `eval`, `eval_launch`, `compare`, `mirror_probe`, `haa_finetune`, and `haa_eval`. `summarize_haa` is the only additional approval key whose closure imports the finalizer.

**Independent verification performed:**

- Read the review briefings, approved plans and relevant experiment records, prior findings, fix prompt, Coder report, and complete fix-cycle diff.
- Ran `git diff --check 0cecf8e..HEAD`, `bash -n tools/exp06_haa_pipeline.sh`, and in-memory compilation of all eight changed Python files: passed.
- Checked all nineteen branch commits: each is at most 200 added-plus-deleted lines; maximum **186**. Every commit carries both required trailers.
- Checked forbidden/pinned files against `main`: unchanged. The exp_03 exact twelve-file closure regression passed; all eleven files in the training import closure are byte-identical to `main`.
- Confirmed the modified-file list contains only four existing exp_06 test files and `exp06_finalize.py`, `exp06_haa_pipeline.sh`, and `exp06_summarize_haa.py`; the two exp_09 regression modules are additions.
- Replayed all ten simulated-evaluation routes and source checks under current and proposed approvals: **10/10 passed in each mode**, with 690 source-file checks per mode.
- Ran `verify_job(..., sensitivity=False, inputs={})` using the reviewed modules against the recorded main-tree paths: **12/12 historical HAA jobs and 93/93 children admitted**. Current approvals remained unchanged; historical approval identities were preserved. Wrong-child-closure and unapproved-historical-closure probes were refused.
- Compared base-versus-tip exp_06 dry-run queues byte for byte. Yawaug seed, zero-shot, complete-queue and mixed-frame goldens passed: **31 children and four jobs**, correct exp_09 paths, room frame, and no child heading argument.
- Replayed frame/init refusals: room specs or children carrying headings, heading specs without headings, wrong per-sample frame, wrong pipeline checkpoint, wrong reused approval identity, and wrong summary-job init were rejected. Executing `JOB_SPEC_PY` in memory produced a heading-free room spec with the actual approved checkpoint digest.
- Rechecked E1 strict zero/margin boundaries, independent harm and non-inferiority conclusions, and void/unconverged suppression. E2 retained eleven cells and `0.05/11`; exp_09 analysis retained A/B/C/E, eleven E3 cells and 44 side-split cells. Missing yawaug was refused. Python, NumPy and Torch global RNG states were unchanged.

All Python checks used the specified interpreter, CPU-only environment and thread limits. Read-only accommodations disabled cache writes, used memory-backed temporary approval copies, and replaced signal-based liveness queries with `/proc` reads.

The pytest replays, invoked through `pytest.main` with `-q -s -p no:cacheprovider`, were:

| Selection | Independent result |
|---|---|
| `test_exp09_sim_eval_closures.py` plus the exp_03 closure regression | **4 passed** |
| Read-only-compatible cases from approvals, finalizer, HAA consumer, pipeline, summariser and aug-checkpoint modules | **94 passed, 458 deselected** |
| exp_06 drift guard plus exp_07 runtime-approval parametrisation | **4 failed**: expected three-key exp_06 drift, plus all three exp_07 guards |
| exp_07 runtime-approval parametrisation in the main checkout | **3 failed**, identically |

The initial unadapted test attempt encountered sandbox cache/temp-write failures. I did not replay the full write-dependent suite.

**Coder-reported results, distinguished from my replay:** the full suite at `db4121e` reported **3 failed, 3263 passed, 50 skipped**; the final-tip four-module rerun reported **494 passed**. Separate reported counts were summariser 98, aug checkpoint 8, simulated closures 3, and approvals API 50. My current guard replay establishes the additional exp_07 failure described in F4; it does not reproduce the report’s earlier failure set.

**Exact blocking list:** no remaining Coder implementation fixes for F1–F3. **F4 is the sole additional outstanding launch-readiness finding**, owned by the Planner/exp_07 coordination.

After merging the reviewed tip, the Planner must still recompute and re-fill exactly the three moved keys at the merge tip, record an acceptable CPU-suite result after resolving F4, and complete the bounded two-epoch class_room yawaug fine-tune plus four-sample hallway evaluation. Both exploratory receipts must succeed, with readback confirming room frame, null headings and correct checkpoint lineage. Once those conditions pass, the reviewed implementation is suitable for the exp_09 HAA launch and exp_09 summariser.

