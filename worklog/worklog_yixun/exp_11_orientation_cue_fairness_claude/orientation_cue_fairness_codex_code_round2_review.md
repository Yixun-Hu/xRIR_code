**Reviewer:** OpenAI Codex, GPT-6, API session. Exact model identifier and reasoning-effort setting are not exposed in the available invocation metadata; I cannot attest to `gpt-6-astra` / `ultra`.  
**Sandbox/resources:** Read-only repository; authorized temporary/cache writes only under `/tmp`; CUDA disabled; CPU concurrency ≤4; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, clean before and after review.  
**Range:** `c491396..c8acda24137c6b14c7e9fb2dc14be770261de6bc`, 42 commits.  
**Date/time:** 2026-09-27, approximately 02:08–02:15 EDT.

**Verdict: request changes.** Five blockers remain. The model implementation and numerical orchestration look sound, but adapter zero-shot completion fails, and several admission and launcher safeguards are incomplete.

1. **[P1] J/K zero-shot evaluations cannot finalize.**  
   [tools/exp11_finalize.py:540](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:540)

   The pipeline supplies the historical A/E checkpoint for adapter zero-shot evaluation. Evaluation correctly uses the strict base-loading mode, but finalization unconditionally demands the complete `simple_adapter` state.

   A CPU check using the actual control checkpoint reproduced:

   ```text
   266 parameters are not the 268 of build_xrir_exp11('simple_adapter', 8)
   ```

   This prevents the required J/K zero-shot jobs from completing, blocking Phase 1b and final publication. Finalization must apply the same contextual loading contract: strictly validate the approved base checkpoint for adapter zero-shot initialization, while requiring complete adapter state for subsequent evaluations. Add regression coverage through evaluation admission for both J and K.

2. **[P1] Several exp_11 approval values are never enforced.**  
   [tools/exp06_summarize_haa.py:1429](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1429), [tools/exp11_haa_pipeline.sh:169](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_haa_pipeline.sh:169)

   `exp11_approvals()` verifies committed bytes and required non-null fields, but never compares exp_11’s `code.summarize_haa` with the producer. The producer check uses exp_06’s record instead. The summary also never compares exp_11’s three `reused` identities with the approvals records and receipt actually consumed.

   The pipeline checks exp_11’s reused exp_06 digest, but K’s exp_04 resolution uses exp_06’s pin without checking exp_11’s exp_04 pin.

   A temporary committed fixture containing an all-zero summary digest, incorrect reused digests, and a nonexistent receipt path passed `exp11_approvals()`.

   Enforce these identities at their consuming gates, retaining the historical checks. Add incorrect-but-populated digest tests; null-field tests do not cover this failure.

3. **[P2] Diagnostic receipt identity comparisons were dropped.**  
   [tools/exp11_finalize.py:806](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:806)

   The receipt’s runner digest is checked only for SHA-256 syntax. It is never compared with provenance; `git_head` is not required; and receipt/provenance `exploratory` values are not compared. The frozen exp_06 finalizer performs all three comparisons.

   A synthetic receipt with a different runner digest, contradictory exploratory status, and no `git_head` passed both exp_11 receipt validators. Restore these bindings and add individual mutation tests. This is a regression from the promised identical completion contract.

4. **[P2] The 36-hour ceiling can be disabled or exceeded.**  
   [tools/exp11_launch.sh:36](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:36)

   `EXP11_FULL_CEILING_S` is passed directly to GNU `timeout` without validation. A dry-run with value `0` produced `timeout --kill-after=60 0 …`; GNU documents that zero disables the timeout. Larger values also exceed the registered ceiling.

   Validate a positive duration no greater than `129600` seconds before launch. Overrides may tighten the ceiling.

5. **[P2] Recovery can promote an attempt under the wrong arm.**  
   [tools/exp11_launch.sh:275](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:275)

   Recovery validates `ATTEMPT` independently, then promotes its basename under the root selected by `--arm`. A dry-run demonstrated:

   ```text
   finalize --arm I --attempt <H-root>/attempt_H
   PROMOTE <I-root>/final -> attempt_H
   ```

   A valid H attempt can therefore lead to a dangling or unrelated I link, potentially replacing an existing valid `I/final`. Require the resolved attempt directory and validated recipe/profile to match the selected arm before promotion.

**Assessment of the remaining requested contracts**

| Area | Assessment |
|---|---|
| Oriented model | Frozen azimuth helper reused; non-persistent planes; five-channel embedding; concatenation into parent forward; pool guard present. Derived increment **526,336**. |
| Adapter | Composition around pinned SimpleViT; correct token-angle convention and radians conversion; zero weight **and** bias; correct post-LayerNorm insertion; **1,536** additional parameters. Initialization equality and both strict loading modes pass. No permissive loading or silent reset path found. |
| Registry | Separate `BACKBONES_EXP11`; inherited class identities preserved. `BACKBONES_EXP06` and its digest remain unchanged. |
| Profiles | Eight code keys present; `launch_sh` binds both shell files; committed-blob binding and all-null template are correct. Requirement matrix permits both pretrainings and J/K/Phase 1b with H/I pins null, H with its own pin, and final publication only with both pins. Enforcement gap is blocker 2. |
| Recipes | H/I match the two historical argument records, including loader length **9,261**. H’s retention change is declared. I preserves saving, resume, and augmentation-counter restrictions. Type checks, unknown-field rejection, and mixture rejection pass. |
| Pretraining | Frozen numerical helpers and loop order preserved. Preflight checks reviewed HEAD, cleanliness outside `worklog/`, live PID files, approvals, and exclusive-card census for probe/full. Promotion follows external finalization, subject to blockers 4–5. |
| Full completion | Preserves input inventories and revalidation; source/orchestration closure checks; committed approvals; agreement among startup, retained, and checkpoint arguments; exact epoch/budget requirements; final metadata; checkpoint key/dtype/shape/value equality; zero exit; closed log and receipt; final log rehash. |
| HAA completion | Own run types, dispatch, specification validation, lineage and completeness checks are present. Heading/cache binding and mixed-adapter-heading rejection are correct. Frozen helpers are imported without production monkeypatching. Blocker 1 prevents adapter zero-shot completion. |
| HAA orchestration | Optimizer/scheduler order, accumulation tail step, validation cadence, epoch-zero selection, strict improvement, and final checkpoint behavior match the frozen loop. Evaluation uses the pinned numerical helpers. |
| Pipeline | Correct H/I heading routes and J/K room-plus-adapter routes; cue state resets across mixed queues; correct roots and entry points; explicit census before every job. Golden dry-runs pass. |
| Summarizer | Historical arms retain exp_06 admission; H/I/J/K use exp_11 admission. Serialized types are distinguished from common semantic roles. Historical code-key checks and room-frame rejection remain active. Q/P definitions, phase separation, and A′ copying are correct apart from blocker 2. |
| Closure hygiene | Only exp_06 `summarize_haa` moved against `9f98bbb`; nineteen other keys are unchanged. Its closure now includes `tools/exp11_finalize.py`. All four simulated-evaluation closures remain unchanged. No frozen production file was edited. |

The copied-loop parity test is acceptable **as bounded CPU orchestration evidence** after checking its transcription against frozen `main`. The source-hash guard prevents unnoticed changes to the original; it does not independently prove that the transcription was correct. Manual comparison and the passing trace/checkpoint tests support this implementation. GPU behavior still requires the planned smokes.

Using the `probe` kind for the smaller training smoke is acceptable: its entry point, no-save requirement, resource limits, and permanent non-admissibility are explicit. Delegating `child-exit`, including the shared `EXP06_CHILD_EXIT` marker, is also appropriate.

The bundled smoke shell creates an adapter fixture but only executes oriented HAA smokes. The direct smoke runner supports adapters; retain explicit reviewed adapter training/evaluation smoke commands before the J/K queue.

**Verification and process**

- Selected CPU suite: **187 passed**, 152.92 seconds, using `-p no:cacheprovider`, disabled bytecode writes, and `/tmp` caches.
- This includes model, adapter, registry, profile, recipe, training, finalizer, HAA parity, shell, admission, summarizer, and `test_exp09_sim_eval_closures.py` checks.
- Pinned oracles preserved exp_06, exp_09, and exp_11 Phase-1 statistical payloads and rendered summaries.
- Additional synthetic assembly completed for Phase 1b: nine arms, four decisions, four screens, one external row; and final: eleven arms, fourteen decisions, thirteen screens, one external row.
- `git diff --check` passed. HEAD and repository cleanliness remained unchanged.

The reported full-suite result is **3614 passed, 55 skipped, two failures**. Independent closure recomputation supports the expected exp_06 approvals-drift failure. The SIGTERM failure, load ≈33, and subsequent twelve-pass retry are **reported evidence**: I found no matching committed round-2 transcript and did not rerun signaling tests.

The process record needs correction:

- The requested `coder_prompts/round2_opus_prompt.md` is absent; only the `_DRAFT.md` file is present.
- Red-first compliance is not fully evidenced. Three preceding test commits are labeled red, but finalizer, parity, and summarizer tests land after their implementations. No corresponding round-2 red transcripts were found.
- There are **ten production-code commits exceeding 200 changed lines**, excluding tests and generated records: `769a6ef`, `b59e234`, `3458146`, `b39b7d0`, `216aef3`, `c7b1d7b`, `c0b10a9`, `8f5ff9a`, `e978c2e`, and `e7cb7f4`. The four-exception account is incomplete.

**Approval refill manifest**

These are independently recomputed values at the reviewed tip, **not authorization to fill approvals now**. Fixes will change closures; recompute at the eventual reviewed merge.

Exp_06’s sole refill key is:

```text
code.summarize_haa
393d01f0f0aa2ceb2ad4dfc3d180c4a0e9cb7eee92b18a175d3f8eb5cf327636
```

Exp_11’s eight code keys:

| Key | SHA-256 at `c8acda2` |
|---|---|
| `code.train` | `d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571` |
| `code.finalize` | `0b9fb51a600bb24f5f5f0d7d0efb9a74811e983c0108eefd8c0e182e72447c3f` |
| `code.haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` |
| `code.haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` |
| `code.haa_pipeline_sh` | `163d1fda1bc5700c3faf7812d9f924972a76e172e335a90254c4fd4e7c08ee14` |
| `code.launch_sh` | `5e950c224e212c6f932a2068ad2083be8ceb2792c09cca546e762154b5a6d9ab` |
| `code.smoke` | `62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e` |
| `code.summarize_haa` | `393d01f0f0aa2ceb2ad4dfc3d180c4a0e9cb7eee92b18a175d3f8eb5cf327636` |

Additional exp_11 leaves:

```text
reused.approved_digests_exp04:
2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314

reused.legacy_receipt.path:
ckpt/exp06/legacy_receipt.json

reused.legacy_receipt.sha256:
5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540
```

Set `reused.approved_digests_exp06` to the hash of the **refilled** exp_06 record. For reference, replacing only its current summary digest with the value above, preserving all other bytes, would yield:

```text
c6aaacfc65b76837137287929b7d6e8b0af67a276900b6ccb5599ba5aa1b8a43
```

That conditional value must also be recomputed after fixes. Keep both `artifacts.simpor_epoch_012` and `artifacts.simpor_yaw_epoch_012` records null until their completed checkpoints are reviewed.

No review file or repository changes were saved.
tokens used
150,804
**Reviewer:** OpenAI Codex, GPT-6, API session. Exact model identifier and reasoning-effort setting are not exposed in the available invocation metadata; I cannot attest to `gpt-6-astra` / `ultra`.  
**Sandbox/resources:** Read-only repository; authorized temporary/cache writes only under `/tmp`; CUDA disabled; CPU concurrency ≤4; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, clean before and after review.  
**Range:** `c491396..c8acda24137c6b14c7e9fb2dc14be770261de6bc`, 42 commits.  
**Date/time:** 2026-09-27, approximately 02:08–02:15 EDT.

**Verdict: request changes.** Five blockers remain. The model implementation and numerical orchestration look sound, but adapter zero-shot completion fails, and several admission and launcher safeguards are incomplete.

1. **[P1] J/K zero-shot evaluations cannot finalize.**  
   [tools/exp11_finalize.py:540](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:540)

   The pipeline supplies the historical A/E checkpoint for adapter zero-shot evaluation. Evaluation correctly uses the strict base-loading mode, but finalization unconditionally demands the complete `simple_adapter` state.

   A CPU check using the actual control checkpoint reproduced:

   ```text
   266 parameters are not the 268 of build_xrir_exp11('simple_adapter', 8)
   ```

   This prevents the required J/K zero-shot jobs from completing, blocking Phase 1b and final publication. Finalization must apply the same contextual loading contract: strictly validate the approved base checkpoint for adapter zero-shot initialization, while requiring complete adapter state for subsequent evaluations. Add regression coverage through evaluation admission for both J and K.

2. **[P1] Several exp_11 approval values are never enforced.**  
   [tools/exp06_summarize_haa.py:1429](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1429), [tools/exp11_haa_pipeline.sh:169](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_haa_pipeline.sh:169)

   `exp11_approvals()` verifies committed bytes and required non-null fields, but never compares exp_11’s `code.summarize_haa` with the producer. The producer check uses exp_06’s record instead. The summary also never compares exp_11’s three `reused` identities with the approvals records and receipt actually consumed.

   The pipeline checks exp_11’s reused exp_06 digest, but K’s exp_04 resolution uses exp_06’s pin without checking exp_11’s exp_04 pin.

   A temporary committed fixture containing an all-zero summary digest, incorrect reused digests, and a nonexistent receipt path passed `exp11_approvals()`.

   Enforce these identities at their consuming gates, retaining the historical checks. Add incorrect-but-populated digest tests; null-field tests do not cover this failure.

3. **[P2] Diagnostic receipt identity comparisons were dropped.**  
   [tools/exp11_finalize.py:806](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:806)

   The receipt’s runner digest is checked only for SHA-256 syntax. It is never compared with provenance; `git_head` is not required; and receipt/provenance `exploratory` values are not compared. The frozen exp_06 finalizer performs all three comparisons.

   A synthetic receipt with a different runner digest, contradictory exploratory status, and no `git_head` passed both exp_11 receipt validators. Restore these bindings and add individual mutation tests. This is a regression from the promised identical completion contract.

4. **[P2] The 36-hour ceiling can be disabled or exceeded.**  
   [tools/exp11_launch.sh:36](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:36)

   `EXP11_FULL_CEILING_S` is passed directly to GNU `timeout` without validation. A dry-run with value `0` produced `timeout --kill-after=60 0 …`; GNU documents that zero disables the timeout. Larger values also exceed the registered ceiling.

   Validate a positive duration no greater than `129600` seconds before launch. Overrides may tighten the ceiling.

5. **[P2] Recovery can promote an attempt under the wrong arm.**  
   [tools/exp11_launch.sh:275](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:275)

   Recovery validates `ATTEMPT` independently, then promotes its basename under the root selected by `--arm`. A dry-run demonstrated:

   ```text
   finalize --arm I --attempt <H-root>/attempt_H
   PROMOTE <I-root>/final -> attempt_H
   ```

   A valid H attempt can therefore lead to a dangling or unrelated I link, potentially replacing an existing valid `I/final`. Require the resolved attempt directory and validated recipe/profile to match the selected arm before promotion.

**Assessment of the remaining requested contracts**

| Area | Assessment |
|---|---|
| Oriented model | Frozen azimuth helper reused; non-persistent planes; five-channel embedding; concatenation into parent forward; pool guard present. Derived increment **526,336**. |
| Adapter | Composition around pinned SimpleViT; correct token-angle convention and radians conversion; zero weight **and** bias; correct post-LayerNorm insertion; **1,536** additional parameters. Initialization equality and both strict loading modes pass. No permissive loading or silent reset path found. |
| Registry | Separate `BACKBONES_EXP11`; inherited class identities preserved. `BACKBONES_EXP06` and its digest remain unchanged. |
| Profiles | Eight code keys present; `launch_sh` binds both shell files; committed-blob binding and all-null template are correct. Requirement matrix permits both pretrainings and J/K/Phase 1b with H/I pins null, H with its own pin, and final publication only with both pins. Enforcement gap is blocker 2. |
| Recipes | H/I match the two historical argument records, including loader length **9,261**. H’s retention change is declared. I preserves saving, resume, and augmentation-counter restrictions. Type checks, unknown-field rejection, and mixture rejection pass. |
| Pretraining | Frozen numerical helpers and loop order preserved. Preflight checks reviewed HEAD, cleanliness outside `worklog/`, live PID files, approvals, and exclusive-card census for probe/full. Promotion follows external finalization, subject to blockers 4–5. |
| Full completion | Preserves input inventories and revalidation; source/orchestration closure checks; committed approvals; agreement among startup, retained, and checkpoint arguments; exact epoch/budget requirements; final metadata; checkpoint key/dtype/shape/value equality; zero exit; closed log and receipt; final log rehash. |
| HAA completion | Own run types, dispatch, specification validation, lineage and completeness checks are present. Heading/cache binding and mixed-adapter-heading rejection are correct. Frozen helpers are imported without production monkeypatching. Blocker 1 prevents adapter zero-shot completion. |
| HAA orchestration | Optimizer/scheduler order, accumulation tail step, validation cadence, epoch-zero selection, strict improvement, and final checkpoint behavior match the frozen loop. Evaluation uses the pinned numerical helpers. |
| Pipeline | Correct H/I heading routes and J/K room-plus-adapter routes; cue state resets across mixed queues; correct roots and entry points; explicit census before every job. Golden dry-runs pass. |
| Summarizer | Historical arms retain exp_06 admission; H/I/J/K use exp_11 admission. Serialized types are distinguished from common semantic roles. Historical code-key checks and room-frame rejection remain active. Q/P definitions, phase separation, and A′ copying are correct apart from blocker 2. |
| Closure hygiene | Only exp_06 `summarize_haa` moved against `9f98bbb`; nineteen other keys are unchanged. Its closure now includes `tools/exp11_finalize.py`. All four simulated-evaluation closures remain unchanged. No frozen production file was edited. |

The copied-loop parity test is acceptable **as bounded CPU orchestration evidence** after checking its transcription against frozen `main`. The source-hash guard prevents unnoticed changes to the original; it does not independently prove that the transcription was correct. Manual comparison and the passing trace/checkpoint tests support this implementation. GPU behavior still requires the planned smokes.

Using the `probe` kind for the smaller training smoke is acceptable: its entry point, no-save requirement, resource limits, and permanent non-admissibility are explicit. Delegating `child-exit`, including the shared `EXP06_CHILD_EXIT` marker, is also appropriate.

The bundled smoke shell creates an adapter fixture but only executes oriented HAA smokes. The direct smoke runner supports adapters; retain explicit reviewed adapter training/evaluation smoke commands before the J/K queue.

**Verification and process**

- Selected CPU suite: **187 passed**, 152.92 seconds, using `-p no:cacheprovider`, disabled bytecode writes, and `/tmp` caches.
- This includes model, adapter, registry, profile, recipe, training, finalizer, HAA parity, shell, admission, summarizer, and `test_exp09_sim_eval_closures.py` checks.
- Pinned oracles preserved exp_06, exp_09, and exp_11 Phase-1 statistical payloads and rendered summaries.
- Additional synthetic assembly completed for Phase 1b: nine arms, four decisions, four screens, one external row; and final: eleven arms, fourteen decisions, thirteen screens, one external row.
- `git diff --check` passed. HEAD and repository cleanliness remained unchanged.

The reported full-suite result is **3614 passed, 55 skipped, two failures**. Independent closure recomputation supports the expected exp_06 approvals-drift failure. The SIGTERM failure, load ≈33, and subsequent twelve-pass retry are **reported evidence**: I found no matching committed round-2 transcript and did not rerun signaling tests.

The process record needs correction:

- The requested `coder_prompts/round2_opus_prompt.md` is absent; only the `_DRAFT.md` file is present.
- Red-first compliance is not fully evidenced. Three preceding test commits are labeled red, but finalizer, parity, and summarizer tests land after their implementations. No corresponding round-2 red transcripts were found.
- There are **ten production-code commits exceeding 200 changed lines**, excluding tests and generated records: `769a6ef`, `b59e234`, `3458146`, `b39b7d0`, `216aef3`, `c7b1d7b`, `c0b10a9`, `8f5ff9a`, `e978c2e`, and `e7cb7f4`. The four-exception account is incomplete.

**Approval refill manifest**

These are independently recomputed values at the reviewed tip, **not authorization to fill approvals now**. Fixes will change closures; recompute at the eventual reviewed merge.

Exp_06’s sole refill key is:

```text
code.summarize_haa
393d01f0f0aa2ceb2ad4dfc3d180c4a0e9cb7eee92b18a175d3f8eb5cf327636
```

Exp_11’s eight code keys:

| Key | SHA-256 at `c8acda2` |
|---|---|
| `code.train` | `d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571` |
| `code.finalize` | `0b9fb51a600bb24f5f5f0d7d0efb9a74811e983c0108eefd8c0e182e72447c3f` |
| `code.haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` |
| `code.haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` |
| `code.haa_pipeline_sh` | `163d1fda1bc5700c3faf7812d9f924972a76e172e335a90254c4fd4e7c08ee14` |
| `code.launch_sh` | `5e950c224e212c6f932a2068ad2083be8ceb2792c09cca546e762154b5a6d9ab` |
| `code.smoke` | `62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e` |
| `code.summarize_haa` | `393d01f0f0aa2ceb2ad4dfc3d180c4a0e9cb7eee92b18a175d3f8eb5cf327636` |

Additional exp_11 leaves:

```text
reused.approved_digests_exp04:
2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314

reused.legacy_receipt.path:
ckpt/exp06/legacy_receipt.json

reused.legacy_receipt.sha256:
5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540
```

Set `reused.approved_digests_exp06` to the hash of the **refilled** exp_06 record. For reference, replacing only its current summary digest with the value above, preserving all other bytes, would yield:

```text
c6aaacfc65b76837137287929b7d6e8b0af67a276900b6ccb5599ba5aa1b8a43
```

That conditional value must also be recomputed after fixes. Keep both `artifacts.simpor_epoch_012` and `artifacts.simpor_yaw_epoch_012` records null until their completed checkpoints are reviewed.

No review file or repository changes were saved.

