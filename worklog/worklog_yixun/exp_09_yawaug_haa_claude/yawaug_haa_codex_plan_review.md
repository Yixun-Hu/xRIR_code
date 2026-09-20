# Codex plan review — exp_09 yawaug_haa plan v1

- Reviewer: OpenAI Codex `gpt-6-astra`, `reasoning effort: xhigh` (CLI override), sandbox read-only, main tree at HEAD 3b7d829
- Prompt: `review_prompts/plan_prompt.md`; run 2026-09-19 21:20:57 → 21:27:03, log `yawaug_haa_2026-09-19_21:20:57_codex_plan_review.log`, Codex session 01a0bc66-f0d4-77c1-8204-6e7855875406

## Final message

**Reviewer:** OpenAI Codex; configured model `gpt-6-astra`, reasoning `xhigh`; current API session.  
**Sandbox:** read-only; CUDA disabled; numerical-library threads capped at four; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code`, branch `main`; existing worklog modifications and untracked exp_09 record.  
**HEAD:** `3b7d82953ac2709d8a9dee26407f461058f48952`  
**Review snapshot:** 2026-09-19 21:24:30 −04:00.

**Verdict: request changes.** The experiment is scientifically appropriate and does not require another pretraining run. V1 has concrete checkpoint-binding and orchestration gaps, plus statistical details that should be fixed before implementation.

1. **Replace the nonexistent `AUG['sha256']` lookup with the approved exp_04 checkpoint binding.**

   [`exp04_profiles.AUG`](/home/yixunhu/codespace/xRIR_code/tools/exp04_profiles.py:89) contains the checkpoint path, epoch, backbone and role, **but no `sha256` key**. Both proposed uses—in the pipeline gate and arm registry—would fail.

   The authority should be exp_04’s approved `checkpoints.aug` record. At launch and summarisation, require its approvals-file digest to equal exp_06’s `reused.approved_digests_exp04`—normalised by the API to `exp04_approved_digests_sha256`—then require epoch 12 and the actual checkpoint hash to match that record. Bind this dependency into the published analysis.

   I verified that the current exp_04 approvals file matches the reused pin, and that the checkpoint actually hashes to:

   `f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299`

   **No new approvals key is necessary.** Do not modify the pinned exp_04 profile to add one, and do not compare E against `artifacts.epoch_012`, which identifies C. Checking the reused pin must be explicit: the existing `haa_children` producer gate does not perform this exp_04 identity check for you.

2. **Specify a complete E-only pipeline route, including zero-shot and record paths.**

   Adding `yawaug` to `init_of` is insufficient. Bare `zeroshot` currently expands explicitly to `cyl_or control_hf cyl_hf`; it does not select the preceding init. The proposed `yawaug:0 zeroshot` command therefore needs defined new semantics.

   The least disruptive design is an explicit job such as `yawaug:zeroshot`, preserving bare `zeroshot` for the existing exp_06 arms. Register the exp_09 queue as three fine-tuning jobs plus that one zero-shot job: **31 children and four job completions**.

   Assign the frame on every init selection, including mixed queues. For E, write `frame: room` and omit the job-spec heading; omit `--heading-json-dir` from every child. Retain the wrappers’ existing `heading: null` metadata rather than deleting required metadata keys.

   Make the reproduction command explicitly set the exp_09 output root. Also route child logs to the exp_09 record: the sourced launcher currently fixes `RECORD` to exp_06, and `child_log` uses an `oriented_cyl_…` prefix.

   **Keep `HEADING_COMPLETE` unchanged.** The least-change solution is to supply the four existing approved heading files to the *producer gate*, while supplying none to the room-frame job or children. The gate’s artifact verification does not rotate data. `headings=None` currently refuses, deliberately; a room-frame exception would require additional API changes and regression coverage without helping this run.

3. **Choose the summariser flag and define arm-specific roots and unchanged exp_06 defaults.**

   I recommend **`--experiment exp09` on `tools/exp06_summarize_haa.py`** under the existing `summarize_haa` approval key.

   A thin `exp09_summarize_haa.py` importing exp_06 is not automatically covered by that key. [`source_identity`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:172) identifies the exp_06 entry module; both approval source maps also name that module. A wrapper containing experiment-specific decisions could execute unbound code unless the closure registration is deliberately extended. That is possible, but exceeds the claimed simple “same producer, no new key” design.

   Root routing also needs an explicit change. [`load_new_arm`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:663) currently takes the supplied common root and appends only the registry path’s basename. Registering `ckpt/exp09/sim2real/yawaug` alone does **not** make it load from that location.

   Freeze separate experiment configurations: exp_06 retains its current arms, contrasts and defaults; exp_09 loads A/B from the legacy receipt, C from exp_06, and E from exp_09. Avoid changing module globals at runtime. Retain closure checks for room-frame arms; only heading-specific requirements are conditional.

   Preserve exclusive publication and input revalidation. Protect exp_06’s canonical output paths. The proposed exp_06 “context row” should be an explicitly sourced exp_09 addendum or link, since exp_06’s existing canonical JSON does not contain E.

4. **Resolve the statistical implementation details before delegating them to the Coder.**

   E1’s intended rule is well posed, but [`decision_cell`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:799) currently returns a margin-based `pass/fail`; it does not implement the proposed two-sided headline. Register separate outputs for the E1 category and the +0.23 dB statement.

   Also specify the scope of the void rule. Exp_06 applies its 1% invalidity/cohort rule to H1/H1b. Its [`screen_cells`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:820) does **not** apply that rule to nonempty screen cohorts. V1’s general “void rule as in exp_06” sentence leaves this ambiguous. For literal protocol reuse, explicitly apply it to E1 and retain exp_06’s descriptive-screen treatment for E2. Extending it to E2 would also be defensible, but must be registered as an exp_09 change before observing E.

**Design and comparability.** The saved arguments confirm the claimed pretraining-budget match:

| Setting | Exp_01 control | Exp_04 yaw augmentation |
|---|---:|---:|
| Epochs | 12 | 12 |
| Microbatch × accumulation | 32 × 2 | 32 × 2 |
| LR / weight decay | 1e−3 / 1e−4 | Same |
| Decay parameters | gamma 0.1, decay_epochs 3 | Same |
| Seed / TF32 | 0 / enabled | Same |
| Workers | 12 | 12 |
| Resume / truncation | None / disabled | Same |

Both histories contain epochs 1–12 exactly once. The control log and augmented arguments report 9,261 training batches per epoch. The checkpoint-save cadence differs—operationally, not in optimisation budget. The scheduler implements exponential decay each epoch, reaching a factor of 0.1 every three epochs; retain that implementation rather than interpreting the prose as a step schedule.

Thus E − A is a useful matched-budget augmentation comparison. Qualify “isolates augmentation” as **conditional on these two pretrained checkpoints**: three fine-tuning seeds do not estimate pretraining-run variability. Harm would also not establish loss of the absolute-heading cue by itself; exp_04 already found a canonical-heading C50 cost.

The room-frame HAA recipe is appropriate: unchanged two-stage training and validation selection, four complete test splits, K=8, deterministic `eval_seed=0` references, three fine-tuning seeds and zero-shot. The pinned dataset selects references from `(eval_seed, room, query index)`, independently of model construction. Fine-tuning adds no yaw augmentation.

E − C appropriately remains descriptive because architecture, frame and pretraining differ. No E heading-frame arm or repeated simulated-parity evaluation is required for the requested question. Retain the exp_02 caveat about absolute comparisons with the paper’s Table 2.

**Room-frame feasibility.** The wrappers already support this path: absent heading arguments select `HAADataset` and `frame='room'`. [`_heading_binding`](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1082) rejects a nonempty room-frame heading. Job-spec validation and lineage likewise already support room-frame jobs. There is no reason to modify the numerical entry points or weaken their validators.

**The blob-at-commit fix is correct, with a bounded sufficiency claim.** I reproduced the current refusal using the real `cyl_or/seed0/stage1` provenance. Its reviewed commit is `b0bae26d…`; the historical approvals hash is `2d258f04…`, while the current file hashes to `673c3648…`. The child’s `haa_finetune` closure equals the approved value in **both** versions, yet the loader refuses because the current approvals bytes differ from the historical blob. This confirms the [2026-09-18 19:03 notebook diagnosis](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_worklog.md:697).

Implement historical reading in [`haa_approvals`](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1135) from the committed blob, validate those exact bytes, and compute their digest. Preserve the existing completion receipt’s `path`, `sha256` and `committed_at` values and shape: `verify_child` compares the reconstructed evidence against the original completion.

Do not merely remove the current-file equality check or silently substitute today’s approvals. Keep the live producer gate’s working-tree checks intact. If a shared byte-parsing helper requires editing `exp06_profiles.py`, include that file and all resulting digest propagation in the reviewed scope.

This is sufficient for the documented **approvals-only refill** failure within the proposed scope. I checked saved training/evaluation source records for C/D/F: they match current source bytes and contain none of the proposed edited files. It is not a general solution for historical replay after changing a child’s executable dependencies: `verify_source_closure`, input revalidation and `check_arm_identities` still impose independent checks.

An integration regression must admit historical C and newly produced E together under the new producer approvals, without restoring any working file or rewriting historical completions. Do not bind historical approvals hashes to the current-file path in the summariser’s ordinary mutable-input map; that would recreate a conflicting binding.

**Decision rules after clarification.** E1 should use the converged seed-0 two-way 95% interval:

- Lower bound `> 0`: detected harm.
- Upper bound `< 0`: detected improvement.
- Otherwise: no detected difference.
- Independently, upper bound `< 0.23`: non-inferior at the exp_06 margin.

Equality does not satisfy either strict inequality. “Detected harm” and “non-inferior at +0.23” can coexist; report both without changing the headline. Failure to establish non-inferiority is not evidence of inferiority. Void or unconverged E1 results must suppress **both** conclusions, even when a nominal interval remains available.

E2 labels must use the Bonferroni-11 interval, with tails at `0.05/22` and `1−0.05/22`, including hallway C50 in the eleven-cell family. Do not select nominal versus adjusted intervals after seeing results. E3 remains descriptive, without a success criterion. Preserve the fixed convergence tolerance, seeds and single quadrupling; handle zero-width intervals as a named refusal/withheld result.

The optional mirror diagnostic is acceptable. Give it an exp_09 output path and retain its diagnostic label. Substituting E for the historical control can invalidate the probe’s original anchor reproduction; its resulting G1 status must not gate E’s training or reinterpret E1.

**Required validation.** Extend the plan’s generic “tests for every item” into explicit acceptance cases:

- Correct E checkpoint admitted; wrong checkpoint, wrong exp_04 approvals identity and wrong epoch refused.
- E-only zero-shot scheduling, mixed-arm frame reset, correct output/log roots, and unchanged legacy queue expansion.
- Room-frame train/eval/job completion admission; nonempty heading rejected.
- Historical approval X followed by refill Y: old receipt unchanged and old/new arms jointly admitted; source/data/receipt tampering still refused.
- E1 boundaries, overlapping harm/non-inferiority, void and convergence suppression; exact E2 tails.
- Default exp_06 analysis unchanged, exp_09 roots correct, and publication refuses changed inputs or existing canonical outputs.

Recompute actual closure digests after implementation rather than assuming only three keys move. Identify the exact permitted exp_07 test failure; do not use “known guard” as a blanket exception. A bounded room-frame smoke/readback should precede the full queue once a card is available.

**Cost and schedule.** The GPU estimate is credible. Retained exp_06 SimpleViT jobs contain approximately 49–55 minutes of training plus evaluation, excluding finalisation; the notebook reports roughly one hour per complete job. Three jobs plus zero-shot reasonably fit about 3–3.5 exclusive GPU-hours. Reserve roughly four hours and budget about 4 GB for retained training artifacts, plus test/log headroom.

Code-ready around 03:00 is plausible but optimistic given historical-verification regression work, review fixes and a full CPU suite previously taking about 52 minutes. GPU handover remains the dominant external dependency; results timing should be relative to the actual slot.

This review used source inspection, saved-artifact checks and a read-only reproduction of the approvals-loader refusal. Full tests were not run; optional import-based closure discovery encountered Matplotlib’s writable-cache requirement, so recorded closure membership and hashes were inspected directly. No files were saved, GPU work launched, or processes signalled.
tokens used
163,369
**Reviewer:** OpenAI Codex; configured model `gpt-6-astra`, reasoning `xhigh`; current API session.  
**Sandbox:** read-only; CUDA disabled; numerical-library threads capped at four; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code`, branch `main`; existing worklog modifications and untracked exp_09 record.  
**HEAD:** `3b7d82953ac2709d8a9dee26407f461058f48952`  
**Review snapshot:** 2026-09-19 21:24:30 −04:00.

**Verdict: request changes.** The experiment is scientifically appropriate and does not require another pretraining run. V1 has concrete checkpoint-binding and orchestration gaps, plus statistical details that should be fixed before implementation.

1. **Replace the nonexistent `AUG['sha256']` lookup with the approved exp_04 checkpoint binding.**

   [`exp04_profiles.AUG`](/home/yixunhu/codespace/xRIR_code/tools/exp04_profiles.py:89) contains the checkpoint path, epoch, backbone and role, **but no `sha256` key**. Both proposed uses—in the pipeline gate and arm registry—would fail.

   The authority should be exp_04’s approved `checkpoints.aug` record. At launch and summarisation, require its approvals-file digest to equal exp_06’s `reused.approved_digests_exp04`—normalised by the API to `exp04_approved_digests_sha256`—then require epoch 12 and the actual checkpoint hash to match that record. Bind this dependency into the published analysis.

   I verified that the current exp_04 approvals file matches the reused pin, and that the checkpoint actually hashes to:

   `f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299`

   **No new approvals key is necessary.** Do not modify the pinned exp_04 profile to add one, and do not compare E against `artifacts.epoch_012`, which identifies C. Checking the reused pin must be explicit: the existing `haa_children` producer gate does not perform this exp_04 identity check for you.

2. **Specify a complete E-only pipeline route, including zero-shot and record paths.**

   Adding `yawaug` to `init_of` is insufficient. Bare `zeroshot` currently expands explicitly to `cyl_or control_hf cyl_hf`; it does not select the preceding init. The proposed `yawaug:0 zeroshot` command therefore needs defined new semantics.

   The least disruptive design is an explicit job such as `yawaug:zeroshot`, preserving bare `zeroshot` for the existing exp_06 arms. Register the exp_09 queue as three fine-tuning jobs plus that one zero-shot job: **31 children and four job completions**.

   Assign the frame on every init selection, including mixed queues. For E, write `frame: room` and omit the job-spec heading; omit `--heading-json-dir` from every child. Retain the wrappers’ existing `heading: null` metadata rather than deleting required metadata keys.

   Make the reproduction command explicitly set the exp_09 output root. Also route child logs to the exp_09 record: the sourced launcher currently fixes `RECORD` to exp_06, and `child_log` uses an `oriented_cyl_…` prefix.

   **Keep `HEADING_COMPLETE` unchanged.** The least-change solution is to supply the four existing approved heading files to the *producer gate*, while supplying none to the room-frame job or children. The gate’s artifact verification does not rotate data. `headings=None` currently refuses, deliberately; a room-frame exception would require additional API changes and regression coverage without helping this run.

3. **Choose the summariser flag and define arm-specific roots and unchanged exp_06 defaults.**

   I recommend **`--experiment exp09` on `tools/exp06_summarize_haa.py`** under the existing `summarize_haa` approval key.

   A thin `exp09_summarize_haa.py` importing exp_06 is not automatically covered by that key. [`source_identity`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:172) identifies the exp_06 entry module; both approval source maps also name that module. A wrapper containing experiment-specific decisions could execute unbound code unless the closure registration is deliberately extended. That is possible, but exceeds the claimed simple “same producer, no new key” design.

   Root routing also needs an explicit change. [`load_new_arm`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:663) currently takes the supplied common root and appends only the registry path’s basename. Registering `ckpt/exp09/sim2real/yawaug` alone does **not** make it load from that location.

   Freeze separate experiment configurations: exp_06 retains its current arms, contrasts and defaults; exp_09 loads A/B from the legacy receipt, C from exp_06, and E from exp_09. Avoid changing module globals at runtime. Retain closure checks for room-frame arms; only heading-specific requirements are conditional.

   Preserve exclusive publication and input revalidation. Protect exp_06’s canonical output paths. The proposed exp_06 “context row” should be an explicitly sourced exp_09 addendum or link, since exp_06’s existing canonical JSON does not contain E.

4. **Resolve the statistical implementation details before delegating them to the Coder.**

   E1’s intended rule is well posed, but [`decision_cell`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:799) currently returns a margin-based `pass/fail`; it does not implement the proposed two-sided headline. Register separate outputs for the E1 category and the +0.23 dB statement.

   Also specify the scope of the void rule. Exp_06 applies its 1% invalidity/cohort rule to H1/H1b. Its [`screen_cells`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:820) does **not** apply that rule to nonempty screen cohorts. V1’s general “void rule as in exp_06” sentence leaves this ambiguous. For literal protocol reuse, explicitly apply it to E1 and retain exp_06’s descriptive-screen treatment for E2. Extending it to E2 would also be defensible, but must be registered as an exp_09 change before observing E.

**Design and comparability.** The saved arguments confirm the claimed pretraining-budget match:

| Setting | Exp_01 control | Exp_04 yaw augmentation |
|---|---:|---:|
| Epochs | 12 | 12 |
| Microbatch × accumulation | 32 × 2 | 32 × 2 |
| LR / weight decay | 1e−3 / 1e−4 | Same |
| Decay parameters | gamma 0.1, decay_epochs 3 | Same |
| Seed / TF32 | 0 / enabled | Same |
| Workers | 12 | 12 |
| Resume / truncation | None / disabled | Same |

Both histories contain epochs 1–12 exactly once. The control log and augmented arguments report 9,261 training batches per epoch. The checkpoint-save cadence differs—operationally, not in optimisation budget. The scheduler implements exponential decay each epoch, reaching a factor of 0.1 every three epochs; retain that implementation rather than interpreting the prose as a step schedule.

Thus E − A is a useful matched-budget augmentation comparison. Qualify “isolates augmentation” as **conditional on these two pretrained checkpoints**: three fine-tuning seeds do not estimate pretraining-run variability. Harm would also not establish loss of the absolute-heading cue by itself; exp_04 already found a canonical-heading C50 cost.

The room-frame HAA recipe is appropriate: unchanged two-stage training and validation selection, four complete test splits, K=8, deterministic `eval_seed=0` references, three fine-tuning seeds and zero-shot. The pinned dataset selects references from `(eval_seed, room, query index)`, independently of model construction. Fine-tuning adds no yaw augmentation.

E − C appropriately remains descriptive because architecture, frame and pretraining differ. No E heading-frame arm or repeated simulated-parity evaluation is required for the requested question. Retain the exp_02 caveat about absolute comparisons with the paper’s Table 2.

**Room-frame feasibility.** The wrappers already support this path: absent heading arguments select `HAADataset` and `frame='room'`. [`_heading_binding`](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1082) rejects a nonempty room-frame heading. Job-spec validation and lineage likewise already support room-frame jobs. There is no reason to modify the numerical entry points or weaken their validators.

**The blob-at-commit fix is correct, with a bounded sufficiency claim.** I reproduced the current refusal using the real `cyl_or/seed0/stage1` provenance. Its reviewed commit is `b0bae26d…`; the historical approvals hash is `2d258f04…`, while the current file hashes to `673c3648…`. The child’s `haa_finetune` closure equals the approved value in **both** versions, yet the loader refuses because the current approvals bytes differ from the historical blob. This confirms the [2026-09-18 19:03 notebook diagnosis](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_worklog.md:697).

Implement historical reading in [`haa_approvals`](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1135) from the committed blob, validate those exact bytes, and compute their digest. Preserve the existing completion receipt’s `path`, `sha256` and `committed_at` values and shape: `verify_child` compares the reconstructed evidence against the original completion.

Do not merely remove the current-file equality check or silently substitute today’s approvals. Keep the live producer gate’s working-tree checks intact. If a shared byte-parsing helper requires editing `exp06_profiles.py`, include that file and all resulting digest propagation in the reviewed scope.

This is sufficient for the documented **approvals-only refill** failure within the proposed scope. I checked saved training/evaluation source records for C/D/F: they match current source bytes and contain none of the proposed edited files. It is not a general solution for historical replay after changing a child’s executable dependencies: `verify_source_closure`, input revalidation and `check_arm_identities` still impose independent checks.

An integration regression must admit historical C and newly produced E together under the new producer approvals, without restoring any working file or rewriting historical completions. Do not bind historical approvals hashes to the current-file path in the summariser’s ordinary mutable-input map; that would recreate a conflicting binding.

**Decision rules after clarification.** E1 should use the converged seed-0 two-way 95% interval:

- Lower bound `> 0`: detected harm.
- Upper bound `< 0`: detected improvement.
- Otherwise: no detected difference.
- Independently, upper bound `< 0.23`: non-inferior at the exp_06 margin.

Equality does not satisfy either strict inequality. “Detected harm” and “non-inferior at +0.23” can coexist; report both without changing the headline. Failure to establish non-inferiority is not evidence of inferiority. Void or unconverged E1 results must suppress **both** conclusions, even when a nominal interval remains available.

E2 labels must use the Bonferroni-11 interval, with tails at `0.05/22` and `1−0.05/22`, including hallway C50 in the eleven-cell family. Do not select nominal versus adjusted intervals after seeing results. E3 remains descriptive, without a success criterion. Preserve the fixed convergence tolerance, seeds and single quadrupling; handle zero-width intervals as a named refusal/withheld result.

The optional mirror diagnostic is acceptable. Give it an exp_09 output path and retain its diagnostic label. Substituting E for the historical control can invalidate the probe’s original anchor reproduction; its resulting G1 status must not gate E’s training or reinterpret E1.

**Required validation.** Extend the plan’s generic “tests for every item” into explicit acceptance cases:

- Correct E checkpoint admitted; wrong checkpoint, wrong exp_04 approvals identity and wrong epoch refused.
- E-only zero-shot scheduling, mixed-arm frame reset, correct output/log roots, and unchanged legacy queue expansion.
- Room-frame train/eval/job completion admission; nonempty heading rejected.
- Historical approval X followed by refill Y: old receipt unchanged and old/new arms jointly admitted; source/data/receipt tampering still refused.
- E1 boundaries, overlapping harm/non-inferiority, void and convergence suppression; exact E2 tails.
- Default exp_06 analysis unchanged, exp_09 roots correct, and publication refuses changed inputs or existing canonical outputs.

Recompute actual closure digests after implementation rather than assuming only three keys move. Identify the exact permitted exp_07 test failure; do not use “known guard” as a blanket exception. A bounded room-frame smoke/readback should precede the full queue once a card is available.

**Cost and schedule.** The GPU estimate is credible. Retained exp_06 SimpleViT jobs contain approximately 49–55 minutes of training plus evaluation, excluding finalisation; the notebook reports roughly one hour per complete job. Three jobs plus zero-shot reasonably fit about 3–3.5 exclusive GPU-hours. Reserve roughly four hours and budget about 4 GB for retained training artifacts, plus test/log headroom.

Code-ready around 03:00 is plausible but optimistic given historical-verification regression work, review fixes and a full CPU suite previously taking about 52 minutes. GPU handover remains the dominant external dependency; results timing should be relative to the actual slot.

This review used source inspection, saved-artifact checks and a read-only reproduction of the approvals-loader refusal. Full tests were not run; optional import-based closure discovery encountered Matplotlib’s writable-cache requirement, so recorded closure membership and hashes were inspected directly. No files were saved, GPU work launched, or processes signalled.

