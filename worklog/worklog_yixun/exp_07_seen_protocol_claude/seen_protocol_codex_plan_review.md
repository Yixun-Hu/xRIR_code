**Reviewer:** OpenAI Codex gpt-6-astra (read-only plan review) · **Date:** 2026-09-15 · log `seen_protocol_2026-09-15_09:59:07_codex_plan_review.log`

**Reviewer:** OpenAI Codex, read-only · **Date:** 2026-09-15  
**Verdict: request changes.** The experiment is scientifically reasonable, but the proposed tooling reuse would currently fail or bind the wrong training data.

## Findings

1. **Blocker — The evaluation pipeline cannot run the seen split unchanged.**

   `build_manifest(seen_dataset, …)` works: it uses `file_list`, `ir_path`, and filesystem naming—not `get_ir_and_location_for_other_sources`. However, [the evaluator’s dataset builder](/home/yixunhu/codespace/xRIR_code/eval_yaw_rotation.py:307) explicitly constructs the **unseen** test dataset. [ManifestDataset](/home/yixunhu/codespace/xRIR_code/tools/reference_manifest.py:193) checks exact query-list equality; it does not simply accept the manifest’s list. The read-only audit reproduced its rejection: **6,217 versus 6,337 queries**. Additionally, [the evaluation launcher](/home/yixunhu/codespace/xRIR_code/tools/exp04_eval_launch.py:244) hard-codes `split="unseen"`.

   **Required:** specify a seen-capable entry point and launcher integration that construct the actual seen dataset and bind the correct split. Reuse the numerical evaluation functions unchanged; preserve the frozen files. Add tests for correct dispatch, cross-split rejection, canonical padding, and numerical parity under identical inputs.

2. **Blocker — Training provenance and probe reuse require additional implementation.**

   [train_data_identity](/home/yixunhu/codespace/xRIR_code/tools/provenance.py:267) always instantiates the unseen training dataset. [The launcher](/home/yixunhu/codespace/xRIR_code/tools/exp04_launcher.py:294) uses exp_04’s inventory cache and derives loader length from its count. Recovery also requires **296,334** files. Adding only the trainer flag and a 9,265 assertion will therefore fail.

   Likewise, [exp05_probe](/home/yixunhu/codespace/xRIR_code/tools/exp05_probe.py:17) accepts only S/L, fixes augmentation off, and projects with 9,261 batches. [exp05_gates](/home/yixunhu/codespace/xRIR_code/tools/exp05_gates.py:73) returns no timing limits for M.

   **Required:** explicitly assign protocol-aware training identity, cache isolation, recovery validation, and M/seen probe/receipt support to files and tests. Bind `seen_test_split.pkl`, the seen dataset source, and the training input inventory, including geometry dependencies. Ensure import-derived closures actually include the selected dataset—an import executed only inside `main()` would escape current discovery.

3. **Blocker — A profile registry alone does not implement the proposed approval contract.**

   [results_table](/home/yixunhu/codespace/xRIR_code/tools/results_table.py:25) unconditionally accesses `checkpoints.aug`, special-cases that role, and emits `profile_name="TABLE_V1"`. [paired_compare admission](/home/yixunhu/codespace/xRIR_code/tools/paired_compare.py:314) requires training bindings only for role `aug`; it does not establish the proposed three-arm training-contract checks. The exp_04 approval loader also rejects the proposed schema.

   **Required:** define an exp_07 approval loader and role-aware admission layer. For **each new arm**, verify:

   - approved checkpoint path/hash and epoch 12;
   - checkpoint hash equals the bound completion’s `outputs["epoch_012.pth"]`;
   - completion binds the correct training manifest and `args.json`;
   - protocol, backbone, yaw settings, recipe, full-run constraints, and data identity match;
   - training closures agree across the new arms and match the approved training pin; launcher closures belong to the reviewed allowlist.

   Preserve A9’s runtime-read, committed approval JSON and non-circular pinning sequence. Give the released checkpoint an explicit external-reference provenance policy.

4. **Should-fix — Split counts are correct; the update count and single-delta wording need correction.**

   The file audit confirmed:

   | Quantity | Verified |
   |---|---:|
   | Seen training files | 296,454 |
   | Seen test files | 6,217 |
   | Seen test rooms | 131 |
   | Seen-test files in unseen training | 5,124 |
   | Seen-test files in unseen testing | 1,093 |
   | Seen train/test target overlap | 0 |

   All 131 test rooms contain seen-training examples. The actual split holds out **630 receivers**, none appearing in seen training. Thus “other receivers” is accurate, and the 1,093 queries are also in training-seen rooms under this protocol.

   With the existing accumulation behavior, the correct budget is **4,633 updates/epoch × 12 = 55,596**, versus 55,572 previously: **24 additional updates, approximately 0.0432%**.

   **Required:** record `B=9265`, `t=(epoch−1)B+batch_idx`, domain `0…111179`, and the unchanged mixer, seed, and overflow guard. The changed offsets are a consequence of protocol-dependent loader length; at matching batch indices, counters diverge from epoch 2. Also disclose the final micro-batch changing from 14 to 6 examples.

   Describe this as the **same recipe and epoch budget with protocol as the intended change**. Historical training trajectories are not paired. Define historical default normalization, operational exclusions, and separately validated derived fields; the stated “exactly protocol” argument difference is insufficient.

5. **Should-fix — The descriptive estimand needs explicit cohort and uncertainty rules.**

   Five reference/Griffin–Lim seeds are appropriate for table mean ± sample SD. They do **not** estimate training-seed variability. Query and room bootstrap intervals remain conditional on these trained checkpoints; the room interval does not establish generalization to untrained rooms.

   The plan says “mean over 6,217 queries,” but [TABLE_V1](/home/yixunhu/codespace/xRIR_code/tools/results_table.py:68) uses each seed’s finite queries and permits a finite-count difference of two. Paired comparisons need their own all-five-seeds, both-arms finite mask. Also, existing `rho_bootstrap` computes a **relative** difference, whereas this plan specifies paired differences without defining their scale.

   **Required:** freeze the metrics, absolute-versus-relative statistic, table finite-value policy, paired masks, exclusions, empty-cell refusal, room weighting, and percentile convention. Explicitly select epoch 12 and exclude test-selected checkpoints. Keep intervals pointwise and descriptive, without significance stars or superiority/equivalence claims. No additional hypothesis is necessary for the requested table; any later inferential claim needs a prospectively registered rule.

6. **Should-fix — The released checkpoint is a meaningful reference, but “contamination” is unsupported.**

   Comparing `xRIR_seen.pth` on identical queries, references, and phases gives a useful **checkpoint-performance reference**. It cannot isolate architectural or augmentation effects because its training budget, realization, selection procedure, and complete provenance are unknown.

   The sentence “Contamination of the released checkpoint’s row is the authors’ own protocol” should be removed. The inspected seen protocol excludes these test targets. References from other held-out queries at the same receiver are part of the few-shot task, not evidence of training leakage.

   **Required:** mark the released row’s training budget/epoch as unknown, pin its actual hash, and state these limitations. Cite the existing [full-split reproduction](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results.md:7): **0.0389 s EDT, 1.029 dB C50, 7.27% T60**. Add a calibration/readback check for the new evaluation path and explain the change from historical random references, phases, and batch size. That historical single run does not establish a variance band.

7. **Should-fix — The budget and both schedules underestimate known work.**

   Recorded training histories total approximately **28.22 h SimpleViT, 31.36 h cylindrical, and 27.41 h augmented**. A better initial estimate is therefore **87 GPU-h training + 5.3 GPU-h evaluation ≈ 92 GPU-h**, before validation and contingencies. Forty evaluations correctly include the released row.

   Exp_06’s [current plan](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/plan_oriented_cyl.md:175) budgets approximately **42 GPU-h**, with about 50 reserved, rather than the 37 used here.

   **Required:** recompute both orderings using per-arm estimates and actual handoffs, then replace estimates with certified probe projections. B remains the sensible default because it preserves the existing GPU-1 commitment. A reallocates that commitment and delays exp_06 even though it moves to GPU 0. Both appear feasible at central estimates; the deadline is not guaranteed by the permitted ceilings. State that the one-retry limit and cumulative ceiling both apply, and include re-probe, validation, and finalization time.

8. **Should-fix — Reporting reuse needs concrete changes, especially units and multiple records.**

   The current table writer supports **one** generated block; [its preservation logic](/home/yixunhu/codespace/xRIR_code/tools/results_table.py:148) rejects multiple matching blocks. Its EDT values are stored in **milliseconds**, while the requested LaTeX table uses **seconds**. Exp_04’s binder assumes one augmented training attempt, and its generators hard-code exp_04 profiles and hypothesis outputs.

   **Required:** specify profile-specific block handling, correct EDT conversion for both means and SDs, and exp_07 renderers/binding that cover three training attempts and the external checkpoint. Validate the existing unseen canonical JSON through its original provenance and preserve its numbers and labels. Test combined-table regeneration and preservation of the existing generated block.

9. **Should-fix — The plan’s SOP handoff is stale and incomplete.**

   The plan assigns Codex as Coder and Fable as code Reviewer, contrary to [the current repository directive](/home/yixunhu/codespace/xRIR_code/CLAUDE.md:7): **Opus 5 Coder in the worktree; Codex code Reviewer**.

   **Required:** correct the roles and expand the per-file/per-function test plan to cover the missing evaluation, provenance, probe, approval, and reporting work. Include the audit CLI change currently mentioned only in validation. Explicitly require closed review rounds and final integrative review before expensive launches, logged parity/resource/production acceptance criteria, and launch-time commands and parameters. Deferred GPU parity must pass before production. Name the review artifacts and canonical table/pairs outputs among the deliverables.

## Exact changes required before approval

1. Replace the unchanged-evaluator claim with a tested seen dataset dispatch and correctly bound evaluation split.
2. Add protocol-aware training inventories, source/split bindings, recovery checks, and M/seen probe receipts.
3. Specify exp_07 approval loading and complete per-arm training/checkpoint admission checks.
4. Correct **55,590 → 55,596** and document the counter, partial-batch, normalization, and historical-realization differences.
5. Freeze descriptive statistics, cohorts, units, exclusions, checkpoint selection, and claim limits.
6. Correct the released-row caveats and add the existing reproduction reference plus evaluation calibration.
7. Revise budget and schedules using measured arm times and current exp_06 reservations.
8. Specify multiple-block reporting, EDT conversion, and binding of all new and reused artifacts.
9. Update SOP roles, implementation/test coverage, review gates, acceptance criteria, and deliverable names.

**Validation:** source inspection and read-only execution of extracted split/manifest logic against the local data listing. No files were modified.
179,071
**Reviewer:** OpenAI Codex, read-only · **Date:** 2026-09-15  
**Verdict: request changes.** The experiment is scientifically reasonable, but the proposed tooling reuse would currently fail or bind the wrong training data.

## Findings

1. **Blocker — The evaluation pipeline cannot run the seen split unchanged.**

   `build_manifest(seen_dataset, …)` works: it uses `file_list`, `ir_path`, and filesystem naming—not `get_ir_and_location_for_other_sources`. However, [the evaluator’s dataset builder](/home/yixunhu/codespace/xRIR_code/eval_yaw_rotation.py:307) explicitly constructs the **unseen** test dataset. [ManifestDataset](/home/yixunhu/codespace/xRIR_code/tools/reference_manifest.py:193) checks exact query-list equality; it does not simply accept the manifest’s list. The read-only audit reproduced its rejection: **6,217 versus 6,337 queries**. Additionally, [the evaluation launcher](/home/yixunhu/codespace/xRIR_code/tools/exp04_eval_launch.py:244) hard-codes `split="unseen"`.

   **Required:** specify a seen-capable entry point and launcher integration that construct the actual seen dataset and bind the correct split. Reuse the numerical evaluation functions unchanged; preserve the frozen files. Add tests for correct dispatch, cross-split rejection, canonical padding, and numerical parity under identical inputs.

2. **Blocker — Training provenance and probe reuse require additional implementation.**

   [train_data_identity](/home/yixunhu/codespace/xRIR_code/tools/provenance.py:267) always instantiates the unseen training dataset. [The launcher](/home/yixunhu/codespace/xRIR_code/tools/exp04_launcher.py:294) uses exp_04’s inventory cache and derives loader length from its count. Recovery also requires **296,334** files. Adding only the trainer flag and a 9,265 assertion will therefore fail.

   Likewise, [exp05_probe](/home/yixunhu/codespace/xRIR_code/tools/exp05_probe.py:17) accepts only S/L, fixes augmentation off, and projects with 9,261 batches. [exp05_gates](/home/yixunhu/codespace/xRIR_code/tools/exp05_gates.py:73) returns no timing limits for M.

   **Required:** explicitly assign protocol-aware training identity, cache isolation, recovery validation, and M/seen probe/receipt support to files and tests. Bind `seen_test_split.pkl`, the seen dataset source, and the training input inventory, including geometry dependencies. Ensure import-derived closures actually include the selected dataset—an import executed only inside `main()` would escape current discovery.

3. **Blocker — A profile registry alone does not implement the proposed approval contract.**

   [results_table](/home/yixunhu/codespace/xRIR_code/tools/results_table.py:25) unconditionally accesses `checkpoints.aug`, special-cases that role, and emits `profile_name="TABLE_V1"`. [paired_compare admission](/home/yixunhu/codespace/xRIR_code/tools/paired_compare.py:314) requires training bindings only for role `aug`; it does not establish the proposed three-arm training-contract checks. The exp_04 approval loader also rejects the proposed schema.

   **Required:** define an exp_07 approval loader and role-aware admission layer. For **each new arm**, verify:

   - approved checkpoint path/hash and epoch 12;
   - checkpoint hash equals the bound completion’s `outputs["epoch_012.pth"]`;
   - completion binds the correct training manifest and `args.json`;
   - protocol, backbone, yaw settings, recipe, full-run constraints, and data identity match;
   - training closures agree across the new arms and match the approved training pin; launcher closures belong to the reviewed allowlist.

   Preserve A9’s runtime-read, committed approval JSON and non-circular pinning sequence. Give the released checkpoint an explicit external-reference provenance policy.

4. **Should-fix — Split counts are correct; the update count and single-delta wording need correction.**

   The file audit confirmed:

   | Quantity | Verified |
   |---|---:|
   | Seen training files | 296,454 |
   | Seen test files | 6,217 |
   | Seen test rooms | 131 |
   | Seen-test files in unseen training | 5,124 |
   | Seen-test files in unseen testing | 1,093 |
   | Seen train/test target overlap | 0 |

   All 131 test rooms contain seen-training examples. The actual split holds out **630 receivers**, none appearing in seen training. Thus “other receivers” is accurate, and the 1,093 queries are also in training-seen rooms under this protocol.

   With the existing accumulation behavior, the correct budget is **4,633 updates/epoch × 12 = 55,596**, versus 55,572 previously: **24 additional updates, approximately 0.0432%**.

   **Required:** record `B=9265`, `t=(epoch−1)B+batch_idx`, domain `0…111179`, and the unchanged mixer, seed, and overflow guard. The changed offsets are a consequence of protocol-dependent loader length; at matching batch indices, counters diverge from epoch 2. Also disclose the final micro-batch changing from 14 to 6 examples.

   Describe this as the **same recipe and epoch budget with protocol as the intended change**. Historical training trajectories are not paired. Define historical default normalization, operational exclusions, and separately validated derived fields; the stated “exactly protocol” argument difference is insufficient.

5. **Should-fix — The descriptive estimand needs explicit cohort and uncertainty rules.**

   Five reference/Griffin–Lim seeds are appropriate for table mean ± sample SD. They do **not** estimate training-seed variability. Query and room bootstrap intervals remain conditional on these trained checkpoints; the room interval does not establish generalization to untrained rooms.

   The plan says “mean over 6,217 queries,” but [TABLE_V1](/home/yixunhu/codespace/xRIR_code/tools/results_table.py:68) uses each seed’s finite queries and permits a finite-count difference of two. Paired comparisons need their own all-five-seeds, both-arms finite mask. Also, existing `rho_bootstrap` computes a **relative** difference, whereas this plan specifies paired differences without defining their scale.

   **Required:** freeze the metrics, absolute-versus-relative statistic, table finite-value policy, paired masks, exclusions, empty-cell refusal, room weighting, and percentile convention. Explicitly select epoch 12 and exclude test-selected checkpoints. Keep intervals pointwise and descriptive, without significance stars or superiority/equivalence claims. No additional hypothesis is necessary for the requested table; any later inferential claim needs a prospectively registered rule.

6. **Should-fix — The released checkpoint is a meaningful reference, but “contamination” is unsupported.**

   Comparing `xRIR_seen.pth` on identical queries, references, and phases gives a useful **checkpoint-performance reference**. It cannot isolate architectural or augmentation effects because its training budget, realization, selection procedure, and complete provenance are unknown.

   The sentence “Contamination of the released checkpoint’s row is the authors’ own protocol” should be removed. The inspected seen protocol excludes these test targets. References from other held-out queries at the same receiver are part of the few-shot task, not evidence of training leakage.

   **Required:** mark the released row’s training budget/epoch as unknown, pin its actual hash, and state these limitations. Cite the existing [full-split reproduction](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results.md:7): **0.0389 s EDT, 1.029 dB C50, 7.27% T60**. Add a calibration/readback check for the new evaluation path and explain the change from historical random references, phases, and batch size. That historical single run does not establish a variance band.

7. **Should-fix — The budget and both schedules underestimate known work.**

   Recorded training histories total approximately **28.22 h SimpleViT, 31.36 h cylindrical, and 27.41 h augmented**. A better initial estimate is therefore **87 GPU-h training + 5.3 GPU-h evaluation ≈ 92 GPU-h**, before validation and contingencies. Forty evaluations correctly include the released row.

   Exp_06’s [current plan](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/plan_oriented_cyl.md:175) budgets approximately **42 GPU-h**, with about 50 reserved, rather than the 37 used here.

   **Required:** recompute both orderings using per-arm estimates and actual handoffs, then replace estimates with certified probe projections. B remains the sensible default because it preserves the existing GPU-1 commitment. A reallocates that commitment and delays exp_06 even though it moves to GPU 0. Both appear feasible at central estimates; the deadline is not guaranteed by the permitted ceilings. State that the one-retry limit and cumulative ceiling both apply, and include re-probe, validation, and finalization time.

8. **Should-fix — Reporting reuse needs concrete changes, especially units and multiple records.**

   The current table writer supports **one** generated block; [its preservation logic](/home/yixunhu/codespace/xRIR_code/tools/results_table.py:148) rejects multiple matching blocks. Its EDT values are stored in **milliseconds**, while the requested LaTeX table uses **seconds**. Exp_04’s binder assumes one augmented training attempt, and its generators hard-code exp_04 profiles and hypothesis outputs.

   **Required:** specify profile-specific block handling, correct EDT conversion for both means and SDs, and exp_07 renderers/binding that cover three training attempts and the external checkpoint. Validate the existing unseen canonical JSON through its original provenance and preserve its numbers and labels. Test combined-table regeneration and preservation of the existing generated block.

9. **Should-fix — The plan’s SOP handoff is stale and incomplete.**

   The plan assigns Codex as Coder and Fable as code Reviewer, contrary to [the current repository directive](/home/yixunhu/codespace/xRIR_code/CLAUDE.md:7): **Opus 5 Coder in the worktree; Codex code Reviewer**.

   **Required:** correct the roles and expand the per-file/per-function test plan to cover the missing evaluation, provenance, probe, approval, and reporting work. Include the audit CLI change currently mentioned only in validation. Explicitly require closed review rounds and final integrative review before expensive launches, logged parity/resource/production acceptance criteria, and launch-time commands and parameters. Deferred GPU parity must pass before production. Name the review artifacts and canonical table/pairs outputs among the deliverables.

## Exact changes required before approval

1. Replace the unchanged-evaluator claim with a tested seen dataset dispatch and correctly bound evaluation split.
2. Add protocol-aware training inventories, source/split bindings, recovery checks, and M/seen probe receipts.
3. Specify exp_07 approval loading and complete per-arm training/checkpoint admission checks.
4. Correct **55,590 → 55,596** and document the counter, partial-batch, normalization, and historical-realization differences.
5. Freeze descriptive statistics, cohorts, units, exclusions, checkpoint selection, and claim limits.
6. Correct the released-row caveats and add the existing reproduction reference plus evaluation calibration.
7. Revise budget and schedules using measured arm times and current exp_06 reservations.
8. Specify multiple-block reporting, EDT conversion, and binding of all new and reused artifacts.
9. Update SOP roles, implementation/test coverage, review gates, acceptance criteria, and deliverable names.

**Validation:** source inspection and read-only execution of extracted split/manifest logic against the local data listing. No files were modified.
