**Reviewer:** OpenAI Codex gpt-6-astra (read-only plan review, round 2) · **Date:** 2026-09-15 · log `seen_protocol_2026-09-15_10:07:05_codex_plan_round2_review.log`

**Reviewer:** OpenAI Codex (GPT-6), read-only review · **Date:** 2026-09-15  
**Reviewed:** `plan_seen_protocol.md`, v2

## Verdict: approve with changes

The experiment’s design is sound. The remaining issues concern executable contracts, calibration, and reporting. I found no fundamental scientific blocker.

### Verified facts

The dataset code and local file inventory confirm:

| Protocol | Training targets | Test targets | Micro-batches/epoch | Updates over 12 epochs |
|---|---:|---:|---:|---:|
| Unseen | 296,334 | 6,337 | 9,261 | 55,572 |
| Seen | 296,454 | 6,217 | 9,265 | 55,596 |

The seen split contains **131 rooms and 630 held-out receivers**. Every test room has training receivers; train/test target and receiver overlap are both zero. Exactly **5,124** seen-test targets belong to unseen-protocol training, and **1,093** belong to unseen-protocol testing. Retraining is therefore necessary.

The single intended change is defensible as **protocol, with the same recipe and epoch budget**. It is not an equal-update or paired-training comparison. V2 correctly records the final micro-batch sizes, +24 updates, and yaw-counter domain `0…111179`. Retain loader-derived `B=9265`; retaining `9261` would violate the established counter rule.

## Numbered findings

1. **Should-fix — The normalized-argument comparison contradicts the per-arm contract.**  
   [Plan §2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/plan_seen_protocol.md:12) says every arm is compared against exp_01’s control, but its permitted differences omit `backbone`. Literally implemented, this rejects `seen_cyl`. The current [launcher](/home/yixunhu/codespace/xRIR_code/tools/exp04_launcher.py:303) already selects the historical comparator by backbone.

   Specify the comparator for each arm explicitly. Prefer each arm’s unseen twin: exp_01 SimpleViT, exp_01 cylindrical, and exp_04 augmented. Distinguish this protocol comparison from the within-seen architecture/augmentation comparisons. Include exp_05 A1’s resolved historical defaults, particularly `yaw_aug_seed → seed`.

2. **Should-fix — Manifest reuse works, but the evaluator construction contract needs completion.**  
   The rationale in [§2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/plan_seen_protocol.md:13) cites the shared reference-selection method. Actually, [`build_manifest`](/home/yixunhu/codespace/xRIR_code/tools/reference_manifest.py:99) uses only `file_list` and `ir_path`, and reconstructs candidates itself. The seen dataset supplies compatible paths. I checked K=1 and K=8 seed-42 manifests: both had 6,217 valid queries, no invalid paths, and usable `ManifestDataset` samples.

   Specify these details:

   - Construct `xRIR_Dataset(split="test", num_shot=K, max_len=9600)`. Its default `num_shot=4` would fail `ManifestDataset` validation.
   - Run construction from the repository/worktree root: the frozen seen module opens the pickle relative to the current directory.
   - Assign manifest generation/indexing to a named reviewed entry point.
   - State how the new evaluator owns dataset construction. [`run_exp04`](/home/yixunhu/codespace/xRIR_code/tools/exp04_eval.py:114) has no dataset-factory hook and directly calls the unseen builder.
   - Define numerical parity with an explicit metadata exception list; complete output files cannot be byte-identical because their provenance and elapsed-time fields differ.
   - Test the seen split’s **nine-real-sample final batch padded to 16**, as well as both K values.

   **Reuse unchanged:** manifest/reference helpers, `ManifestDataset`, numerical evaluation functions, checkpoint prefix stripping, and evaluation `data_identity`. All reference candidates at these held-out receivers belong to the seen-test query set, so a common evaluation inventory across K/seeds is appropriate.

   **New or adapted:** dataset orchestration, split-aware launcher checks, training inventory/probes, exp_07 approval profiles and admission, descriptive producers, and renderers. V2 correctly identifies most of this work; existing producer entry points cannot accept exp_07 merely by receiving different profile values.

3. **Should-fix — Calibration presently assumes the explanation it should test.**  
   [§2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/plan_seen_protocol.md:13) says differences from historical reproduction will be attributed to references/phases/batch “not to the evaluator.” Unseen parity does not establish that conclusion for the new seen loading path.

   Define a calibration acceptance rule and failure action before examining the new results, satisfying [SOP evaluation integrity](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:101). Separate deterministic implementation parity from historical reproduction, whose reference draws and phases were uncontrolled. Investigate discrepancies rather than assigning their cause in advance; budget any additional diagnostic runs.

   The released checkpoint is a **meaningful performance reference under the same seen evaluation protocol**. Its differences from `seen_simple` do not isolate training recipe, augmentation, architecture, or compute efficiency because its training budget, realization, and selection are unknown. Record its epoch as unknown. K=1 is evaluation with one reference, not evidence of a separately matched K=1 training run. Its full file digest matches the plan’s `1762c702…` prefix.

4. **Should-fix — Clarify the room estimand and distinguish comparison means from table means.**  
   “Equal-weight 131-room cluster bootstrap” in [§2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/plan_seen_protocol.md:14) is ambiguous. The [existing helper](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:178) draws rooms uniformly but retains every selected query, computing a **query-weighted** mean. Averaging room means equally defines another estimand. Seen-room query counts range from **30 to 200**, so this distinction matters.

   Prefer the existing whole-room resampling convention for a secondary interval around the registered query-weighted statistic. If equal-room means are intended, give them their own estimate and label. Report retained rooms after masking rather than assuming all 131 remain.

   Also show both arms’ means on each paired cohort: these can differ from the table’s independently finite per-seed means. Specify paired metrics, quantile convention, and the convergence diagnostic’s tolerance/failure treatment. Pointwise descriptive intervals need no superiority test, but replace “never compared inferentially” with wording that consistently permits checkpoint-conditional intervals while excluding causal or method-superiority claims.

5. **Should-fix — Close the provenance and SOP artifact details.**  
   The new approval design is appropriate: runtime-read pins avoid circular closure hashes, training closures must agree, and launcher closures may use a reviewed allowlist. Complete its specification:

   - Bind the **actual pickle path and full digest** as a revalidated input through training, evaluation, recovery, and producer admission. Merely recording a digest in metadata does not invoke [`provenance.revalidate`](/home/yixunhu/codespace/xRIR_code/tools/provenance.py:300).
   - Make the released checkpoint’s exception apply only to training provenance; retain every evaluation manifest/completion/data/protocol check.
   - Document that existing `train_data_identity` hashes training **WAVs only**. Bind training metadata/depth separately or explicitly disclose that inherited provenance limitation.
   - Add `tools/yaw_aug.py` to the file-change table. Give its audit an exp_07 output path, explicit three-batch arguments, cohort/offset digest, and inclusion in the final binding report. Its current default points into exp_04’s artifacts.
   - State the reviewed pin-finalization sequence and acceptance checks for missing/unapproved pins.
   - Include all three probe receipts, parity/calibration/audit evidence, and failed attempts in the deliverable record. Add an explicit check for the promised **one-retry maximum**.

   The plan already covers the principal SOP deliverables, review rounds, epoch-12 selection, launch records, and canonical rendering.

6. **Nit — Preserve visible uncertainty in the LaTeX table.**  
   Three decimal places in **seconds** can turn a nonzero EDT SD into `0.000`. Keep enough SD precision, or explicitly explain the rounding while linking the full-precision canonical values.

### Budget and GPU orderings

The central budget is consistent: approximately **87 training + 5.3 evaluation + 2 validation = 94 GPU-hours**. Both schedules fit at those estimates. **B is the appropriate default** because it preserves exp_06’s reservation; A requires an explicit scheduling change. The deadline caveat is necessary: probe ceilings and retries do not guarantee either calendar. Any extra calibration work should be added to these estimates.

## Exact changes required before approval

1. Specify per-arm historical comparators and correct the argument-normalization contract.
2. Complete the manifest/evaluator construction contract, including K, working directory, reviewed builder, orchestration, metadata parity exceptions, and final-batch testing.
3. Replace automatic calibration attribution with a preregistered acceptance/failure rule; clarify released-checkpoint and K=1 labels.
4. Freeze room weighting, paired-cohort reporting, paired metrics, quantiles, and convergence handling.
5. Complete split-file revalidation, provenance scope, approval sequencing, audit/probe bindings, and retry acceptance checks.

Finding 6 is optional.

**Verification limits:** No files were modified and no GPU runs were launched. Ordinary imports encountered read-only library-cache restrictions; manifest/readback checks used unchanged source definitions with unused cache-writing imports omitted. These checks do not replace the planned full-environment parity gates.
173,053
**Reviewer:** OpenAI Codex (GPT-6), read-only review · **Date:** 2026-09-15  
**Reviewed:** `plan_seen_protocol.md`, v2

## Verdict: approve with changes

The experiment’s design is sound. The remaining issues concern executable contracts, calibration, and reporting. I found no fundamental scientific blocker.

### Verified facts

The dataset code and local file inventory confirm:

| Protocol | Training targets | Test targets | Micro-batches/epoch | Updates over 12 epochs |
|---|---:|---:|---:|---:|
| Unseen | 296,334 | 6,337 | 9,261 | 55,572 |
| Seen | 296,454 | 6,217 | 9,265 | 55,596 |

The seen split contains **131 rooms and 630 held-out receivers**. Every test room has training receivers; train/test target and receiver overlap are both zero. Exactly **5,124** seen-test targets belong to unseen-protocol training, and **1,093** belong to unseen-protocol testing. Retraining is therefore necessary.

The single intended change is defensible as **protocol, with the same recipe and epoch budget**. It is not an equal-update or paired-training comparison. V2 correctly records the final micro-batch sizes, +24 updates, and yaw-counter domain `0…111179`. Retain loader-derived `B=9265`; retaining `9261` would violate the established counter rule.

## Numbered findings

1. **Should-fix — The normalized-argument comparison contradicts the per-arm contract.**  
   [Plan §2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/plan_seen_protocol.md:12) says every arm is compared against exp_01’s control, but its permitted differences omit `backbone`. Literally implemented, this rejects `seen_cyl`. The current [launcher](/home/yixunhu/codespace/xRIR_code/tools/exp04_launcher.py:303) already selects the historical comparator by backbone.

   Specify the comparator for each arm explicitly. Prefer each arm’s unseen twin: exp_01 SimpleViT, exp_01 cylindrical, and exp_04 augmented. Distinguish this protocol comparison from the within-seen architecture/augmentation comparisons. Include exp_05 A1’s resolved historical defaults, particularly `yaw_aug_seed → seed`.

2. **Should-fix — Manifest reuse works, but the evaluator construction contract needs completion.**  
   The rationale in [§2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/plan_seen_protocol.md:13) cites the shared reference-selection method. Actually, [`build_manifest`](/home/yixunhu/codespace/xRIR_code/tools/reference_manifest.py:99) uses only `file_list` and `ir_path`, and reconstructs candidates itself. The seen dataset supplies compatible paths. I checked K=1 and K=8 seed-42 manifests: both had 6,217 valid queries, no invalid paths, and usable `ManifestDataset` samples.

   Specify these details:

   - Construct `xRIR_Dataset(split="test", num_shot=K, max_len=9600)`. Its default `num_shot=4` would fail `ManifestDataset` validation.
   - Run construction from the repository/worktree root: the frozen seen module opens the pickle relative to the current directory.
   - Assign manifest generation/indexing to a named reviewed entry point.
   - State how the new evaluator owns dataset construction. [`run_exp04`](/home/yixunhu/codespace/xRIR_code/tools/exp04_eval.py:114) has no dataset-factory hook and directly calls the unseen builder.
   - Define numerical parity with an explicit metadata exception list; complete output files cannot be byte-identical because their provenance and elapsed-time fields differ.
   - Test the seen split’s **nine-real-sample final batch padded to 16**, as well as both K values.

   **Reuse unchanged:** manifest/reference helpers, `ManifestDataset`, numerical evaluation functions, checkpoint prefix stripping, and evaluation `data_identity`. All reference candidates at these held-out receivers belong to the seen-test query set, so a common evaluation inventory across K/seeds is appropriate.

   **New or adapted:** dataset orchestration, split-aware launcher checks, training inventory/probes, exp_07 approval profiles and admission, descriptive producers, and renderers. V2 correctly identifies most of this work; existing producer entry points cannot accept exp_07 merely by receiving different profile values.

3. **Should-fix — Calibration presently assumes the explanation it should test.**  
   [§2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/plan_seen_protocol.md:13) says differences from historical reproduction will be attributed to references/phases/batch “not to the evaluator.” Unseen parity does not establish that conclusion for the new seen loading path.

   Define a calibration acceptance rule and failure action before examining the new results, satisfying [SOP evaluation integrity](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:101). Separate deterministic implementation parity from historical reproduction, whose reference draws and phases were uncontrolled. Investigate discrepancies rather than assigning their cause in advance; budget any additional diagnostic runs.

   The released checkpoint is a **meaningful performance reference under the same seen evaluation protocol**. Its differences from `seen_simple` do not isolate training recipe, augmentation, architecture, or compute efficiency because its training budget, realization, and selection are unknown. Record its epoch as unknown. K=1 is evaluation with one reference, not evidence of a separately matched K=1 training run. Its full file digest matches the plan’s `1762c702…` prefix.

4. **Should-fix — Clarify the room estimand and distinguish comparison means from table means.**  
   “Equal-weight 131-room cluster bootstrap” in [§2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/plan_seen_protocol.md:14) is ambiguous. The [existing helper](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:178) draws rooms uniformly but retains every selected query, computing a **query-weighted** mean. Averaging room means equally defines another estimand. Seen-room query counts range from **30 to 200**, so this distinction matters.

   Prefer the existing whole-room resampling convention for a secondary interval around the registered query-weighted statistic. If equal-room means are intended, give them their own estimate and label. Report retained rooms after masking rather than assuming all 131 remain.

   Also show both arms’ means on each paired cohort: these can differ from the table’s independently finite per-seed means. Specify paired metrics, quantile convention, and the convergence diagnostic’s tolerance/failure treatment. Pointwise descriptive intervals need no superiority test, but replace “never compared inferentially” with wording that consistently permits checkpoint-conditional intervals while excluding causal or method-superiority claims.

5. **Should-fix — Close the provenance and SOP artifact details.**  
   The new approval design is appropriate: runtime-read pins avoid circular closure hashes, training closures must agree, and launcher closures may use a reviewed allowlist. Complete its specification:

   - Bind the **actual pickle path and full digest** as a revalidated input through training, evaluation, recovery, and producer admission. Merely recording a digest in metadata does not invoke [`provenance.revalidate`](/home/yixunhu/codespace/xRIR_code/tools/provenance.py:300).
   - Make the released checkpoint’s exception apply only to training provenance; retain every evaluation manifest/completion/data/protocol check.
   - Document that existing `train_data_identity` hashes training **WAVs only**. Bind training metadata/depth separately or explicitly disclose that inherited provenance limitation.
   - Add `tools/yaw_aug.py` to the file-change table. Give its audit an exp_07 output path, explicit three-batch arguments, cohort/offset digest, and inclusion in the final binding report. Its current default points into exp_04’s artifacts.
   - State the reviewed pin-finalization sequence and acceptance checks for missing/unapproved pins.
   - Include all three probe receipts, parity/calibration/audit evidence, and failed attempts in the deliverable record. Add an explicit check for the promised **one-retry maximum**.

   The plan already covers the principal SOP deliverables, review rounds, epoch-12 selection, launch records, and canonical rendering.

6. **Nit — Preserve visible uncertainty in the LaTeX table.**  
   Three decimal places in **seconds** can turn a nonzero EDT SD into `0.000`. Keep enough SD precision, or explicitly explain the rounding while linking the full-precision canonical values.

### Budget and GPU orderings

The central budget is consistent: approximately **87 training + 5.3 evaluation + 2 validation = 94 GPU-hours**. Both schedules fit at those estimates. **B is the appropriate default** because it preserves exp_06’s reservation; A requires an explicit scheduling change. The deadline caveat is necessary: probe ceilings and retries do not guarantee either calendar. Any extra calibration work should be added to these estimates.

## Exact changes required before approval

1. Specify per-arm historical comparators and correct the argument-normalization contract.
2. Complete the manifest/evaluator construction contract, including K, working directory, reviewed builder, orchestration, metadata parity exceptions, and final-batch testing.
3. Replace automatic calibration attribution with a preregistered acceptance/failure rule; clarify released-checkpoint and K=1 labels.
4. Freeze room weighting, paired-cohort reporting, paired metrics, quantiles, and convergence handling.
5. Complete split-file revalidation, provenance scope, approval sequencing, audit/probe bindings, and retry acceptance checks.

Finding 6 is optional.

**Verification limits:** No files were modified and no GPU runs were launched. Ordinary imports encountered read-only library-cache restrictions; manifest/readback checks used unchanged source definitions with unused cache-writing imports omitted. These checks do not replace the planned full-environment parity gates.
