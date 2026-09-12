# Codex plan review — param_efficiency (exp_05), plan v1

**Reviewer:** OpenAI Codex CLI 0.154.0 (`codex exec -s read-only`, model gpt-6-astra, reasoning ultra) · **Date:** 2026-09-12 15:33:05 -0400 · prompt: review_prompts/plan_prompt.md

---

**Reviewer:** OpenAI Codex, GPT-6, API session, read-only · **Date:** 2026-09-12
**Verdict: request changes.**

I reviewed the [exp_05 plan](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/plan_param_efficiency.md), query and notebook, SOP, requested prior records, checkpoint arguments, models, trainer, and evaluation/statistics code. The announcements directory is empty. No files were modified.

The three-tier design is reasonable. Historical M reuse is acceptable with the stated training-realization limitations; it does not independently require retraining. Epoch-12 selection, paired five-seed evaluation, and K=1 as a separate secondary analysis are appropriate.

1. **Blocker — §§4, 6: S/L evaluation is missing from the implementation plan.**
   The reused evaluator [constructs M with default arguments and loads strictly](/home/yixunhu/codespace/xRIR_code/eval_yaw_rotation.py:569). Trainer flags alone cannot make S/L checkpoints evaluable, and neither exp_05’s file table nor exp_04’s planned evaluator changes supplies this capability.
   **Minimal change:** assign an evaluator extension or separate entry point that constructs the registered tier, validates its configuration against checkpoint provenance, and records that configuration. Include loading/mismatch tests for all tiers and evaluation at both K values. Preserve an explicit legacy-M configuration.

2. **Blocker — §§2, 3, 9: the 16×4 fallback breaks the single-delta contract.**
   AudioEnc uses ResNet-18 BatchNorm, so changing microbatch size from 32 to 16 changes training even with effective batch 64. There is also a concrete optimizer difference: the [trainer divides losses by the accumulation factor](/home/yixunhu/codespace/xRIR_code/train_xRIR_backbone.py:117), including the final incomplete group. The last 14-sample batch receives weight ½ with 32×2 and ¼ with 16×4. This confounds cross-tier H2 comparisons.
   **Minimal change:** require 32×2 for every confirmatory tier. If L cannot fit, revise its capacity or explicitly redesign the recipe and comparators before launching. Applying the fallback to both L backbones preserves only their within-tier comparison.

3. **Blocker — §5: H2’s family and equivalence levels are not fully registered.**
   “Each hypothesis is a separate family” plus `m=2` is ambiguous when H2 makes EDT/C50 decisions for two pairings. These could be four metric-level claims or two conjunction claims requiring both metrics. “TOST at the same level” also leaves its tails ambiguous: the [existing helper](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:286) uses a different interval convention from two-sided superiority.
   **Minimal change:** enumerate the hypothesis units and exact tails. A straightforward choice is one four-cell H2 family: superiority interval `[Q0.00625, Q0.99375]`; TOST with each one-sided test at `0.0125`, requiring `[Q0.0125, Q0.9875]` strictly inside ±3%. Other family definitions are defensible if explicitly stated. Register both target definitions, recomputation of the relative statistic inside each paired resample, bootstrap seeds 0/1, and convergence checks covering every decision-driving interval.

4. **Should-fix — §§4, 6, 10: provenance and tooling reuse need one enforceable contract.**
   “If available” permits either exp_04’s execution-bound workflow or weaker historical binding, while the producer’s listed checks do not establish checkpoint, tier, epoch, precision, grid, or closure identity. Exp_04’s analysis profiles also implement different hypotheses. Furthermore, adding `count_parameters` to `model/xRIR_cyl.py` changes an exp_03 closure member; protecting only `tools/yaw_rotation.py` is insufficient. The [historical binder](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:76) checks both bytes and subsequent history.
   **Minimal change:** require exp_04-equivalent provenance and dedicated exp_05 profiles, regardless of implementation reuse. Explicitly validate completed outputs, approved closures, checkpoint/configuration, dataset/query identity, seeds/manifests, K, grids, canonical batch and TF32. Put parameter counting in an experiment-owned helper and specify a compatible evaluator/closure strategy. Reuse M evaluations only after those checks pass; otherwise rerun them. Preserve historical acceptance rules and statistics.

5. **Should-fix — §§6, 7: the ladder does not establish fit, throughput, or sufficient M parity.**
   Three smoke batches at unspecified batch size/backbone do not establish that all four new arms fit the intended recipe or provide stable timing. The epoch-time comparison lacks a projection definition, although recorded epoch time includes testing and saving. M parity currently checks shapes/forward, primarily against the simple checkpoint.
   **Minimal change:** prescribe storage-light forward/backward and intended-batch fit/timing probes for all four arms, with warm-up and test/save overhead defined. Check both M backbones against a pinned reference for loss, gradients, buffers and RNG behavior, plus normalized recipe/environment differences with an explicit allowlist. Clarify that full-model steps require GPU; CPU tests can cover encoder components.

6. **Should-fix — §§2, 6, 8: parameter matching and trainable-count statements are inaccurate.**
   The rounded table values check out, but S has **2,766,080 versus 2,777,984 encoder parameters: +0.430%**, exceeding the claimed ≤0.3%. M and L differ by approximately 0.242% and 0.163%. Each full model also contains **20 frozen frequency parameters**, so trainable is `full − 20`, not exactly full.
   **Minimal change:** register an honest matching tolerance—e.g. ≤0.5%, retaining these tiers—and report exact integer counts alongside rounded values. Define trainable using `requires_grad`; retain the stated `source_network`/full-system boundary without changing the models.

7. **Should-fix — §5: plotted curves and fixed targets need explicit cohort rules.**
   The all-seed paired mask is appropriate, but different comparisons can retain different queries. Consequently, a baseline’s plotted mean, fixed target and comparison-cell mean could differ without explanation. Exclusion reporting and empty-cell refusal are also missing.
   **Minimal change:** specify the cohorts used for curve means, seed SD and targets; reconcile or label comparison-specific masks. Report excluded queries by arm/seed/metric and retained query/room counts, and refuse empty cells. This extends exp_05’s reporting without changing prior statistics.

8. **Should-fix — §§3, 4, 9: the budget mixes incompatible timings and omits work.**
   `1.4 s × 9,261 × 12` is **43.2 hours**, not 28.2 hours. The latter is the observed control runtime and should be identified separately. Eight evaluation hours cover the 60 standalone runs but omit four five-angle yaw blocks—roughly another three hours at exp_04’s cited block timing, before tier-specific adjustments. Restarting from epoch 1 also has no cumulative budget limit.
   **Minimal change:** reconcile measured timing assumptions, budget the complete matrix and failed attempts, and specify cumulative limits and stop conditions. For yaw, either label M’s seed-0 result as historical context or budget matching M seed-42 blocks. Keep the 60-run allowance unless M reuse actually passes the final closure checks.

9. **Nit — §§1, 5: H1’s verdict should name its scope.**
   The registered rule can pass on EDT while C50 worsens; it also tests only three capacities.
   **Minimal change:** report “dominance supported on EDT/C50 at the three tested tiers,” with the other metric’s outcome stated separately. Avoid implying dominance throughout the intervening parameter range.

**Required before asking the user to approve:** resolve findings **1–8** in the revised plan and notebook, including their implementation/test assignments and budget consequences, then verify those revisions. Finding **9** is optional wording cleanup.

Verification here covered source inspection, parameter counts from the model definitions, and successful strict loading of both M checkpoints. No training or GPU validation was performed.
codex exit 0 at 15:36:43
