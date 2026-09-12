# Codex plan review — param_efficiency (exp_05), plan v2, round 2

**Reviewer:** OpenAI Codex CLI 0.154.0 (`codex exec -s read-only`, model gpt-6-astra, reasoning ultra) · **Date:** 2026-09-12 15:39:21 -0400 · prompt: review_prompts/plan_round2_prompt.md

---

**Reviewer:** OpenAI Codex, GPT-6, API session, read-only · **Date:** 2026-09-12

I read the requested SOP, experiment records, v2 plan/query/notebook, checkpoint arguments, models, trainer, and evaluation/statistics code. The announcements directory is empty. No files were modified.

Round-1 closure against the [v2 plan](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/plan_param_efficiency.md):

| Finding | Status | Assessment |
|---|---|---|
| 1. S/L evaluation | **Closed** | §§4, 6 assign tier construction, strict loading, mismatch refusal, both K values, and explicit legacy-M handling. |
| 2. Microbatch fallback | **Closed** | §§3, 9 require 32×2 throughout; an L fit failure requires redesign before launch. |
| 3. H2 family/levels | **Closed** | §5 specifies four cells, exact superiority/TOST tails, paired resampling, seeds 0/1, and convergence checks. |
| 4. Provenance/tooling | **Closed** | Mandatory exp_04-equivalent provenance, dedicated profiles, protected exp_03 closure files, and conditional M-output reuse address the finding. |
| 5. Ladder/parity | **Partially closed** | Four-arm probes and both-backbone parity are assigned; timing and configuration-comparison defects remain below. |
| 6. Parameter counts | **Partially closed** | Matching tolerance and trainable definition are corrected, but several exact integers are wrong. |
| 7. Cohorts/targets | **Partially closed** | Masks, exclusions, counts, and empty-cell refusal are specified; H2 still tests against a different target from the declared one. |
| 8. Budget | **Closed in substance** | Measured timing, six yaw blocks, and cumulative attempt limits are covered. M fallback arithmetic needs a minor correction. |
| 9. H1 wording | **Closed** | Verdicts are metric-specific and restricted to the three tested tiers. |

New and residual findings:

1. **Should-fix — §5, H2 and cohorts; §6, producer: the declared target and tested target differ.**
   Fixed targets use the baseline’s own cohort, while H2 uses its mean on the paired cohort. Displaying both means does not make “reaches the target” accurate.

   For example, repeat these errors across all five seeds: baseline `[1,1,3,3]`, cylindrical `[invalid,invalid,2.4,2.6]`. The declared target is **2**, and cylindrical’s mean is **2.5**. Nevertheless, the paired baseline mean is **3**, so H2 passes superiority. The repository bootstrap produces an adjusted interval of **[−20%, −13.3%]** under both registered seeds; convergence also passes.

   **Minimal change:** define H2’s target on its registered paired cohort and explicitly scope the target-reaching/parameter-ratio claim to that cohort. Keep the own-arm curve mean separately labelled. Alternatively, use a common cohort throughout. Assign this unequal-cohort example as a producer test. [Plan location](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/plan_param_efficiency.md:43)

2. **Should-fix — §§2, 5–6: incorrect exact parameter counts.**
   Encoder instantiation and the existing M checkpoints give:

   | Tier | SimpleViT | CylindricalViT |
   |---|---:|---:|
   | S | 2,766,080 | 2,777,984 |
   | M | **19,703,296** | **19,750,912** |
   | L | **43,709,184** | 43,780,608 |

   V2 understates both M counts by 512 and overstates Simple-L by 256. Consequently, its assigned exact-count tests would reject the registered models.

   **Minimal change:** correct the integers in §2, H2’s parameter comparisons, and test expectations. The tiers, rounded ratios, and ≤0.5% tolerance remain valid. [Plan location](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/plan_param_efficiency.md:19)

3. **Should-fix — §7: epoch-1 acceptance uses the whole-run projection.**
   The defined projection covers twelve epochs, but epoch 1 is allowed to take `1.05 ×` that projection. A projected 50-hour run would therefore admit a 52.5-hour first epoch.

   **Minimal change:** define `T_epoch = 9261 × t_micro + t_test + t_save` and `T_run = 12 × T_epoch`; apply the epoch-1 gate to `1.05 × T_epoch`, with an explicit stop action on failure. [Plan location](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/plan_param_efficiency.md:60)

4. **Should-fix — §§3, 6–7: the strict parity schema does not handle the planned additions.**
   New parameter-count fields, and potentially exp_04’s disabled augmentation/`no_save` fields, are absent from historical M `args.json`. The stated allowlist does not define their normalization, so a literal comparison can fail despite recipe parity.

   **Minimal change:** register one normalized schema: validate derived counts separately; normalize missing legacy augmentation/save flags to explicitly checked defaults; define permitted GPU placement differences. Require rejection of a nonzero augmentation setting. [Plan location](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/plan_param_efficiency.md:60)

5. **Nit — §9: M fallback evaluations are double-counted and miscounted.**
   The 60 standalone runs already include M. Its subset is **2 arms × 2 K values × 5 seeds = 20 runs**, approximately **2 h 40 min**, not ten additional runs.

   **Minimal change:** budget 60 standalone runs ≈8 h, including M fallback, plus six yaw blocks ≈4.5 h. Deduct reuse only after admission checks pass. [Plan location](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/plan_param_efficiency.md:68)

6. **Nit — §§4, 6: literal M-output byte identity is impossible.**
   Outputs add tier/provenance metadata, and the existing evaluator records variable elapsed time. Aggregate equality alone also does not establish paired-data equality.

   **Minimal change:** require exact query/index order, per-sample values, validity masks, and aggregate metrics, excluding an explicit metadata allowlist. [Plan location](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/plan_param_efficiency.md:52)

**Verdict: approve with changes.**

**Required before asking the user to approve:** resolve new findings **1–4** in the plan and notebook, including their implementation/test assignments, and verify the revisions. Findings **5–6** are optional wording/arithmetic cleanup.

Historical M checkpoint reuse remains acceptable with the disclosed training-realization limitations. Verification included source inspection, CPU parameter/checkpoint checks, and the synthetic H2 counterexample; no training or GPU validation was performed.
codex exit 0 at 15:43:27
