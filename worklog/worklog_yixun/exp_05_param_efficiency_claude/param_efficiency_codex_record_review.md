# exp_05 param_efficiency — Codex record review (round 1)

**Reviewer:** OpenAI Codex `gpt-6-astra` (reasoningeffort:xhigh), `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt05` (worktree merged with main `a327ccf`; record read on the main tree), prompt `review_prompts/record_review_prompt.md`, log `param_efficiency_2026-09-18_20:04:42_codex_record_review.log`. **Date:** 2026-09-18 20:04–20:20 EDT.

**Reviewer:** OpenAI Codex (GPT-6, API agent; read-only review, scratch execution under `/tmp`) · **Date:** 2026-09-18 EDT · **Marker:** `record`

**Verdict: record not approved.** The canonical numerical evidence checks out, but the interpretation, reproducibility claims, and experiment bookkeeping require corrections.

1. **High — Unsupported causal and capacity conclusions.** [param_efficiency_analysis.md:17](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_analysis.md:17), also line 23. “Saturates at the S tier,” “adding capacity … does nothing,” and attributing the gain definitively to positional code exceed three tested capacities with one unpaired training realization per arm. “Released SimpleViT” also misidentifies the evaluated exp_01 M control as released weights. **Minimal fix:** describe the observed points under the fixed training budget; label the mechanism as a hypothesis; name the exp_01 control accurately. Qualify line 7’s outside-encoder equality because `src_proj` scales with tier, as the plan discloses.

2. **Medium — Analysis wording departs from the registered statistical scope.** [param_efficiency_analysis.md:11](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_analysis.md:11), lines 13–15.
   - K = 8 H1 is **partial on EDT and partial on C50**. Call the passing individual cells *superiority*, rather than “dominance” at individual tiers.
   - K = 1 is **not supported on EDT; not supported on C50**. All point estimates favor SimpleViT, but L EDT is +0.58% with interval [−0.03%, +1.20%]. “Worse at every tier” needs that qualification.
   - “Largest at S” holds for EDT; the largest C50 reversal is at M (+2.76%, versus S +1.18%). “Tier-independent” is too broad.
   - S T60 is descriptively close, not established equal: 9.64 versus 9.63.
   - H2 claims and parameter ratios need explicit scope to each metric’s paired cohort—here 6,337 queries across 17 rooms. At K = 1, only M_cyl versus L_simple C50 reaches its target, through equivalence; neither pairing qualifies for a parameter-ratio claim.

   **Minimal fix:** revise these sentences to the registered wording and reference the complete generated K = 1 tables.

3. **Medium — Incorrect yaw summary.** [param_efficiency_analysis.md:19](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_analysis.md:19); repeated in [param_efficiency_worklog.md:180](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_worklog.md:180). Across the four nonzero rotations, EDT changes range **−0.27% to +3.24%**, and C50 **+2.32% to +5.77%**. Thus rotation does not cost every arm 1–3% EDT. The listed grid also omits `480`. The descriptive sweep does not establish a universal absence of robustness benefit. **Minimal fix:** report the correct ranges and complete grid, and say that no consistent advantage was observed in this single-seed descriptive sweep.

4. **Medium — Mixed record revisions defeat the claimed exact reproduction at `73e90d4`.** [param_efficiency_01_results.html:15](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_01_results.html:15). Markdown reproduces exactly at `73e90d4`; HTML instead cites `8835f00` in both HEAD fields. The [binding report:827](/home/yixunhu/codespace/xRIR_code/ckpt/exp05/reports/binding_report_20260918T234858147838Z.json:827) likewise records `8835f00`, contradicting analysis line 3’s attribution to `73e90d4`. The HTML’s only differences are those two HEAD strings; `8835f00` changed only the notebook. **Minimal fix:** append the accurate chronology, regenerate from one fixed reviewed revision, and publish a new consistent binding.

5. **Medium — Four PDFs fail byte-for-byte reproduction.** [make_figures.py:71](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/make_figures.py:71). All four PNGs reproduce exactly. All four PDFs differ because `savefig` embeds the current `/CreationDate`; masking that field makes each pair identical. **Minimal fix:** make PDF metadata deterministic, review that change, verify two exports byte-for-byte, regenerate the figures, and rebind.

6. **Medium — Required command, setup, and commit records are incomplete.** [param_efficiency_command.md:5](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_command.md:5), [commits_param_efficiency.md:1](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/commits_param_efficiency.md:1). The command file supplies a placeholder training template and record commands, omitting executed probe, L_simple, evaluation-queue, and producer commands. The commit index contains only the plan-handoff commit. `param_efficiency_params_set_up.md` is absent. **Minimal fix:** reconstruct complete commands/configuration from manifests and logs, include the five probes, four trainings, 66 evaluations, producers and record run, and populate the commit index through record closure. Mark retrospective additions honestly; do not imply they were recorded at launch.

7. **Medium — Notebook facts and timing disclosures need an erratum.** [param_efficiency_worklog.md:122](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_worklog.md:122), lines 125 and 162–166; [param_efficiency_analysis.md:27](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_analysis.md:27).
   - S_simple’s epoch-12 test loss is **0.0159933**; 0.0158928 is epoch 11.
   - S_cylindrical’s epoch-12 loss is **0.0159770**; 0.0158748 is its earlier minimum.
   - The L queue contains **20 k0 runs plus two yaw runs**, not 22 plus two. Every L manifest names reviewed commit **`3e2c28b`**; `076cd40` is the runtime HEAD for 21 runs.
   - The analysis omits the documented S_simple CPU-contention window: **September 13, 15:29–approximately 16:40**, with epoch 4 taking 68.84 minutes. The notebook’s L co-tenancy statement supplies no comparable precise window.

   **Minimal fix:** append corrections, distinguish reviewed commits from runtime HEADs, record the S/M queue’s actual completion and `ddfcb66` revision, and disclose documented timing windows while identifying unrecorded ones.

8. **Medium — SOP sequencing exceptions need explicit disclosure.** [param_efficiency_worklog.md:65](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_worklog.md:65), also line 73. Rounds 2 and 3 started before preceding reviews closed, contrary to the SOP’s sequencing rule. The retained `full` review is the September 18 record-tooling review; I found no pre-training integrative `full` review in this record. **Minimal fix:** disclose the historical sequencing deviations and subsequent closures; link any existing pre-launch integrative review, or explicitly record its absence. A later review cannot establish pre-launch compliance.

## Verified

- `check_record.py ckpt/exp05/reports` **passed, exit 0**. The report digest matches `60810502078f4bcc0a287e626d88182a1e33da50796595bb822b80847eec29ef`.
- Regenerated Markdown is byte-identical. HTML has identical numerical content, with only the two revision differences above. Both documents cite all **five product digests and four completion digests**.
- Independently checked **162 curve points and 92 comparison cells** against evaluation arrays: means, seed SDs, effect estimates, target values, decision flags and recorded convergence gates agree. All 66 runs share the registered query/index ordering; acoustic values are finite throughout.
- Approval bytes match `61f4d1b` and `73e90d4`. Recomputed evaluator, writer, training, producer and launcher closures match the pins and manifests. Historical recomputation confirms **`0bff0d4f…` at `f19b9b6`** and **`9dadbef0…` at `862923e`**; the correction of `b09937de…` is justified.
- All four checkpoint hashes match approvals and completions. Four certified trainings have 12 epochs, 32 × 2 execution, passing epoch-one and cumulative-hour gates; all five probe receipts are clean. L’s approximately 44.4 GiB fit is supported.
- Historical M provenance, unpaired training realizations and evaluation-seed SD limitations are disclosed. All **13 saved reviews** contain identity headers.

The paper must not claim universal curve dominance, established capacity saturation or positional-code causation, generalization across training seeds or untested capacities, proven yaw equivalence, K = 1 parameter-efficiency ratios, or encoder reduction factors as full-system or runtime speedups.

**Exact required corrections:** findings **1–8**, followed by regeneration of affected assets, preservation of the existing report, a new binding, and a passing checker. All review-created files remained under `/tmp`.
