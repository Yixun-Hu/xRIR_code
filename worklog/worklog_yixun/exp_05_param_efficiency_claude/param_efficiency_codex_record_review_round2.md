# exp_05 param_efficiency — Codex record review (round 2)

**Reviewer:** OpenAI Codex `gpt-6-astra` (reasoningeffort:xhigh), `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt05` (worktree at `ec75704` = main), prompt `review_prompts/record_review_round2_prompt.md`, log `param_efficiency_2026-09-18_21:46:17_codex_record_review_round2.log`. **Date:** 2026-09-18 21:46–22:03 EDT.

**Reviewer:** OpenAI Codex (GPT-6, API agent), read-only · **Date:** 2026-09-18 EDT · **Marker:** `record`, round 2  
**Scope:** worktree `ec75704`; record read from main; regeneration at `9b25738`.

**Verdict: record approved with corrections.** The numerical evidence, regenerated artifacts and binding pass. Five documentation corrections remain.

1. **Medium — Command reconstruction remains incomplete.** [param_efficiency_command.md:67](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_command.md:67), also lines 72–79 and 9–17. The five probes have receipt summaries but only an ellipsis-based launcher template. The 66 evaluations have one child command and a reference to an unretained scratch queue script. The producer section names another scratch script, and the new `9b25738` record run is absent. This does not complete round-1 correction 6’s reproduction record.  
   **Minimal fix:** retain expanded commands, or the executed scripts with concrete invocation tables and environments, covering all probes/evaluations/producers; append the fixed-revision record run and annotate the earlier mixed-revision run. Keep the retrospective designation.

2. **Medium — Commit index stops before the closure it claims to cover.** [commits_param_efficiency.md:4](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/commits_param_efficiency.md:4). Its newest entry is `1f3792d`; subsequent corrections, relocation records, round-9 fix/review/merge, `9b25738`, and regeneration commit `ec75704` are missing. The shared launcher fix `0e35e2c`, named in the round-2 close-out, is also absent.  
   **Minimal fix:** complete the index through `ec75704`, include shared experiment commits, and label that cutoff explicitly.

3. **Medium — L_cylindrical’s GPU-sharing disclosure is incorrect.** [param_efficiency_analysis.md:27](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_analysis.md:27), repeated in [param_efficiency_worklog.md:196](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_worklog.md:196). All **46 exp_04 evaluation manifests specify GPU 0**; L_cylindrical trained on GPU 1. The [contemporaneous notebook:260](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_worklog.md:260) confirms the GPU-0 evaluation chain. Sharing the host does not establish sharing the card.  
   **Minimal fix:** correct the analysis to describe shared-host overlap and append a notebook erratum; retain the qualification that individual contention windows were not recorded.

4. **Medium — SOP disclosure introduces an impossible review chronology.** [param_efficiency_worklog.md:197](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_worklog.md:197). Round 5’s review occurred **September 14**, so it cannot have preceded the September 13 launches.  
   **Minimal fix:** append the correct chronology: round-2 close-out at approximately 00:53 September 13 covered the launch path; round-3 close-out followed at approximately 01:05. Round 5 preceded L_simple’s later launch. Preserve the explicit absence of a pre-launch integrative `full` review and the sequencing-deviation disclosure.

5. **Low — Setup table mislabels cumulative hours as training wall time.** [param_efficiency_params_set_up.md:5](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_params_set_up.md:5). The L values **41.8 / 46.2 h** are ledger totals including probes. Certified training wall times are **41.6716 / 46.0963 h**, correctly shown in the generated results.  
   **Minimal fix:** rename the column “Cumulative hours including probes,” or use training wall times consistently.

## Verified

- **Round-1 corrections 1–3:** revised scientific wording and quoted numerical values agree with the canonical products/generated tables: H1 K=8 is **partial/partial**, K=1 **not supported/not supported**; L’s K=1 EDT interval includes zero; H2 ratios are cohort-scoped and absent at K=1. Historical M provenance, `src_proj` scaling, mechanism-as-hypothesis, T60 qualification and corrected yaw ranges/grid are present.
- Independently recomputed **162 curve means/seed SDs and 92 comparison effects**, checked decision flags and convergence calculations, and verified identical query/index ordering across all **66 runs**. Quoted effects and interval bounds match.
- **Corrections 4–5:** Markdown, HTML, four PNGs and four PDFs regenerated at `9b25738` match the main-tree and committed artifacts **byte-for-byte**. Both documents cite the same revision. A second export reproduced all eight figure files exactly.
- **`check_record.py ckpt/exp05/reports` passed, exit 0**, with the four certified attempts behind NAS directory symlinks. The latest report’s SHA-256 is `5e7ddc4510582136c675269a04208598ea6cc2af25a63adbac70766924e84002`. The previous report is unchanged; only documents, figures and HEAD differ between reports.
- **Corrections 6–7:** four recorded trainer commands and the evaluation example match manifests; all five probe summaries and recorded recipes check out. Epoch-12 losses are **0.0159933 / 0.0159770**. S_simple epoch 4 is **68.8357 min**; L_simple epoch 10 **213.8871 min**. Queue logs confirm **20 k0 + 2 yaw** L runs and **40 k0 + 4 yaw** S/M runs. L reviewed/runtime revisions and S/M `ddfcb66` match the erratum.
- **Correction 8:** premature round starts and the missing pre-launch `full` review are disclosed, subject to finding 4. The retained round-6–8 notes exist; all 15 review files have identity headers.

The paper must not claim universal curve dominance, established capacity saturation or positional-code causation, general superiority across training seeds or untested capacities/rooms, significant K=1 inferiority in every cell, proven yaw equivalence or absence of benefit, K=1 efficiency ratios, encoder reductions as full-system/runtime speedups, or complete pre-launch SOP compliance.

**Exact required corrections: findings 1–5 above.** Round-1 corrections 1–5 are closed; 6–8 remain partially closed pending these documentation fixes. All review-created files stayed under `/tmp`; the temporary clone was deleted.
