# exp_05 param_efficiency — Codex record review (round 3, closure)

**Reviewer:** OpenAI Codex `gpt-6-astra` (reasoningeffort:xhigh), `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt05` (worktree at `615579a` = main), prompt `review_prompts/record_review_round3_prompt.md`, log `param_efficiency_2026-09-18_22:04:46_codex_record_review_round3.log`. **Date:** 2026-09-18 22:04–22:21 EDT.

**Reviewer:** OpenAI Codex (GPT-6, API agent), read-only · **Date:** 2026-09-18 EDT · **Marker:** `record`, round 3  
**Scope:** worktree `615579a`; record read from main; canonical regeneration at `9b25738`.

**Verdict: record approved with corrections.** Two documentation corrections remain.

1. **Medium — Restore two omitted commits.** The regenerated [commit index](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/commits_param_efficiency.md:4) correctly labels cutoff `ec75704` and includes `0e35e2c`, but drops two entries present before this correction:
   - `a3eb253` — Expose trainer ViT tiers with historical M step parity.
   - `dc0c895` — test(exp05): compare tier evaluation and smoke checkpoints on GPU.

   `git log` over the declared tools/tests/record/shared-launcher scope yields **130 entries; the index contains 128**. Restore these two.

2. **Low — Correct the wall-time explanation and notebook summary.** The [new footnote](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_params_set_up.md:17) incorrectly calls the generated results’ wall times “the trainer-side sum of epoch times.” They come from **`completion.json.wall_hours`**, measured by the launcher through post-run validation. They exceed the manifest→execution intervals by **37–44 seconds**.

   Keep the setup table’s correct values: **11.3964 / 11.9857 / 41.6595 / 46.0840 h**. Replace the footnote’s explanation accordingly, and append an erratum to the [notebook summary](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_worklog.md:213): its L values **41.67 / 46.10** are completion-based; the stated timestamp calculation gives **41.6595 / 46.0840**.

Verification completed:

- **Commands:** all six retained scripts match their original scratch copies byte-for-byte. Five probes, four trainings, both evaluation queues (**22 + 44 runs**) and five producer input sets match receipts, manifests and logs. Both record runs are correctly identified as superseded/canonical.
- **GPU disclosure:** all **46 exp_04 manifests specify GPU 0**; L_cylindrical used GPU 1. Shared-host overlap is supported.
- **Chronology:** review headers confirm September 13 close-outs and September 14 rounds 4–5, supporting the corrected launch ordering. The approximate 00:53 round-2 time comes from the notebook; its review header says 00:58. The missing pre-launch `full` review and premature round starts remain disclosed.
- **Hours:** cumulative values match the ledgers; all four new wall-time values match manifest/execution timestamps.
- **Artefacts:** Markdown, HTML and all eight figure files are byte-identical to the canonical `9b25738` regeneration committed in `ec75704`. The binding SHA-256 remains `5e7ddc4510582136c675269a04208598ea6cc2af25a63adbac70766924e84002`.
- **`check_record.py ckpt/exp05/reports` passed, exit 0**, from the recorded main-checkout paths using code verified identical to the reviewed worktree.

**Exact required corrections: items 1–2 above.** No repository files were changed. Temporary caches were removed; no clone was created.
