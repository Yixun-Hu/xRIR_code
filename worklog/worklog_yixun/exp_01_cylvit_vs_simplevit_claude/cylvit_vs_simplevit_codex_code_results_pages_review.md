**Reviewer:** OpenAI Codex (codex-cli 0.144.1, `codex exec`, read-only sandbox, model `gpt-5.6-sol`, reasoning effort `ultra`; session 01a074b5-2588-7ad0-a8c2-6231f64f431c) · **Date:** 2026-09-05 23:30 (−04:00) · **Round:** batched review of the two results-page generators (exp_01 + exp_02), Planner-written one-offs · **Briefing:** SOP, both experiments' records, producers (`tools/compare_eval.py`, `tools/summarize_epochs.py`, `sim_to_real/summarize_haa.py`), input JSONs under `ckpt/`. Prompt: `review_prompts/results_pages_prompt.md` (exp_01 folder).

> Verdict: **Reject**. Findings 1–2 also correct claims made in the exp_01 record (references were not identical across models; room-level inference not established). Planner response and fix commits: see `_worklog.md` entries after 2026-09-05T23:30.

## Summary verdict

**Reject.**

The arithmetic on the currently saved arrays is mostly correct, including experiment A’s epoch-5–12 averaging and the omission of all-NaN metrics. However, the claimed paired evaluations are not reproducibly paired through waveform reconstruction, the bootstrap units do not support the stated verdicts, and experiment B publishes an incomplete run set as a final-looking result.

## Blocking findings

1. **The waveform metrics are not nuisance-paired as claimed.**

   - In experiment A, the DataLoader iterator is created only after architecture-specific model construction ([eval_xRIR_backbone.py:95](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:95), [eval_xRIR_backbone.py:100](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:100), [eval_xRIR_backbone.py:110](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:110)). The CylViT constructor consumes a different amount of Torch RNG state ([xRIR_cyl.py:21](/home/yixunhu/codespace/xRIR_code/model/xRIR_cyl.py:21)), which changes worker NumPy seeds ([eval_xRIR_backbone.py:50](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:50)) and therefore reference selection ([treble_xRIR_dataset.py:170](/home/yixunhu/codespace/xRIR_code/treble_multi_room_dataset/treble_xRIR_dataset.py:170)). A replay found different reference multisets for roughly 16% of queries. The JSON records neither reference IDs nor RNG state ([eval_xRIR_backbone.py:146](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:146)), while the page claims the same query/reference inputs ([make_results_html.py:174](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:174)).
   - Both experiments reconstruct waveforms with random-initialized Griffin–Lim. A does so after the architecture-dependent RNG consumption ([eval_xRIR_backbone.py:119](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:119)); B never seeds Torch before reconstruction ([eval_haa.py:48](/home/yixunhu/codespace/xRIR_code/sim_to_real/eval_haa.py:48), [eval_haa.py:73](/home/yixunhu/codespace/xRIR_code/sim_to_real/eval_haa.py:73)). Consequently, EDT/C50/T60 differences include uncontrolled phase draws.
   - **Fix:** deterministically seed reference selection and Griffin–Lim per query, or use a deterministic phase initialization; persist reference IDs and phase policy/seed; then rerun checkpoint evaluation and regenerate all summaries/pages.

2. **The bootstrap units do not match the advertised inference or verdicts.**

   - A bootstraps 6,337 queries IID ([make_results_html.py:27](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:27)) although they are nested within only 17 held-out rooms, then turns exclusion of zero into “supported” ([make_results_html.py:134](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:134)). A room-cluster bootstrap makes the displayed EDT and loss intervals cross zero.
   - B concatenates repeated evaluations of the same queries across seeds ([make_results_html.py:107](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:107)) and bootstraps every row as independent ([summarize_haa.py:50](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:50)). For classroom T60, the page’s naive interval excludes zero, whereas a two-way query/seed bootstrap crosses zero.
   - **Fix:** state the intended estimand and retain cluster structure. Use a room-cluster/hierarchical bootstrap for A and a query/seed-aware bootstrap for B; report per-seed effects and counts. Replace “tie” with “no detected difference” unless an equivalence margin and test are defined.

3. **Experiment B silently publishes an incomplete, stale snapshot.**

   - The plan requires three seeds per main initialization plus a `released_repomaps` side row ([plan_sim2real_transfer.md:18](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/plan_sim2real_transfer.md:18)).
   - `load_runs` accepts any partial set ([summarize_haa.py:23](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:23)), and the generator immediately computes final verdicts from whatever seeds currently exist ([make_results_html.py:83](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:83), [make_results_html.py:128](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:128)). The model list omits `released_repomaps` entirely ([make_results_html.py:22](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:22)).
   - The page also claims provenance from `summary.txt`, `_results.md`, and `_analysis.md` ([make_results_html.py:159](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:159)), although those artifacts were absent when the reviewed page was generated.
   - **Fix:** encode the expected run manifest and fail final generation unless all required seeds, rooms, metrics, and side runs are present. If partial rendering is needed, require an explicit draft mode that prominently suppresses verdicts.

4. **The claim of numerical self-consistency is false.**

   - A uses 2,000 bootstrap replicates ([make_results_html.py:19](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:19)), while the results record and producer use 10,000 ([cylvit_vs_simplevit_results.md:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results.md:3), [summarize_epochs.py:35](/home/yixunhu/codespace/xRIR_code/tools/summarize_epochs.py:35)). The global RNG is also consumed by per-epoch bootstraps before the reported averaged intervals, so merely changing the count will not reproduce the summary.
   - B hardcodes 5,000 replicates ([make_results_html.py:116](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:116)) versus the producer’s 10,000 default ([summarize_haa.py:61](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:61)). Its `data.json` omits the differences, CIs, counts, and bootstrap configuration actually displayed ([make_results_html.py:124](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:124)).
   - **Fix:** create one canonical structured statistics artifact containing all displayed values and configuration; generate Markdown and HTML from it and assert exact rounded agreement.

5. **Experiment B omits a published reference value.**

   `PAPER` sets complex-room T60 to `None` ([summarize_haa.py:17](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:17)), so the page renders a dash. The published complex-room T60 is **4.33**; only dampened-room T60 is unavailable ([CVPR 2025 paper, Table 2](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Hearing_Anywhere_in_Any_Environment_CVPR_2025_paper.pdf)).

   **Fix:** set complex T60 to 4.33 and regenerate the artifacts.

## Non-blocking findings

- Neither generator validates paired identity before positional subtraction ([A:38](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:38), [B:107](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:107)). Current files align, but both should assert lengths, indices, paths, split, seed, K, and—after fixing finding 1—reference/phase provenance.
- B’s `nanmean` path admits infinities ([make_results_html.py:39](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:39)). Filter with `np.isfinite`, record valid/total counts, and fail on zero valid pairs.
- A’s non-zero-truncated bar axes ([make_results_html.py:78](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:78)) visually magnify small gaps. Label them as zoomed or anchor them at zero.
- Report that the multiple per-room/per-metric intervals are nominal and unadjusted; the current verdict scoreboard encourages family-wise interpretation.

## Questions

1. Are the intended CIs conditional on these exact rooms, queries, and training seeds, or intended to generalize over rooms and seeds?
2. Is experiment B meant to support an interim preview? If so, should draft pages be permitted only with an explicit watermark and no verdict?
