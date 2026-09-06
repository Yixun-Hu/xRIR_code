**Reviewer:** OpenAI Codex (codex-cli 0.144.1, `codex exec`, read-only sandbox, model `gpt-5.6-sol`, reasoning effort `ultra`; session 01a074cb-ba0f-7481-9dcc-b638b95ea152) · **Date:** 2026-09-05 23:45 (−04:00) · **Round:** 2 of the results-page generator review (verifies round-1 fixes) · **Briefing:** SOP, round-1 review, changed files, updated records; prompt `review_prompts/results_pages_r2_prompt.md`.

**Reviewer:** OpenAI Codex (GPT-5, API workspace agent, read-only sandbox) · **Date:** 2026-09-05 · **Round:** 2

## Summary verdict

**Reject.**

The current artifacts are numerically fresh and mutually agree at displayed precision. Exp_01’s room-cluster analysis and finding 5 are fixed. Blocking defects remain in exp_02’s seed inference, final-run completeness gate, canonical-source enforcement, and draft verdict suppression.

## Round-1 findings

1. **Partially fixed — nuisance pairing.**

   Exp_01 now discloses the defect and schedules a properly paired exp_03 k=0 rerun ([results.md:5](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results.md:5), [exp_03 plan:83](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:83)). The original arrays remain unpaired, however, and the regenerated producer output still says “identical references” ([summarize_epochs.py:58](/home/yixunhu/codespace/xRIR_code/tools/summarize_epochs.py:58), [summary.txt:1](/home/yixunhu/codespace/xRIR_code/ckpt/backbone_comparison_summary.txt:1)).

   The correction’s claim that receivers with “≤8 candidates get the same set” is also wrong for fewer than eight: the dataset falls back to random sampling with replacement ([treble_xRIR_dataset.py:183](/home/yixunhu/codespace/xRIR_code/treble_multi_room_dataset/treble_xRIR_dataset.py:183)). “No systematic direction” should be qualified as an expectation; the realized nuisance effect is unknown.

   Exp_02 reference selection is deterministic and asserted, but Griffin–Lim remains unseeded ([eval_haa.py:73](/home/yixunhu/codespace/xRIR_code/sim_to_real/eval_haa.py:73)). The page now states that caveat, so this is an acceptable interim disclosure, not a completed pairing fix.

2. **Partially fixed — bootstrap units.**

   Exp_01 is fixed. Whole rooms are resampled correctly while retaining their queries ([summarize_epochs.py:27](/home/yixunhu/codespace/xRIR_code/tools/summarize_epochs.py:27)). Independent 200,000-resample checks reproduced the stored room intervals within Monte Carlo error, and raw means/per-room effects exactly matched the canonical JSON.

   Exp_02 correctly clusters repeated rows by query, but never resamples training seeds ([summarize_haa.py:50](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:50)). Thus its CI is conditional on the realized seeds, not query-and-seed-aware inference. On the current two-seed classroom T60 snapshot:

   - Implemented query-only CI: `[-0.626, -0.220]`
   - Two-way query/seed CI: approximately `[-0.97, +0.08]`

   The nominal conclusion changes. Per-seed effects are shown, but per-seed valid counts are not.

3. **Partially fixed — completeness and draft gating.**

   The current page is prominently marked DRAFT, the producer requires `--allow-partial`, and `released_repomaps` is represented.

   The final gate remains unsafe: `EXPECTED_RUNS` specifies only minimum counts, while `completeness()` checks merely that four room keys exist ([summarize_haa.py:67](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:67)). It does not enforce exact seed IDs, matched cyl/control seed sets, required metrics, split sizes, metadata, or reject extra runs. An in-memory manifest containing arbitrary seed names and empty `{room: {}}` payloads returned `(True, [])`. Cyl/control seeds are only intersected ([summarize_haa.py:125](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:125)), so disjoint count-complete sets could produce a FINAL page with no paired results. Zero-valid required cells are silently skipped.

4. **Partially fixed — canonical consistency.**

   Bootstrap calculations are centralized at 10,000 replicates, and the current JSON, summary, Markdown, and HTML numbers agree.

   The advertised invariant is not enforced:

   - Exp_01 duplicates paper/released constants and rereads history/baseline despite claiming every displayed statistic is canonical ([make_results_html.py:18](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:18), [make_results_html.py:36](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:36)). No assertion checks `_results.md`.
   - Exp_02 rereads raw runs and recomputes displayed means, standard deviations, and charts ([make_results_html.py:32](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:32)). Its one-way, six-decimal mean check does not validate seed labels, key completeness, paper values, missing-list agreement, or paired CIs ([make_results_html.py:94](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:94)).
   - Both command records still omit the required `--json <path>` invocation.

5. **Fixed — paper complex-room T60.**

   `4.33` is present in the producer, canonical JSON, and generated HTML.

## New blocking finding

Draft verdict suppression is incomplete. The HTML retains significance stars and their interpretation, while the draft `data.json` exports all eleven textual verdicts without a draft flag ([make_results_html.py:113](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:113), [data.json:558](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/data.json:558)).

## Non-blocking

- Exp_01 still performs positional subtraction without identity/meta assertions; current 12×2 inputs do align.
- Exp_02 does not assert `split` or persist reference/phase provenance.
- Finite filtering, zoomed-axis labels, “no detected difference,” and multiplicity caveats are fixed.
- Bootstrap seeds and unique-cluster counts should be recorded in both canonical artifacts.
- No regression tests cover the new bootstrap, completeness, or canonical-agreement logic.

## Questions

1. Are exp_02 CIs intentionally conditional on these exact training seeds? If not, seeds must also be resampled.
2. Does “suppress verdicts” include significance stars and the exported `data.json`? The current implementation treats them differently.
