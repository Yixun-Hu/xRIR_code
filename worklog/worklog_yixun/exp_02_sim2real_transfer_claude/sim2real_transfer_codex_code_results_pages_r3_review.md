**Reviewer:** OpenAI Codex (codex-cli 0.144.1, `codex exec`, read-only sandbox, model `gpt-5.6-sol`, reasoning effort `ultra`; session 01a074d7-6de2-7762-9494-9d536927a758) · **Date:** 2026-09-05 23:57 (−04:00) · **Round:** 3 (verifies round-2 fixes) · **Briefing:** SOP, rounds 1–2, fix logs, changed files, artefacts; prompt `review_prompts/results_pages_r3_prompt.md`.

**Reviewer:** OpenAI Codex (GPT-5, API workspace agent, read-only sandbox) · **Date:** 2026-09-05 · **Round:** 3

## Summary verdict

**Reject.** The current artifacts are numerically consistent, and the two-way bootstrap is correct. However, the final completeness gate can still certify malformed or misconfigured runs, canonical-source guarantees remain incomplete, and draft significance interpretation is not fully suppressed.

## Round-2 items

1. **Partially fixed — nuisance-pairing wording.**

   Producer and summary now say references were not identical ([summarize_epochs.py:53](/home/yixunhu/codespace/xRIR_code/tools/summarize_epochs.py:53)). The candidate-count correction and expectation/realized-effect distinction are correct in [results.md:5](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results.md:5) and [analysis.md:17](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_analysis.md:17).

   The generated exp_01 page still states categorically that the nuisances act “without a systematic direction” ([make_results_html.py:178](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:178)). Exp_02 similarly says unseeded Griffin–Lim has “no systematic bias” ([make_results_html.py:161](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:161)). Both should say this is the expectation and the realized nuisance contribution is unknown.

2. **Fixed — two-way query × seed bootstrap.**

   Independent query and seed multiplicities are correctly multiplied at [summarize_haa.py:69](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:69)–[81](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:81). Pairwise finite filtering and counts are correct at [194](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:194)–[214](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:214).

   A separate direct Cartesian-resampling implementation reproduced all 11 stored interval pairs with maximum endpoint difference `1.11e-16`. Classroom T60 reproduced `−0.4203146`, query CI `[−0.6262733, −0.2195104]`, and two-way CI `[−0.9726263, +0.0829201]`.

3. **Partially fixed — completeness gate; blocking.**

   Exact directory seed IDs, expected rooms/counts, `split == test`, finite required metrics, extra runs/groups, matched seed labels, and required zero-valid aborts now work ([summarize_haa.py:87](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:87)–[127](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:127), [199](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:199)–[208](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:208)). The current four missing runs are correctly marked draft.

   Two blocking holes remain:

   - Only `len(index)` is checked. Required metric and `ir_path` lengths, unique identities, and exact split membership are not. A synthetic released hallway run with 423 indices but one EDT value passed `completeness()` and completed as FINAL using that one-point mean.
   - Expected protocol metadata is not enforced. Synthetic K=4, `eval_seed=999`, and bogus backbone metadata passed FINAL because paired checks require only equality, not planned values ([summarize_haa.py:191](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:191)). Training seed and depth-map variant are also not verified; `depth_variant` is omitted from per-sample metadata ([eval_haa.py:99](/home/yixunhu/codespace/xRIR_code/sim_to_real/eval_haa.py:99)–[111](/home/yixunhu/codespace/xRIR_code/sim_to_real/eval_haa.py:111)), so the `released_repomaps` row cannot be authenticated.

4. **Partially fixed — canonical-source enforcement; blocking.**

   Both command records now include `--json`, and current raw, canonical, extract, Markdown, and HTML values agree.

   Enforcement remains weaker than claimed:

   - Exp_01 searches summary rows independently anywhere in Markdown rather than asserting a contiguous verbatim block, and never binds `summary.txt` to `stats.json` ([make_results_html.py:41](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:41)–[47](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py:47)).
   - Exp_02 checks only draft state and the missing list against disk ([make_results_html.py:86](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:86)–[90](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:90)), despite claiming exact per-run agreement. Its chart also recomputes the displayed seed mean ([line 75](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:75)).
   - The exp_02 command record lacks the `--allow-partial` invocation needed to reproduce the present interim artifact.

5. **Partially fixed — draft suppression.**

   Actual interval stars, verdict cells/summary, and exported verdicts are suppressed; `data.json` has `"draft": true`.

   The draft HTML still unconditionally renders the significance legend `* = 95% bootstrap CI excludes 0` ([make_results_html.py:169](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py:169)), retaining the interpretation explicitly identified in round 2.

## Non-blocking

- `paired()` itself assumes finite, nonempty inputs; its sole production caller filters correctly.
- Exp_01 still lacks index/path identity assertions before positional subtraction.
- Exp_01’s page says multiplicity spans four displayed metrics although the producer evaluates five, including log-STFT MSE.
- Deferred regression tests were not treated as a finding.

## Questions

None; the blocking fixes are mechanically defined.
