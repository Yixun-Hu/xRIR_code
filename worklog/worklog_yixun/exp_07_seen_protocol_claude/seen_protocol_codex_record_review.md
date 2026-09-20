# exp_07 seen_protocol — Codex record review (round 1)

**Reviewer:** OpenAI Codex `gpt-6-astra` (reasoningeffort:xhigh), `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07` (worktree merged with main `b50202c`), prompt `review_prompts/record_review_prompt.md`, log `seen_protocol_2026-09-20_02:26:58_codex_record_review.log`. **Date:** 2026-09-20 02:27–02:38 EDT.

**Reviewer:** OpenAI Codex `gpt-6-astra`, codex-cli 0.154.0, `codex exec`, read-only review · **Date:** 2026-09-20 · **Marker:** `record`  
Reviewed code and MAIN record at `b50202c`; regenerated outputs at recorded revision `0e47341`.

**Verdict: record approved with corrections.** The numerical artefacts, approval pins and binding pass verification. Findings 1–7 below are the required corrections.

1. **Medium — The interpretation exceeds the registered descriptive scope.**  
   [seen_protocol_analysis.md:20](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_analysis.md:20), [seen_protocol_worklog.md:88](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_worklog.md:88).  
   The calibration reproduced **exp_01’s historical reproduction**, not the published Table-1 row: the retained paper screenshot gives K = 8 seen values **0.038 / 0.940 / 8.13**, whereas the calibration targets **0.0389 / 1.029 / 7.27**. Also, “leaves less room for the encoder’s geometry prior” presents an untested explanation as a result. “Cylindrical … worst at K = 1” needs qualification: the released reference has higher EDT and C50.
   
   **Minimal fix:** distinguish historical calibration from the paper’s reported numbers; state its registered rule, `|mean − historical| ≤ 3 SD + 2% |historical|`; describe observed checkpoint differences, label the geometry explanation as a hypothesis, and qualify rankings as “among the three recipe arms.”

2. **Medium — The analysis omits the released-reference pairing at K = 1.**  
   [seen_protocol_analysis.md:18](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_analysis.md:18).  
   The other pairings cover both K values, but released-versus-simple stops at K = 8. This conceals the change in direction for T60.
   
   **Minimal fix:** add released minus seen_simple at K = 1: **EDT +9.07% [8.05, 10.10], C50 +8.30% [7.58, 9.02], T60 −2.30% [−3.12, −1.48]**, using descriptive query intervals. Note that the T60 room interval **[−5.78, +0.81]%** includes zero.

3. **Medium — The recorded evaluation revision is wrong for seen_aug.**  
   [seen_protocol_analysis.md:28](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_analysis.md:28), [seen_protocol_worklog.md:159](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_worklog.md:159).  
   All ten manifests and the queue-start log name **`483b37c`**, not `e9a0ecf`, as `reviewed_commit`. Likewise, “all at `0e47341`” in the end-game entry conflates phases: released K = 1 evaluations used **`f5e1cf6`**; the pin commit and subsequent production used `0e47341`.
   
   **Minimal fix:** correct the analysis and append a dated notebook correction separating evaluation revisions from the production revision. The evaluator/writer closures remain identical across these revisions.

4. **Medium — The record claims a 40,000-resample rerun that did not occur.**  
   [seen_protocol_worklog.md:161](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_worklog.md:161), [tools/exp07_pairs.py:238](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_pairs.py:238).  
   All **60 absolute/relative statistic objects** retain `n_boot = 20000`; none needed reconvergence. The largest convergence ratio is **0.01826**, below **0.10**. The summary footer misleadingly prints “Reconvergence re-run at n_boot 40000” whenever `--reconverge` was supplied, even when no cell was rerun.
   
   **Minimal fix:** append an explicit erratum covering the notebook and all three summary footers: **20,000 resamples; optional 40,000-resample retry enabled but never triggered**. Preserve the historical bound artefacts; do not imply that the flag proves a retry occurred.

5. **Medium — The command and setup records are incomplete under the SOP.**  
   [seen_protocol_command.md:27](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_command.md:27), [seen_protocol_command.md:49](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_command.md:49), [seen_protocol_params_set_up.md:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_params_set_up.md:3).  
   The command file omits the pre-training parity/audit/released-K8/calibration phase and the **September 18 09:38 refused seen_cyl smoke**. Evaluation and production entries contain placeholders. Their pointer to `eval_manifest.json` supplies the **child evaluator command**, not the complete launcher invocation. The setup file is explicitly retrospective and omits settings such as architecture dimensions, training K/max length and yaw seed/width.
   
   **Minimal fix:** add complete commands or pinned executable script invocations for the missing phases, refusals and restarts; supply the complete configuration and explicit receipt/ledger paths. Mark reconstructed entries as retrospective and acknowledge the launch-time documentation deviation rather than backdating them.

6. **Low — An estimated co-tenant finish is presented as an observed timing boundary.**  
   [seen_protocol_analysis.md:28](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_analysis.md:28).  
   The notebook’s [18:36 entry](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_worklog.md:112) calls **01:30 September 18 an expected finish**, not a confirmed stop. The analysis turns this into an actual overlap window and omits the date rollover.
   
   **Minimal fix:** identify the observed September 17 start and label the September 18 finish as estimated/unconfirmed. Keep the separately documented **19:22–21:14** test-suite overlap.

7. **Medium — The “complete” commit index excludes experiment implementation and restoration commits.**  
   [commits_seen_protocol.md:4](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/commits_seen_protocol.md:4).  
   Its path filter misses commits that touched shared modules before A1: **`e122355`, `d7511fb`, `2fe4395`, `9a0fbe0`, `e4832f2`, `a035d7f`, `0a2e518`, `25b5402`, `18dacdc`**. These include the original changes and their restorations, so their omission obscures A1’s implementation history.
   
   **Minimal fix:** add those commits, the relevant rerun revision `02bf1e2`, and the final record commit `b50202c`; update the count and cutoff.

**Verified**

- Regenerated Markdown, HTML and LaTeX under `/tmp` using the generators at **`0e47341`**: **all three diffs empty**.
- LaTeX follows the supplied Table-1 column layout, includes both K values, K-labelled nonzero SD footnotes, correct unrounded-mean bolding among the three recipe arms, and canonical-digest caption pointers.
- **`check_record.py ckpt/exp07` passed, exit 0**, from the canonical MAIN paths. The worktree and MAIN’s 21 exp_07 Python tool/record files are byte-identical.
- Independently recomputed every table mean/sample SD and all paired point estimates/cohort means from the per-query files: matched. All 40 runs contain **6,217 finite queries** with matching query order.
- All six approval closures recomputed at the fill revision match; all three checkpoint hashes match approvals and certified completions. Evaluation and training manifests match the corresponding pins.
- Calibration passes all three registered inequalities and predates training; parity records **9/9 passes**; alignment audit records **0 changed delays / 768 pairs**.
- Three certified 12-epoch attempts, their probes and ledgers, and the failed cylindrical smoke are retained. All epoch-1 and cumulative timing gates passed.
- The nine dirty-tree evaluation refusals and subsequent completions are present. Cylindrical evaluations correctly split **1 at `38a6727` / 9 at `02bf1e2`**.
- All 14 prior review files contain identity/date headers. A1’s initial merge changed no pre-existing non-worklog file.
- Review writes stayed under `/tmp`; the temporary clone was deleted.

The paper must not claim method superiority/equivalence, training-seed robustness, a demonstrated geometry mechanism, seen-room results as unseen-room generalisation, yaw robustness from these k = 0 evaluations, or runtime speedups from the contended timings. K = 1 denotes evaluation with one reference, not separately trained K = 1 models; the released checkpoint has unknown training budget.

**Required corrections: findings 1–7. No retraining or reevaluation is indicated by this review.**
