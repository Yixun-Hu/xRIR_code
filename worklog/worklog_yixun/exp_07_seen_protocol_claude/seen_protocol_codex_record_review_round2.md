# exp_07 seen_protocol — Codex record review (round 2)

**Reviewer:** OpenAI Codex `gpt-6-astra` (reasoningeffort:xhigh), `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07` (worktree at `515eac9`), prompt `review_prompts/record_review_round2_prompt.md`, log `seen_protocol_2026-09-20_02:39:10_codex_record_review_round2.log`. **Date:** 2026-09-20 02:39–02:46 EDT.

**Reviewer:** OpenAI Codex `gpt-6-astra`, codex-cli 0.154.0, `codex exec`, read-only · **Date:** 2026-09-20 · **Marker:** `record`, round 2  
Reviewed MAIN record and worktree at `515eac9`, including `42d142d`.

**Verdict: record approved with corrections. Two corrections remain, within round-1 items 1 and 5.**

1. **Medium — The analysis still claims reproduction of the paper’s row.**  
   [seen_protocol_analysis.md:20](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_analysis.md:20) retains “the same recipe reproduces and exceeds the paper’s seen row.” This contradicts the corrected distinction between historical calibration and the published row, and exceeds the registered checkpoint-descriptive scope.  
   **Minimal fix:** replace that conclusion with: “At K = 8, these three retrained checkpoints have lower mean errors than the released checkpoint on all three metrics.” The historical calibration rule, geometry-hypothesis label and qualified K = 1 ranking are otherwise corrected.

2. **Medium — The refused-smoke command records the wrong revision and remains abbreviated.**  
   [seen_protocol_command.md:55](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_command.md:55) specifies `076cd40`, but the [retained chain log:131](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_20260916T202013_gpu0_chain.log:131) records `ddfcb66718a560b0a59c2e66176f391eeb2fc4ab` for the September 18 09:38 refusal.  
   **Minimal fix:** use that revision and replace the ellipsis with `--log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260918T093826smoke`. The other added command phases, retained launcher scripts, setup values, receipt/ledger paths and retrospective disclosures are verified.

**Verified closures:**

- **Item 2:** Released-minus-simple K = 1 estimates and intervals match the canonical JSON; the T60 room interval includes zero.
- **Item 3:** Manifests and logs confirm `483b37c` for seen_aug, `f5e1cf6` for released K = 1, and `0e47341` for production. The dated erratum correctly separates them.
- **Item 4:** All 60 statistics use **20,000** resamples; maximum convergence ratio **0.0182558**. No retry occurred. The errata explain all three unchanged summary footers.
- **Item 6:** The finish is explicitly estimated/unconfirmed on September 18; the separate test-suite overlap is retained.
- **Item 7:** The regenerated index matches all **192** entries through `b50202c`, includes the nine previously omitted commits, and separately documents `02bf1e2`.

Regeneration at **`0e47341`** produced byte-identical Markdown, HTML and LaTeX. The retained figure and binding digest are unchanged. **`check_record.py ckpt/exp07` passed, exit 0.** Review writes stayed under `/tmp`; the temporary clone was deleted.
