# exp_05 param_efficiency — Codex code review, round 7 close (record tooling) + integrative merge verdict

**Reviewer:** OpenAI Codex `gpt-6-astra` (reasoningeffort:xhigh), `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt05`, prompt `review_prompts/code_round7_record_close_prompt.md`, log `param_efficiency_2026-09-18_16:06:07_codex_code_round7_record_close.log`. **Scope:** round-7 commits through `8535fae` on `exp05-record`; merge rehearsal onto main `8ce27e7`. **Date:** 2026-09-18 16:06–16:32 EDT.

**Reviewer:** OpenAI Codex (GPT-6, API agent; read-only repository review with isolated scratch execution) · **Date:** 2026-09-18

Reviewed the seven specified round-7 commits through `8535fae`, plus the integrative `full` merge against main `8ce27e7`.

**Round-7 verdict: request changes. Integrative verdict: not approved.**  
Round-6 findings **1–7 are resolved**. **The exact blocking list is new finding 1 below.**

1. **Blocker — Moving an attempt behind a directory symlink breaks the bound record.**  
   [bind_provenance.py:202](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/bind_provenance.py:202), [dependency comparison:463](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/bind_provenance.py:463).

   Starting with a passing synthetic binding, I moved `S_simple/attempt_*` to a simulated NAS directory and replaced its original location with a directory symlink. No evidence bytes changed. Both `check_record` and fresh `collect` failed because the binder looked for `cumulative_hours.json` under the resolved destination’s parent. Probe receipts, other attempts and `final` are likewise located from that parent.

   Making those siblings accessible exposed a second failure: the binder rejected the original `args.json` dependency because its recomputed dependency keys used resolved NAS paths, while the published sidecars retained the original paths. Restoring the directory made the unchanged report pass again.

   **Why it matters:** `check_record` would **not** survive the Planner’s proposed post-bind migration, despite unchanged evidence.

   **Minimal fix:** preserve stable logical paths in persisted bindings and dependency maps; locate arm metadata from the logical approved arm directory; retain digest and filesystem-identity validation. Add a regression that binds, relocates an attempt behind a directory symlink, then successfully checks the **unchanged report and products**.

**Verified**

- **Overwrite protection:** 216 document attempts covering both generators, 36 concrete inputs, and direct/hardlink/symlink destinations were refused without changing input bytes. All eight figure destinations were checked against seven consumed inputs through both alias types: **112 refusals before any additional figure write**. Eight direct figure collisions using a valid renamed companion were also refused.
- **Producer and closure admission:** producer drift now fails `collect` and `check_record`. Appending to the actual cloned `tools/param_curve.py` fails live identity validation; independently holding that identity fixed still triggers the declared-file digest check. Repaired evaluator/writer swaps and forged closure records are refused. Explicit legacy admission was verified at the helper level only; the current producer prevents an end-to-end positive legacy case.
- **Profiles and presentation:** K=123 is refused by shared admission. Inline SVG mark styles override the imported blue rule. Both pages correctly report S_cyl EDT/C50/T60 cohorts of **12/11/8 queries**, **3/3/2 rooms**, and **0/1/4 exclusions per seed**, for both K families. H1 own-cohort rooms and H2 baseline rooms and room intervals match canonical fields. The README workaround is removed.
- **Live record:** fresh MAIN-tree bind passed in **443.5 s**; independent `check_record` passed in **434.2 s**, reproducing the report exactly. Coverage remains **66 evaluations, four trainings, five ledger-listed probes, two historical M checkpoints, five products, two documents and eight figures**. Execution evidence and approval sections match round 6.
- **Numerical traceability:** all twelve HTML/Markdown tables agree; numerical fields trace to canonical products or digest-checked attempt evidence. M rows use exp_05 products. Both documents cite all nine required digests. All four PNGs are byte-identical to round 6. Live H1 remains “partial” for both K=8 metrics and “not supported” for both K=1 metrics.
- **Fresh checks:** `static_checks.sh` exited **0**: **86 passed**; both collection orders **307 passed, 1 skipped** each. The rehearsed merged tree also passed **86 record tests**.
- **Full-suite evidence:** inspected the coder’s completed **3,191 passed / 49 skipped / 2 failed** log. Independently reproduced both failures on main `8ce27e7` **without this branch**: exp_06’s stale `compare`/`eval_launch` pins and exp_07’s null `seen_simple` approval. This was not a fresh full-suite rerun.
- **Integration and hygiene:** `git merge --no-ff --no-commit 8535fae` onto main `8ce27e7` was conflict-free and staged exactly **ten additions**. The literal current-main diff has thirteen paths because main subsequently added three review-record changes; the actual merge preserves those. Whitespace checks pass. The twelve pinned exp_03 files have empty post-`62c9107` history. Python **3.8.20**, CPU only, scratch-only writes; reviewed source and live evidence remain unchanged.

The supplied exp_04 attempt appeared as an ordinary directory to `stat`/`readlink` in this environment. The relocation finding comes from the independent exp_05 reproduction above. The disclosed missing separate red observation for item 5 remains a process deviation, not an additional blocker.

[Reproductions, commands and verification notes](/tmp/exp05-round7-review-atEbPQ/review_notes.md).
