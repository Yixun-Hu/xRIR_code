# exp_05 param_efficiency — Codex code review, round 6 (record tooling) + integrative merge verdict

**Reviewer:** OpenAI Codex `gpt-6-astra` (reasoningeffort:xhigh), `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt05`, prompt `review_prompts/code_round6_record_prompt.md`, log `param_efficiency_2026-09-18_13:34:20_codex_code_round6_record.log` (a first run, log `…_12:47:56_…_KILLED143.log`, was terminated externally at 77 % of its suite run; this second run recovered that clone's completed suite result). **Scope:** `d9e9759..840e345` on `exp05-record`. **Date:** 2026-09-18 13:34–13:54 EDT.

**Reviewer:** OpenAI Codex (GPT-6, API agent; read-only repository review with isolated scratch execution) · **Date:** 2026-09-18

Reviewed `d9e9759..840e345` and merge integration with main `ecbcc5f`.

**Round-6 verdict: request changes. Integrative verdict: not approved.**  
**Blocking findings: 1, 2, 3 and 4.**

1. **Blocker — Generators can overwrite input evidence.**  
   [make_results_md.py:290](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/make_results_md.py:290), [make_figures.py:84](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/make_figures.py:84).  
   The document guard omits directly consumed `history.jsonl`, `args.json` and `train_manifest.json`. Actual generator calls overwrote each synthetic input. The figure guard checks directories only: a predicted PNG destination hardlinked or symlinked to `CURVE_K8.json` overwrote that JSON. This can destroy the evidence being published.  
   **Minimal fix:** protect every consumed input and preflight every concrete destination using filesystem identity. Add direct-path, hardlink and symlink regressions.

2. **Blocker — Producer-source drift escapes `check_record`.**  
   [bind_provenance.py:347](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/bind_provenance.py:347).  
   `producer_declarations()` checks the sidecar’s internally consistent closure claims, then reuses recorded hashes without verifying producer files. Appending to `tools/param_curve.py` in the isolated clone left the binding report identical, and `check_record` passed. Thus a declared input can change without detection.  
   **Minimal fix:** recompute `producer_identity('tools.param_curve')`, require agreement with the approval and sidecar, and verify producer files at their declared digests.

3. **Blocker — Unapproved evaluator/writer closures are accepted.**  
   [bind_provenance.py:405](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/bind_provenance.py:405).  
   Any non-current evaluator digest is treated as legacy for tier M, without matching an explicitly approved `evaluator_exp04` pin. The writer approval is not enforced. After repairing enclosing bindings, synthetic records passed with an M evaluator replaced by the writer closure and, separately, an S writer replaced by the evaluator closure.  
   **Minimal fix:** validate closure-record consistency and approved evaluator/writer identities for every run. Admit legacy M only against its explicit approval.

4. **Blocker — Cohort reporting violates A7 for valid unequal cohorts.**  
   [make_results_md.py:143](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/make_results_md.py:143).  
   The six-arm row uses EDT’s cohort for all three metrics. A product generated through `param_curve` with one invalid S_cyl C50 query had C50 **n=11**, EDT/T60 **n=12**; the row reported **12**. Both pages also omit H2’s baseline-own room count, per-seed exclusions and H2 room-cluster intervals. These omissions obscure the cohorts and secondary inference required by the plan.  
   **Minimal fix:** render metric-specific query/room counts and the missing canonical fields; test unequal cohorts through the producer fixture.

5. **Should-fix — Registered-profile admission is not shared with generators.**  
   [exp05_record.py:72](/home/yixunhu/codespace/xRIR_code_wt05/tools/exp05_record.py:72).  
   Changing CURVE_K8’s embedded `num_shot` to `123` and refreshing its profile/sidecar digests is accepted by generation, which prints K=123. Only the binder’s separate registered-digest check rejects it.  
   **Minimal fix:** enforce `profile_digest(name)` in shared admission, with synthetic registration injected at the fixture boundary.

6. **Should-fix — HTML uncertainty marks lose their backbone colors.**  
   [make_results_html.py:88](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/make_results_html.py:88).  
   Imported CSS sets `.mark` fill and stroke to blue, overriding the SVG presentation attributes emitted here. Cylindrical dots and uncertainty bars therefore lose their intended orange color; trajectory marks have the same conflict.  
   **Minimal fix:** use backbone-specific CSS rules or inline styles that take precedence.

7. **Nit — The README’s worktree workaround is insufficient.**  
   [README.md:99](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/README.md:99).  
   Passing MAIN’s `--approved` path still leaves producer dependencies rooted in the worktree. Replaying that binding stage rejects MAIN’s `tools/exp04_profiles.py` path.  
   **Minimal fix:** remove this alternative and retain the main-checkout rule, or implement complete checkout routing.

**Verified**

- Read all ten new files and the required plan/notebook/reference context. Both diff checks pass. All ten paths are absent from main; no existing source changed. The exact twelve exp_03 pinned files have empty post-`62c9107` history.
- Python 3.8, CPU only. Preserved completed `static_checks.sh` results: **71 passed**; both collection orders **292 passed, 1 skipped** each. Compilation and shell syntax checks passed.
- Fresh merge rehearsal: `git merge --no-ff --no-commit 840e345` onto **`ecbcc5f`**, **no conflicts**. Fresh record tests on that merged tree: **71 passed**.
- Recovered and inspected the interrupted review’s completed `840e345` clone [full-suite run](/tmp/exp05-round6-review-pq1uu5v4/full-suite.log): **3,050 passed, 49 skipped, 2 failed**, in 42m04s. Both failures reproduce without this branch: exp_07’s known `seen_simple` approval guard, and exp_06’s dry-run golden now encountering completed live stages and printing `SKIP`.
- Fresh live bind: **429.7 s**; independent `check_record`: **425.6 s**, identical report. Coverage includes **66 evaluations, four certified trainings, all five ledger-listed probes, two pinned historical M checkpoints, five products, two documents and eight figures**. Branch tools were loaded from scratch with checkout roots routed to MAIN; outputs stayed in scratch.
- All twelve Markdown/HTML tables agree. Displayed numerical fields trace to canonical JSON or digest-checked attempt evidence. M rows use exp_05’s products; figures share HTML’s canonical `series()`. Both K families are present. Live H1 interval decisions correctly give “partial” for EDT/C50 at K=8 and “not supported” at K=1.
- Independently checked approval path/SHA/git-blob identity, live producer/profile identities and recorded execution pins. Documents cite all five product digests and four completion digests. Optional figures are recorded in `inputs`.
- Fresh mutations confirmed curve/interval/verdict presentation changes and refusals for parameter-count, throughput and epoch-loss corruption. Foreign-K declarations and omissions are refused. Synthetic aborted attempts and external-log drift are bound.
- The fixture publishes through `param_curve.admit/analyze/publish`. Markdown tests co-landed with implementation; the disclosed lack of red-first testing remains an SOP process deviation.

Reproductions and verification details are retained in the [review scratch notes](/tmp/exp05-round6-rereview-z4md5a6e/review_notes.md). The source worktree and live checkpoint evidence were left unchanged.
