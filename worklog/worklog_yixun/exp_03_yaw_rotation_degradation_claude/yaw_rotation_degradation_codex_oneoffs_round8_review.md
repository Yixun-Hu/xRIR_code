# Codex review — Planner one-offs and final documents, round 8

Reviewer: OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra), independent of the Planner (Claude) and Coder (Claude Opus subagent).
Prompt: review_prompts/oneoffs_round8_prompt.md · started 2026-09-06 23:53:37

---

## Verdict: request changes

The retained quantitative evidence is sound, but round-7 binder requirements 1–2 remain incompletely implemented and tested. No evaluator, sweep, or statistical-summary rerun is needed.

### 1. Round-7 closure table

| # | Status | Assessment |
|---|---|---|
| 1 | **Partial — blocking** | `last_migration` idempotence and pre-write validation are fixed. However, a subsequent legitimate binding migration appends the sidecar’s `binding_history` at [bind_provenance.py:350](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:350), while `_deep_drift()` treats lists atomically at [line 174](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:174). `report_drift()` therefore rejects the appended history at lines 214–221. I reproduced both resulting drift errors. |
| 2 | **Partial — blocking coverage defect** | The filesystem test at [test_exp03_record_tools.py:465](/home/yixunhu/codespace/xRIR_code/tests/test_exp03_record_tools.py:465) hand-constructs `pv2` by adding `extra` without performing the sidecar-history append that `main()` performs. Its refusal arm merely elects not to call `commit_transaction()` and then observes no write. It therefore misses the defect and is not an integrated projection→validation→commit/refusal test. |
| 3 | **Closed** | `run_start` must be a nonempty exact string at [check_sweep_acceptance.py:244](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:244); inventory counts are strict non-boolean integers at [line 268](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:268); gate `batch_canonical is True` is required at [line 343](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:343). |
| 4 | **Closed** | All requested adversarial forms are covered at [test_exp03_record_tools.py:501](/home/yixunhu/codespace/xRIR_code/tests/test_exp03_record_tools.py:501); the focused log reports 31 passed. |
| 5 | **Partial — record defect** | The command section and retractions are present. But the substantive 23:48 and 23:49 worklog entries omit several required SOP fields, and the 23:47 Version Control field is incomplete; compare [worklog:452](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:452) with [SOP:47](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:47). The retained log also contains only one v7 binder invocation, so the separately claimed “immediate rerun” is not retained. |
| 6 | **Not closed — record defect** | The disclosure at [analysis.md:14](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:14) gives the wrong history. The three sweeps were already bound by v2 at [20:20 log:36](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_2026-09-06_20:20:18_bind_provenance.log:36); all 13 were bound by v3 at [21:06 log:112](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_2026-09-06_21:06:20_bind_provenance_v3.log:112), then rebound by v4. No current sidecar or report contains `binding_history`; the report’s surviving `written` is the v5 migration time at [binding_report.json:2250](/home/yixunhu/codespace/xRIR_code/ckpt/yaw_rotation/binding_report.json:2250). |
| 7 | **Closed** | Fresh binder, recomputing acceptance, focused-test, static-check, and full-suite logs exist. All four canonical/copy hashes match: full stats `5d4afce8…`, summary `52f1e62e…`, gate stats `248fc987…`, binding report `f4dcb560…`. |
| 8 | **Open** | The remaining auxiliary gaps were not added to the analysis as open tooling items, as required by the decision rule at [worklog:443](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:443). |

### 2. Evidence-path decision

For the retained record itself, I found no defect in:

- The confirmatory numbers or verdicts: H1 is correctly “not supported” and H2 correctly “not evaluable”; the four room-level C50 exceptions are also correctly reported.
- The current canonical artifacts or their copies.
- The current 13 sidecars, their report records, artifact/checkpoint/log digests, or source/data binding.
- Acceptance v7. The retained run recomputes the gate and ends `ACCEPTANCE PASS` in [the v7 log](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_2026-09-06_23:48:25_acceptance_v7.log:1). I found no malformed-input path capable of making different scientific bytes pass as this record.

A defect nevertheless remains in the binder’s required additive-migration contract. It is fail-closed—it cannot make a wrong current record pass—but it directly repeats the class of binder defect that round 7 required fixing.

### 3. Remaining auxiliary items

| Item | Classification |
|---|---|
| Launcher copy/refresh steps remain prose/manual, and the decision command uses the working-tree producer | **Defect** |
| No real clean launcher end-to-end run; real guard and rename-failure paths remain unexercised | **Residual** |
| Renderers still accept single-row deletion, all-null TOST payloads, and empty final verdict/bound payloads | **Defect** |
| Raw/truncated spectral identifiers remain in Markdown headings | **Defect** |
| Missing renderer-corruption and clean-launcher integration tests | **Residual** |
| Sidecars/report are replaced sequentially, so the transaction is not crash-atomic across files | **Residual** |
| “All-or-nothing” / “all atomically” wording contradicts that crash residual | **Defect** |
| A changed sidecar digest is accepted when the embedded old/new content is identical; this cannot pass wrong current bytes but is an immutability-hardening gap | **Defect** |
| Incomplete SOP fields, inaccurate binding-history disclosure, unsupported retained-rerun claim, and failure to copy the auxiliary list into the analysis | **Defect** |

Fix the history-append comparison, test the actual projection path, correct the provenance chronology, and complete the record disclosures before committing. I made no file changes.
codex exit 0 at 00:04:18
