# Codex review — Planner one-offs and final documents, round 4

Reviewer: OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra), independent of the Planner (Claude) and Coder (Claude Opus subagent).
Prompt: review_prompts/oneoffs_round4_prompt.md · started 2026-09-06 21:28:39

---

The v3 numerical result remains credible and unchanged, but the record is not commit-ready. The current 13 runs are internally healthy; the remaining blockers are fail-open acceptance/provenance contracts and incomplete producer-only rendering. No files were modified.

Independent checks:

- Current acceptance: `ACCEPTANCE PASS`.
- Targeted adversarial suite: 9 passed, though it does not cover the holes below.
- `bash -n` and `git diff --check`: clean.
- Canonical/copy SHA-256 pairs match: stats `0fc42a6…`, summary `52f1e62e…`, binding report `68ceecbd…`.
- The analysis, Markdown, and HTML numerical results agree with v3 JSON. `delay_flips_max` is correctly absent.
- Current raw gate/sweep arrays are finite and aligned with their actual manifests; no current-run corruption was found.

## Findings

1. **blocker — The v3 gate can still be fabricated or become stale.**

   **File:** [check_sweep_acceptance.py:83](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:83), [check_sweep_acceptance.py:212](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:212), [check_sweep_acceptance.py:250](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:250)

   **What is wrong:** The checker trusts `gate_pass` rather than recomputing the v3 decision. `band_rule.startswith("v3")` accepts arbitrary v3-prefixed text. A nonexistent `summary_path` skips digest and chronology checks; I verified that changing the real gate JSON to a nonexistent summary produced no validation errors. Gate-run arrays, seed-1/2 manifest order, aggregates, environment, and full provenance are not checked. Neither the gate producer, the two exp_01 per-sample inputs, nor the complete gate-input digest set is bound.

   **Why it matters:** An old passed summary can authorize replaced or failing gate artifacts.

   **Minimal fix:** Recompute the exact v3 decision from the ten bound runs, three manifests, pinned exp_01 inputs, baseline, and pinned producer; compare every derived row/reason and the retained summary. Require the exact band-rule value, an existing canonical summary, canonical/retained gate-stat identity, and verified chronology. Add stale-artifact, stale-manifest, missing-summary, and nonfinite-gate tests.

2. **blocker — Sweep content validation still has fail-open cases.**

   **File:** [check_sweep_acceptance.py:125](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:125), [check_sweep_acceptance.py:171](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:171), [check_sweep_acceptance.py:294](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:294)

   **What is wrong:** Spectral `n_valid`/`n_nan` are not reconciled. All-null acoustic arrays accept an arbitrary aggregate mean, including `NaN`/`Infinity`. Integer coercion lets fractional angle grids pass. Decomposition does not require `k=32` and `n_batches=1`. Adversarial in-memory probes for these mutations returned no failures.

   **Why it matters:** A malformed, complete-looking sweep can still print `ACCEPTANCE PASS`.

   **Minimal fix:** Enforce exact non-boolean types and grids; reconcile counts for every metric; require `mean is null` exactly when zero values are valid and finite otherwise; pin decomposition metadata and consistency; add tests for each case.

3. **blocker — Retained provenance is improved but still does not bind all historical evidence.**

   **File:** [check_sweep_acceptance.py:187](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:187), [bind_provenance.py:76](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:76), [bind_provenance.py:129](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:129), [bind_provenance.py:175](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:175)

   **What is wrong:** Acceptance does not compare live sidecars with the retained binding report; environment values are checked only for presence, `scipy` is omitted, and any truthy inventory token passes. I verified that invented environment and inventory values pass. The binder inventories dataset `(path, size)`, not file bytes, so same-size changes are invisible. Logs are not hash-bound, and the gate producer/exp_01 inputs are absent.

   **Why it matters:** The report proves current artifact/checkpoint/sidecar consistency, but not every decision-driving byte or historical runtime claim.

   **Minimal fix:** Content-hash all referenced dataset files or bind a trusted immutable dataset version; retain inventory/chronology evidence; bind gate-producer, exp_01 inputs, and retained logs; compare the exact 13 live sidecars and their hashes against both binding-report copies; recompute data identity during acceptance. Continue labeling environment capture as post hoc.

4. **blocker — The producer-only rendering rule remains unmet.**

   **File:** [make_results_md.py:46](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:46), [make_results_md.py:59](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:59), [make_results_md.py:93](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:93), [make_results_html.py:167](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:167), [make_results_html.py:224](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:224), [make_results_html.py:233](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:233)

   **What is wrong:** Most display names now use producer fields, but several visible sites still iterate or print raw `control/cyl/released`, raw `edt/c50`, `.upper()` metric names, and hard-coded `EDT`, `C50`, `s`, and `dB`. HTML H2 visibly renders lowercase metric keys. Scientific H1/H2 headings remain locally authored. HTML silently falls back for missing note fields, while Markdown does not require `decomposition_note`.

   **Why it matters:** There is no current numerical mismatch, but changing producer labels can produce internally inconsistent consumers while they still render FINAL.

   **Minimal fix:** Iterate from `config.labels`; use `run_descriptions`, `metric_names`, and `metric_units` at every visible site; serialize remaining scientific headlines/glosses required by the stated rule; require every producer field without fallback. Add a test that deliberately changes producer labels, then regenerate Markdown, HTML, and `data.json`.

5. **should-fix — Launcher v3 cannot execute a clean gate→sweep workflow and still soft-fails abort preservation.**

   **File:** [launch_yaw_sweep_v3.sh:69](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:69), [launch_yaw_sweep_v3.sh:92](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:92), [launch_yaw_sweep_v3.sh:136](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:136), [bind_provenance.py:145](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:145)

   **What is wrong:** Fresh gate sidecars lack the closure binding required by `gate_ok()`, while the binder refuses to operate until the three not-yet-created sweep runs also exist. Thus a clean v3 gate cannot become eligible to launch a sweep. Future v3 runs would also conflict with the checker’s historical fixed launcher HEAD. Missing logs or failed abort renames remain warnings rather than checked failures.

   **Why it matters:** Existing-directory refusal tests pass, but the launcher cannot reproduce the experiment from a clean state or guarantee mandated abort evidence.

   **Minimal fix:** Add an exact ten-gate pre-sweep binding phase or write full bindings from the launcher; distinguish historical and future expected launcher versions; make abort preservation fail explicitly. Test a clean transition and evaluator/checkpoint/provenance abort paths.

6. **should-fix — Preregistration chronology and H2 wording remain inconsistent.**

   **File:** [yaw_rotation_degradation_params_set_up.md:28](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_params_set_up.md:28)

   **What is wrong:** H2 and gate-v1 are still dated `13:45`, while the retained history establishes `13:41`. The H2 aggregate sentence abbreviates the other three branches rather than stating the exact four-way rule.

   **Why it matters:** It leaves the preregistration record internally contradictory.

   **Minimal fix:** Use `13:41` and spell out supported / partially supported / not supported / not evaluable exactly as in plan §12.

7. **should-fix — The reproduction record still contains shorthand and an inaccurate sidecar claim.**

   **File:** [yaw_rotation_degradation_command.md:35](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:35), [yaw_rotation_degradation_command.md:44](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:44), [yaw_rotation_degradation_command.md:88](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:88)

   **What is wrong:** Numerous log names still use `...`/brace/wildcard shorthand, the final section has an unfinished `21:03–` interval, and the final `21:27:38` acceptance invocation is absent. Line 35 says exact commands are in every sidecar, but five older sidecars contain placeholders and five later gate sidecars have no command.

   **Why it matters:** Round-3 required exact historical reproduction, and the present narrative overstates the surviving evidence.

   **Minimal fix:** Expand every retained filename and final command; complete the time interval; distinguish verbatim sidecar commands from post-hoc reconstructions and explicitly record the missing-command limitation.

8. **should-fix — The append-only worklog is still SOP-incomplete and contains false closure claims.**

   **File:** [yaw_rotation_degradation_worklog.md:258](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:258), [yaw_rotation_degradation_worklog.md:273](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:273), [yaw_rotation_degradation_worklog.md:281](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:281), [yaw_rotation_degradation_worklog.md:287](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:287), [yaw_rotation_degradation_worklog.md:300](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:300), [experiment_SOP.md:47](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:47)

   **What is wrong:** Post-round-3 substantive entries omit combinations of Goal, Version Control, Command/Validation, pre-run Acceptance criteria, Analysis, and Next. The worklog also falsely claims the gate cannot be fabricated, abort renaming is checked, every visible label is producer-owned, and no shorthand remains. “Two attempts” names three aborted binder attempts.

   **Why it matters:** The record still does not meet the SOP lab-notebook contract and incorrectly reports closure.

   **Minimal fix:** Preserve existing entries and append full-field corrective entries that explicitly retract each overbroad claim and correct the attempt count.

9. **should-fix — Source-closure size is misstated and one causal attribution remains too strong.**

   **File:** [yaw_rotation_degradation_analysis.md:9](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:9), [yaw_rotation_degradation_analysis.md:10](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:10), [yaw_rotation_degradation_params_set_up.md:37](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_params_set_up.md:37), [yaw_rotation_degradation_command.md:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:3), [launch_yaw_sweep_v3.sh:5](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:5)

   **What is wrong:** The binding report contains 12 files total—evaluator plus 11 repo-local imports—not evaluator plus 12 imports. Analysis line 10 still calls the remaining effect “attributable” to geometry-seeing components, despite the requested conclusion being only that alignment’s measured contribution is negligible and an unquantified CUDA residual remains.

   **Why it matters:** These overstate both the evidence set and the mechanism attribution.

   **Minimal fix:** Say “12-file closure (evaluator plus 11 repo-local imports)” and residual-qualify or remove the attribution clause.

## Round-3 closure table

| Round-3 finding | Round-4 status |
|---|---|
| 1 — retained provenance | **Partial** — exact 13-run records and copies exist; full byte/input/log binding remains open |
| 2 — self-attested gate | **Not closed** |
| 3 — fail-open sweep validation | **Partial / not closed** |
| 4 — producer-only consumers | **Partial / not closed** |
| 5 — plan §12 H2 rule | **Closed in the plan**; parameter consistency remains open |
| 6 — launcher contract | **Partial / not closed** |
| 7 — plan/parameter chronology | **Partial** |
| 8 — exact reproduction commands | **Partial / not closed** |
| 9 — SOP-complete worklog | **Not closed** |
| 10 — analysis overstatement | **Partial** — three requested corrections landed; attribution remains |

The round-3 “exactly required” items 1–8 are therefore partial or open. Item 9’s commands were run, but its adversarial coverage is insufficient to demonstrate closure.

## Verdict

**request changes**

Exactly required before the record commit:

1. Make the gate decision independently recomputable or completely input-hash-bound, and add the missing adversarial tests.
2. Close every sweep schema/content/provenance hole described in findings 2–3.
3. Strengthen retained data/log/gate-input binding and compare live evidence against the retained report.
4. Finish producer-only rendering, require the fields fail-closed, and regenerate every consumer.
5. Repair the clean launcher workflow and hard-fail/fault-test abort preservation.
6. Correct parameters, command history, source-closure count, and the remaining analysis attribution.
7. Append SOP-complete worklog corrections and explicit retractions.
8. Re-run the full suite, new adversarial tests, static checks, binder, and acceptance; retain the new logs and reverify every canonical/copy digest.

All SOP deliverable classes are present. No sweep rerun is presently indicated; one is required only if the strengthened binding or validation exposes a mismatch. The commits file can be updated after the record commit as planned.
codex exit 0 at 21:42:03
