# Codex review — Planner one-offs and final documents, round 5

Reviewer: OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra), independent of the Planner (Claude) and Coder (Claude Opus subagent).
Prompt: review_prompts/oneoffs_round5_prompt.md · started 2026-09-06 22:12:33

---

The current artefacts are internally consistent and the reported numbers remain credible, but the record is not commit-ready. I found no quantitative discrepancy between the v4 JSON, analysis, Markdown, or HTML. The four canonical/copy pairs match, and all 13 current sidecars, artefact hashes, logs, and recorded dataset hashes agree with the current binding report. The remaining blockers are fail-open contracts and incomplete producer-only rendering—not evidence that the present numerical result is wrong.

## Findings

1. **blocker — Gate acceptance can still print PASS without establishing the gate.**

   **File:** [check_sweep_acceptance.py:286](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:286), [check_sweep_acceptance.py:317](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:317), [check_sweep_acceptance.py:395](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:395), [check_sweep_acceptance.py:408](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:408)

   **What is wrong:** Gate-run validation omits spectral and acoustic `n_nan`, permits a non-null mean for an all-null acoustic array, never checks P=E at k=0, and accepts fractional/boolean grids via `int()` coercion. In-memory probes accepted `n_nan=777`, an all-null mean of `123`, `[0.25]`, and `[False]`. Invalid sweep-log timestamps make `sweep_start_time()` return `None`, silently disabling chronology. Finally, `--no-recompute-gate` skips the principal independent check while the program can still emit `ACCEPTANCE PASS`.

   **Why it matters:** The default current run is sound because recomputation succeeds, but the published acceptance interface has a direct bypass and malformed-data paths.

   **Minimal fix:** Apply the exact sweep count/null/P=E/type rules to gate runs; require every sweep start timestamp; require the canonical summary path; and make skipped recomputation produce a non-accepting status and nonzero exit. Add adversarial tests for each case.

2. **blocker — The accepted decomposition is not necessarily the decomposition consumed by the producer.**

   **File:** [check_sweep_acceptance.py:449](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:449), [summarize_yaw.py:1657](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:1657)

   **What is wrong:** Acceptance validates `metrics_yaw.json["decomposition"]`, while `summarize_yaw.py` builds the result from `per_sample_yaw.json["decomposition"]`. Those fields are not compared.

   **Why it matters:** A changed or missing per-sample decomposition can be rebound and accepted even though it changes the final scientific output.

   **Minimal fix:** Require the metrics and per-sample decompositions to be identical, then validate the per-sample value actually consumed by the producer. Add mismatch/missing-field tests.

3. **blocker — Re-running the binder can overwrite historical evidence rather than detecting drift.**

   **File:** [bind_provenance.py:208](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:208), [bind_provenance.py:216](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:216), [bind_provenance.py:232](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:232), [bind_provenance.py:240](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:240)

   **What is wrong:** The “existing binding agrees” guard checks only the source-closure and evaluator digests. Environment, data identity, run start, and log digest are overwritten with freshly observed values, and the report is then regenerated.

   **Why it matters:** An accidental post-binding change to a log, dataset, or environment can be laundered into the new historical record by simply rerunning the binder.

   **Minimal fix:** Once a sidecar/report is bound, require every previously bound field to match exactly and refuse otherwise. Any migration must preserve the previous record and explicitly explain the change. Add drift-and-rebind tests.

4. **blocker — Gate producer and gate-sidecar provenance are still incomplete.**

   **File:** [bind_provenance.py:140](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:140), [check_sweep_acceptance.py:265](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:265), [check_sweep_acceptance.py:308](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:308), [check_sweep_acceptance.py:330](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:330), [summarize_yaw.py:41](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:41)

   **What is wrong:** Only `tools/summarize_yaw.py` is extracted from `5febf47`; it imports the working-tree `tools/paired_stats.py`. The current and historical `paired_stats.py` do match (`016b375a…`), but that is neither bound nor asserted. Gate sidecars also receive only shallow validation: no exact closure record, live environment, seed-specific live data identity, or log-digest comparison. The report checker does not validate its recorded producer digest.

   **Why it matters:** Recomputed gate output can depend on an unpinned implementation, and correspondingly edited gate sidecars/report copies can pass.

   **Minimal fix:** Bind and execute the complete repo-local gate-producer closure; validate its commit and hashes from the report; and apply deep provenance checks to every gate sidecar. No experiment rerun is needed.

5. **blocker — Launcher v3 still cannot execute its claimed clean workflow.**

   **File:** [launch_yaw_sweep_v3.sh:140](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:140), [launch_yaw_sweep_v3.sh:141](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:141), [launch_yaw_sweep_v3.sh:142](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:142), [bind_provenance.py:188](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:188)

   **What is wrong:** Immediately after measurements, the launcher calls the binder. The binder already requires `gate_stats.json`, `gate_summary.txt`, and the decision copies, but the next launcher line merely prints the decision command. A clean run therefore fails; pre-existing decisions can instead make it bind stale inputs. The fault test also counts all historical `*_ABORTED_*` logs and ignores the exact code file, so stale logs can mask a failed new preservation attempt; it does not exercise checkpoint/evaluator-change branches.

   **Why it matters:** Existing-directory refusal and two abort paths worked, but the future clean gate→decision→binding→sweep contract is not reproducible.

   **Minimal fix:** Remove the circular ordering through a pre-decision/post-decision binding split or generate the decision before final binding. Test a clean temporary workflow and evaluator, checkpoint, provenance, missing-log, and rename-failure paths using only newly created logs.

6. **blocker — Producer-only rendering remains incomplete.**

   **File:** [make_results_md.py:60](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:60), [make_results_md.py:71](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:71), [make_results_md.py:94](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:94), [make_results_md.py:116](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:116), [make_results_html.py:169](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:169), [make_results_html.py:198](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:198), [make_results_html.py:217](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:217), [make_results_html.py:239](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:239), [make_results_html.py:283](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:283), [test_exp03_record_tools.py:228](/home/yixunhu/codespace/xRIR_code/tests/test_exp03_record_tools.py:228)

   **What is wrong:** Markdown still exposes raw model keys in multiple tables and hard-codes metric names/units and cylindrical/control headings. HTML silently drops missing labels/metrics, has visible-name fallbacks, hard-codes `control` as primary, and prints raw margin labels. Its H2 table displays only query-level TOST despite the producer rule saying “query / room.” The new mutation test covers only HTML and checks presence, not leaked originals, Markdown, `data.json`, or nested missing mappings.

   **Why it matters:** Consumers can render `FINAL` while disagreeing with producer-owned labels or omitting producer results.

   **Minimal fix:** Require complete nested mappings; derive logic from `config.roles`; use `config.run_descriptions`, metric names, and units at every visible site; remove fallbacks/silent filtering; render both TOST levels; add `--out` support for Markdown and test all consumers with every label changed and missing fields. Colour slots and numeric precision are presentation and are not findings.

7. **should-fix — The command and source-closure narrative remains inaccurate.**

   **File:** [yaw_rotation_degradation_command.md:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:3), [yaw_rotation_degradation_command.md:90](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:90), [yaw_rotation_degradation_command.md:94](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:94), [yaw_rotation_degradation_command.md:110](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:110), [yaw_rotation_degradation_params_set_up.md:37](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_params_set_up.md:37), [launch_yaw_sweep_v3.sh:10](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:10)

   **What is wrong:** Three files still say evaluator plus 12 imports; the closure is 12 files total—evaluator plus 11 imports. `_command.md` says “two” aborted binder attempts while listing three, retains a wildcard for four launcher logs, and records the fault test using an unretained `<stub ...>` placeholder.

   **Why it matters:** These are factual and reproducibility defects in the permanent record.

   **Minimal fix:** Correct the count and attempt number; enumerate the four filenames; retain the exact stub or explicitly record that its source was ephemeral and the test is not exactly reproducible.

8. **should-fix — Recent worklog entries remain SOP-incomplete and renew overbroad claims.**

   **File:** [yaw_rotation_degradation_worklog.md:316](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:316), [yaw_rotation_degradation_worklog.md:332](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:332), [yaw_rotation_degradation_worklog.md:341](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:341), [yaw_rotation_degradation_worklog.md:347](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:347), [yaw_rotation_degradation_worklog.md:353](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:353), [yaw_rotation_degradation_worklog.md:360](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:360), [experiment_SOP.md:47](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:47)

   **What is wrong:** Substantive actions omit required combinations of Goal, Change, Version Control, Command/Validation, Acceptance criteria, Analysis, and Next. Lines 323–329 overclaim binding every decision-driving byte, line 334 overclaims validation against every sidecar, and line 348 overclaims producer-only/fail-closed rendering.

   **Why it matters:** The append-only lab notebook again reports closure that the implementation does not supply.

   **Minimal fix:** Append—do not rewrite—full-field corrective entries retracting those claims, documenting these round-5 findings, and stating that source/data/log/environment binding was post hoc.

9. **nit — Two small bookkeeping statements are stale.**

   **File:** [bind_provenance.py:102](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:102), [yaw_rotation_degradation_worklog.md:356](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:356)

   **What is wrong:** The data-identity docstring still says `(relpath, size)` although the implementation correctly hashes bytes. The worklog says 83 commits since `5bf4640`; `git rev-list --count 5bf4640..HEAD` returns 82.

   **Why it matters:** Low risk, but both misdescribe retained evidence.

   **Minimal fix:** Correct them in the next append-only correction/code pass. This finding is not independently blocking.

## Round-4 closure

| Round-4 finding | Round-5 status | Closure assessment |
|---|---|---|
| 1 — fabricated/stale gate | **Partial** | Exact rule, required summary, copies, ten-run validation, and default recomputation landed; finding 1 and producer-closure gaps remain. |
| 2 — sweep content fail-open | **Partial** | Original count/null/grid/decomposition checks landed; the new per-sample decomposition mismatch remains. |
| 3 — provenance incomplete | **Partial** | Current byte hashes are substantially improved and presently match; immutability, deep gate validation, and producer closure remain open. |
| 4 — producer-only rendering | **Not closed** | Several raw/fallback/silent-omission paths remain. |
| 5 — launcher workflow/abort evidence | **Partial** | Hard-fail abort handling and two fault paths landed; clean ordering and complete fault coverage remain open. |
| 6 — params chronology/H2 wording | **Closed** | 13:41 and the four-way rule are present. |
| 7 — command record | **Partial** | Sidecar limitations and final invocations landed; count, wildcard, and ephemeral-stub defects remain. |
| 8 — SOP worklog | **Partial** | Prior retractions landed; new entries are incomplete and contain new overclaims. |
| 9 — closure count/attribution | **Partial** | Analysis wording and attribution are corrected; the wrong closure count remains in three other files. |

Round-4 required items 1–8 are therefore: **1 partial, 2 partial, 3 partial, 4 open, 5 partial, 6 partial, 7 partial, 8 performed but insufficient to demonstrate closure**.

## Residual limitations

These cannot be eliminated retroactively without rerunning the experiment, but they do **not** block the commit once accurately disclosed and the fixable contract defects above are closed:

- Environment, source closure, dataset inventory, and log hashes were captured post hoc. They establish consistency of the currently retained bytes, not that those exact bytes/environment existed at execution time.
- Five gate commands are reconstructed, and five v1 sidecars contain only a GPU-placeholder template.
- Launcher v3 was never used for a real run. A clean synthetic integration test is sufficient for commit; eliminating the historical limitation would require a real rerun.
- CUDA nondeterminism remains unquantified.

The gate-producer dependency closure, binder immutability, and current-data validation are fixable without rerunning measurements. No full experiment rerun is presently indicated unless the strengthened checks expose a mismatch.

## Verdict

**request changes**

Before the record is committed, findings **1–8** must be addressed, affected artefacts/consumers and binding-report copies regenerated, and the expanded adversarial suite, full suite, binder, default acceptance, `py_compile`, `bash -n`, and static checks rerun with retained logs.

The commits-file sequencing is acceptable: after the record commit, immediately append the ten producer commits after `5febf47` plus the record-commit SHA and commit that bookkeeping update. The file’s current omission is therefore not itself a precommit blocker.
codex exit 0 at 22:45:04
