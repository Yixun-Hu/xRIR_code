# Codex review — Planner one-offs and final documents, round 6

Reviewer: OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra), independent of the Planner (Claude) and Coder (Claude Opus subagent).
Prompt: review_prompts/oneoffs_round6_prompt.md · started 2026-09-06 23:01:21

---

## Verdict

**request changes**

The retained numerical result still appears internally consistent: all four canonical/copy digest pairs match, the final binder and acceptance logs report success, and all 13 current checkpoint/meta and run-log associations are correct. The defects below are contract and record-integrity failures, not a detected quantitative discrepancy.

## Findings

1. **blocker — Binder migrations are neither immutable nor transactional.**

   **File:** [bind_provenance.py:163](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:163), lines 163–184 and 260–297; [test_exp03_record_tools.py:295](/home/yixunhu/codespace/xRIR_code/tests/test_exp03_record_tools.py:295)

   **What is wrong:** `report_drift()` iterates only over the new report, so removal of a previously retained top-level field, run, or gate input is not drift. Its null-SHA exception allows the entire gate-input record—including its path—to change, rather than only `sha256: null → digest`. A read-only probe changing a path and deleting retained fields returned `[]`.

   The advertised `run_set="gate" → "all"` migration is also rejected as drift. Worse, sidecars are written at lines 272–279 before report drift is checked at 285–289, leaving partial mutations on refusal. Additive sidecar changes overwrite `bound_at`, and report migrations replace the sole report without preserving prior state.

   **Why it matters:** Historical evidence can be dropped or altered during a purportedly additive migration, while the clean gate→sweep workflow cannot create its final all-runs report.

   **Minimal fix:** Compare every previously retained field recursively; whitelist only same-record null-SHA→digest and monotonic gate→all/run additions; preserve original binding timestamps and migration history; validate the complete projected transaction before writing any sidecar or report. Add deletion, path-change, gate→all, timestamp-preservation, and “nothing written on refusal” tests.

2. **blocker — The clean launcher workflow still cannot reach acceptance.**

   **File:** [launch_yaw_sweep_v3.sh:153](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:153), lines 153–155 and 183; [check_sweep_acceptance.py:375](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:375), lines 375–396 and 418–429.

   **What is wrong:**

   - The launcher prints the post-decision binder before the decision command.
   - The decision command creates neither the mandatory retained v3 decision copy nor the assets gate-JSON copy.
   - It invokes the working-tree `tools/summarize_yaw.py`, while acceptance recomputes with the closure pinned at `5febf471…`. The current producer differs and changes gate-summary text.
   - The subsequent default binder hits finding 1’s `gate → all` refusal.
   - No clean synthetic gate→decision→bind→sweep integration was run.
   - `evaluator_changed` and `checkpoint_changed` fault injections directly call `abort_run()` before the real guards at lines 109–110; rename failure remains untested.

   **Why it matters:** The historical artifacts pass because all required copies already exist, but the retained launcher cannot reproduce its claimed workflow from an empty decision state.

   **Minimal fix:** Use one explicitly pinned decision producer throughout; emit decision, required copies, and post-decision binding in executable order; support the monotonic binder migration; then retain a clean isolated end-to-end integration log. Exercise the actual evaluator/checkpoint guards and rename-failure branch.

3. **blocker — Deep provenance can bind the wrong checkpoint or an unrelated “log.”**

   **File:** [bind_provenance.py:187](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:187), lines 187–200 and 235–259; [check_sweep_acceptance.py:238](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:238), lines 238–266 and 353–371; [launch_yaw_sweep_v3.sh:172](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:172).

   **What is wrong:**

   - Metrics meta and provenance validate checkpoint paths independently; neither requires `provenance.checkpoint == metrics.meta.checkpoint == profile checkpoint`. An in-memory `gate_control` sidecar changed to the released checkpoint and its real digest passed `check_provenance()` with no failures.
   - `run_start()` accepts any existing filename containing a timestamp. It does not require the record folder, `.log`, correct mode/tag, or a non-aborted log. A retained gate-decision `.txt` was accepted as a sweep log.
   - Launcher preflight calls `gate_ok()` without live environment/data context or retained-report validation. Its gate `git_sha` check is tautological (`head=pv["git_sha"]`).

   **Why it matters:** A first binding can canonically associate outputs with a different checkpoint or unrelated evidence, and the launcher can authorize a costly sweep using stale/fabricated gate provenance. Current retained associations happen to be correct.

   **Minimal fix:** Tie all three checkpoint identities together; require the exact normalized `{timestamp}_{gate|sweep}_{tag}.log` path under the record folder; and make launcher preflight use the same live environment, data-identity, report, and sidecar checks as final acceptance.

4. **blocker — Producer-only rendering remains fail-open.**

   **File:** [make_results_html.py:182](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:182), lines 182–228, 239–268 and 304; [make_results_md.py:68](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:68), lines 68 and 91–117; [test_exp03_record_tools.py:318](/home/yixunhu/codespace/xRIR_code/tests/test_exp03_record_tools.py:318).

   **What is wrong:**

   - HTML silently converts missing nested metric sections into empty tables and skips missing `h1.bounds`.
   - Both generators convert a missing room-cluster TOST object into the displayed verdict “not established,” rather than refusing absence.
   - Markdown still uses raw run IDs, hard-coded `s EDT`/`dB C50`, raw spectral column names, and raw-name fallbacks.
   - HTML still hard-codes visible `TOST r(cyl)`.
   - The consumer test mutates all units but checks only `uedt`; it does not remove scientific rows/bounds, mutate actual label keys, or exercise `data.json` because `--out` suppresses its generation.

   **Why it matters:** A malformed producer artifact can still render `FINAL` while silently omitting results or turning missing evidence into a negative TOST result.

   **Minimal fix:** Validate exact nested label×condition×metric coverage, H1 bounds, H2 rows, delay flips, decomposition, and both TOST levels; remove remaining raw identifiers/fallbacks; test every label/name/unit plus missing result structures and `data.json`; regenerate Markdown, HTML, and `data.json`.

5. **record-blocker — The accepted historical residuals are not fully or accurately disclosed.**

   **File:** [yaw_rotation_degradation_analysis.md:9](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:9), lines 9 and 13; [yaw_rotation_degradation_params_set_up.md:37](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_params_set_up.md:37); [yaw_rotation_degradation_command.md:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:3), lines 3, 35 and 63.

   **What is wrong:** Analysis discloses post-hoc source/environment capture and CUDA nondeterminism, but not post-hoc dataset-inventory/log hashing or the unexercised real launcher. Params labels only the source closure as post hoc. Command line 3 categorically says every run “used” the closure, stronger than the evidence establishes. The reconstructed-command/template-sidecar limitation is accurately stated only in the command record.

   Chronology is likewise inferred post hoc from mutable filename timestamps and filesystem mtime.

   **Why it matters:** Round 5 classified these limitations as non-blocking only once accurately disclosed.

   **Minimal fix:** In analysis, params, and command records, state that the source closure, environment, dataset inventory, log digests, and chronology evidence were captured post hoc and prove consistency of retained/current bytes—not execution-time identity. Also state the five reconstructed commands/five GPU-template sidecars, that launcher v3/v4 was not used for a real experiment run, and the CUDA nondeterminism limitation.

6. **record-blocker — Command and worklog records remain SOP-incomplete and contain incorrect evidence references.**

   **File:** [experiment_SOP.md:37](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:37), [experiment_SOP.md:47](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:47); [yaw_rotation_degradation_command.md:94](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:94); [yaw_rotation_degradation_worklog.md:385](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:385), lines 385–409.

   **What is wrong:**

   - `_command.md` still uses the round-5-flagged wildcard at line 94 and ends at the round-4 commands; none of the v5 binder/checker, launcher-v4, generator-v5, 24-test, final acceptance, or 229-test commands is recorded.
   - The recent acceptance, launcher, generator, and final-validation entries omit required SOP fields.
   - Worklog line 395 names nonexistent `22:53:25` exit-7 and `22:53:28` no-output logs and uses `22:54:2x`. Actual new files are `22:53:22`, `22:53:25`, `22:53:32`, `22:53:35`, `22:53:42_sweep`, and `22:54:33_gate`.
   - Lines 394 and 400 overclaim launcher ordering/branch coverage and fail-closed rendering.
   - No retained log supports the reported 24-test, 229-test, or static-validation passes, despite round 5 requiring retained logs.

   **Why it matters:** The permanent record cannot reproduce or audit the final correction pass and remains noncompliant with the append-only notebook SOP.

   **Minimal fix:** Append full-field corrective/retraction entries; add exact commands, environment, outputs, exit statuses, and full filenames to `_command.md`, explicitly marking late reconstruction; enumerate the four older refusal logs; rerun the final focused/full/static validation with timestamped retained output.

7. **should-fix — Narrow acceptance paths remain fail-open.**

   **File:** [check_sweep_acceptance.py:269](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:269), lines 269–296, 309–340 and 375–443.

   **What is wrong:**

   - `check_report()` omits retained `decision_v1` and `decision_v2`, despite claiming every gate input matches.
   - Gate-producer closure validation ignores its recorded working-tree digests.
   - Boolean `n_nan=False`, `max_samples=False`, `gl_seed=False`, and—for the batch-1 gate—`batch_size=True` pass integer comparisons.
   - Gate recomputation accepts matching output files without requiring subprocess exit zero.

   **Why it matters:** These are literal fail-open contract edges, although current retained values and failed-decision hashes are correct.

   **Minimal fix:** Validate every retained gate input and complete closure record; use strict non-boolean integer checks for counts/profile scalars; require recomputation exit zero; add adversarial tests.

## Round-5 closure

| Round-5 finding | Round-6 status |
|---|---|
| 1 — gate acceptance could pass without establishing the gate | **Closed for the cited cases**; exact k=0 content, chronology, canonical path, and non-accepting skip landed. New strict-type edge is finding 7. |
| 2 — producer-consumed decomposition unchecked | **Closed.** |
| 3 — binder could overwrite history | **Partial, blocking**; finding 1. |
| 4 — gate producer/sidecar provenance incomplete | **Closed for the enumerated producer-closure gaps**, but new association/preflight/report gaps remain in findings 3 and 7. |
| 5 — launcher clean workflow | **Not closed, blocking**; finding 2. |
| 6 — producer-only rendering | **Not closed, blocking**; finding 4. |
| 7 — command/source narrative | **Partial**; closure count, three attempts, and retained stub are fixed; wildcard, missing final commands, and disclosure remain. |
| 8 — SOP worklog | **Not closed**; retractions landed, but new entries remain incomplete and inaccurate. |
| 9 — stale bookkeeping | **Closed**; byte-hash docstring corrected and 83→82 explicitly retracted. |

## Exactly what must change before commit

Blocking defects:

1. Make binder migration monotonic, history-preserving, and all-or-nothing.
2. Repair and prove the clean launcher workflow with a retained isolated integration test.
3. Tie checkpoints/logs to their run metadata and use full live gate validation before sweep.
4. Make both renderers reject missing producer results and remove remaining local/raw presentation semantics; regenerate all consumers.
5. Complete the residual-limitations disclosures.
6. Append SOP-complete corrective worklog and command records with exact filenames.
7. Retain logs for the expanded focused suite, full suite, syntax/static checks, binder, and default recomputing acceptance.

Finding 7 should also be closed in that pass because the record claims a fail-closed acceptance contract.

Residual limitations to record, not fix:

- Source closure, environment, dataset inventory, log hashes, and chronology evidence were captured post hoc.
- Five gate commands are reconstructed; five v1 sidecars contain a GPU-placeholder template.
- Launcher v3/v4 was never used for a real experiment run.
- CUDA run-to-run nondeterminism remains unquantified.

Those residuals do not require an experiment rerun. A rerun becomes necessary only if the strengthened checks expose a mismatch. The delayed `commits_yaw_rotation_degradation.md` update remains acceptable: after the record commit, append the ten post-`5febf47` producer commits plus the record-commit SHA in the immediate bookkeeping commit.
codex exit 0 at 23:13:48
