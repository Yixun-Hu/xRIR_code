# Codex review — Planner one-offs and final documents, round 3

Reviewer: OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra), independent of the Planner (Claude) and Coder (Claude Opus subagent).
Prompt: review_prompts/oneoffs_round3_prompt.md · started 2026-09-06 20:41:20

---

The current numerical result is credible and unchanged, but the record is not commit-ready. The four requested tables match the v2 producer exactly; the remaining blockers concern fail-closed validation, retained provenance, and preregistration integrity. No files were modified.

Verified:

- Canonical/copy JSON SHA-256: `662b93631172e5759c06e6a93be5992d433a8dd60a0951b47148965c59e933d0`.
- Canonical/copy summary SHA-256: `52f1e62eced6cdc55417e97876d3ca85fbbac37832a3a7271dfa8ed745db45f9`; the JSON’s summary binding is valid.
- Markdown and HTML reproduce byte-for-byte from their generators.
- The distance-to-threshold, k=0, H2/TOST, and run-settings numerical cells are producer fields and are correct.
- The main analysis numbers, including all four room-level +10% exceptions, match the JSON.
- The actual seed-0 manifest and all current query/index arrays agree.
- The actual ten gate runs have the expected profiles and matching artifact hashes; the retained v3 decision predates the sweep.
- Launcher v3 passes `bash -n`. `git diff --check` fails only on the plan’s extra blank line at EOF.

## Findings

1. **Blocker — retained provenance does not bind all decision-driving bytes.** [bind_provenance.py:25](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:25), [bind_provenance.py:111](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:111), [bind_provenance.py:161](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:161), [binding_report.json:107](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/binding_report.json:107).
   **What is wrong:** The three local sweep sidecars credibly match the 12-file closure, but the retained report records only `unchanged: true` per run—not the sidecar/artifact/checkpoint/manifest hashes, commands, logs, or run times. The raw sidecars live under ignored `ckpt/`. “Exact runs” is also enforced only by basename, the report changes its `written` timestamp on an otherwise identical rerun, the reviewed commit remains abbreviated, and none of the ten gate runs is closure-bound. The environment omits runtime dependencies such as torchvision/einops and lacks a dataset inventory identity.
   **Why it matters:** A checkout of the committed record cannot prove which raw bytes were bound, and the gate has weaker provenance than the sweep it authorized.
   **Minimal fix:** Use full SHA `62c9107b4150e44c4ac410ff4cab359c1e71cc10`; bind the exact canonical sweep and ten-gate paths; retain full per-run binding records or copied/hash-bound sidecars; include semantic manifest/data identity and relevant environment versions; make/report idempotence explicit; rerun the binder and retain/copy its evidence. No run rerun is needed unless the expanded binding fails.

2. **Blocker — the prerequisite gate check is self-attested.** [check_sweep_acceptance.py:64](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:64), [launcher v3:121](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:121).
   **What is wrong:** A fabricated JSON containing `mode="k0-gate"`, `gate_pass=true`, the pinned manifest string, and any existing file with its own matching hash passes. Neither checker nor launcher requires band-v3, the exact ten input runs/settings, gate sidecar/artifact/checkpoint hashes, the retained passed-decision digest, or that the gate predates the sweep.
   **Why it matters:** A stale v1/v2 result or invented gate can authorize acceptance despite the real current gate being valid.
   **Minimal fix:** Independently validate the exact ten-run v3 gate or bind to a retained immutable gate report; require the v3 rule, exact run profiles and provenance, decision/summary hashes, and chronology. Add adversarial tests for minimal/fabricated and stale-rule gates.

3. **Blocker — sweep content validation remains fail-open.** [check_sweep_acceptance.py:41](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:41), [check_sweep_acceptance.py:85](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:85), [check_sweep_acceptance.py:91](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:91), [check_sweep_acceptance.py:118](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:118), [check_sweep_acceptance.py:120](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:120).
   **What is wrong:** The checker never loads or hashes the actual manifest; arbitrary unique queries/indices spanning 17 invented prefixes pass. Acoustic arrays accept `NaN`/±`inf`, then discard them during reconciliation; an all-nonfinite array skips aggregate checking. Validity counts are not reconciled, `stft`/`decay` are omitted, P=E at k=0 is not required, and delay flips can be booleans, negative, or inconsistent with per-sample data. Closure records and environment are checked only shallowly.
   **Why it matters:** Malformed or non-manifest artifacts can still print `ACCEPTANCE PASS`. The current artifacts are healthy, but the claimed contract is not.
   **Minimal fix:** Load and semantic-hash the manifest; compare every query/index exactly; permit only finite numbers or null; reconcile means and validity counts for every emitted metric; check P=E k0, delay/decomposition top-level data, bounded non-boolean flip counts, and full provenance records. Add adversarial regression tests and retain a new acceptance log.

4. **Blocker under the stated producer-only criterion — the generators still author scientific facts locally.** [make_results_md.py:17](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:17), [make_results_md.py:54](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:54), [make_results_md.py:90](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:90), [make_results_html.py:22](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:22), [make_results_html.py:254](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:254), [HTML:844](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_01_results.html:844).
   **What is wrong:** The requested tables’ numerical payloads are correct producer fields, but model/metric labels, units, grid counts, design semantics, same-reference/phase claims, exp_01 history, and decomposition interpretation remain hard-coded. HTML also inserts `datetime.date.today()`, making regeneration date-dependent. Its delay heading says “queries,” although the producer counts `(query, reference)` pairs. Results §6 contains selected metadata, not all “evaluation settings.”
   **Why it matters:** This contradicts the explicit worklog/review acceptance criterion that no number, label, or headline comes from outside the JSON. One heading also changes the counted entity.
   **Minimal fix:** Serialize all retained data-dependent labels, counts, scientific descriptions, and run-history assertions in the producer schema; derive grid counts; remove or serialize the date; change the delay heading to `(query, reference) pairs`; say “selected evaluation settings”; regenerate Markdown, HTML, and `data.json`.

5. **Blocker — plan §12 rewrites the H2 preregistration incorrectly.** [plan:120](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:120), [contemporaneous review:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_codex_code_evaluator_review.md:3), [summarize_yaw.py:519](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:519).
   **What is wrong:** The plan now says H2 is “not supported” if any H1-passing cell fails. The contemporaneous preregistration and implementation say: all pass → supported; a nonempty proper subset passes → partially supported; none pass → not supported; no H1-passing cells → not evaluable.
   **Why it matters:** This creates a false retrospective preregistration record. The present outcome is unaffected because H1 has zero passing cells.
   **Minimal fix:** Restore that exact four-way rule in plan §12 and spell it out consistently in parameters.

6. **Should-fix — launcher v3’s ten-run schedule is correct, but its full contract is incomplete.** [launcher v3:14](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:14), [launcher v3:37](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:37), [launcher v3:58](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:58), [launcher v3:109](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:109), [launcher v3:119](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:119).
   **What is wrong:** The exact ten band-v3 measurements and their model/manifest/batch/TF32/GL-seed settings are correct, and `GATE_MEASUREMENTS_DONE` is correctly distinct from a pass. However, the launcher verifies only the entry evaluator rather than the closure, its printed decision command omits explicit `--band-rule v3`, sweep mode inherits finding 2’s gate weakness, and `abort_run` does not check that its required log exists or that `mv` succeeds.
   **Why it matters:** A future rerun could use changed dependencies/wrong gate semantics or fail to preserve the mandated `_ABORTED_...` log.
   **Minimal fix:** Verify/store the complete closure before and after every run, call the strengthened gate validator, specify `--band-rule v3`, use full SHAs, and make log renaming checked. Fault-test post-evaluator provenance/hash failures.

7. **Should-fix — plan/parameters chronology remains stale.** [plan:77](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:77), [plan:88](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:88), [plan:121](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:121), [plan:126](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:126), [params:28](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_params_set_up.md:28).
   **What is wrong:** Two “19-angle” references remain; the gate is said to precede reading any rotated angle although the smoke had already read all angles; `14:5x`, `20:xx`, and `12:5x` remain; params date the v2 amendment `15:1x` although it was specified at 14:53 before the 14:54 nuisance launch. The H2 amendment was recorded at 13:40/13:41, not 13:45. `git diff --check` also reports a blank line at plan line 127.
   **Why it matters:** Round-2 finding 16 required exact, chronology-preserving amendment history.
   **Minimal fix:** Correct both counts; say the gate preceded the full confirmatory sweep; replace placeholders with retained timestamps; correct amendment times; remove the extra EOF blank line.

8. **Should-fix — the reproduction commands are still not exact.** [command:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:3), [command:8](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:8), [command:26](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:26), [command:36](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:36), [command:50](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:50), [command:51](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:51), [command:68](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:68).
   **What is wrong:** The blanket claim that every run used HEAD `62c9107` conflicts with section-specific HEADs `b9884c7`, `a1db897`, and `5febf47`; `62c9107` identifies the reviewed evaluator bytes. SHAs remain abbreviated, seed-1/2 manifest and sweep commands are elided, timestamps retain `x` placeholders, and the v1/v2 decision commands omit `--band-rule v1`/`v2`—so replaying them against today’s v3 default does not reproduce those decisions. The binding-report copy is absent.
   **Why it matters:** The committed record cannot reproduce the historical gate decisions or exact sweep after ignored sidecars disappear.
   **Minimal fix:** Record full 40-character SHAs and per-section launch state; expand every command; add explicit v1/v2/v3 flags; replace all timestamp/tag shorthand; include the exact sweep commands and final report-copy command.

9. **Should-fix — corrective worklog entries remain SOP-incomplete.** [worklog:184](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:184), [worklog:209](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:209), [worklog:216](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:216), [worklog:223](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:223), [worklog:230](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:230), [SOP:47](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:47).
   **What is wrong:** The acceptance, launcher, generator, record-text, and regeneration actions omit Version Control, Command/Validation, Change, and/or Analysis fields. The worklog also claims producer-only output and exact commands despite findings 4 and 8, retains `20:0x`, and abbreviates the manifest hash in a command.
   **Why it matters:** Round-2 exactly-required item 7 and the append-only lab-notebook contract remain unmet.
   **Minimal fix:** Do not rewrite prior entries. Append supplemental corrective entries with all required fields, full hashes, exact commands/evidence, and explicit corrections of the overbroad claims.

10. **Should-fix — residual analysis overstatement.** [analysis:9](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:9), [analysis:10](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:10), [analysis:12](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:12), [analysis:22](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:22).
    **What is wrong:** “Every per-sample difference is attributable to the model” conflicts with the admitted unquantified CUDA nondeterminism; “the effect sits in the geometry branches” is stronger than the measured negligible E–P difference; “metrics move 2.5× more” applies to EDT/C50, not T60/loss; and comparing the +5.34% C50 yaw effect with “the k=0 gap” is metric-ambiguous—the paired C50 gap is only −0.8%, while −4.4% is EDT.
    **Why it matters:** These statements turn bounded observations into broad attribution or cross-metric comparisons.
    **Minimal fix:** Say changes belong to the observed model/evaluation pipeline subject to CUDA residuals; say alignment was negligible on the measured grid; restrict 2.5× to EDT/C50; remove or explicitly qualify the cross-metric comparison. The recommended next steps are otherwise appropriately conditional.

## Closure table

| Prior finding | Status at round 3 |
|---|---|
| Round 2 #1 — interval labels | **Closed** |
| Round 2 #2 — query vs room headline | **Closed** |
| Round 2 #3 — TOST wording | **Closed** |
| Round 2 #4 — k=0 units/precision | **Closed** |
| Round 2 #5 — ten-run gate launcher | **Closed** |
| Round 2 #6 — fail-closed/logging launcher | **Partial** — core handling fixed; closure/gate/checked rename remain |
| Round 2 #7 — source/environment closure | **Partial** — sweep locally bound; gate/future/full-SHA/data evidence remain |
| Round 2 #8 — exact set/idempotence/report | **Partial** — sidecar mechanics fixed; basename-only set and insufficient/non-idempotent retained report remain |
| Round 2 #9 — acceptance exact runs/gate/pins | **Partial** — sweep roles/head/checkpoints pinned; gate and canonical paths remain fail-open |
| Round 2 #10 — content validation | **Partial** — many checks added; manifest, nonfinite/count/delay/omitted-array holes remain |
| Round 2 #11 — producer-only Markdown | **Partial / not closed** |
| Round 2 #12 — numerical headline errors | **Closed** for the enumerated errors; finding 10 is residual wording |
| Round 2 #13 — reliability wording | **Closed** for the enumerated errors; finding 10 is residual wording |
| Round 2 #14 — causal/angular/practical claims | **Closed** for the prior defects |
| Round 2 #15 — decomposition/exp_02/H2 overclaim | **Closed** |
| Round 2 #16 — plan/params history | **Not closed** |
| Round 2 #17 — worklog/commands | **Not closed** |
| Round 1 #5 — config/units/interval labels | **Closed** |
| Round 1 #8 — evaluator/checkpoint provenance | **Partial** |
| Round 1 #12 — setup/hash/file preflight | **Closed** |
| Round 1 #13 — status/ABORTED/temp handling | **Partial** — unchecked/missing abort-log rename path |

## Verdict

**request changes**

Exactly required before the record commit:

1. Strengthen and test the binder/report; retain complete bindings for the three sweep and ten gate runs.
2. Make acceptance independently validate the v3 gate, actual manifest, complete finite content, aggregates, delay data, and full provenance; retain a new passing log.
3. Finish launcher v3’s closure checking, explicit v3 decision, strengthened gate call, full SHAs, and checked abort rename.
4. Complete the producer schema/generator correction, fix the delay entity/date/selected-settings wording, and regenerate all affected artifacts.
5. Correct the H2 four-way rule, 18-angle references, timestamps, gate-order wording, parameters, and plan whitespace.
6. Replace all reproduction shorthand with exact commands, explicit band rules, full SHAs, and the report-copy step.
7. Append—not rewrite—SOP-compliant corrective worklog entries.
8. Soften the four residual analysis statements.
9. Run the full suite plus new adversarial validator tests, `py_compile`, `bash -n`, and `git diff --check`; re-run the binder/acceptance and reverify copied hashes. A sweep rerun is required only if expanded binding or validation fails.

All SOP deliverable classes are present. After the record commit, update [commits_yaw_rotation_degradation.md:78](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/commits_yaw_rotation_degradation.md:78) with `7f3532c…`, `631050d…`, `cfc5e4f…`, and the record commit SHA, then commit that bookkeeping update as requested. Exclude the unrelated currently modified exp_01/exp_02 commit manifests from the exp_03 record commit.
codex exit 0 at 20:59:57
