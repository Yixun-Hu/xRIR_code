# exp_07 seen_protocol — Codex code review, round 7 close + repeated integrative verdict (`full`)

**Reviewer:** OpenAI Codex `gpt-6-astra` (Ultra), `codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07`, prompt `review_prompts/code_round7_close_prompt.md`, log `seen_protocol_2026-09-15_19:52:57_codex_code_round7_close.log`. **Scope:** exp07-window `9c3b458..d58c2fd` (round 7) and the whole branch for the integrative verdict. **Date:** 2026-09-15 19:53–20:27 EDT.

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

Reviewed all changed lines in the 12 commits `9c3b458..d58c2fd`, replayed the round-6 reproductions, and repeated the integration checks.

**Round-7 verdict: request changes. Round 7 remains open.**  
**FINAL INTEGRATIVE verdict (`full`): not approved.**  
**Exact blocking findings: 1, 2, 3, and 4 below.**

1. **Blocker — product coverage still permits missing contracts and dependencies.**  
   [bind_provenance.py:453](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:453)

   The new checks enforce the expected `run_flags` keys, but do not enforce equivalent coverage of `contracts` or all required dependencies. Dependency admission also uses the entire record’s allowed inputs instead of restricting them to the product’s runs.

   Independently reproduced, retaining approval pins and repairing outer hashes:

   - Remove every table contract: accepted.
   - Remove a required reference-manifest dependency: accepted.
   - Add a released-run dependency to the cylindrical/simple pairing: accepted.

   **Why:** complete run flags still do not establish complete, accurate product provenance.

   **Minimal fix:** require exact contract coverage and validate contract identities. Derive each product’s required input map from its expected runs, training evidence, approval and producer closure; reject omissions and unrelated dependencies.

2. **Blocker — parity can be certified without the tests running.**  
   [exp07_parity.py:103](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_parity.py:103), [bind_provenance.py:188](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:188)

   The producer accepts a pre-existing JUnit file and inherits `PYTEST_ADDOPTS`. The binder hashes the XML but never verifies its outcomes against the receipt.

   Independently reproduced:

   - XML containing a failed case, with the receipt still claiming nine passes: accepted.
   - XML containing no test cases: accepted.
   - From a clean clone, pre-create passing XML and set `PYTEST_ADDOPTS=--help`: the **real pytest subprocess runs zero tests**, exits 0, and the producer writes a passing receipt which the binder accepts.

   Only CUDA availability was stubbed for that last reproduction; the subprocess and evidence checks were real.

   **Minimal fix:** require fresh invocation-owned XML and reserve output paths before launching pytest; control ambient pytest options. In the binder, parse the XML, enforce the registered successful cases, and require agreement with the receipt.

3. **Blocker — attempt-state handling accepts missing evidence and rejects legitimate aborted probes.**  
   [bind_provenance.py:72](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:72), [bind_provenance.py:124](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:124)

   Three counterexamples remain:

   - Delete a certified probe’s external log: collection succeeds with `present=False`.
   - Remove the log block and execution receipt from a `guard_epoch_one` abort: it is accepted as a setup failure. Neither `reason == "setup_failed"` nor the documented directory/log shape is required.
   - Retain a legitimately aborted probe in the ledger: collection refuses it because every probe must have a completion-linked receipt. The launcher writes that receipt only after `execute_attempt` succeeds.

   **Why:** the binder admits incomplete certification evidence while preventing publication of valid failure history.

   **Minimal fix:** require existing certified logs at mandatory recorded digests; validate the documented setup-failure shape explicitly; require completion-linked probe receipts only for successfully completed probes, retaining aborted probes through their abort evidence.

4. **Blocker — calibration bypasses table validation beyond the intended waiver.**  
   [exp07_calibration.py:104](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_calibration.py:104)

   Reusing `run_contract` and `admit_run` preserves their checks, but calibration omits validations performed later by `build_table`.

   Independently reproduced with the synthetic producer fixture:

   - Remove `loss` and `log_mse` from the released runs: calibration passes; the table refuses.
   - Make one seed’s finite count differ by four: calibration passes; the table refuses the registered tolerance violation.

   **Why:** this gate can approve evaluations that the publication protocol subsequently rejects.

   **Minimal fix:** share the table’s metric-coverage and five-seed cohort validation with calibration, while retaining independence from trained-arm approval pins.

5. **Should-fix — the calibration provenance sidecar is not bound into the report.**  
   [bind_provenance.py:237](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:237)

   The binder reads the sidecar but does not include its digest in the returned record. Changing its generation command and removing all producer closure records leaves the binding report **identical**.

   **Minimal fix:** include the sidecar’s stamp in the report and validate its producer closure records against the recomputed identity.

## Verified

The original findings close as follows:

| Round-6 finding | Round-7 status |
|---|---|
| 1 — coverage/dependencies | Partial; finding 1 above remains |
| 2 — calibration/parity | Producers added; findings 2 and 4 remain |
| 3 — attempt history/logs | Partial; finding 3 remains |
| 4 — split agreement | Closed: launcher and binder refuse disagreement |
| 5 — trainer parity harness | Closed structurally: actual `run_trainer` route exercised; GPU execution deferred |
| 6 — hardlink overwrite | Closed for Markdown, HTML and LaTeX |
| 7 — repeated `--evidence` | Closed, including equals syntax |
| 8 — positive SD printing | Closed |
| 9 — redundant `--entry` | Closed |

The unchanged round-6 file produced **17 passed / 6 failed**, correcting the reported 18/5 count. Those six failures are obsolete assertions expecting the holes to remain. Updating those expectations in a separate review copy allowed **all 23 replay cases** to pass, including every reproduction inside the compound tests. The original file remains unchanged.

Additional checks refused an internally consistent calibration summary unrelated to the bound runs, stale parity-log digests, mismatched parity commits, missing ledger directories, wrong probe-completion linkage, and disagreement in just one output meta. Earlier forgery refusals remain effective.

Reproductions and logs: [review tests](/home/yixunhu/codespace/xRIR_code_wt07/.review-round7/test_new_adversarial.py), [replay and adversarial log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round7/adversarial-final.log), [missing-dependency reproduction](/home/yixunhu/codespace/xRIR_code_wt07/.review-round7/missing-dependency.log).

Calibration’s approval independence is sound in principle:

- The real approval file remains all-null.
- Reviewed evaluator/writer closure derivation succeeds.
- The calibration producer’s 25-file closure excludes the approval JSON.
- Synthetic calibration succeeds without accessing the approval loader or trained checkpoint pins.
- The released checkpoint was rehashed and matches registered SHA-256 `1762c702ee23a8f584e67c4b5b052b7483237e6471bb5e58774b66d514aec2f8`.
- Failed calibration writes its evidence and exits 1; the binder refuses `passed=False`.

Validation used Python **3.8.20**, empty `CUDA_VISIBLE_DEVICES`, disabled bytecode, and pytest’s cache provider disabled.

| Check | Result |
|---|---|
| Full suite in clone at `d58c2fd`: `python -m pytest tests -q -p no:cacheprovider` | **2,551 passed, 46 skipped, 1 clone-setup failure**, 1,774.95 seconds |
| Setup failure: `test_reviewed_ref_resolution[main]` | Clone lacked local `main`; after `git branch main origin/main`, targeted rerun: **1 passed** |
| `static_checks.sh` | **Exit 0** |
| Publication subset | **264 passed** |
| Both record-test collection orders | **190 passed, 1 skipped each** |
| Python 3.8 parsing, shell syntax, working-tree and round-7 `git diff --check` | Passed |

Thus every non-skipped suite case passed across the full run and the corrected targeted rerun; the initial full invocation itself was not zero-failure. [Full log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round7/full-suite.log), [targeted rerun](/home/yixunhu/codespace/xRIR_code_wt07/.review-round7/clone-main-rerun.log), [static checks](/home/yixunhu/codespace/xRIR_code_wt07/.review-round7/static-checks.log).

Merge preservation remains verified:

- Dry-run `git merge --no-ff --no-commit d58c2fd` into main **`f35196c`**: no conflicts.
- On the merged tree, exp_04’s `check_record.py ckpt/yaw_aug`: **exit 0**.
- Exp_06’s `test_shared_flags_and_defaults_equal_the_trainers`: **1 passed**.
- Direct comparison of merged-tree bytes: **15 closure files across 46 exp_04 evaluations match**.
- All nine previously restored shared modules still match main.
- Against current main, all **33 changed non-worklog paths are new**. Existing-path differences are confined to worklog records.
- No tracked code edits or GPU work were performed; review writes stayed under `.review-round7/`.

The README’s executable parity/calibration commands and the prior `eval07` helper’s arguments match the current interfaces. The actual default receipt name is `${PARITY%.log}.receipt.json`.

**Round 7: request changes. Integrative `full`: not approved. Blocking findings: 1, 2, 3, 4.**
