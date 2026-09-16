# exp_07 seen_protocol — Codex code review, round 8 close + repeated integrative verdict (`full`)

**Reviewer:** OpenAI Codex `gpt-6-astra` (Ultra), `codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07`, prompt `review_prompts/code_round8_close_prompt.md`, log `seen_protocol_2026-09-15_22:29:34_codex_code_round8_close.log`. **Scope:** exp07-window `d58c2fd..fd6e521` (round 8) and the whole branch for the integrative verdict. **Date:** 2026-09-15 22:29–23:04 EDT.

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

Reviewed every changed line in the nine commits `d58c2fd..fd6e521`, replayed the prior adversarial cases, and repeated the integration checks.

**Round-8 verdict: request changes. Round 8 remains open.**  
**FINAL INTEGRATIVE verdict (`full`): not approved.**  
**Exact blocking findings: 1 and 2 below.**

1. **Blocker — product dependencies remain incomplete, and the allowlist is broader than claimed.**  
   [bind_provenance.py:445](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:445), [training_dependencies:473](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:473)

   The hours ledger and probe receipt are optional under `required ⊆ declared ⊆ allowed`, although `run_contract` unconditionally reads and declares both.

   Reproduced with valid producer outputs, retaining approval pins and repairing outer hashes: remove every consumed probe-receipt and ledger declaration from TABLE or any PAIRS product. **Both `collect` and `check_record` accept the resulting record.** The mutation matrix found 16 accepted omissions: six for TABLE, four for cyl/simple, four for aug/simple, and two for released/simple.

   The upper bound also includes the **entire** `attempt_declarations` map: unused epoch checkpoints, `history.jsonl`, and aborted-attempt files. Adding the failed first attempt’s `abort.json` to cyl/simple passes end to end. Foreign-arm checkpoints, released-run inputs in cyl/simple, and unrelated evaluation completions are correctly refused; unused files from a used arm remain admissible.

   **Why:** the overall report binds these files, but each product can misstate its actual inputs. Equality against a product-specific projection remains possible despite the larger attempt inventory.

   **Minimal fix:** require the consumed ledger and probe receipt. Derive the exact product map from its expected evaluations, consumed training-contract inputs, approval, and producer closure. Keep complete attempt history in the overall report without admitting unused history files as product dependencies.

2. **Blocker — calibration still skips finite-cohort validation for `loss` and `log_mse`.**  
   [exp07_calibration.py:121](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_calibration.py:121)

   `metric_names(cells)` requires all five columns, but the `seed_finite_means` loop covers only EDT, C50, and T60.

   With an otherwise valid **6,217-query synthetic producer fixture**, reducing one seed’s finite `loss` or `log_mse` count to **6,212** still produces a calibration JSON with `passed: true`. `build_table` refuses `finite count tolerance exceeded`. The smaller fixture also demonstrates acceptance with **zero finite queries** for either spectral metric. The binder’s `calibration.measure` route accepts these cohorts too.

   **Why:** the pre-launch gate can still approve evaluations that publication rejects. Round-7 blocker 4 remains partially open.

   **Minimal fix:** validate every registered table metric for nonempty cohorts, finite-count tolerance, and finite seed means before selecting the three historical-calibration metrics. Preserve independence from approval pins.

## Verified

Round-7 findings **2 (parity), 3 (attempt evidence), and 5 (calibration sidecar)** close. Contract coverage closes; dependency coverage remains open under finding 1 above. Calibration remains open under finding 2. Round-6 findings 4–9 remain closed.

- **Parity:** pre-existing XML and existing/dangling symlink reservation paths are refused without overwriting targets. With ambient `PYTEST_ADDOPTS=--help`, the real pytest subprocess in a clean clone selected all nine cases; CUDA was hidden, so all nine skipped and no receipt was issued. Failed/empty XML, same-size XML edits with preserved mtime, mismatched reviewed/git heads, and nonancestor receipts were refused. Matching ancestor receipts remain accepted, preserving the original commit across later pin/bookkeeping commits.
- **Attempts:** missing or edited certified logs are refused. Edited renamed abort logs change the report and fail `check_record`. Legitimate aborted probes remain publishable; crossed completion digests and forged aborted-probe receipts are refused. Stray setup-failure logs are accepted with their bytes bound: a named existing log disables the logless waiver; an internal stray appears in the file inventory. Both invalidate an existing report.
- **Calibration:** unknown/missing metrics, wrong roles/K, and acoustic-metric finite-count violations are refused. The approval-loader trap was never called. Actual evaluator/writer closure derivation succeeds; the 25-file calibration closure excludes the still-all-null approval JSON. The released checkpoint rehash matches `1762c702ee23a8f584e67c4b5b052b7483237e6471bb5e58774b66d514aec2f8`.
- **Sidecar:** changing its generation command changes the report and fails the checker. Removing its producer closure is refused.
- **Table extraction:** numerical `rows` JSON is byte-identical between `d58c2fd` and `fd6e521`, using identical fixture and producer metadata. Full JSON differs only by the three added contract identity fields. Isolating the helper extraction from those additions makes the entire JSON byte-identical.

Validation used Python **3.8.20**, `CUDA_VISIBLE_DEVICES=''`, disabled bytecode/pytest caching, and worktree-local temporary/cache directories.

| Check | Result |
|---|---|
| Clone at `fd6e521`, local `main` created first: `python -m pytest tests -q -p no:cacheprovider` | **2,582 passed / 46 skipped / 0 failed**, 1,811.72 s |
| `static_checks.sh` | **Exit 0**; publication subset **294 passed** |
| Record tests in exp_07 → exp_04 → exp_03 order and reverse | **212 passed / 1 skipped each** |
| Unchanged round-6/7 reviewer files | **21 passed / 13 failed**; failures are obsolete expectations that repaired holes remain |
| Updated replay copies plus new characterization tests | **69 passed**, including all **34 prior cases** and **1,032 coverage mutations** |
| Additional 6,217-query calibration and receipt-copy tests | **2 passed**, confirming the behaviors above |
| Python 3.8 parsing/compilation, shell syntax, working-tree/round/branch `git diff --check` | Passed |

The characterization tests explicitly demonstrate the blockers; their passing count does not mean those defects are fixed. Evidence: [adversarial log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round8/adversarial-final.log), [6,217-query reproduction](/home/yixunhu/codespace/xRIR_code_wt07/.review-round8/boundary-6217.log), [full suite](/home/yixunhu/codespace/xRIR_code_wt07/.review-round8/full-suite.log), [static checks](/home/yixunhu/codespace/xRIR_code_wt07/.review-round8/static-checks.log).

Merge preservation remains verified:

- `git merge --no-ff --no-commit fd6e521` into main **`90bf4ec`** in an isolated clone: no conflicts.
- On the merged tree, exp_04’s `check_record.py ckpt/yaw_aug`: **exit 0**; exp_06’s `test_shared_flags_and_defaults_equal_the_trainers`: **1 passed**.
- Direct comparison: **15 closure files across 46 exp_04 evaluations match** their recorded bytes.
- Against that main, **58 differing paths include 33 non-worklog paths, all new**: every corresponding `git cat-file -e main:<path>` fails.
- No tracked code edits or GPU work were performed. Review artifacts and the [saved review](/home/yixunhu/codespace/xRIR_code_wt07/.review-round8/seen_protocol_codex_code_round8_close_review.md) are under `.review-round8/`; prior reviewer files remain unchanged.
