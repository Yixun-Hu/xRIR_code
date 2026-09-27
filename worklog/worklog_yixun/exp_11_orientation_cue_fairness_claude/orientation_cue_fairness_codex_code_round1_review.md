**Reviewer:** OpenAI Codex `gpt-6-astra`, codex-cli 0.154.0, `codex exec`  
**Reasoning effort:** `ultra`  
**Sandbox/resources:** read-only worktree; authorized test/cache writes only under `/tmp`; CUDA disabled; CPU computation limited to four threads; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`  
**Reviewed:** `c6233e3` → `5db0c827ac72a81981c424a0bb447e2120b23b45` — 14 commits  
**Date/time:** 2026-09-26, 20:24–20:29 EDT

**Verdict: request changes.** Two statistical error-handling defects block Round 1 closure. The pipeline extension, historical-output preservation, R1 copying, and closure hygiene otherwise check out.

1. **[P2] Apply exp_11 screen suppression before running convergence.**  
   [tools/exp06_summarize_haa.py:1399](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1399)

   `exp11_screen_cells` first computes the entire historical `screen_cells` result. Its void checks therefore run **after** the historical convergence helper, which raises on zero-width intervals.

   Reproduced with three seeds and six queries: `G = D = arange(6) + 1`, `E = A = arange(6)`. N1i correctly returns `unavailable`, but the subsequent G − D screen raises:

   ```
   ValueError: a zero-width interval cannot carry a convergence verdict
   ```

   Making one G query non-finite leaves a 5/6 cohort and explicit void reasons; the screen still raises before applying suppression. Thus a cell that must be withheld can abort the whole phase summary.

   **Required:** evaluate invalidity before screen convergence; withhold genuinely degenerate exp_11 screen cells without changing historical `screen_cells` semantics or the frozen bootstrap. Add full-analysis regressions for cancellation and a nonempty, void cohort with constant paired differences. The existing cancellation test stops at `exp11_decision`, so it misses this failure.

2. **[P2] Restrict `unavailable` to finite, genuinely zero-width intervals.**  
   [tools/exp06_summarize_haa.py:1279](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1279)

   The guard re-raises only when `hi > lo`. That also admits NaN endpoints and reversed intervals; it does not establish zero width.

   Using the actual source functions and frozen bootstrap, six finite values of `1e308` against zero produced:

   ```
   status: unavailable
   diff: inf
   two_way: {lo: nan, hi: nan, ...}
   reason: ... seed 0 produced a non-finite or reversed interval
   ```

   This is numerical failure, not the registered cancellation case.

   **Required:** verify finite, equal endpoints and swallow only the documented zero-width refusal. Propagate non-finite, reversed, and unrelated failures. Extend the existing positive-width refusal test to cover those cases.

**The remaining implementation checks passed.**

- **Pipeline:** [init selection](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_haa_pipeline.sh:64) resets the frame on every call. E and G share the checkpoint variable/default; G uses SimpleViT and heading frame. Both use the exp_04 resolver. G’s job specification records all four heading rolls, and its children receive the heading directory. Bare `zeroshot` retains C/D/F, and the exp_11 log prefix is correct. Independent base-versus-tip comparisons found byte-identical dry-run stdout for room-frame `yawaug` fine-tuning, zero-shot, its complete queue, and historical queue cases.
- **G admission:** [expected_inits](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:611) resolves both E and G through the approved exp_04 record and binds that source. The CLI forwards the digest to admission, which checks each job’s initialization. Room-frame G and missing-heading metadata are refused.
- **Historical behavior:** phase dispatch preserves the old analysis path. The moved rendering blocks retain their content. Independent comparisons against **`c6233e3`**, using the existing fixtures, produced identical complete exp_06/exp_09 payloads, key order, and rendered summaries in both normal and exploratory modes.
- **Decision notation:** category direction, `−L < m`, `U < −m`, and strict equivalence containment match §3. Boundary equality does not establish the corresponding claim. Void/unconverged decisions withhold their fields; independently forcing unconverged contrast and interaction cells confirmed this. Only N2/N3 carry margins, and both are hallway C50.
- **N1i:** [interaction_rows](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1293) supplies exactly `(g − e, d − a)` to the frozen paired bootstrap. It uses one mask across four arms and three seeds, reports original per-arm exclusions, enforces the ≥99% joint cohort, and retains both component invalidity checks. Pairing checks cover indices, IR paths, evaluation seed, and K within each seed; admission separately binds training seeds and protocol.
- **S1:** the three families and Bonferroni-11 adjustment are correct. Ordinary void/unconverged labels are suppressed, and historical screen behavior is unchanged. Finding 1 concerns failures reached before that suppression.
- **R1:** [historical_rows](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1460) selects by contrast/room/metric, copies the registered fields exactly, preserves historical field meanings, and records nulls, selectors, and source hashes. Missing sources, duplicate matches, and mismatched summary hashes are refused. Both JSONs and summaries become publication inputs and are rechecked. Real canonical-source tests passed; no inference is recomputed.
- **Outputs:** Phase 1 is registered under `ckpt/exp11/phase1/`. Cross-experiment canonical-path protection handles resolved paths, including the shared `ckpt` symlink. Exclusive publication remains unchanged.

**Closure and approvals hygiene passed.** All 20 code keys were recomputed at the base and tip. Exactly two changed:

| Key | Digest at reviewed HEAD |
|---|---|
| `haa_pipeline_sh` | `f33d4ffe015ab76c1fc1115f8918983d6048351ca12bf96c47739e89a8eb573e` |
| `summarize_haa` | `32937f96910dc5b0452204ad457231422cd32b4ce84867bed3ef7276f9a62fc2` |

`finalize` remains `ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093`. The four simulated-evaluation closures, frozen files, and `BACKBONES_EXP06` are unchanged.

**Verification:** 78 selected tests passed: all 49 exp_11 summarizer tests, eight existing summarizer tests, 18 pipeline tests, and all three `test_exp09_sim_eval_closures.py` tests. `bash -n` and `git diff --check` passed. I excluded lifecycle tests that probe processes to honor the no-signals constraint. The additional failure reproductions used the actual source functions and unchanged bootstrap.

I inspected, but did not rerun, the reported full suite: **1 failed, 3323 passed, 50 skipped**. Its approvals-drift failure is consistent with the two intentionally moved keys; it must pass after the reviewed merge/refill.

**Tests should also be strengthened before merge.**

- [The new historical regression test](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:677) uses an [oracle loaded from mutable `main`](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_summarize_haa.py:1434). After merge, it compares the implementation with itself. Pin `c6233e3` or retain immutable expected fixtures, and cover normal as well as exploratory mode. Both stronger comparisons pass today.
- Add direct unconverged `exp11_cell` tests for all requested fields; current new coverage tests only unconverged screens.
- Add the explicit wrong-initialization **summarizer admission** case for G. Current coverage separately proves resolver identity and pipeline refusal.
- Extend interaction tests to IR-path/K corruption, cross-seed misalignment, and distinct exclusions where both component cohorts pass but the joint cohort fails.

**TDD and commit discipline:** tests co-land with the substantial implementation changes, which the SOP permits. The commits do not preserve red-run output or independently establish execution order; red-first execution remains the Coder’s attestation. The later acceptance-only commits also make the report’s “every case” wording too broad.

The four oversized commits are confirmed: `fff7948` 202 lines, `968e167` 208, `824dce9` 248, and `afd5a71` 202. They are cohesive, with production changes below 200 lines. Record the exceptions; I would not rewrite reviewed history solely for size. Future cycles can place tests in the immediately preceding commit.

**The three open schema questions need no change:** keep contrast strings as screen-family keys, keep `result['decisions']` as the ordered statement-name list, and keep `margin: null` for N1/N1i. Assigning them 0.23 would imply an unregistered margin decision.

After the blockers are fixed and reverified, recompute and refill the two moved approvals at the reviewed merge state. The planned runbook, results generators, CPU-suite completion, and G smoke remain outstanding launch work. This review does not close Round 1 or authorize publication from the current tip.

No review or repository changes were saved.
tokens used
134,856
**Reviewer:** OpenAI Codex `gpt-6-astra`, codex-cli 0.154.0, `codex exec`  
**Reasoning effort:** `ultra`  
**Sandbox/resources:** read-only worktree; authorized test/cache writes only under `/tmp`; CUDA disabled; CPU computation limited to four threads; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`  
**Reviewed:** `c6233e3` → `5db0c827ac72a81981c424a0bb447e2120b23b45` — 14 commits  
**Date/time:** 2026-09-26, 20:24–20:29 EDT

**Verdict: request changes.** Two statistical error-handling defects block Round 1 closure. The pipeline extension, historical-output preservation, R1 copying, and closure hygiene otherwise check out.

1. **[P2] Apply exp_11 screen suppression before running convergence.**  
   [tools/exp06_summarize_haa.py:1399](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1399)

   `exp11_screen_cells` first computes the entire historical `screen_cells` result. Its void checks therefore run **after** the historical convergence helper, which raises on zero-width intervals.

   Reproduced with three seeds and six queries: `G = D = arange(6) + 1`, `E = A = arange(6)`. N1i correctly returns `unavailable`, but the subsequent G − D screen raises:

   ```
   ValueError: a zero-width interval cannot carry a convergence verdict
   ```

   Making one G query non-finite leaves a 5/6 cohort and explicit void reasons; the screen still raises before applying suppression. Thus a cell that must be withheld can abort the whole phase summary.

   **Required:** evaluate invalidity before screen convergence; withhold genuinely degenerate exp_11 screen cells without changing historical `screen_cells` semantics or the frozen bootstrap. Add full-analysis regressions for cancellation and a nonempty, void cohort with constant paired differences. The existing cancellation test stops at `exp11_decision`, so it misses this failure.

2. **[P2] Restrict `unavailable` to finite, genuinely zero-width intervals.**  
   [tools/exp06_summarize_haa.py:1279](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1279)

   The guard re-raises only when `hi > lo`. That also admits NaN endpoints and reversed intervals; it does not establish zero width.

   Using the actual source functions and frozen bootstrap, six finite values of `1e308` against zero produced:

   ```
   status: unavailable
   diff: inf
   two_way: {lo: nan, hi: nan, ...}
   reason: ... seed 0 produced a non-finite or reversed interval
   ```

   This is numerical failure, not the registered cancellation case.

   **Required:** verify finite, equal endpoints and swallow only the documented zero-width refusal. Propagate non-finite, reversed, and unrelated failures. Extend the existing positive-width refusal test to cover those cases.

**The remaining implementation checks passed.**

- **Pipeline:** [init selection](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_haa_pipeline.sh:64) resets the frame on every call. E and G share the checkpoint variable/default; G uses SimpleViT and heading frame. Both use the exp_04 resolver. G’s job specification records all four heading rolls, and its children receive the heading directory. Bare `zeroshot` retains C/D/F, and the exp_11 log prefix is correct. Independent base-versus-tip comparisons found byte-identical dry-run stdout for room-frame `yawaug` fine-tuning, zero-shot, its complete queue, and historical queue cases.
- **G admission:** [expected_inits](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:611) resolves both E and G through the approved exp_04 record and binds that source. The CLI forwards the digest to admission, which checks each job’s initialization. Room-frame G and missing-heading metadata are refused.
- **Historical behavior:** phase dispatch preserves the old analysis path. The moved rendering blocks retain their content. Independent comparisons against **`c6233e3`**, using the existing fixtures, produced identical complete exp_06/exp_09 payloads, key order, and rendered summaries in both normal and exploratory modes.
- **Decision notation:** category direction, `−L < m`, `U < −m`, and strict equivalence containment match §3. Boundary equality does not establish the corresponding claim. Void/unconverged decisions withhold their fields; independently forcing unconverged contrast and interaction cells confirmed this. Only N2/N3 carry margins, and both are hallway C50.
- **N1i:** [interaction_rows](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1293) supplies exactly `(g − e, d − a)` to the frozen paired bootstrap. It uses one mask across four arms and three seeds, reports original per-arm exclusions, enforces the ≥99% joint cohort, and retains both component invalidity checks. Pairing checks cover indices, IR paths, evaluation seed, and K within each seed; admission separately binds training seeds and protocol.
- **S1:** the three families and Bonferroni-11 adjustment are correct. Ordinary void/unconverged labels are suppressed, and historical screen behavior is unchanged. Finding 1 concerns failures reached before that suppression.
- **R1:** [historical_rows](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1460) selects by contrast/room/metric, copies the registered fields exactly, preserves historical field meanings, and records nulls, selectors, and source hashes. Missing sources, duplicate matches, and mismatched summary hashes are refused. Both JSONs and summaries become publication inputs and are rechecked. Real canonical-source tests passed; no inference is recomputed.
- **Outputs:** Phase 1 is registered under `ckpt/exp11/phase1/`. Cross-experiment canonical-path protection handles resolved paths, including the shared `ckpt` symlink. Exclusive publication remains unchanged.

**Closure and approvals hygiene passed.** All 20 code keys were recomputed at the base and tip. Exactly two changed:

| Key | Digest at reviewed HEAD |
|---|---|
| `haa_pipeline_sh` | `f33d4ffe015ab76c1fc1115f8918983d6048351ca12bf96c47739e89a8eb573e` |
| `summarize_haa` | `32937f96910dc5b0452204ad457231422cd32b4ce84867bed3ef7276f9a62fc2` |

`finalize` remains `ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093`. The four simulated-evaluation closures, frozen files, and `BACKBONES_EXP06` are unchanged.

**Verification:** 78 selected tests passed: all 49 exp_11 summarizer tests, eight existing summarizer tests, 18 pipeline tests, and all three `test_exp09_sim_eval_closures.py` tests. `bash -n` and `git diff --check` passed. I excluded lifecycle tests that probe processes to honor the no-signals constraint. The additional failure reproductions used the actual source functions and unchanged bootstrap.

I inspected, but did not rerun, the reported full suite: **1 failed, 3323 passed, 50 skipped**. Its approvals-drift failure is consistent with the two intentionally moved keys; it must pass after the reviewed merge/refill.

**Tests should also be strengthened before merge.**

- [The new historical regression test](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:677) uses an [oracle loaded from mutable `main`](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_summarize_haa.py:1434). After merge, it compares the implementation with itself. Pin `c6233e3` or retain immutable expected fixtures, and cover normal as well as exploratory mode. Both stronger comparisons pass today.
- Add direct unconverged `exp11_cell` tests for all requested fields; current new coverage tests only unconverged screens.
- Add the explicit wrong-initialization **summarizer admission** case for G. Current coverage separately proves resolver identity and pipeline refusal.
- Extend interaction tests to IR-path/K corruption, cross-seed misalignment, and distinct exclusions where both component cohorts pass but the joint cohort fails.

**TDD and commit discipline:** tests co-land with the substantial implementation changes, which the SOP permits. The commits do not preserve red-run output or independently establish execution order; red-first execution remains the Coder’s attestation. The later acceptance-only commits also make the report’s “every case” wording too broad.

The four oversized commits are confirmed: `fff7948` 202 lines, `968e167` 208, `824dce9` 248, and `afd5a71` 202. They are cohesive, with production changes below 200 lines. Record the exceptions; I would not rewrite reviewed history solely for size. Future cycles can place tests in the immediately preceding commit.

**The three open schema questions need no change:** keep contrast strings as screen-family keys, keep `result['decisions']` as the ordered statement-name list, and keep `margin: null` for N1/N1i. Assigning them 0.23 would imply an unregistered margin decision.

After the blockers are fixed and reverified, recompute and refill the two moved approvals at the reviewed merge state. The planned runbook, results generators, CPU-suite completion, and G smoke remain outstanding launch work. This review does not close Round 1 or authorize publication from the current tip.

No review or repository changes were saved.

