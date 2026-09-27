**Reviewer:** OpenAI Codex, GPT-6; reasoning-effort setting not exposed in this session.  
**Sandbox/resources:** read-only repository; test writes under `/tmp`; CUDA disabled; ≤4 CPU threads; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`  
**HEAD:** `af052c07a18de695d025ba61cf2fef192a2f66b0`  
**Scope:** five fix commits, `5db0c82..af052c0`; historical baseline `c6233e3`  
**Date/time:** 2026-09-26, 22:22–22:28 EDT

**Verdict: request changes. Round 1 cannot be CLOSED, and this tip is not approved for merge.** One part of the original exception-handling blocker remains.

1. **[P2] Propagate non-zero-width failures from the convergence helper.**  
   [tools/exp06_summarize_haa.py:1299](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1299)

   The fallback catches every `ValueError` and checks only the recomputed **seed-0** interval. Finite, equal seed-0 endpoints do not establish that the helper raised its documented zero-width refusal: the helper validates both seeds first.

   Independently reproduced with the actual source and unchanged bootstrap, without monkeypatching:

   ```python
   rows = {
       "a": np.array([0., 0., 0., 0., 0., 1e308]),
       "b": np.zeros(6),
       "clusters": np.array([0, 1, 0, 1, 0, 1]),
       "seeds": np.repeat(subject.SEEDS, 2),
       "cohort": 2, "n_test": 2,
       "excluded": {}, "per_seed_diff": {},
   }
   ```

   At the accepted, non-default `n_boot=1`:

   - `constant_difference(rows)` is false.
   - Seed 0 produces `[0.0, 0.0]`.
   - Seed 1 produces `[NaN, NaN]`, with bootstrap mean `inf`.
   - `converged_two_way` raises:
     ```text
     the interval function did not return endpoints: seed 1 produced a non-finite or reversed interval
     ```
   - `exp11_cell` instead returns `status: unavailable`, retaining that numerical-failure message as its reason.

   An independently injected unrelated `ValueError` is also swallowed when nonconstant rows produce a finite zero-width nominal interval. The existing unrelated-failure regression exercises only a positive-width interval.

   **Required:** restrict the fallback to the documented `ZERO_WIDTH_REASON`, retain the finite/equal endpoint guard, and re-raise every other failure. Add regressions for the real seed-1 failure above and an unrelated exception with finite/equal seed-0 endpoints.

**Round-1 item verification:**

| Item | Result |
|---|---|
| **1. Convergence ordering and screen suppression** | **Pass.** The gate checks void first, then `constant_difference` **and** `zero_width`, then calls the frozen helper. `exp11_screen_cells` uses shared helpers plus `screen_cell_base`, without calling historical `screen_cells`. |
| **2. Finite zero-width restriction and propagation** | **Partial; blocker remains above.** `zero_width` correctly rejects NaN, ±inf, reversed endpoints, and non-float endpoints. The original six-values-of-`1e308` reproduction now raises. The late catch still suppresses failures originating from another bootstrap seed. |
| **3. Historical oracle** | **Pass.** [The oracle is pinned to `c6233e3`](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:677), with the pre-round structural assertion and all four exp06/exp09 × normal/exploratory cases. Independent comparisons also matched complete payloads, every nested key order, and rendered summaries. |
| **4. Unconverged decisions** | **Pass.** [Direct coverage](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:883) checks all four field types and every registered field across N1/N1i/N2/N3, while retaining descriptive values. |
| **5. G initialization admission** | **Pass.** [The admission test](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:916) admits matching initialization, refuses the wrong digest through `load_new_arm`, and includes the exploratory control. |
| **6. Interaction integrity** | **Pass.** [The strengthened cases](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:957) cover IR-path/K corruption, cross-seed misalignment, missing seeds, and repeated arms. Both component cohorts pass at 198/200 while the joint cohort fails at 196/200, with exactly one joint-cohort reason. |
| **7. Closure hygiene** | **Pass.** All 20 keys were recomputed against `c6233e3`; exactly the expected two changed. Frozen files and `BACKBONES_EXP06` are untouched. |
| **8. Other regressions, rendering, phase-1 output** | **Pass apart from the remaining blocker.** No historical-output regression found. Withheld fields and screen labels render as `not available`. Phase-1 configuration, output contents, summary hashing, canonical-path protection, and exclusive publication tests pass. |

I separately reran both original full-analysis reproductions:

- **Cancellation:** N1/N1i/N3 become `unavailable`; the affected screens withhold their labels and adjusted intervals; analysis and rendering complete.
- **5/6 void cohort with constant differences:** affected decisions become `void`; the screen records `convergence.status = void` with no attempts; analysis and rendering complete.

For both fixtures, historical `screen_cells` still raises its original zero-width error. Replacing that historical function with a sentinel that raises also confirmed exp_11 never calls it.

**Verification:** 39 unique selected tests passed: 16 blocker regressions, 14 historical/output tests, six strengthening tests, and three simulated-evaluation closure tests. The latter verified the ten existing evaluation manifests. `bash -n` and both relevant `git diff --check` ranges passed.

The approvals-drift test was rerun separately and failed as expected: **18 identical keys, exactly two differing keys**. I inspected, but did not rerun, the recorded full suite: **3341 passed, 50 skipped, one approvals-drift failure**.

**Approvals to refill after repair and successful close review:** the two leaves in [approved_digests.json](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json:27) are:

| Approval key | Independently computed digest at current HEAD |
|---|---|
| `code.haa_pipeline_sh` | `f33d4ffe015ab76c1fc1115f8918983d6048351ca12bf96c47739e89a8eb573e` |
| `code.summarize_haa` | `2cdd624206f58c0157a4b15ec1d05e98c74017e2bf46eade97e7afc3ffb5b6b1` |

The required repair will change `code.summarize_haa` again. Recompute both at the reviewed merge commit; the table above identifies current bytes, not approval of this rejected tip.

`finalize` remains `ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093`. The four simulated-evaluation keys—`eval`, `eval_launch`, `compare`, and `mirror_probe`—remain unchanged.

No review or repository changes were saved.
tokens used
85,074
**Reviewer:** OpenAI Codex, GPT-6; reasoning-effort setting not exposed in this session.  
**Sandbox/resources:** read-only repository; test writes under `/tmp`; CUDA disabled; ≤4 CPU threads; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`  
**HEAD:** `af052c07a18de695d025ba61cf2fef192a2f66b0`  
**Scope:** five fix commits, `5db0c82..af052c0`; historical baseline `c6233e3`  
**Date/time:** 2026-09-26, 22:22–22:28 EDT

**Verdict: request changes. Round 1 cannot be CLOSED, and this tip is not approved for merge.** One part of the original exception-handling blocker remains.

1. **[P2] Propagate non-zero-width failures from the convergence helper.**  
   [tools/exp06_summarize_haa.py:1299](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1299)

   The fallback catches every `ValueError` and checks only the recomputed **seed-0** interval. Finite, equal seed-0 endpoints do not establish that the helper raised its documented zero-width refusal: the helper validates both seeds first.

   Independently reproduced with the actual source and unchanged bootstrap, without monkeypatching:

   ```python
   rows = {
       "a": np.array([0., 0., 0., 0., 0., 1e308]),
       "b": np.zeros(6),
       "clusters": np.array([0, 1, 0, 1, 0, 1]),
       "seeds": np.repeat(subject.SEEDS, 2),
       "cohort": 2, "n_test": 2,
       "excluded": {}, "per_seed_diff": {},
   }
   ```

   At the accepted, non-default `n_boot=1`:

   - `constant_difference(rows)` is false.
   - Seed 0 produces `[0.0, 0.0]`.
   - Seed 1 produces `[NaN, NaN]`, with bootstrap mean `inf`.
   - `converged_two_way` raises:
     ```text
     the interval function did not return endpoints: seed 1 produced a non-finite or reversed interval
     ```
   - `exp11_cell` instead returns `status: unavailable`, retaining that numerical-failure message as its reason.

   An independently injected unrelated `ValueError` is also swallowed when nonconstant rows produce a finite zero-width nominal interval. The existing unrelated-failure regression exercises only a positive-width interval.

   **Required:** restrict the fallback to the documented `ZERO_WIDTH_REASON`, retain the finite/equal endpoint guard, and re-raise every other failure. Add regressions for the real seed-1 failure above and an unrelated exception with finite/equal seed-0 endpoints.

**Round-1 item verification:**

| Item | Result |
|---|---|
| **1. Convergence ordering and screen suppression** | **Pass.** The gate checks void first, then `constant_difference` **and** `zero_width`, then calls the frozen helper. `exp11_screen_cells` uses shared helpers plus `screen_cell_base`, without calling historical `screen_cells`. |
| **2. Finite zero-width restriction and propagation** | **Partial; blocker remains above.** `zero_width` correctly rejects NaN, ±inf, reversed endpoints, and non-float endpoints. The original six-values-of-`1e308` reproduction now raises. The late catch still suppresses failures originating from another bootstrap seed. |
| **3. Historical oracle** | **Pass.** [The oracle is pinned to `c6233e3`](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:677), with the pre-round structural assertion and all four exp06/exp09 × normal/exploratory cases. Independent comparisons also matched complete payloads, every nested key order, and rendered summaries. |
| **4. Unconverged decisions** | **Pass.** [Direct coverage](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:883) checks all four field types and every registered field across N1/N1i/N2/N3, while retaining descriptive values. |
| **5. G initialization admission** | **Pass.** [The admission test](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:916) admits matching initialization, refuses the wrong digest through `load_new_arm`, and includes the exploratory control. |
| **6. Interaction integrity** | **Pass.** [The strengthened cases](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_summarize_haa.py:957) cover IR-path/K corruption, cross-seed misalignment, missing seeds, and repeated arms. Both component cohorts pass at 198/200 while the joint cohort fails at 196/200, with exactly one joint-cohort reason. |
| **7. Closure hygiene** | **Pass.** All 20 keys were recomputed against `c6233e3`; exactly the expected two changed. Frozen files and `BACKBONES_EXP06` are untouched. |
| **8. Other regressions, rendering, phase-1 output** | **Pass apart from the remaining blocker.** No historical-output regression found. Withheld fields and screen labels render as `not available`. Phase-1 configuration, output contents, summary hashing, canonical-path protection, and exclusive publication tests pass. |

I separately reran both original full-analysis reproductions:

- **Cancellation:** N1/N1i/N3 become `unavailable`; the affected screens withhold their labels and adjusted intervals; analysis and rendering complete.
- **5/6 void cohort with constant differences:** affected decisions become `void`; the screen records `convergence.status = void` with no attempts; analysis and rendering complete.

For both fixtures, historical `screen_cells` still raises its original zero-width error. Replacing that historical function with a sentinel that raises also confirmed exp_11 never calls it.

**Verification:** 39 unique selected tests passed: 16 blocker regressions, 14 historical/output tests, six strengthening tests, and three simulated-evaluation closure tests. The latter verified the ten existing evaluation manifests. `bash -n` and both relevant `git diff --check` ranges passed.

The approvals-drift test was rerun separately and failed as expected: **18 identical keys, exactly two differing keys**. I inspected, but did not rerun, the recorded full suite: **3341 passed, 50 skipped, one approvals-drift failure**.

**Approvals to refill after repair and successful close review:** the two leaves in [approved_digests.json](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json:27) are:

| Approval key | Independently computed digest at current HEAD |
|---|---|
| `code.haa_pipeline_sh` | `f33d4ffe015ab76c1fc1115f8918983d6048351ca12bf96c47739e89a8eb573e` |
| `code.summarize_haa` | `2cdd624206f58c0157a4b15ec1d05e98c74017e2bf46eade97e7afc3ffb5b6b1` |

The required repair will change `code.summarize_haa` again. Recompute both at the reviewed merge commit; the table above identifies current bytes, not approval of this rejected tip.

`finalize` remains `ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093`. The four simulated-evaluation keys—`eval`, `eval_launch`, `compare`, and `mirror_probe`—remain unchanged.

No review or repository changes were saved.

