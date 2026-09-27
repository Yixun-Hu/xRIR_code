# exp_11 orientation_cue_fairness — Coder round-1 FIX report

**Coder:** Claude Opus 5 (1M context), Agent-tool subagent at max effort
**Date:** 2026-09-26 20:47:07 EDT
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`
**Base of this cycle:** `5db0c82` (the reviewed round-1 tip) — **no history was rewritten**;
these are five commits on top.
**Review answered:** `orientation_cue_fairness_codex_code_round1_review.md` (Codex
`gpt-6-astra`, ultra, 2026-09-26 20:24–20:29 EDT, verdict *request changes*).
**Environment:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`,
`PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout. No GPU, no process signals,
`tools/exp06_bootstrap.py` and every other frozen file untouched.

## 1. Commits

| SHA | Subject | changed lines |
|---|---|---|
| `65c5bbd` | exp11: an unavailable statement needs finite, equal endpoints | 68 |
| `bf516c6` | exp11: decide screen suppression before the frozen helper runs | 201 |
| `0137787` | exp11: pin the historical-regression oracle to the base commit | 63 |
| `3b21a70` | exp11: unconverged decisions and G's initialisation at admission | 64 |
| `af052c0` | exp11: the interaction's pairing and joint-cohort cases | 69 |

Cycle diff vs `5db0c82`: `tools/exp06_summarize_haa.py` +107/−30,
`tests/test_exp11_summarize_haa.py` +303/−15. Nothing else was touched.

## 2. Blocker 1 — exp_11 suppression now precedes convergence (`bf516c6`)

**Reproduced first.** With the reviewer's fixture (three seeds, six queries,
`G = D = arange(6) + 1`, `E = A = arange(6)`) the new full-analysis test failed with
`tools/exp06_bootstrap.py:125: ValueError: a zero-width interval cannot carry a
convergence verdict`, exactly as reported: `exp11_screen_cells` called the historical
`screen_cells` first, which resamples and runs the convergence helper on every non-empty
cohort before any exp_11 gate could look at the cell.

**Fix.** A new gate, `exp11_convergence(rows, nominal, void, alpha, n_boot)`, orders the
three checks explicitly:

1. **invalidity first** — a cell with void reasons returns
   `{'status': 'void', 'n_boot': None, 'interval': None, 'attempts': []}` and the frozen
   helper is never called;
2. **the registered cancellation second** — established two ways, from the rows
   (`constant_difference`: every paired difference the same finite number, so a resample
   of any size at any level averages to it) **and** from the computed interval
   (`zero_width`: two finite, equal endpoints). A set whose arithmetic overflowed
   satisfies the first and not the second, so it is not mistaken for cancellation;
3. **the frozen helper last**, with the narrow late guard of `65c5bbd` as defence.

`exp11_screen_cells` no longer calls `screen_cells`. It assembles its eleven cells from
the same helpers (`cell_rows`, `void_reasons`, `intervals`, `h2_label`) and the extracted
`screen_cell_base`, running the gate before any convergence. `screen_cells` itself is
**unchanged** apart from that pure extraction of its pre-resampling dict; a test asserts
it still raises on the very fixture exp_11 now withholds, so the historical exp_06/exp_09
semantics are preserved by construction, and the pinned-oracle regression (§4) proves
both experiments' payloads and rendered summaries are byte-identical to the base commit.

`exp11_cell` goes through the same gate, so decision and screen cells share one ordering.

**Regressions, through `analyse` (not `exp11_decision`):**

* `test_a_cancelling_hallway_withholds_cells_instead_of_aborting_the_phase` — the phase
  summary is produced: N1, N1i and N3 are `unavailable` with every field withheld, the
  G − D and G − E hallway C50 screen cells are `withheld` with `label: null`,
  `adjusted_two_way: null` and the frozen helper's own reason, the hallway EDT cell and
  the C − G family are unaffected, and `render` still produces the summary.
* `test_a_void_cohort_of_constant_differences_withholds_without_any_convergence` — one
  invalid G query on top of the same fixture: a 5/6 cohort with void reasons, label
  withheld, and `convergence == {'status': 'void', …}` — i.e. no convergence was run at
  all on an invalid cell.
* `test_the_convergence_gate_runs_the_frozen_helper_last` — the helper is monkeypatched to
  raise if called; the void and cancellation branches still return their results.
* `test_the_recorded_reason_is_the_frozen_helpers_own_wording` — `ZERO_WIDTH_REASON` is
  pinned by raising it from `bootstrap.convergence_endpoints`, so a reworded helper fails
  the test instead of letting exp_11 publish a reason nothing raises.
* `test_the_historical_screen_still_raises_on_the_same_cells` — the historical behaviour
  is asserted, not changed.

## 3. Blocker 2 — `unavailable` requires finite, equal endpoints (`65c5bbd`)

`zero_width(interval)` replaces the `hi > lo` test: both endpoints must be `float`,
finite and equal. Everything else propagates — positive width, NaN, ±inf, a reversed
interval, and any unrelated `ValueError`.

Tests: the predicate over equal / reversed / NaN / inf / mixed endpoints; the reviewer's
own reproduction (six finite values of `1e308` against zero) now raising
`non-finite or reversed` instead of publishing `diff: inf` with a NaN interval; a reversed
interval raising; the positive-width and unrelated-failure cases retained from round 1;
and the genuine cancellation still reported as `unavailable`.

## 4. Test strengthening

3. **Pinned oracle** (`0137787`). The historical regression no longer loads
   `main:tools/exp06_summarize_haa.py` (which would compare the implementation with itself
   after the merge) but `git show c6233e3:tools/exp06_summarize_haa.py`, loaded as a module
   from a temp file. It is parametrised over `exp06`/`exp09` × normal/exploratory (4 cases),
   comparing the complete `analyse` payload (`json.dumps(sort_keys=True)`), the key order
   and the rendered summary. `test_the_pinned_oracle_really_predates_this_round` asserts the
   oracle has no arm G, no `exp11` experiment and none of the exp_11 helpers, so a
   degenerate self-comparison cannot pass silently.
4. **Unconverged cells** (`3b21a70`). A direct `exp11_cell` test with the convergence
   status forced to `not_converged`: every requested field withheld for the contrast cell
   and, looping over all four registered statements, for the interaction too, while `diff`
   and the nominal interval are still reported.
5. **Wrong initialisation at admission** (`3b21a70`). `load_new_arm` is exercised for G
   with a stubbed `verify_job`: the matching digest is admitted (positive control, four
   jobs, the heading bound per room), a job recording another checkpoint is refused with
   *"did not start from the registered initialisation"*, and the unpinned (exploratory)
   call still loads. This is the admission path a published summary rests on, distinct
   from the pipeline's launch-time gate.
6. **Interaction cases** (`af052c0`). A private 200-query four-arm fixture separates the
   component cohorts from the joint one (the shared six-query hallway cannot: any
   exclusion there already voids a component). Two excluded queries per component leave
   each at 198/200 with **no** void reason while the joint cohort falls to 196, and the
   ≥ 99 % joint rule is the single recorded reason. Plus: a corrupted `ir_path` and a
   changed `num_shot` refused by the pairing assertions, seeds that measured different
   query orders refused even though every arm agrees within each seed, a missing seed
   named by arm and room, and four arms that are not distinct refused.

Items 4–6 are characterisation tests of guards that already held; they were written and
run before being committed, but they did not go red first because the behaviour they pin
was already correct. Items 1–3 were red first (the two blocker reproductions and the
oracle pin all failed before their fix).

## 5. Verification

* `bash -n tools/exp06_haa_pipeline.sh`, `py_compile` on the changed modules and
  `git diff 5db0c82..HEAD --check` — clean.
* `tests/test_exp11_summarize_haa.py` + `tests/test_exp06_haa_pipeline.py`: **136 passed**
  (4 min 34 s).
* `tests/test_exp06_summarize_haa.py`: **98 passed** (7 min 14 s) — the `screen_cell_base`
  extraction changes nothing exp_06 or exp_09 publishes.
* `tests/test_exp09_sim_eval_closures.py`: **3 passed**.
* Full CPU suite at `af052c0`: **1 failed, 3341 passed, 50 skipped** in 1 h 29 min
  (log `orientation_cue_fairness_2026-09-26_20:50_suite_full_cpu_af052c0.log`). The single
  failure is again
  `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`,
  the exp_06 approvals drift guard: 18 of the 20 filled `code.*` keys are identical and
  exactly the two of §6 differ. It goes green with the re-fill at the merge commit.

## 6. Code-key digests at the fix tip `af052c0`

Recomputed over all 20 keys with `tools.exp06_profiles.compute_code_digests`; exactly two
moved against `c6233e3`, as in round 1:

| key | digest at `af052c0` | was (`c6233e3`) |
|---|---|---|
| `haa_pipeline_sh` | `f33d4ffe015ab76c1fc1115f8918983d6048351ca12bf96c47739e89a8eb573e` | `4a401ee3d80f2e3c77ecf68fd538e0108d6c649c92680c1b0373824626ad8c08` |
| `summarize_haa` | **`2cdd624206f58c0157a4b15ec1d05e98c74017e2bf46eade97e7afc3ffb5b6b1`** | `3fc1e82d374c8e5fb8130d5223b7d744168b0462dabc1d22017423b4a1ad3376` |

`haa_pipeline_sh` is unchanged from the reviewed round-1 tip; **`summarize_haa` moved
again** (it was `32937f96…` at `5db0c82`) — the re-fill at the merge commit must use the
value above, recomputed there. `finalize` remains
`ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093`, and the four sim-eval
closure keys are untouched.

## 7. Commit-size exceptions (review item 7)

The SOP's ≤ 200-line guideline was exceeded by five commits in all; each is cohesive and
each production half is well under 200 lines, and per the reviewer I did not rewrite
reviewed history to split them:

| SHA | lines | production half |
|---|---|---|
| `fff7948` | 202 | 95 |
| `968e167` | 208 | 94 |
| `824dce9` | 248 | 101 |
| `afd5a71` | 202 | 121 |
| `bf516c6` (this cycle) | 201 | 117 |

Round-1 report wording narrowed: it said "tests written first and run red before the
implementation in every case", which is true of the substantive round-1 commits but not of
the later acceptance-only commits (`73a8732`, `5db0c82`) or of items 4–6 here, which pin
behaviour that already held. The accurate statement is: **every behavioural change in this
round was preceded by a failing test; the acceptance-only commits add characterisation
tests for guarantees that already held.**

The three open schema questions are left exactly as the reviewer advised: contrast strings
stay the screen-family keys, `result['decisions']` stays the ordered statement-name list,
and N1/N1i keep `margin: null`.

## 8. Still outstanding (unchanged from round 1)

Approvals re-fill of `code.haa_pipeline_sh` and `code.summarize_haa` at the reviewed merge
state (this is also what makes
`tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`
green again); the record generators and `planner_probes/exp11_runbook.sh`; the merge into
`main`; G's bounded heading-frame smoke on a card; and all of round 2.
