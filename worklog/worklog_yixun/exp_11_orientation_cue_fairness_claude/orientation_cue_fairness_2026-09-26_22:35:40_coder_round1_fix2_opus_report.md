# exp_11 orientation_cue_fairness — Coder round-1 FIX CYCLE 2 report

**Coder:** Claude Opus 5 (1M context), Agent-tool subagent at max effort
**Date:** 2026-09-26 22:35:40 EDT
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`
**Base of this cycle:** `af052c0` (the reviewed fix-cycle-1 tip) — **no history rewritten**;
one commit on top.
**Review answered:** `orientation_cue_fairness_codex_code_round1_close_review.md` (Codex
GPT-6, 2026-09-26 22:22–22:28 EDT, *request changes*, one remaining blocker; items 1 and
3–8 passed).
**Environment:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`,
`CUDA_VISIBLE_DEVICES=''`. No GPU, no process signals, `tools/exp06_bootstrap.py` and
every other frozen file untouched.

## 1. Commit

| SHA | Subject | changed lines |
|---|---|---|
| `91fbd8b` | exp11: only the helper's own zero-width refusal is unavailable | 86 |

`tools/exp06_summarize_haa.py` +28/−5, `tests/test_exp11_summarize_haa.py` +53. Nothing
else touched.

## 2. The blocker, reproduced and fixed

**Reproduced first**, with the real source and the unchanged bootstrap, at the accepted
non-default `n_boot=1` with `a = [0, 0, 0, 0, 0, 1e308]`, `b = zeros(6)`,
`clusters = [0, 1, 0, 1, 0, 1]`, `seeds = repeat(SEEDS, 2)`:

```
constant_difference: False
seed 0 two_way: {'lo': 0.0, 'hi': 0.0, ...}          # finite and equal
seed 1 two_way: {'lo': nan, 'hi': nan, 'mean': inf}  # the mean overflowed
helper raised:  the interval function did not return endpoints:
                seed 1 produced a non-finite or reversed interval
BEFORE -> status: unavailable, reason: <that numerical-failure message>
AFTER  -> ValueError: ... seed 1 produced a non-finite or reversed interval
```

The frozen helper validates **every** seed before it measures a width, so a recomputation
of seed 0 alone can never establish that it raised its zero-width refusal. The fallback
now requires **both** conditions and re-raises everything else unchanged:

```python
    except ValueError as error:
        if str(error) != ZERO_WIDTH_REASON:
            raise
        if not zero_width(intervals(rows, alpha, n_boot)['two_way']):
            raise
```

**The message is no longer a literal in this module.** `tools/exp06_bootstrap.py` is
frozen and publishes the text only by raising it, so `_zero_width_reason()` provokes it
once at import from `bootstrap.convergence_endpoints(lambda seed: (0.0, 0.0))` — a
degenerate interval function, no resampling at all — and `ZERO_WIDTH_REASON` is whatever
the helper said. A reworded helper therefore changes the constant with it instead of
leaving a stale copy behind; a test asserts the string appears nowhere in this module's
source, so the duplication cannot come back.

The exact-equality test is correct for this helper: `convergence_endpoints` raises the
zero-width refusal *after* its per-seed validation block, so that message is never
wrapped in the `the interval function did not return endpoints: …` prefix, while every
per-seed failure always is.

Nothing else changed: the ordering gate of fix cycle 1 (`void` → `constant_difference`
**and** `zero_width` → the helper) is untouched, so in practice the fallback is reached
only by a refusal the pre-gate could not foresee.

## 3. Regressions added

* `test_a_refusal_raised_for_another_bootstrap_seed_is_not_unavailable` — the reviewer's
  case verbatim: the constant-difference gate is false, the seed-0 interval *is* finite
  and zero-width, and `exp11_cell` must still raise.
* `test_an_unrelated_failure_with_finite_equal_seed_zero_endpoints_propagates` — an
  injected unrelated `ValueError` over those same rows propagates (the previous
  unrelated-failure test only covered a positive-width interval).
* `test_the_documented_refusal_is_still_reported_as_unavailable` — the one refusal exp_11
  may publish, raised with the helper's own wording, is still a defined `unavailable`
  statement with that reason.
* `test_the_refusal_text_comes_from_the_frozen_helper_and_is_not_copied` — the constant
  equals what the frozen helper raises, and the literal is absent from the module source.
* The fix-cycle-1 cases are retained: positive width, NaN, ±inf, reversed, the six-values
  of `1e308` reproduction, the genuine cancellation, and the full-analysis suppression
  regressions.

## 4. Verification

* `bash -n tools/exp06_haa_pipeline.sh`, `py_compile` on the changed files,
  `git diff af052c0..HEAD --check` — clean.
* `tests/test_exp11_summarize_haa.py`: **71 passed**.
* `tests/test_exp09_sim_eval_closures.py`: **3 passed**.
* `tests/test_exp11_summarize_haa.py` + `tests/test_exp06_summarize_haa.py` +
  `tests/test_exp06_haa_pipeline.py`: **238 passed** (20 min 52 s).
* Full CPU suite: FINAL_SUITE_LINE

## 5. Code-key digests at `91fbd8b`

All 20 keys recomputed against `c6233e3`; exactly two moved, as in both previous cycles:

| key | digest at `91fbd8b` |
|---|---|
| `haa_pipeline_sh` | `f33d4ffe015ab76c1fc1115f8918983d6048351ca12bf96c47739e89a8eb573e` (unchanged since round 1) |
| `summarize_haa` | **`645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619`** |

`summarize_haa` moved again (round 1 `32937f96…` → fix 1 `2cdd6242…` → **`645e74c0…`**);
the re-fill at the merge commit must use the value recomputed there. `finalize` remains
`ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093`; the four sim-eval
closure keys and `BACKBONES_EXP06` are untouched.

## 6. Outstanding (unchanged)

Approvals re-fill of `code.haa_pipeline_sh` and `code.summarize_haa` at the reviewed merge
state — which is also what makes
`tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`
green; the record generators and `planner_probes/exp11_runbook.sh`; the merge into `main`;
G's bounded heading-frame smoke on a card; and all of round 2.
