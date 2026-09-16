**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 oriented_cyl — round 2a, fix cycle 4 (Codex close review 3, findings 1–2 + nit 3)

Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`, from `a977a68` to
`12e06fc`. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`,
`PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt`,
`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest `-p no:cacheprovider`), no GPU work, no
process signalled, nothing written under `ckpt/` or the main tree except this report. Only
`tools/exp06_finalize.py` and `tests/test_exp06_finalize.py` were touched;
`git diff main...exp06-window --diff-filter=M --name-only` is empty (no file that exists on
`main` was modified), so every pinned closure — exp_03's twelve files, the trainer's import
closure, `tools/provenance.py`, `sim_to_real/*.py`, `tests/test_provenance.py` — is untouched.
The superseded archive folder and the `archive/codex-round2a-2026-09-15` branch were not read.

## 1. Commits (changed lines = insertions + deletions; every one ≤ 200)

| # | Commit | Changed | Subject |
|---|---|---:|---|
| 1 | `3eaec27` | 26 | red job tests for artifact maps that record an extra null (finding 1) |
| 2 | `c91e014` | 9 | compare a child's whole artifact mapping, key set first (finding 1) |
| 3 | `c4ec284` | 63 | red tests for invalid counters that contradict the observations (finding 2) |
| 4 | `6418e11` | 18 | reconcile the invalid-measurement counters with the observations (finding 2) |
| 5 | `027bd5b` | 21 | red CLI test for a JSON integer too wide for a float (nit 3) |
| 6 | `f0898a1` | 16 | name the refusal when a JSON integer cannot become a float (nit 3) |
| 7 | `12e06fc` | 6 | document how the invalid-measurement counters are reconciled (finding 2) |

Maximum 63 changed lines; every commit carries the two required trailer lines
(`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`,
`Claude-Session: https://claude.ai/code/session_01NT7CPAUaJ71xgzoS8JGDVw`). Each
implementation commit follows its own red-test commit (TDD pairs 1/2, 3/4, 5/6); commit 7 is
documentation only and changes no behaviour. Nothing was amended, rebased, reset or pushed.

## 2. Red counts (measured at the red commit, before the paired implementation)

| Red commit | Failing cases | Which |
|---|---:|---|
| `3eaec27` | 2 | `…artifact_map_must_equal_the_re_run_key_for_key[stage1-extra_null, eval/hallway-extra_null]` |
| `c4ec284` | 9 | `…parsed_and_cross_checked[count_c50_under, count_c50_over, count_edt_excess, count_t60_excess]`, `…omitted_t60_room_measures_and_counts_nothing[2 cases]`, `…job_refuses_counters_that_contradict_the_observations[3 cases]` |
| `027bd5b` | 2 | `…integer_too_wide_for_a_float_is_a_named_refusal[edt, mean]` (both `OverflowError: int too large to convert to float`, at `tools/exp06_finalize.py:767` `_numbers` and `:589` `_finite`) |

13 red cases in total. Three of the five finding-1 regression cases (`stage1-extra_hash`,
`stage1-null_for_hash`, `eval/hallway-missing_key`) were already refused by the previous
value comparison and are kept as guards; the two extra-null cases are the review's
reproduction and were green-on-refusal only after the fix.

## 3. Per-finding fix and regression

### Finding 1 — exact artifact-mapping agreement in `verify_child`

`tools/exp06_finalize.py::verify_child` compared the union of the two key sets with
`recorded.get(a) == fresh.get(a)`, so an absent key and a recorded `null` agreed. It now
requires **identical key sets** first (naming both sorted lists) and then compares by **direct
indexing** with `exp06_recipe.strict_equal` (type-strict, so no `True`/`1`/`None` can stand in
for a hash). A child completion whose `artifacts` mapping is not exactly the mapping the
re-run hashed is refused, and nothing is written.

Regression `test_a_child_artifact_map_must_equal_the_re_run_key_for_key` (job level, five
cases): an extra `"nonexistent.pth": null` in `stage1/completion.json`, the same in
`eval/hallway/completion.json`, an extra string-valued artifact, a `null` where a hash is
expected, and a missing key — each refused with a cause containing “artefact”, and no
`completion.json` at the job root.

### Finding 2 — invalid-measurement counters reconciled with the observations

`sim_to_real/eval_haa.py:76–98` (read for this cycle) counts **every** non-finite C50
(`c50` is written only when `np.isfinite(c)`), but counts EDT and T60 **only when the
measurement raised** `ValueError`/`IndexError` — a subset of their non-finite observations —
and never runs T60 at all when `room in NO_T60_ROOMS`, where every `t60` value is the
initialised NaN and the summary is `null`. `haa_metrics` now encodes exactly that with a new
module constant `METRIC_INVALID = {'c50': ('c50_outliers', True), 'edt': ('edt_invalid',
False), 't60': ('t60_invalid', False)}` (the flag = exact equality vs. upper bound):

* `c50_outliers` **must equal** the number of non-finite per-sample `c50` values;
* `edt_invalid` ≤ the number of non-finite `edt` values, `t60_invalid` ≤ that of `t60`;
* in an omitted-T60 room: the summary stays `null` (as before), **every** `t60` observation
  must be non-finite, and `t60_invalid` must be 0.

Legitimate NaN observations are preserved: a NaN the writer did not count is still a real
measurement it read as non-finite, so only contradictions refuse. The bound check against
`n_samples` and the recomputed `{mean, median, n}` summaries are unchanged and still run.

Regressions — standalone (`test_the_metrics_summary_is_parsed_and_cross_checked`, cases
`count_c50_under` = the review's `[1.1, 1.3, NaN]` with `c50_outliers=0`, `count_c50_over`,
`count_edt_excess` = three finite EDT with `edt_invalid=3`, `count_t60_excess` = three finite
T60 with `t60_invalid=2`); omitted-T60 room
(`test_an_omitted_t60_room_measures_and_counts_nothing`: all-NaN T60 with `t60_invalid=2`, and
finite T60 values in `dampened_room`); job admission with refreshed metric-file hashes
(`test_a_job_refuses_counters_that_contradict_the_observations`, three cases, each rewriting
`eval/hallway/metrics_hallway.json` **and** the child completion's hash for it); and
acceptance of a consistent file
(`test_invalid_measurements_may_be_fewer_than_the_nonfinite_observations`: EDT
`[0.05, NaN, 0.07]` with `edt_invalid=1`, T60 `[4.0, NaN, 6.0]` with `t60_invalid=0`, C50 with
one NaN and `c50_outliers=1` — finalisation succeeds and records three samples).

### Nit 3 — numeric-conversion overflow becomes a named refusal

`math.isfinite(10 ** 400)` raises `OverflowError`, which escaped both numeric checks. A new
`_isfinite(label, value)` helper translates it into
`ValueError('<field label> is an integer of <n> digits, too large for a finite number')`, and
both call sites use it: `_finite` (summary means/medians, heading `phi_deg`, history losses)
and `_numbers` (per-sample values). `main()` therefore reports the documented named refusal
and exits 2.

Regression `test_an_integer_too_wide_for_a_float_is_a_named_refusal` runs through
`exp06_finalize.main(...)` for both sites: a per-sample `edt` of `10 ** 400` (cause names
`edt`) and a summary `edt_error_s.mean` of `10 ** 400` (cause names `mean`); both return
status 2 with the cause on stderr and write no completion.

## 4. Validation commands and outcomes

| Command | Outcome |
|---|---|
| `pytest tests/test_exp06_finalize.py -k artifact_map` (at `3eaec27`) | **2 failed, 3 passed** (red) |
| `pytest tests/test_exp06_finalize.py` (at `c91e014`) | **260 passed**, 8 warnings, 249.6 s |
| `pytest tests/test_exp06_finalize.py -k "count_ or omitted_t60 or nonfinite_observations or dampened or contradict or corrupt_per_sample"` (at `c4ec284`) | **9 failed, 9 passed** (red) |
| same selection + `metrics_summary` (at `6418e11`) | **32 passed** |
| `pytest tests/test_exp06_finalize.py -k too_wide` (at `027bd5b`) | **2 failed** (red, `OverflowError`) |
| `pytest tests/test_exp06_finalize.py -k "too_wide or metrics_summary or heading or history"` (at `f0898a1`) | **41 passed** |
| `python -m py_compile tools/exp06_finalize.py tests/test_exp06_finalize.py` | passed |
| `bash -n tools/exp06_launch.sh` | passed |
| `git diff --check` | clean (before and after every commit) |
| `git diff main...exp06-window --diff-filter=M --name-only` | empty |
| `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider` | **540 passed, 3 skipped**, 31 warnings, 478.7 s (the 3 skips are the CUDA-only cases; 523 + 17 new cases this cycle) |

## 5. Discrepancies and deviations

1. **Test-fixture correction required by finding 2.** The helper `write_haa_eval` wrote the
   same finite T60 values for every room, including `dampened_room`, which the pinned writer
   never produces. Enforcing “every T60 observation is NaN in an omitted-T60 room” would have
   made that fixture — and the job fixture's `eval/dampened_room` child — self-contradictory,
   so the helper now writes `[NaN] * n` for `NO_T60_ROOMS` unless the case overrides `t60`
   explicitly. `test_the_dampened_room_records_no_t60` and the nine-child job fixture are
   otherwise unchanged and stay green.
2. **Finding 1 comparison is slightly stronger than the review's minimal fix.** Values are
   compared with `exp06_recipe.strict_equal` rather than `==`, so a boolean or numeric value
   can never equal a hash string. No test depended on the looser comparison.
3. **Nit 3 was fixed in this cycle rather than batched into round 2b**, as the cycle
   instruction asked; it is a two-line translation plus one helper and touches no other
   behaviour. `_isfinite` is used by `_finite` (so heading `phi_deg` and history losses are
   covered as well) and by `_numbers`.
4. No other spec/code discrepancy. Nothing under the main tree was written except this report;
   `ckpt/` was not written to; no process was signalled; no plan deviation.
