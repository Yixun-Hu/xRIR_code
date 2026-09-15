**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# Round 3a — device-preserving probe alignment, the G1 mirror probe, the alpha-aware bootstrap

Worktree `/home/yixunhu/codespace/xRIR_code_wt2b`, branch `exp06-round2b`, base `3df20b8`.
CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), `PYTHONPATH` and
`XRIR_DATA_PATH` as the prompt specifies, `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest with `-p no:cacheprovider`.
The prohibited worktree `/home/yixunhu/codespace/xRIR_code_wt` and the superseded round-2a
archive were never read. No process was signalled; nothing under `ckpt/` was written.

## Commits (base `3df20b8` → `895f512`, 16 commits)

| SHA | changed lines | subject |
|---|---|---|
| `4a8189d` | 51 | exp06: red tests that the post-run gate reads the retained completion (nit 7) |
| `626e8d4` | 20 | exp06: validate the hashes the run published, not the caller's dict (nit 7) |
| `141a166` | 34 | exp06: red tests that a refused preparation leaves a receipt (nit 8) |
| `1bf2532` | 38 | exp06: a refused preparation writes its receipt and aborts the job root (nit 8) |
| `2176329` | 138 | exp06: red tests for the device-preserving probe alignment |
| `8a25004` | 82 | exp06: device-preserving shift_and_align for the CPU probe |
| `792a16d` | 225 | exp06: red tests for the mirror probe's cohort rule, forward and decision |
| `0c89feb` | 188 | exp06: the mirror probe's cohort rule, recomposed forward and G1 decision |
| `d6f19f6` | 287 | exp06: red tests for the mirror statistics, the gate arms and the bound record |
| `ce010d1` | 152 | exp06: the mirror statistics of one arm, with room-frame side labels |
| `c6b41b7` | 82 | exp06: the legacy reproduction and the full-cohort gate arms |
| `abb6d7d` | 109 | exp06: the hash-bound G1 record and its command line |
| `402a42b` | 229 | exp06: red tests for the alpha-aware paired bootstrap |
| `4f4355d` | 167 | exp06: the alpha-aware paired bootstrap and its convergence rule |
| `cb3c770` | 11 | exp06: red tests that an empty batch and an unserialisable record are refused |
| `895f512` | 11 | exp06: refuse an empty C50 batch and an unserialisable completion record |

Every commit message ends with the two required trailer lines. Nothing was amended,
rebased, reset or pushed.

Files touched (`git diff --stat 3df20b8..HEAD`): new `tools/exp06_probe_align.py`,
`tools/exp06_mirror_probe.py`, `tools/exp06_bootstrap.py`,
`tests/test_exp06_probe_align.py`, `tests/test_exp06_mirror_probe.py`,
`tests/test_exp06_bootstrap.py`; edited `tools/exp06_eval_launch.py`,
`tools/exp06_haa_pipeline.sh` (the two permitted cycle-0 files) and their two test files
`tests/test_exp06_eval_launch.py`, `tests/test_exp06_haa_pipeline.py`. A grep of the diff's
file list against the whole HARD-RULE set (`train_xRIR_backbone.py` and its closure,
`tools/exp04_*`, `tools/exp05_*`, `tools/provenance.py`, `tools/paired_compare.py`,
`sim_to_real/*`, exp_03's pinned files, `tests/test_provenance.py`,
`tests/test_exp03_record_tools.py`) and against the concurrently edited round-2 files
(`exp06_finalize.py`, `exp06_train.py`, `exp06_recipe.py`, `exp06_smoke.py`,
`exp06_profiles.py`, `exp06_launch.sh`) returns nothing.

## Cycle 0 — the two round-2b close-review nits

**Nit 7 (`tools/exp06_eval_launch.py`).** `validate_outputs` now begins with `_persisted`,
which re-reads `<run>/completion.json`, refuses it if it is unreadable, is not an object, or
is not byte-for-byte the record `execute_run` returned (`_same`, the existing type-strict
comparison), and then validates the *persisted* record's manifest digest and output hashes.
New refusal reasons `completion_unreadable` and `completion_mismatch` quarantine as before.
Six new regressions (red first): a persisted record whose hash for either output was
replaced while the returned mapping stayed correct; missing, unparsable and non-object
completions; and the converse case where the file on disk was made self-consistent with a
rewritten output but no longer matches what the launcher returned. The `certified` fixture
now publishes `completion.json` and a `persist()` helper republishes it, so the pre-existing
corruption tests still exercise the same path.

**Nit 8 (`tools/exp06_haa_pipeline.sh`).** `prepare_job`'s three gates now call a new
`prepare_failed <root> <reason>`, which prints the same `REFUSED <reason> <root>` line, writes
`<root>/preparation_failure.json` (`schema_version`, `reason`, `job_root`, `aborted_dir`,
`failed_at` ISO timestamp, and `provenance.git_state` of the repo, so HEAD/dirty/diff are
bound), renames the job root to `<root>_ABORTED_prepare_<reason>` (suffixing the launcher pid
if that name is taken), and always returns 1. Failure to write the receipt or to rename is
itself announced (`UNRECORDED` / `UNABORTED`) and never converted into success. The
`_ABORTED_` rename is what removes the review's "dead `launch.pid` in a live-looking job
root" state. Five new regressions: a receipt with the right reason, job root, aborted dir,
HEAD and a parsable timestamp for each of the three gates; two refused preparations of the
same job leaving two distinct `_ABORTED_` directories; and the real refused-heading path
(`flat_dampened`) leaving a `job_spec` receipt and no `job_spec.json`. `bash -n` passes and
the three dry-run golden-argv tests are unchanged (the failure path is inert under
`--dry-run`).

## Cycle 1 — `tools/exp06_probe_align.py`

`shift_and_align_device(model, x, src_loc, ref_ir_locs)` re-implements `xRIR.shift_and_align`
and `xRIR.apply_delay` with `torch.zeros_like(signal)` in place of the pinned
`torch.zeros_like(signal).cuda()`: the same `round((|s| − |r|)/343·22050)` integer delays
(`.int()`), the same `|r|/(|s| + 1e-7)` gains, the same right-shift/left-shift/zero-pad
slicing and the same clipping-by-empty-slice for delays at least as long as the signal.
`integer_delays_and_gains` and `apply_delay_device` are exposed separately. The `model`
argument is accepted only so the helper and the pinned method are interchangeable at a call
site; it is type-checked (`nn.Module` or `None`) and never read, which also catches an
argument passed in the wrong position. Fail-closed checks: all three inputs tensors,
floating, one shared dtype, one shared device, shapes `[B,K,L] / [B,3] / [B,K,3]`.

Tests (27, one GPU-skipped): agreement with an independent pure-NumPy reference for
`float32` (2e-6) and `float64` (1e-12) over batch/shot/length combinations including `K=1`,
`K=8` and `B>1`; a single-impulse structural test over delays `0, ±1, ±7, ±9, ±240, ±400`
that checks the landing index, the gain, zero padding at both ends and total clipping;
delays and gains against hand-computed sample counts; output allocated on the input's device
with the input unmutated; eight malformed-argument refusals; a source check that the module
contains no `.cuda(` call and no `'cuda'` device string; and a
`skipif(not torch.cuda.is_available())` test asserting `torch.equal` with
`model.shift_and_align` on CUDA (skipped here — CPU-only session).

## Cycle 2 — `tools/exp06_mirror_probe.py`

* `composed_forward(model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, align=…)` mirrors
  `xRIR.forward` line for line, transferring the time grid exactly as the pinned forward does
  (`model.times.unsqueeze(0).to(ref_locs.device)`, plan R3.2 — `times` is a plain attribute,
  not a buffer, so `model.to(device)` does not move it). It returns
  `(out_log_spec [B,63,T], tgt_spec [B,63,T], weights [B,N,T], source_out [B,C])`.
* `geometry_feature(model, src_locs, depth_coord)` → `[N, C]`;
  `spectral_c50(mag, onset)` with `C50_FRAMES = round(0.05·22050/31) = 36`;
  `onset_frames(src_loc)` = `round(|s|/343·22050/31)`; `mirror_positions` = `(−x, −y, z)`.
* `legacy_cohort(test_indices, y_room_frame, n=24)` is the 2026-09-14 rule (rank by
  room-frame y per side, `np.linspace(0, len−1, n).round().astype(int)`), returning both
  positions in the split and microphone ids. `HALLWAY_LEGACY_COHORT` freezes both sides;
  a test re-derives them from `~/data_cache/HAA_xrir/hallway` and they match.
* `mirror_stats(model, dataset, room, query_ids, frame_k, side_of_query, device, batch)`
  computes the mirror cosine of the pooled geometry features, the share of
  `mean_t |w_i(t)|` carried by opposite-side references, and the signed spectral-C50 error
  (pred − true), with standard deviations and the reference side counts. Side labels are
  always room-frame signs read from `dataset.data[room]['src_local']` before any roll;
  `frame_k` must equal the roll the dataset itself applies (`k_by_room`, else 0), the
  panorama the dataset serves is checked against the one the probe rotated
  (`torch.equal`), and every drawn reference block is checked against
  `data['rirs'][_pick_refs(...)]` so the side labels provably belong to the references used.
  A dataset without a fixed `eval_seed` and any non-finite statistic are refusals.
* `legacy_reproduction(...)` (CPU, batch 8, room frame, frozen +y cohort) and
  `full_gate(cylor_checkpoint, heading_k, …)` (all +y test microphones, arms `cyl_or`,
  `cyl`, `control`, `cyl_hf`, device `cuda` when available else `cpu`, recorded).
  `g1_decision(stats)` applies the bracketing precondition (`cyl` cosine > 0.90, `control`
  cosine < 0.60, `cyl` share > 0.70, `control` share < 0.35) → otherwise *inconclusive*, then
  passes iff `cyl_or` cosine < 0.80 **and** share < 0.50. Every comparison is strict, so a
  value exactly on a threshold never satisfies it.
* CLI: `python tools/exp06_mirror_probe.py --cylor-checkpoint … --heading-json … --out …
  [--device cuda|cpu] [--legacy-only] [--haa-root …] [--room …] [--batch N] [--num-shot N]
  [--cyl-checkpoint …] [--control-checkpoint …]`. The record binds the sha256 of all three
  checkpoints, the sha256 of the heading JSON (read through
  `exp06_heading.read_heading_json(..., room_dir=…)`, so its recorded cache input hashes are
  re-verified) plus its `k`/`phi_deg`/`decision`, the frozen and full cohorts, the legacy
  report, the per-arm statistics, the decision, the import-derived source closure of
  `tools.exp06_mirror_probe` with `git_state`, `provenance.environment()` and a UTC
  timestamp. It is published with `provenance.write_manifest` (`open(..., 'xb')`), so an
  existing output is refused rather than overwritten. Exit codes: 0 pass, 3 fail,
  4 inconclusive, 2 argparse.

No diagnostic script is imported and no Torch global is patched.

Tests (72; 2 GPU/real-probe skips): the cohort rule against a hand-computed selection and
against the real cache, and five malformed-input refusals; `spectral_c50` against a
synthetic early/late spectrogram and four shape refusals; `geometry_feature` bit-equal to the
manual pinned composition; `composed_forward` bit-equal to
`Σ log_specs · weights` recomputed from its own returned weights and to
`convert_ir_to_spec(tgt_wav)[:,0]`, with a spy confirming the time grid is handed over on the
inputs' device and a check that the injected alignment is the one used; a GPU-skipped parity
test against `model.forward` (`atol/rtol 1e-5`); the mirror/rotation commutation for `k = 0`
and `k = 128`; `mirror_stats` on a synthetic HAA cache (tiny `dim=32, depth=1` model) with the
cosine checked against a manual computation, side labels and reference counts identical in
the room and heading frames, and six refusals (frame mismatch in both directions, wrong side,
unknown room, non-deterministic references, empty cohort); `legacy_reproduction` reporting its
deviation fail-closed; `full_gate` running all four arms; and four CLI tests (full binding and
overwrite refusal, a failed legacy reproduction forcing *inconclusive*, `--legacy-only`
writing no verdict, a heading whose cache moved refused).

### Real-checkpoint legacy reproduction (`EXP06_REAL_PROBE=1`, CPU, 4 threads)

`python -m pytest tests/test_exp06_mirror_probe.py -k published_anchors` → **1 passed in
44.4 s**. The numbers, from `legacy_reproduction(device='cpu')` on the frozen 24-microphone
+y hallway cohort with `eval_seed 0` references (60.9 % of drawn references are −y):

| model | mirror cosine | anchor | Δ | −y weight share | anchor | Δ | signed spectral-C50 error |
|---|---|---|---|---|---|---|---|
| `cyl` (`ckpt/xRIR_cyl_8_shot/epoch_12.pth`) | 0.9745 | 0.975 | −0.0005 | 0.8510 | 0.85 | +0.0010 | +7.44 dB |
| `control` (`ckpt/xRIR_simple_8_shot/epoch_12.pth`) | 0.4354 | 0.435 | +0.0004 | 0.2382 | 0.24 | −0.0018 | +1.71 dB |

All four deviations are inside ±0.01, so the probe's numerics reproduce the 2026-09-14
diagnostic and `reproduced` is `True`. The frozen cohort ids are
`[324, 335, 340, 350, 362, 375, 387, 402, 414, 427, 434, 454, 461, 466, 468, 480, 498, 508,
521, 533, 547, 554, 571, 575]`.

## Cycle 3 — `tools/exp06_bootstrap.py`

`paired_intervals(a, b, clusters, seeds, alpha=0.05, n_boot=10000, seed=0,
return_samples=False)` reproduces `sim_to_real.summarize_haa.paired` exactly: one
`np.random.default_rng(seed)` advanced through the `n_boot` query-cluster draws and then the
`n_boot` two-way draws, float64, `np.unique` label order, `np.bincount(...)[qi]` multiplicity
weights, `wq * ws` products with weighted `np.average` denominators, `np.nan` when a draw's
weights sum to zero, and NumPy's linear `nanpercentile`. Only the tail probabilities are
configurable (`[100·α/2, 100·(1−α/2)]`). It returns the point estimate, both intervals with
their sample mean/sd/NaN count, `frac_a_lower`, `n`, `n_clusters`, `n_seeds`, `alpha`,
`tails`, `n_boot`, `seed`, and optionally the raw sample arrays.
`bonferroni_alpha(alpha, m)`, `convergence_endpoints(interval_fn, seed_a=0, seed_b=1,
tol=0.10)` and `converged_interval(fn_factory, n_boot, …)` (quadruple once, then
`not_converged` with no interval) complete the module.

Tests (29): bit-exact agreement with the pinned `paired` on synthetic data at α = 0.05;
**all eleven canonical exp_02 cells reproduced with `abs diff == 0.0`** on `diff`, `lo`, `hi`,
`lo_two_way`, `hi_two_way` against `ckpt/sim2real/stats.json` (`n_boot` 10000, `boot_seed` 0),
rebuilt from the per-sample files with summarize_haa's own pooling, 10.7 s, skipped when
`ckpt/sim2real` is absent; the eleven cells confirmed to be the required rooms × metrics;
adjusted tails at α = 0.05/11 shown to be percentiles of the very same sample arrays and
strictly wider than the nominal ones; the sample summary; Bonferroni arithmetic and seven
refusals; nine `paired_intervals` degeneracy refusals (NaN, inf, mismatched lengths, empty,
mismatched cluster/seed labels, α, `n_boot`, `seed`); convergence on identical intervals
(ratio 0), shifted equal-width intervals either side of the tolerance, zero-width refusal,
five malformed-interval refusals; and the three `converged_interval` paths.

## Validation commands and outcomes

| command | outcome |
|---|---|
| `python -m pytest tests/test_exp06_eval_launch.py tests/test_provenance.py -q` (baseline, before any edit) | 78 passed |
| `pytest tests/test_exp06_eval_launch.py -q` after the nit-7 red commit | 6 failed, 46 passed |
| `pytest tests/test_exp06_eval_launch.py tests/test_provenance.py -q` after the fix | 84 passed |
| `pytest tests/test_exp06_haa_pipeline.py -k "preparation or refused_heading or second_refused"` after the nit-8 red commit | 5 failed, 3 passed |
| `pytest tests/test_exp06_haa_pipeline.py -q` after the fix | 37 passed (97.9 s) |
| `pytest tests/test_exp06_probe_align.py -q` (red) | 1 collection error (module absent) |
| `pytest tests/test_exp06_probe_align.py -q` (green) | 26 passed, 1 skipped |
| `pytest tests/test_exp06_mirror_probe.py -q` (cycle-2a red) | 1 collection error |
| `pytest tests/test_exp06_mirror_probe.py -q` (cycle-2a green) | 28 passed, 1 skipped |
| `pytest tests/test_exp06_mirror_probe.py -q` (cycle-2b red) | 18 failed, 28 passed, 2 skipped |
| `pytest tests/test_exp06_mirror_probe.py -q` (cycle-2b green) | 46 passed, 2 skipped |
| `EXP06_REAL_PROBE=1 pytest … -k published_anchors -s` | 1 passed (44.4 s) — anchors table above |
| `pytest tests/test_exp06_bootstrap.py -q` (red) | 1 collection error |
| `pytest tests/test_exp06_bootstrap.py -q -k "not canonical_exp02"` (green) | 42 passed |
| `pytest tests/test_exp06_bootstrap.py -q -k canonical_exp02` | 11 passed (10.7 s) |
| `pytest … -k "refuses_a_bad_shape or unserialisable"` (hardening red) | 2 failed |
| `python -m py_compile` on all six new/edited Python files | clean |
| `bash -n tools/exp06_haa_pipeline.sh` | clean |
| `git diff --check`, `git status --porcelain` | clean, empty |
| **`CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider`** | **823 passed, 6 skipped (13 min 5 s)** |

`tests/test_provenance.py` was green at every step. The 128 tests collected from the three
new test files are all part of that 823. The 6 skips are the GPU-only parity tests
(`test_exp06_probe_align`, `test_exp06_mirror_probe`, and three pre-existing exp_06 GPU
tests) plus the `EXP06_REAL_PROBE` slow test, which was run separately and passed.

## Discrepancies, deviations and choices made fail-closed

1. **`np.nanpercentile`, not `np.percentile`.** The prompt describes exp_02's convention as
   "`np.percentile` linear interpolation"; `summarize_haa.paired` actually calls
   `np.nanpercentile` for both schemes. Reproducing the pinned code exactly takes precedence,
   so `exp06_bootstrap` uses `np.nanpercentile`. This is what makes all eleven canonical cells
   reproduce at `abs diff 0.0`.
2. **The adjusted tails cannot be bit-equal to `100·0.05/22`.** `bonferroni_alpha(0.05, 11)`
   is `0.05/11`; `100·((0.05/11)/2)` and `100·0.05/22` differ by one ulp under IEEE division
   (0.2272727272727273 vs 0.22727272727272727), which moves the interpolated quantile by
   ~2e-18. The test therefore asserts the tails and endpoints agree to `rel 1e-12 / abs 1e-15`
   and additionally asserts exactness for α = 0.05 (`tails == [2.5, 97.5]`). The module takes
   only α, so no α-only expression can reproduce the `0.05/22` bit pattern.
3. **`convergence_endpoints` refuses zero-width intervals; `paired_compare.convergence` lets
   identical zero-width intervals pass.** The prompt and plan §7 both say zero width is
   refused, so this module raises `ValueError`. It also raises on non-finite or reversed
   intervals, where `paired_compare.convergence` returns `passed: False`. Both differences are
   documented in the module docstring; `paired_compare` itself is untouched.
4. **A failed legacy reproduction forces G1 *inconclusive*.** `g1_decision` implements exactly
   the four-condition precondition and the two pass conditions the prompt lists. The prompt
   does not say what the *record* should report when part 1 of §6.1 (the validity check of the
   probe's own numerics) fails, so `build_record` overrides the outcome to `inconclusive` with
   the deviations as reasons — a verdict resting on unvalidated numerics is withheld rather
   than published. `--legacy-only` likewise records `inconclusive` and `stats: null`, never a
   pass. This is an addition beyond the literal spec; flag it if the Planner wants the two
   kept independent.
5. **Injection hooks on `legacy_reproduction` / `full_gate`.** Both take `model_factory`,
   `checkpoints`, `root`, `room`, `max_len`, `num_shot`, `batch` and (for the legacy part)
   `cohort_size` / `verify_frozen_cohort`, so the CPU tests can drive them with tiny models on
   a synthetic cache. The defaults are exactly the plan's: the two exp_01 checkpoints,
   `build_xrir_exp06`, the registered HAA root, hallway, 24-microphone frozen cohort with the
   frozen-id check on. The CLI never overrides `model_factory` or `verify_frozen_cohort`.
6. **Three commits exceed the 200-changed-line guidance**, all test-only: `792a16d` (225),
   `d6f19f6` (287) and `402a42b` (229). The implementation commits were split to stay under
   (188 / 152 / 82 / 109 / 153). This repeats the accounting problem the round-2b close review
   raised as nit 6; no history was rewritten.
7. **Two pre-existing exp_06 test files were edited** (`tests/test_exp06_eval_launch.py`,
   `tests/test_exp06_haa_pipeline.py`) to carry the cycle-0 regressions the review asked for.
   The HARD RULE's protected list does not include them, and the nits explicitly call for new
   regressions; no other earlier-round file was touched.
   `tests/test_exp06_haa_pipeline.py` now imports `json` at module level while keeping the
   pre-existing mid-file `import json` of its end-to-end section (harmless duplicate, left
   alone to keep the diff minimal).
8. **`tools/exp06_mirror_probe.py` imports two names that read as private**:
   `eval_xRIR_backbone.load_model_state` (the exp_03-pinned loader `finetune_haa` already
   composes — import-only, as the HARD RULE requires) and
   `tools.exp06_heading._closure_digest` (aliased `closure_digest`), to avoid a second copy of
   the closure-digest expression. It also calls `HAADataset._pick_refs`, the deterministic
   draw the dataset's own `__getitem__` uses; every drawn block is verified against
   `data['rirs'][ref_ids]` before its side labels are believed. Say the word if the Planner
   prefers a local copy of the digest helper.
9. **A small post-cycle hardening pass** (`cb3c770` / `895f512`, red first): `spectral_c50`
   refuses an empty batch instead of raising `RuntimeError` from `.min()`; `mirror_stats`
   takes the shot count from `dataset.num_shot` rather than a loop variable read after the
   loop; `exp06_eval_launch._persisted` treats an unserialisable returned record as
   `completion_mismatch` instead of letting `json.dumps` raise `TypeError`.
10. **Not verified here:** both CUDA parity tests (`shift_and_align_device` bit-exact against
    the pinned method; `composed_forward` against `model.forward`) are `skipif`-guarded and
    were skipped — this session is CPU-only by instruction. They should be run once on a GPU
    before G1 is executed for real, since the full-cohort gate runs on CUDA when available.
