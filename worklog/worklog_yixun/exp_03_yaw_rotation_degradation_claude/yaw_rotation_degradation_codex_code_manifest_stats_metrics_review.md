**Reviewer:** OpenAI Codex (codex-cli 0.144.1, `codex exec`, read-only sandbox, model `gpt-5.6-sol`, reasoning effort `ultra`; session 01a0777d-b229-7e83-b8f5-8dfad783fde1) · **Date:** 2026-09-06 12:21 (−04:00) · **Round:** Coder round 2 (reference manifest, paired stats, per-sample metrics; tests 9–13) · **Briefing:** SOP, plan v2, notebook incl. the round-2 Coder report, round-1 reviews, exp_01 corrected results; prompt `review_prompts/code_round2_prompt.md`.

> Planner answers: (1) yes — round 3's evaluator will take `--manifest-hash` and refuse to run unless the loaded manifest hashes to that recorded digest; (2) `clusters=` support is added to `equivalence_tost` in the fix round so the room-cluster robustness analysis covers the equivalence rows; (3) the commit range in the notebook is corrected to the SHAs actually on `main` (see `commits_yaw_rotation_degradation.md`).

**Reviewer:** OpenAI Codex (GPT-5 API workspace agent; exact serving minor version not exposed, read-only review) · **Date:** 2026-09-06

## Summary verdict

**Approve with changes; round 2 remains open.**

The manifest candidate logic and `ManifestDataset` match `xRIR_Dataset`; real-data spot checks, including truncation, were bit-identical. Per-sample losses and seeded Griffin–Lim match the baseline recipes. The core bootstrap mathematics—ratio inside resamples, cluster multiplicities, shared DiD draws, percentile levels, and \(1-2\alpha\) TOST—is correct.

Python 3.8.20 validation: **69 passed, 8 warnings in 14.22 s**. Real manifest: 6,337 queries, 520 replacement cases, hash `3e0e7f...ee082`.

## Blocking findings

1. [tools/per_sample_metrics.py:111](/home/yixunhu/codespace/xRIR_code/tools/per_sample_metrics.py:111) — `acoustic_metrics` does not enforce exp_01’s first-8,000-sample window; it delegates slicing to the future evaluator. Full prediction/target lengths differ and tail content can change EDT/C50/T60, biasing both the k=0 gate and sweep.

   **Fix:** slice both IRs inside this helper and add a >8,000-sample poisoned-tail regression.

2. [tools/paired_stats.py:56](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:56) — `invalid_fraction(ek)` counts every angle-invalid sample, including samples already invalid at k=0, and returns no count. This cannot implement the preregistered “samples that become invalid” indicator; [valid_mask](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:33) also does not report the specified counts.

   **Fix:** add a paired validity summary reporting baseline-invalid, newly-invalid, recovered, paired-valid counts, and an explicitly denominated newly-invalid fraction.

3. [tools/paired_stats.py:190](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:190) — the approved test/API was `bonferroni_alpha(18) == 0.05/18`, but the implementation requires `(alpha, m)` and [the test](/home/yixunhu/codespace/xRIR_code/tests/test_paired_stats.py:164) silently changes the specification.

   **Fix:** use `bonferroni_alpha(m, alpha=0.05)` and test the registered one-argument call.

4. [tests/test_paired_stats.py:68](/home/yixunhu/codespace/xRIR_code/tests/test_paired_stats.py:68) and [tests/test_paired_stats.py:120](/home/yixunhu/codespace/xRIR_code/tests/test_paired_stats.py:120) — mandatory test 10 does not discriminate the registered estimators. Proportional data cannot distinguish ratio-of-means from mean per-sample ratio; singleton/equal-size clusters cannot distinguish retained-query weighting from equal-room weighting.

   **Fix:** add explicit seeded bootstrap oracles using heterogeneous denominators and unequal cluster sizes. For example, `[1,100]→[2,100]` must give \(1/101\), not \(0.5\).

5. [tools/paired_stats.py:85](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:85), [tools/paired_stats.py:117](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:117), [tools/paired_stats.py:247](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:247) — validation permits fractional `n_boot`/`m`, TOST permits `alpha >= 0.5`, and zero-denominator resamples can silently produce `NaN` bounds despite finite inputs.

   **Fix:** require genuine positive integers, require TOST `0 < alpha < 0.5` and finite margin, and reject any nonfinite bootstrap draw rather than using `nanpercentile`.

## Non-blocking

- [tools/reference_manifest.py:120](/home/yixunhu/codespace/xRIR_code/tools/reference_manifest.py:120) — reference choices are query-stable, but serialized entry order, `index`, and therefore [the hash](/home/yixunhu/codespace/xRIR_code/tools/reference_manifest.py:135) inherit `dataset.file_list` order. A recopied cache can produce a different hash and be rejected. This fails closed and does not bias the current pinned-cache sweep, but contradicts the directory-order-independence claim. Canonicalize entries by relative query path before freezing the manifest, or document native-order pinning.
- [tools/reference_manifest.py:178](/home/yixunhu/codespace/xRIR_code/tools/reference_manifest.py:178) validates with optimization-removable assertions and does not validate per-entry reference count, receiver, candidate membership, or relative/root-contained paths. Prefer explicit structural validation.
- [tools/per_sample_metrics.py:151](/home/yixunhu/codespace/xRIR_code/tools/per_sample_metrics.py:151) can return infinite T60 when `gt_t60` is a NumPy zero, despite the NaN contract. Normalize nonfinite results to NaN.
- With `alpha=0.05/18`, 10,000 bootstrap samples leave only about 14 draws in each ordinary-CI tail. Use substantially more resamples or demonstrate tail/seed convergence before the confirmatory gate.

## Test gaps

- Add a production-dataset forced-choice oracle, synthetic truncation case, permuted-file-list/hash-policy test, and the specified worker count `{0,4}`.
- Compare seeded Griffin–Lim directly, bit-for-bit, with `manual_seed; eval_unseen.griffin_lim`; use a spy proving `want_t60=False` never calls RT60.
- Add exact percentile-level oracles, nonzero correlated DiD and clustered-DiD oracles.
- Set `rel=0`/`rtol=0` in the claimed `1e-7` loss comparisons.

## Questions

1. Will round 3 compare the manifest against a literal externally recorded digest before evaluation, rather than hashing and trusting the supplied file?
2. Is the secondary room-cluster robustness analysis intended to cover TOST? `equivalence_tost` currently has no `clusters=` support.
3. The reported final commit `34b0f16` does not exist in this checkout; the requested diff contains 15 commits ending at `e0d4150`. Should the worklog commit range be corrected?
