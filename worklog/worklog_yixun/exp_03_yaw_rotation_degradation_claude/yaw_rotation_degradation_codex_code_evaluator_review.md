**Reviewer:** OpenAI Codex (codex-cli 0.144.1, `codex exec`, read-only sandbox, model `gpt-5.6-sol`, reasoning effort `ultra`; session 01a077be-dff9-7af1-891c-e481e9e09abe) · **Date:** 2026-09-06 13:40 (−04:00) · **Round:** Coder round 3 (evaluator + summarizer) · **Briefing:** SOP, plan v2, notebook incl. the round-3 Coder report, params draft, rounds-1/2 reviews, exp_01 corrected results; prompt `review_prompts/code_round3_prompt.md`.

> Planner answers (pre-registered for the fix round): (1) H2 is "supported" only if the adjusted DiD upper bound is < 0 at EVERY cell where primary H1 passes; "partially supported" (cells listed) if at some; "not supported" if at none; "not evaluable" if H1 has no passing cell. (2) The exp_01 parity tolerance and the manifest-seed noise bands are encoded in the summarizer's `k0-gate` mode: two extra k=0-only runs of the control checkpoint on manifest seeds 1 and 2 give the reference-draw spread; the gate passes if |mean_k0(seed 0) − mean_exp01(epoch 12)| ≤ 2 × max spread + 1e-6 for EDT/C50/T60/loss, for each model with an exp_01 counterpart (cyl, control), and if the released checkpoint's k=0 means lie within the same band of its exp_01 baseline reproduction. (3) Partial summaries exist only in an explicit `exploratory` mode with verdicts and the canonical hash suppressed; `k0-gate` and `full` are the only canonical modes.

**Reviewer:** OpenAI Codex (GPT-5 API workspace agent; exact serving minor version not exposed, read-only review) · **Date:** 2026-09-06

## Summary verdict

**Approve with changes; keep round 3 open and do not launch the parity gate or sweep.**

The evaluator’s core science is correct: P uses cached k=0 alignment, E performs an ordinary forward on the same rotated tensors, k=0 is bit-exact, and its acoustic computation is shared. Magnitude reconstruction, seeded Griffin–Lim, 8,000-sample window, loss slices, delay-flip direction, and token roll-back match the plan/reference code. The default family is correctly 18 and \(r_k\), DiD, cluster bootstrap, and k=0 absolute bootstrap are mathematically sound.

Full suite: **121 passed, 20 warnings in 35.31 s**. Trained probes found k=0 P/E and angle order bit-exact for all three checkpoints. No new blocker was found in closed rounds 1–2.

## Blocking findings

1. [tests/test_eval_yaw_rotation.py:347](/home/yixunhu/codespace/xRIR_code/tests/test_eval_yaw_rotation.py:347), [tests/test_eval_yaw_rotation.py:388](/home/yixunhu/codespace/xRIR_code/tests/test_eval_yaw_rotation.py:388) — approved T14 is neither implemented nor satisfied.

   The test uses random initialization and batch 8, then weakens “exact acoustic / spectral atol \(10^{-6}\)” to relative `allclose`. The suite itself reports non-exact C50. Trained batch-16 versus batch-1 probes failed for all checkpoints: maximum C50 differences were `3.82e-5` control, `1.08e-4` cylindrical, and `4.41e-5` released; consistency exceeded \(10^{-6}\) for control and released.

   **Fix:** parameterize T14 over all three trained checkpoints at batch 16 versus 1. Either canonicalize compute shape—padding every short batch, including the final 6,337th query, to 16 before slicing—or formally amend the preregistered tolerance before launch and record this numerical noise floor. Do not claim exact invariance.

2. [tools/summarize_yaw.py:218](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:218) — cylindrical TOST uses the four-way DiD validity mask.

   Cylindrical-valid samples are dropped solely because the control is invalid. A discriminating probe with true cylindrical \(r=+5\%\) returned `r_cyl=0` and `equivalent=True`.

   **Fix:** retain the four-way mask for DiD, but use an independent `valid_mask(cyl0, cylk)` and corresponding rooms for cylindrical TOST; report its separate sample count.

3. [tools/summarize_yaw.py:603](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:603) — H2 has no verdict.

   Rows are printed, but the script never applies `d < 0 and hi < 0`, links results to the primary H1-passing metric/angle cells, or emits supported/not-supported wording.

   **Fix:** implement and serialize per-cell and aggregate H2 decisions, including a non-evaluable state when H1 has no passing cells.

4. [tools/summarize_yaw.py:481](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:481) — parity-only runs cannot be summarized, while partial runs silently redefine the family.

   A k=0-only gate yields family size zero and crashes in `bonferroni_alpha`. Conversely, the tests bless a partial family of four instead of the preregistered 18.

   **Fix:** fix confirmatory family size at 18 and validate the exact grid. Add a k0-only mode that produces the paired exp_01 table while marking H1/H2 not evaluated.

5. [tools/summarize_yaw.py:453](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:453), [tools/summarize_yaw.py:459](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:459) — no production acceptance or pairing gate exists.

   A 300-query/five-room smoke run can receive a definitive H1 verdict and canonical hash. Cross-run checks omit `gl_seed`, TF32 policy, indices, grids, array lengths, spectral finiteness, roles/checkpoints, and delay-audit agreement, yet the k=0 table claims identical phases. Missing released data is treated as a failed replication rather than “not evaluated.”

   **Fix:** add explicit `full`, `k0-gate`, and `exploratory` modes. Full mode must enforce 6,337 unique queries, 17 rooms, exact grids, three roles, `gl_seed=0`, TF32 off, aligned indices and arrays, finite spectral metrics, and complete audits. Emit `valid_for_confirmatory` with reasons and suppress verdicts for partial data.

6. [tools/summarize_yaw.py:652](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:652) — convergence validation covers only H1 query intervals.

   H2 DiD and adjusted TOST can remain unstable while the command exits successfully. Failed H1 convergence is detected only after nominal canonical files are written.

   **Fix:** check all decision-driving H1, H2, and TOST bounds, retain both seeds’ endpoints, serialize an explicit convergence status, and publish canonical-success artifacts only after it passes.

## Non-blocking

- [eval_yaw_rotation.py:384](/home/yixunhu/codespace/xRIR_code/eval_yaw_rotation.py:384) correctly controls both relevant PyTorch 2.0.1 TF32 routes; disabling them gives matmul precision `highest`.
- [eval_yaw_rotation.py:543](/home/yixunhu/codespace/xRIR_code/eval_yaw_rotation.py:543) serializes invalid values as bare `NaN`, which is not standards-compliant JSON. Convert them to `null` and use `allow_nan=False`; likewise prevent `Infinity` in summary JSON.
- Summary-text hashing is correct when both outputs are supplied. Require `--json` and `--summary` together and reject identical paths.
- The decomposition measures receiver-view tokens/pooling and only the query-source coordinate embedding—not all reference-coordinate features. Label that scope explicitly.
- The 10% absolute equivalents and clustered TOST verdicts are absent from the printed tables.
- There is no resumability or atomic overwrite protection. At the measured rate, a model takes about 2.35 hours, while default progress updates are roughly 18 minutes apart.

## Test gaps

- Permanent trained-checkpoint T14/T15 coverage for all three models and a spy proving only one k=0 acoustic inversion.
- Asymmetric invalidity for TOST; explicit H2 verdict combinations.
- K0-only mode, fixed family 18, and full three-model/17-room schema.
- Mismatched seeds/precision/grids, malformed lengths, spectral NaNs, convergence failure, strict JSON, interruption/resume, and atomic-output behavior.

## Questions

1. Must H2 pass at every primary-H1 passing metric/angle cell, or is one corresponding cell sufficient?
2. Where will the two extra-manifest-seed noise bands and numerical exp_01 parity tolerances be encoded?
3. Should partial summaries remain supported explicitly, or should this CLI produce only gate/full canonical results?
