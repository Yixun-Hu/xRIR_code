**Reviewer:** OpenAI Codex (codex-cli 0.144.1, `codex exec`, read-only sandbox, model `gpt-5.6-sol`, reasoning effort `ultra` per `~/.codex/config.toml`; session 01a074b2-a0e5-7da1-a03c-cf3b529d4ac8) · **Date:** 2026-09-05 23:25 (−04:00) · **Round:** plan v1 · **Briefing:** SOP, plan, query, notebook, exp_01 results/analysis, exp_02 plan, model/dataset/eval code (prompt saved as `review_prompts/plan_v1_prompt.md`)

> The reviewer self-identified its serving model generically ("GPT-5, API workspace agent"); the CLI banner reported `model: gpt-5.6-sol`, `reasoning effort: ultra`. Planner responses to each finding are in `plan_yaw_rotation_degradation.md` §11 (v2 changelog).

# Plan review — exp_03 `yaw_rotation_degradation`

**Reviewer:** OpenAI Codex (GPT-5, API workspace agent; exact serving minor version not exposed; read-only sandbox) · **Date:** 2026-09-05

## Summary verdict

**Approve with changes.** The core yaw transform is correct for an active positive yaw: panorama columns increase in \(\theta\), with \(x=d\cos\phi\cos\theta,\ y=d\cos\phi\sin\theta\); therefore a positive \(k\)-column roll followed by \(R_z(+2\pi k/W)\) maps the old ray at \(c-k\) onto the ray for column \(c\) ([dataset conversion](/home/yixunhu/codespace/xRIR_code/treble_multi_room_dataset/treble_xRIR_dataset.py:43)). Query and reference locations are receiver-relative, and references retain the query receiver ID ([query coordinates](/home/yixunhu/codespace/xRIR_code/treble_multi_room_dataset/treble_xRIR_dataset.py:138), [reference selection](/home/yixunhu/codespace/xRIR_code/treble_multi_room_dataset/treble_xRIR_dataset.py:170)).

The model analysis is also mostly right. CylindricalViT’s gauge rotation cancels the channel rotation and leaves an azimuthal token shift at patch-aligned angles ([gauge alignment](/home/yixunhu/codespace/xRIR_code/model/cylindrical_vit.py:136), [forward](/home/yixunhu/codespace/xRIR_code/model/cylindrical_vit.py:172)). Full-model invariance is then broken by the learned token-position pool and raw componentwise XYZ embedding ([pooling](/home/yixunhu/codespace/xRIR_code/model/xRIR.py:189), [coordinate features](/home/yixunhu/codespace/xRIR_code/model/xRIR.py:210)).

The blockers are in isolating that transform, evaluation parity, and confirmatory statistics.

## Blocking findings

1. **Plan §§2, 4, 6 — `shift_and_align` is not numerically invariant.**

   The claim of exact invariance in [plan §2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:11) is true algebraically, but the implementation recomputes float32 norms and rounds delays to integer samples ([xRIR.py:254–264](/home/yixunhu/codespace/xRIR_code/model/xRIR.py:254)). A real held-out candidate pair, `Bedrooms_idx_18` S008/R0012 versus S003/R0012, gives an unrotated delay value of exactly 80.5, rounded to 80, but 80.500015 after the proposed \(k=4\) float32 rotation, rounded to 81. Thus rotation can change the reference waveform by one sample independently of either ViT or coordinate embedding.

   **Fix:** separate geometry coordinates from alignment coordinates. Reuse the \(k=0\) aligned reference audio, or add a backward-compatible forward input allowing alignment to use the original coordinates while the geometry branches receive rotated coordinates. Audit all real source/reference pairs and angles; a random synthetic assertion is insufficient.

2. **Plan §§7–9 — exp_01 parity and cross-model reference pairing are not achievable as written.**

   Exp_01 evaluates with batch size 1 ([evaluator loader](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:95)); exp_03 proposes batch 16 ([plan cost/run](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:81)). References are drawn by worker-local `np.random.choice` on every dataset call ([dataset sampling](/home/yixunhu/codespace/xRIR_code/treble_multi_room_dataset/treble_xRIR_dataset.py:183)), so seed and worker count do not preserve reference identities when batching or worker assignment changes. Existing artifacts record target index/path but not reference IDs ([saved fields](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:107)), and `compare_eval.py` checks only indices ([comparison check](/home/yixunhu/codespace/xRIR_code/tools/compare_eval.py:34)).

   Batch-16 per-sample loss also needs a new reduction: the reused losses currently reduce across the entire batch ([STFT loss](/home/yixunhu/codespace/xRIR_code/utils/spec_utils.py:423), [decay loss](/home/yixunhu/codespace/xRIR_code/utils/spec_utils.py:561)).

   **Fix:** persist a per-query reference manifest and verify its IDs/hashes across models; use a dedicated loader generator independent of model initialization. Validate every per-sample batch-16 metric against materialized batch-1 inputs. Run the complete \(k=0\) gate before the rotated sweep, as required by the SOP—not inside the full run ([SOP parity order](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:79)).

3. **Plan §§4, 8, 9 — Griffin–Lim randomness is not paired across angles.**

   The inherited wrapper creates Griffin–Lim without specifying phase initialization ([eval_unseen.py:11](/home/yixunhu/codespace/xRIR_code/eval_unseen.py:11)); in the pinned environment its default is `rand_init=True`. The evaluator seeds Torch once and then reconstructs sequentially ([seeding](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:80), [reconstruction](/home/yixunhu/codespace/xRIR_code/eval_xRIR_backbone.py:119)). Looping over angles therefore assigns different random phases to \(k=0\) and \(k\ne0\), mixing reconstruction noise into EDT/C50/T60 degradation.

   **Fix:** use identical phase initialization per sample across all angles and models, or a validated deterministic initialization. Require angle-order invariance and identical-magnitude/identical-metric tests. Treat legacy exp_01 numerical reproduction separately if its random stream cannot be preserved.

4. **Plan §§3–4 — the registered rule does not statistically establish “substantial” degradation.**

   A point estimate ≥10% with a CI merely excluding zero establishes positive degradation, not degradation of at least 10% ([plan threshold](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:22)). “Some angle” across five nonzero acoustic angles, two metrics, and two SimpleViT models also creates an undeclared multiple-testing family. The reused helper bootstraps only an absolute mean difference ([compare_eval.py:57](/home/yixunhu/codespace/xRIR_code/tools/compare_eval.py:57)).

   The 6,337 queries come from only 17 rooms ([exp_01 results](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results.md:3)); query-IID resampling cannot support population-of-unseen-rooms inference.

   **Fix:** define \(r_k=(\bar e_k-\bar e_0)/\bar e_0\), recompute it inside each paired bootstrap, and require its multiplicity-adjusted lower bound to exceed 10% if the claim is confirmatory. Use paired room-cluster/hierarchical resampling, or explicitly restrict inference to this fixed split. Pre-register invalid-metric masks; the current helper silently drops nonfinite pairs ([filtering](/home/yixunhu/codespace/xRIR_code/tools/compare_eval.py:52)).

5. **Plan §§3–4 — H1/H2 and the angle grid are not fully decidable.**

   H1 does not say whether both SimpleViT checkpoints must pass. H2 does not define “less” or “nearly unchanged,” nor identify which SimpleViT is primary. Only the exp_01 control is training-matched to CylindricalViT. Cross-model robustness must be a degradation difference-in-differences, not merely cylindrical error minus simple error.

   The spectral grid omits \(k=416=-67.5^\circ\), the only missing counterpart among its signed pairs ([angle set](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:28)); acoustic metrics cover only positive yaw through \(180^\circ\) ([full metrics](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:30)), although the model has no reflection symmetry guaranteeing \(+\Delta\) and \(-\Delta\) behave alike.

   **Fix:** make the same-budget SimpleViT the primary H2 comparator and the released model a replication; define disagreement outcomes and an equivalence margin for “nearly unchanged.” Add representative signed acoustic pairs or narrow the claims to the listed directions.

6. **Plan §§6–9 — the evaluator lacks mandatory TDD and review gates.**

   The plan explicitly assigns no unit test to the experiment’s central evaluator ([plan §6](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/plan_yaw_rotation_degradation.md:63)), contrary to mandatory TDD for new functions ([SOP](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:65)).

   **Fix:** factor the evaluator into testable functions for batch reduction, sample/result indexing, reference manifests, phase control, and invalid metrics. Add the SOP’s per-round reviews and final integrative review before launch.

## Non-blocking findings

- Call the convention “active \(+\Delta\) yaw”; a passive camera-heading change uses the inverse sign. Do not reuse the dataset’s nonzero rotation branches, which rotate the x–z plane about y and conflict with the panorama’s z-vertical convention ([dataset helper](/home/yixunhu/codespace/xRIR_code/treble_multi_room_dataset/treble_xRIR_dataset.py:62)).
- SimpleViT sensitivity is not caused only by absolute positional codes. Its unconstrained flattened XYZ patch projection also fails to commute with x/y channel rotation ([simple_vit.py:127](/home/yixunhu/codespace/xRIR_code/model/simple_vit.py:127)).
- The unchanged-RIR premise is reasonable for this single-channel setting; the official dataset describes omnidirectional sources and monaural receivers ([AcousticRooms README](https://github.com/facebookresearch/AcousticRooms)).
- The cited ≈2% adjacent-epoch movement is checkpoint drift, not fixed-evaluation noise ([exp_01 analysis](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_analysis.md:19). Give the thresholds their absolute equivalents and present them as practical margins.
- Three approximately two-hour jobs imply about 6 GPU-hours and closer to 3.5–4 wall-clock hours on two GPUs, before the separate parity gate; concurrent CPU Griffin–Lim may add contention.
- Conclusions must be scoped to these epoch-12, single-seed checkpoints; exp_01 explicitly leaves seed variance unknown ([analysis limitation](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_analysis.md:17)).

## Missing tests/controls

- Direct oracle: `Rz(Δ)·roll(convert(depth),k) == convert(roll(depth,k))`, including z and seam columns.
- Covariance of all three actual model views: source, receiver, and reference.
- Cylindrical tokens reshaped to `[B,16,16,C]` and rolled only along azimuth; a flat 256-token roll crosses elevation rows.
- Full-model \(k=W\) versus \(k=0\) equality with cached alignment and common Griffin–Lim phase.
- Reference-ID/hash equality across models and repeated runs.
- Batch-1 versus batch-16 equality for every per-sample metric.
- Threshold-boundary, simultaneous-inference, clustered-bootstrap, and invalid-metric tests.
- A trained `lin_proj_0` azimuth-shift/commutator audit, not only token-level equivariance.

## Questions for the Planner

1. Must evidence show the true degradation exceeds 10%, or is 10% only a point-estimate label?
2. Is inference about this fixed 17-room split or a population of unseen rooms?
3. What is H1’s verdict if the two SimpleViT checkpoints disagree?
4. How will original-coordinate alignment, reference identities, and Griffin–Lim phase draws be held fixed?
5. Why omit negative-yaw acoustic controls and \(k=416\)?

No files were modified.
