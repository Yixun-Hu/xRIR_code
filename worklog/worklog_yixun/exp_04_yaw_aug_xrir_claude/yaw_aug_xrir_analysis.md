# Analysis — yaw_aug_xrir (exp_04)

Planner analysis (Claude Fable 5.1), 2026-09-15. Every number below is copied from the canonical producer JSON under `ckpt/yaw_aug/results/` (bound in `ckpt/yaw_aug/binding_report_20260915T084205518027Z.json`, `check_record.py` exit 0); the tables are in `yaw_aug_xrir_results.md` and the page `yaw_aug_xrir_01_results.html`. Pre-registered rules: plan §3; amendments A1–A10 in plan §11.

## 1. Question and answer

Does training the SimpleViT xRIR with per-sample active yaw augmentation (offsets d ~ U{0..511} on the whole receiver-centred scene, audio untouched; the FLAC exp_17 recipe ported) keep accuracy at the canonical heading (H1, non-inferiority +3 %), reduce the yaw-rotation degradation measured in exp_03 (H2, C50 primary at ±22.5°/±45°), and make the model near-invariant to yaw (TOST ±2 %)?

- **H1 — "non-inferior on EDT only" at K = 8 and at K = 1.** EDT is non-inferior (K = 8: ρ = −1.33 %, one-sided upper −0.22 %; superior at query level, the 17-room interval [−3.11, +0.55] % includes 0; K = 1: +1.83 %, upper +2.50 %). C50 is **not**: the augmented model is 4.5 % worse than the same-budget control at K = 8 (ρ = +4.47 %, upper +5.53 %, companion [+3.39, +5.53] %, rooms [+0.85, +7.74] %) and 4.2 % worse at K = 1 (upper +4.62 %). T60 (descriptive) is 2.6 % better at K = 8, unchanged at K = 1.
- **H2 — "partially supported".** D_k = r_k(aug) − r_k(control) for C50 is negative at all four diagonal headings (−1.02, −2.26, −1.50, −0.49 percentage points at +22.5°, +45°, −45°, −22.5°) but only the +45° cell's adjusted upper bound is below zero (−0.62 pp; the others +0.41, +0.08, +0.85). The supportive EDT cells all fail (uppers +0.35 to +1.76 pp); T60 (descriptive) is negative with bounds below zero at +22.5° and +45°.
- **TOST — equivalent only at the axis-aligned headings.** For C50 and EDT the augmented model is within ±2 % at 90°, 180° and −90° (and EDT at −22.5°); at ±22.5°/±45° equivalence is not established (C50 r_k = +2.0 to +3.1 %, upper bounds +3.6 to +5.0 %).

Reading: yaw augmentation trades canonical-heading C50 accuracy (about −4.5 % relative) for a modest, mostly non-significant reduction of the diagonal-heading C50 penalty (from the control's ≈ 4–5 % in exp_03 to ≈ 2–3 %), a small EDT/T60 gain, and no invariance at the diagonals. It does not deliver the "free" robustness the FLAC exp_17 result suggested; on this model the augmentation behaves as a regulariser that costs the clarity metric.

## 2. Table (five evaluation seeds, mean ± SD; unseen split, 6 337 queries, epoch 12, condition P, k = 0)

| Model | K | T60 (%) | C50 (dB) | EDT (ms) |
|---|---|---|---|---|
| SimpleViT (control) | 8 | 9.575 ± 0.009 | 1.238 ± 0.005 | 47.02 ± 0.19 |
| CylindricalViT | 8 | 9.443 ± 0.014 | 1.231 ± 0.006 | 45.06 ± 0.25 |
| YawAug-xRIR | 8 | 9.322 ± 0.006 | 1.294 ± 0.006 | 46.39 ± 0.15 |
| SimpleViT (control) | 1 | 13.38 ± 0.10 | 1.890 ± 0.013 | 68.28 ± 0.61 |
| CylindricalViT | 1 | 13.58 ± 0.09 | 1.942 ± 0.014 | 69.76 ± 0.79 |
| YawAug-xRIR | 1 | 13.39 ± 0.08 | 1.969 ± 0.015 | 69.52 ± 0.73 |

(`TABLE_V1.json`; the living table `worklog/worklog_yixun/model_comparison.md` is rendered from it.) The seed SDs are an order of magnitude below the between-model differences on C50 and EDT, so the paired comparisons above are not seed noise.

## 3. What the augmented model does across headings (descriptive, seed 42, `GRID_SEED42.json`)

C50 r_k at +22.5°/+45°/−45°/−22.5° = +2.8 / +3.3 / +3.2 / +2.1 %; EDT r_k within ±0.7 % everywhere; both ≈ 0 at 90°, 180°, −90°. The shape is the same as exp_03's control curve (peaks at diagonal headings, zero at multiples of 90°), just lower for C50. The epoch-9 checkpoint (41 679 updates, the FLAC "40k" convention; `EPOCH9_K8.json`): EDT 46.89 ± 0.16 ms, C50 1.261 ± 0.006 dB, T60 9.41 ± 0.01 % — C50 is better at epoch 9 than at epoch 12 (1.294), i.e. the C50 deficit grows in the last low-learning-rate epochs, while the control improves there.

## 4. Why the augmentation costs C50 (interpretation, not tested)

The diagonal-heading degradation in exp_03 was attributed to distribution shift from axis-aligned training rooms acting through the learned position pool and the absolute-coordinate embeddings. Augmenting with uniform yaw makes the training distribution heading-invariant, but the test rooms are still axis-aligned: the model loses the prior that walls are at 0°/90° and pays for it exactly on the metric most sensitive to early reflections (C50), while the decay-shaped metrics (EDT, T60) are unaffected or slightly better. Half the augmentation budget is spent on headings that never occur at test time. A milder or structured augmentation (small angles, or 90° multiples only) is the obvious follow-up; it is not part of this experiment.

## 5. Validity and limitations

- All decision-driving intervals passed the two-seed convergence gate (maximum endpoint movement / width ≤ 0.021 against the 0.10 tolerance); no query was excluded in any cell (all 6 337 valid for every seed and arm).
- Single training realisation per arm (plan §2): the control is exp_01's run, the augmented model is one run; run-to-run training noise is not estimated. The C50 effect (4.5 %, upper bound 5.5 %) is far larger than exp_01/exp_03's between-run differences, the EDT effect (1.3 %) is within their range.
- Room-cluster intervals (17 rooms) include zero for EDT and for every H2 cell; the C50 penalty is the only effect whose room interval excludes zero.
- Protocol deviations recorded in the notebook: epoch 3 and the S-tier co-tenant slowed training by ≤ 4 min/epoch (no gate affected); three evaluation launches were refused by the dirty-tree gate during a Coder round and re-run identically; the grid run used exp_03's 10-angle acoustic grid after a queue-script correction; the L-tier training of exp_05 shared the host during evaluation (evaluations are not timing-sensitive).
- The reported checkpoint is epoch 12 (plan §3); the best test-loss epoch was 5 (0.01599 vs 0.01604 at epoch 12).

## 6. Provenance

Training: attempt `ckpt/xRIR_simple_yawaug_8_shot/attempt_20260913T102051` (→ `final`), 27.43 h, launch commit `f19b9b6`, probe receipt `_probe_20260913T101537_probe.json` (overhead 0.78 %), alignment audit 0 / 12 800. Evaluations: 46 runs under `ckpt/yaw_aug/eval/` with immutable manifests and completions (evaluator closure `0b05245c…`, writer `15f9c984…`). Approval record `yaw_aug_xrir_results_assets/approved_digests.json` (commit `fa219ef`). Producers at `daa7eef`/`cab6f42`. Reviews: nine Fable 5.1 code reviews (`yaw_aug_xrir_fable_code_*_review.md`), three Codex plan reviews.
