# Analysis — yaw_aug_xrir (exp_04)

Planner analysis (Claude Fable 5.1), 2026-09-15. Every number below is copied from the canonical producer JSON under `ckpt/yaw_aug/results/` (bound in `ckpt/yaw_aug/binding_report_20260915T084205518027Z.json`, `check_record.py` exit 0); the tables are in `yaw_aug_xrir_results.md` and the page `yaw_aug_xrir_01_results.html`. Pre-registered rules: plan §3; amendments A1–A10 in plan §11.

## 1. Question and answer

Does training the SimpleViT xRIR with per-sample active yaw augmentation (offsets d ~ U{0..511} on the whole receiver-centred scene, audio untouched; the FLAC exp_17 recipe ported) keep accuracy at the canonical heading (H1, non-inferiority +3 %), reduce the yaw-rotation degradation measured in exp_03 (H2, C50 primary at ±22.5°/±45°), and make the model near-invariant to yaw (TOST ±2 %)?

- **H1 — "non-inferior on EDT only" at K = 8 and at K = 1.** EDT is non-inferior (K = 8: ρ = −1.33 %, one-sided upper −0.22 %; superior at query level, the 17-room interval [−3.11, +0.55] % includes 0; K = 1: +1.83 %, upper +2.50 %). C50 is **not**: the augmented model is 4.5 % worse than the same-budget control at K = 8 (ρ = +4.47 %, upper +5.53 %, companion [+3.39, +5.53] %, rooms [+0.85, +7.74] %) and 4.2 % worse at K = 1 (upper +4.62 %). T60 (descriptive) is 2.6 % better at K = 8, unchanged at K = 1.
- **H2 — "partially supported".** D_k = r_k(aug) − r_k(control) for C50 is negative at all four diagonal headings (−1.02, −2.26, −1.50, −0.49 percentage points at +22.5°, +45°, −45°, −22.5°) but only the +45° cell's adjusted upper bound is below zero (−0.62 pp; the others +0.41, +0.08, +0.85). The supportive EDT cells all fail (uppers +0.35 to +1.76 pp); T60 (descriptive) is negative with bounds below zero at +22.5° and +45°.
- **TOST — equivalent only at the axis-aligned headings.** For C50 and EDT the augmented model is within ±2 % at 90°, 180° and −90° (and EDT at −22.5°); at ±22.5°/±45° equivalence is not established (C50 r_k = +2.0 to +3.1 %, upper bounds +3.6 to +5.0 %).

Reading: yaw augmentation trades canonical-heading C50 accuracy (about −4.5 % relative) for a modest, mostly non-significant reduction of the diagonal-heading C50 penalty (the paired five-seed control degrades by 3.70 / 5.31 / 4.57 / 2.52 % at +22.5° / +45° / −45° / −22.5° in `H2_K8.json`; the augmented model by ≈ 2–3 %), a small EDT/T60 gain, and no invariance at the diagonals. It does not deliver the "free" robustness the FLAC exp_17 result suggested; on this model the augmentation behaves as a regulariser that costs the clarity metric.

## 2. Table (five evaluation seeds, mean ± SD; unseen split, 6 337 queries, epoch 12, condition P, k = 0)

| Model | K | T60 (%) | C50 (dB) | EDT (ms) |
|---|---|---|---|---|
| SimpleViT (control) | 8 | 9.575 ± 0.009 | 1.238 ± 0.005 | 47.02 ± 0.19 |
| CylindricalViT | 8 | 9.443 ± 0.014 | 1.231 ± 0.006 | 45.06 ± 0.25 |
| YawAug-xRIR | 8 | 9.322 ± 0.006 | 1.294 ± 0.006 | 46.39 ± 0.15 |
| SimpleViT (control) | 1 | 13.38 ± 0.10 | 1.890 ± 0.013 | 68.28 ± 0.61 |
| CylindricalViT | 1 | 13.58 ± 0.09 | 1.942 ± 0.014 | 69.76 ± 0.79 |
| YawAug-xRIR | 1 | 13.39 ± 0.08 | 1.969 ± 0.015 | 69.52 ± 0.73 |

(`TABLE_V1.json`; the living table `worklog/worklog_yixun/model_comparison.md` is rendered from it.) The seed SDs are well below the between-model differences (C50: ≈ 10× at both K; EDT: 3–4× at K = 8, < 2× at K = 1), so the paired comparisons above are not seed noise; the paired bootstrap, not these SDs, carries the inference.

## 3. What the augmented model does across headings (descriptive, seed 42, `GRID_SEED42.json`)

C50 r_k at +22.5°/+45°/−45°/−22.5° = +2.8 / +3.3 / +3.2 / +2.1 %; EDT r_k between −0.85 % (−5.6°) and +0.7 % over the 18 angles; both ≈ 0 at 90°, 180°, −90°. The shape is the same as exp_03's control curve (peaks at diagonal headings, zero at multiples of 90°), just lower for C50. The epoch-9 checkpoint (41 679 updates, the FLAC "40k" convention; `EPOCH9_K8.json`): EDT 46.89 ± 0.16 ms, C50 1.261 ± 0.006 dB, T60 9.41 ± 0.01 % — C50 is better at epoch 9 than at epoch 12 (1.294), i.e. the augmented model's C50 worsens over the last low-learning-rate epochs (the control's epoch-9 value is not available under this five-seed protocol; exp_01's single-seed per-epoch evaluations are not paired with these runs).

## 4. Why the augmentation costs C50 (interpretation, not tested)

The diagonal-heading degradation in exp_03 was attributed to distribution shift from axis-aligned training rooms acting through the learned position pool and the absolute-coordinate embeddings. Augmenting with uniform yaw makes the training distribution heading-invariant, but the test rooms are still axis-aligned: the model loses the prior that walls are at 0°/90° and pays for it exactly on the metric most sensitive to early reflections (C50), while the decay-shaped metrics (EDT, T60) are unaffected or slightly better. Half the augmentation budget is spent on headings that never occur at test time. A milder or structured augmentation (small angles, or 90° multiples only) is the obvious follow-up; it is not part of this experiment.

## 5. Validity and limitations

- All decision-driving intervals passed the two-seed convergence gate (maximum endpoint movement / width 0.0277, at the TOST EDT −90° cell, against the 0.10 tolerance); no query was excluded in any cell (all 6 337 valid for every seed and arm).
- Single training realisation per arm (plan §2): the control is exp_01's run, the augmented model is one run; run-to-run training noise is not estimated. The C50 effect (4.5 %, upper bound 5.5 %) is far larger than exp_01/exp_03's between-run differences, the EDT effect (1.3 %) is within their range.
- Room-cluster intervals (17 rooms) include zero for the H1 EDT cells and for every decision-driving H2 cell (C50, EDT); the H1 C50 penalty's room interval excludes zero at both K. Among the non-decision cells, the H2 T60 differences at +22.5°/+45° and the TOST C50 cells at ±45° also have room intervals excluding zero.
- Protocol deviations recorded in the notebook: epoch 3 ran ≈ 5 min slower under foreign CPU load and the S-tier co-tenant cost ≤ 1 min/epoch (no gate affected); three evaluation launches were refused by the dirty-tree gate during a Coder round and re-run with identical commands; the grid run used exp_03's 10-angle acoustic grid after a queue-script correction; exp_05's L-tier training shared the host during all evaluations and a foreign FLAC job shared GPU 0 during the last five (evaluations are deterministic given the manifests and not timing-sensitive).
- The reported checkpoint is epoch 12 (plan §3); the best test-loss epoch was 5 (0.01599 vs 0.01604 at epoch 12).

## 6. Provenance

Training: attempt `ckpt/xRIR_simple_yawaug_8_shot/attempt_20260913T102051` (→ `final`), 27.43 h, launch commit `f19b9b6`, probe receipt `_probe_20260913T101537_probe.json` (overhead 0.78 %), alignment audit 0 / 12 800. Evaluations: 46 runs under `ckpt/yaw_aug/eval/` with immutable manifests and completions (evaluator closure `0b05245c…`, writer `15f9c984…`). Approval record `yaw_aug_xrir_results_assets/approved_digests.json` (commit `fa219ef`). Producer runs at `87d3e54` (H2, TOST, EPOCH9), `fa219ef` (H1 ×2, TABLE_V1) and `f851be1` (GRID), as recorded in the sidecars; generators and binding at `f851be1` (committed in `cab6f42`). Reviews: 14 Fable 5.1 code/record reviews (`yaw_aug_xrir_fable_*_review.md`), three Codex plan reviews.
