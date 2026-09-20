# Analysis — yawaug_haa (exp_09): the yaw-augmented xRIR on the real-room (HAA) benchmark

Status: complete, 2026-09-20 06:30. Every number in §3 is copied from the canonical `ckpt/exp09/stats.json`
(sha256 f2faca78f895b7fbe8adcfa2b42690c3154d8873cc9d062c6c32208a9015afc9, producer `tools/exp06_summarize_haa.py --experiment exp09`
at commit 4ac1748, drift none, deviations none); the full tables are in `yawaug_haa_results.md` / `yawaug_haa_01_results.html`.
Relative percentages in §4 are derived here from the producer's means and are marked as such.

## 1. Question and design (plan v2, `plan_yawaug_haa.md`)

Does the yaw-augmented SimpleViT (exp_04's `ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth`, arm **E**) transfer to the
four real Hearing-Anything-Anywhere rooms as well as the plain SimpleViT control (arm **A**, exp_02) under the paper's
two-stage fine-tuning recipe, and where does it sit relative to the oriented CylindricalViT (arm **C**, exp_06)?
E is fine-tuned in the **room frame** with no heading (the augmentation was trained to be heading-agnostic; the heading
frame of exp_06 is not applied), three fine-tuning seeds (0, 1, 2) plus one zero-shot job, on GPU 1, via
`tools/exp06_haa_pipeline.sh 1 yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot` with `EXP06_HAA_OUT=ckpt/exp09/sim2real`.
Recipe (unchanged from exp_02/exp_06): stage 1 class_room + hallway + complex_room, 1000 epochs, validation every 10;
stage 2 per room, 200 epochs, validation every 2; AdamW 1e-4, weight decay 1e-4, ×0.1 every 50 epochs, full batch,
TF32; selection on the DiffRIR validation split; K = 8, `eval_seed` 0, unseeded Griffin-Lim (exp_02's primary phase
policy). Arms A, B (cylindrical, exp_02) and C (oriented cylindrical, exp_06) are re-read from their existing
completions; exp_06's canonical JSON is not edited.

Pre-registered outputs (plan §3): **E1** hallway C50, E − A, converged seed-0 two-way 95 % interval → `category`
(detected harm / detected improvement / no detected difference) and, separately, `non_inferior_at_margin` at +0.23 dB;
**E2** all 11 room × metric cells E − A with nominal and Bonferroni-11 two-way intervals (labels only);
**E3** descriptive: E − C on all cells, zero-shot room-frame side split (−y / +y) of E next to A, B, C, and E's per-seed values.

## 2. Provenance chain (what makes the numbers evidence)

- Code: worktree branch `exp09-yawaug` (Coder Claude Opus 5, round 1 `0cecf8e` → fix cycle `9e41c65`), Codex `gpt-6-astra`
  (xhigh) code review `yawaug_haa_codex_code_round1_review.md` (F1–F3) and close review
  `yawaug_haa_codex_code_round1_close_review.md` (approve with changes, all closed). Merged into main **85f2155**
  (`--no-ff`), approvals re-filled at **c13e681** (`finalize` ea8146d8…, `haa_pipeline_sh` 4a401ee3…, `summarize_haa`
  3fc1e82d…; the 17 other keys unchanged; `tools/exp06_approvals_api.py` byte-identical, so exp_06's ten sim-eval manifests
  still verify).
- Launch tip **4e4110f** (main; the five commits after c13e681 are the peer's record-only exp_07 commits — no file outside
  `worklog/` changed). Every child binds the approvals **blob at its reviewed commit** (`haa_approvals`, new in exp_09),
  records the E checkpoint through the exp_04 approvals identity (`exp04_aug_checkpoint` resolver in `tools/exp06_finalize.py`),
  and writes an atomic `completion.json`; the summariser refuses anything not admissible.
- Suite at c13e681 (CPU): 1 failed / 3266 passed / 49 skipped. The failure is the peer's
  `tests/test_exp07_table.py::test_the_cli_refuses_the_unapproved_committed_profile_and_writes_nothing`, whose premise
  (the committed exp_07 profile is unapproved) their pin fill `0e47341` removed; verified to pass at `f5e1cf6` and fail at
  `0e47341` with no exp_09 code involved, and the exp_09 merge touches no exp_07 file. Accepted as the exp_07 baseline
  (worklog 03:17). The three exp_07 `runtime_approval` guards the plan named as the expected red tests now pass.
  The peer fixed that test on main at 04:5x with the test-only merge `237ef9b` (renamed, hermetic all-null record); the only
  non-worklog change since c13e681 is `tests/test_exp07_table.py`, the exp_06 approvals blob is identical and no code key moves.
- Bounded room-frame smoke on GPU 1 (fine-tune 2 epochs on class_room + 4-sample hallway eval, exploratory receipts) passed
  twice: pre-merge on the unchanged entry points (01:43) and post-merge at 4e4110f (03:17, `ckpt/exp09/_smoke/2026-09-20_03:16:28`).
- Queue launched 03:22:17 by the armed chain (`exp09_launch_chain.sh`; pipeline pid 3932953, stamp 20260920T072147);
  logs `yawaug_haa_2026-09-20_03:21:47_launch.log`, per-stage `yawaug_haa_20260920T072147_haa_yawaug_*.log`.

## 3. Results (from `ckpt/exp09/stats.json` only)

**Run.** 35/35 completions under `ckpt/exp09/sim2real/yawaug/` (3 seeds × [stage 1 + 4 stage-2 rooms + 4 evals + job] + zero-shot
[4 evals + job]), every child and job `admissible_arm: true`, exit 0; queue 03:22:17–06:07:48 on GPU 1 (2 h 46 min).
Summariser: 10 000 bootstrap draws (50 000 for the adjusted tails), seeds 0/1, convergence tolerance 0.10, alpha 0.05,
family 11, margin +0.23 dB, cohort 423/423 hallway test queries, no exclusions (`excluded` = 0 queries for both arms and all seeds).

**Fine-tuned arms (mean ± sd over seeds 0–2; EDT s / C50 dB / T60 %).**

| Arm | Classroom | Dampened | Hallway | Complex |
|---|---|---|---|---|
| A control (SimpleViT, exp_02) | 0.0812±0.0007 / 1.501±0.008 / 4.31±0.13 | 0.0384±0.0002 / 2.847±0.028 / – | 0.0611±0.0009 / 1.135±0.008 / 2.68±0.00 | 0.0765±0.0003 / 1.666±0.021 / 4.94±0.07 |
| B cylindrical (exp_02) | 0.0887±0.0008 / 1.630±0.038 / 4.00±0.24 | 0.0385±0.0021 / 2.863±0.132 / – | 0.0874±0.0018 / 2.052±0.084 / 4.16±0.09 | 0.0821±0.0006 / 1.736±0.030 / 5.05±0.22 |
| C oriented cylindrical (exp_06) | 0.0788±0.0005 / 1.498±0.011 / 4.52±0.07 | 0.0379±0.0013 / 2.885±0.073 / – | 0.0621±0.0007 / 1.137±0.006 / 2.72±0.05 | 0.0797±0.0012 / 1.753±0.052 / 4.84±0.03 |
| **E yaw-augmented SimpleViT (exp_09)** | 0.0814±0.0004 / 1.516±0.014 / 4.33±0.23 | 0.0370±0.0005 / 2.785±0.041 / – | **0.0705±0.0007 / 1.412±0.010 / 3.14±0.02** | 0.0805±0.0013 / 1.741±0.031 / 5.05±0.05 |

**E1 (primary, hallway C50, E − A): +0.277 dB, two-way 95 % [+0.204, +0.352] → category `detected harm`;
`non_inferior_at_margin` (+0.23 dB) = no.** Converged at 10 000 draws (seed 0 [+0.204, +0.352], seed 1 [+0.204, +0.350];
movement 0.0018 on a width of 0.148, ratio 0.012 < 0.10), no void reasons, no retry. Query-cluster interval [+0.208, +0.351].
Per-seed differences +0.278 / +0.262 / +0.289 dB (seeds 0 / 1 / 2): every seed is above the margin.

**E2 (screen, E − A, Bonferroni-11 adjusted two-way intervals).**

| Room | EDT (s) | C50 (dB) | T60 (%) |
|---|---|---|---|
| Classroom | +0.0001 [−0.0045, +0.0048] no diff. | +0.015 [−0.069, +0.096] no diff. | +0.02 [−0.47, +0.53] no diff. |
| Dampened | −0.0014 [−0.0070, +0.0040] no diff. | −0.062 [−0.361, +0.236] no diff. | – |
| Hallway | **+0.0095 [+0.0032, +0.0167] harm** | **+0.277 [+0.175, +0.387] harm** | **+0.46 [+0.16, +0.81] harm** |
| Complex | +0.0041 [−0.0016, +0.0099] no diff. (nominal [+0.0002, +0.0077]) | +0.074 [−0.029, +0.175] no diff. (nominal [+0.005, +0.141]) | +0.11 [−0.21, +0.40] no diff. |

Three detected-harm cells, all in the hallway; the two complex-room cells whose nominal intervals exclude zero do not survive
the family adjustment; no cell shows a detected improvement.

**E3 (descriptive, E − C, nominal two-way).** Hallway EDT +0.0085 [+0.0049, +0.0121], C50 +0.275 [+0.206, +0.343],
T60 +0.42 [+0.24, +0.61]; complex T60 +0.22 [+0.02, +0.43]; the other seven cells include zero (classroom EDT +0.0026
[−0.0012, +0.0063], C50 +0.018 [−0.052, +0.090], T60 −0.19 [−0.47, +0.05]; dampened EDT −0.0009, C50 −0.100; complex EDT +0.0009, C50 −0.012).

**Zero-shot (the un-fine-tuned inits; EDT / C50 / T60).** E: classroom 0.1263 / 2.044 / 7.33, dampened 0.0665 / 4.347 / –,
hallway 0.1535 / 2.820 / 7.12, complex 0.0971 / 1.801 / 6.05; A: 0.1179 / 1.933 / 7.35, 0.0723 / 4.670 / –, 0.1869 / 2.389 / 7.47,
0.1561 / 2.765 / 6.92; C: 0.1355 / 2.154 / 8.10, 0.0585 / 4.019 / –, 0.1300 / 1.998 / 4.85, 0.1093 / 2.017 / 6.44.
Room-frame side split of the zero-shot hallway C50 (−y side, n = 238 / +y side, n = 185): A 2.37 / 2.42, B 1.04 / 8.23,
C 1.23 / 2.99, **E 1.07 / 5.07**; hallway EDT: A 0.247 / 0.110, B 0.144 / 0.235, C 0.167 / 0.083, E 0.162 / 0.143;
hallway T60: A 9.27 / 5.15, B 4.82 / 16.45, C 5.38 / 4.16, E 5.87 / 8.73.

## 4. Reading

1. **The yaw augmentation does not transfer better to the real rooms; on the hallway it transfers worse.** After the paper's
   two-stage fine-tuning, E equals the control in the classroom, dampened and complex rooms (eight `no detected difference`
   cells) and is worse on every hallway metric: C50 +0.277 dB (≈ +24 % of the control's 1.135 dB, derived), EDT +9.5 ms
   (≈ +15 %, derived), T60 +0.46 pp (≈ +17 %, derived). The pre-registered primary cell is `detected harm`, and the
   +0.23 dB non-inferiority margin that the oriented cylindrical model met in exp_06 (C − A +0.002 dB) is missed by
   every seed. Fine-tuning shrinks the gap but does not close it: zero-shot hallway C50 2.82 vs 2.39 dB (E − A +0.43),
   fine-tuned 1.41 vs 1.14 (+0.28).
2. **Where E sits.** On the hallway E lies between the control / oriented cylindrical pair (1.135 / 1.137 dB) and the plain
   cylindrical model (2.052 dB) — roughly a third of the way to the cylindrical deficit that exp_06 attributed to
   wrong-side reference attention. E − C is the same size as E − A (+0.275 vs +0.277 dB) because C = A on the hallway.
3. **Consistent with the simulated results.** exp_04 found E non-inferior to the control on EDT only, with C50 ≈ 4.5 % worse at
   k = 0; exp_07 (seen protocol) found E slightly worse at K = 8 (EDT +1.9 %) and best at K = 1; here the in-distribution
   precision cost reappears on the one real room where orientation matters and is larger. The augmentation's benefit —
   reduced degradation under heading shift — is visible only zero-shot: before fine-tuning E is much better than the control in
   the complex room (C50 1.80 vs 2.77 dB, EDT 0.097 vs 0.156 s) and better in the dampened room (4.35 vs 4.67 dB), consistent with
   exp_04's reduced degradation under geometry shift; 1000 + 200 epochs of fine-tuning remove that advantage
   entirely (complex C50 E − A +0.074 dB, no detected difference).
4. **A side asymmetry that looks like the cylindrical model's (hypothesis, not evidence).** Zero-shot on the hallway the
   control is side-symmetric (C50 2.37 / 2.42 dB on the −y / +y sides), the plain cylindrical model is grossly asymmetric
   (1.04 / 8.23) and E is asymmetric in the same direction (1.07 / 5.07); the oriented cylindrical model is in between
   (1.23 / 2.99). A model trained to satisfy f(R·panorama, R·source) = f(panorama, source) for every yaw R cannot separate a
   source from its 180°-rotated twin when the panorama is itself nearly 180°-symmetric, which a long corridor is; the
   cylindrical gauge alignment discards absolute azimuth for the same reason. The real hallway is not acoustically symmetric, so a
   yaw-invariant prior can cost C50 on one side; which side, and by how much, is an empirical matter that the side split only
   describes. The pre-registered optional diagnostic — the exp_06 mirror probe with E in the control slot, exploratory, under
   `ckpt/exp09/_diag/` — is the direct test of this reading (predicted: a high hallway mirror cosine and wrong-side attention
   share for E, near the plain cylindrical model's 0.944 / 0.824 rather than the control's 0.436 / 0.254). It was not run.
5. **What this means for the paper's HAA story.** For the real-room table the orientation-relative channel (exp_06) is the
   mechanism that keeps the control's hallway accuracy while fixing the cylindrical model's side confusion; yaw augmentation
   of the SimpleViT is not a substitute — it introduces a milder version of the same confusion and buys nothing after
   fine-tuning. If a yaw-robust variant is wanted on HAA, the natural next candidate is yaw augmentation *with* the
   orientation channel (augment in the heading frame), which would keep the absolute-side information the augmentation
   removes.
6. **Caveats.** Absolute numbers are not comparable with the paper's Table 2 (test-split definition and selection leakage there;
   see exp_02 and the `xRIR_pdf.md` discussion of 2026-09-19); the inference is about these fixed DiffRIR test splits under
   two-way (seed × query) resampling with three fine-tuning seeds; Griffin-Lim phases are unseeded (exp_02's primary policy,
   shared by all four arms); the relative percentages above are derived from producer means, not producer fields.

## 5. Deviations from the plan and decisions taken

- Plan §4.5 / §5.3 named one exp_07 guard as the only tolerated red test; by launch time the tolerated set had changed twice
  (three guards at 01:05, then the single exp_07 table test at 03:17) because the peer's exp_07 end-game moved main
  underneath. None of these tests touches exp_09 code; each change is recorded with its evidence in the worklog.
- The pre-launch chain stopped on the second change (its filter only allowed the three guards); the Planner ran the
  post-merge smoke by hand at the merged tip and wrote `GO_GPU1` (03:17:57) after verifying the baseline, instead of editing
  and re-arming the chain. Nothing in the launch path itself was changed.
- Main moved by five peer record-only commits between the re-fill and the launch (c13e681 → 4e4110f); the approvals blob
  and every closure digest are identical at both, so the launch tip 4e4110f is the reviewed code.

## 6. Open items

- Optional diagnostic (plan §3, not evidence): the exp_06 mirror probe with E in the control slot, `--exploratory`, under
  `ckpt/exp09/_diag/` — not run; now motivated by §4.4 (predicted high hallway mirror cosine / wrong-side share for E).
- The summariser flagged no deviations (`deviations: []`); the launch tip (4e4110f) and the summarise tip (4ac1748) differ only
  in the peer's record files and `tests/test_exp07_table.py`, with identical approvals blob and closure digests.
- exp_06's S1 seeded-phase sensitivity remains deferred (no loader for the legacy arms); exp_09 inherits the same
  unseeded-Griffin-Lim primary phase policy, so the comparison across arms is like-for-like.
