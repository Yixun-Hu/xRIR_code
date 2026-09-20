# Parameters and set-up — yawaug_haa (exp_09)

## Arm E `yawaug` (new)
- Init: `ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth` (exp_04 certified attempt `attempt_20260913T102051`, 12 epochs; sha256 `f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299`; registered `tools.exp04_profiles.AUG`, pinned in exp_04's approvals `checkpoints.aug`).
- Pretraining budget identical to exp_01's control (both `args.json`: lr 1e-3, wd 1e-4, decay 3 epochs × 0.1, 12 epochs, batch 32 × 2 accumulation, TF32, seed 0, K = 8); the only difference is `--yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512`.
- Backbone `simple`; frame `room` (no heading rotation, no orientation channel); outputs `ckpt/exp09/sim2real/yawaug/{seed0,seed1,seed2,zeroshot}`.

## Fine-tuning and evaluation recipe (frozen; = exp_02 = exp_06 §6.2)
- Stage 1: rooms class_room + hallway + complex_room, `--epochs 1000 --val-every 10 --tf32`, full batch (batch_size 0), AdamW lr 1e-4, wd 1e-4, ExponentialLR ×0.1 per 50 epochs; selection on the DiffRIR validation split.
- Stage 2: per room, `--epochs 200 --val-every 2 --tf32`, same optimiser, init = stage-1 `best.pth`.
- Evaluation: test split, K = 8, `eval_seed 0`, `max_len 9600`, `depth_variant default`, unseeded Griffin-Lim (primary phase policy), EDT / C50 / T60 (T60 skipped for dampened_room); test sizes 463 / 198 / 423 / 198.
- Seeds 0, 1, 2 (fine-tuning); zero-shot evaluation of the init.
- Pipeline: `tools/exp06_haa_pipeline.sh <gpu> yawaug:0 yawaug:1 yawaug:2 zeroshot` with `EXP06_HAA_OUT=ckpt/exp09/sim2real` (names per the Coder's round-1 report), each child finalised by `tools/exp06_finalize.py`; approvals gate `enforce_producer haa_children` at the launch tip.

## Statistics (exp_06 §7 machinery)
- `tools/exp06_bootstrap.py`: two-way and query-cluster percentile bootstraps, `n_boot` 10 000 (50 000 adjusted), seeds 0/1, convergence tolerance 0.10 with one quadrupling, void rule 1 % / 99 %.
- E1: hallway C50, E − A, two-way 95 %: detected harm / improvement / no detected difference; plus non-inferiority statement at +0.23 dB.
- E2: 11 cells E − A, Bonferroni-11. E3: E − C descriptive; zero-shot side split.

## Schedule (agreed with the peer session 2026-09-19 21:2x)
- GPU 1 after exp_07's seen_aug evaluations (peer's "GPU 1 released" ≈ 04:00 Sep 20), ≈ 3.3 h exclusive; GPU 0 not used.

## As run (2026-09-20)
- Code: main **85f2155** (merge of `exp09-yawaug` 9e41c65), approvals re-fill **c13e681**; launch tip 4e4110f (record-only commits above c13e681); summarise tip 4ac1748 (adds the peer's test-only `tests/test_exp07_table.py`); approvals blob and all 20 exp_06 code keys identical at c13e681, 4e4110f and 4ac1748.
- Queue: `tools/exp06_haa_pipeline.sh 1 yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot` (`EXP06_HAA_OUT=ckpt/exp09/sim2real`, `EXP09_YAWAUG_CKPT=ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth`, init sha256 f8e64052…), GPU 1 exclusive, 03:22:17 → 06:07:48 (2 h 46 min; ≈ 55 min per seed: stage 1 ≈ 27 min at ≈ 36 GB, stage 2 ≈ 5 min per room at ≈ 13 GB, evals ≈ 1–2 min; zero-shot ≈ 6 min); 35/35 completions, all `admissible_arm: true`; 3.6 GB on disk.
- Job spec (identical for the three seeds apart from `seed`): backbone simple, frame room, heading none, rooms class_room/complex_room/dampened_room/hallway, expect finetune; stage-1 args epochs 1000, val_every 10, lr 1e-4, wd 1e-4, batch_size 0 (full), TF32, depth_variant default, num_shot 8, eval_seed 0.
- Summariser: `tools/exp06_summarize_haa.py --experiment exp09 --exp09-root ckpt/exp09/sim2real --approved <exp_06 approved_digests.json> --approved-commit 4ac1748 --gate-g1 ckpt/exp06/gate_g1.json` → `ckpt/exp09/stats.json` (sha256 f2faca78f895b7fbe8adcfa2b42690c3154d8873cc9d062c6c32208a9015afc9), `ckpt/exp09/summary.txt` (sha256 94963f7d…); E1 converged at 10 000 draws (movement ratio 0.012), no void, no deviations.
- Per-seed E1 differences (hallway C50, E − A): seed 0 +0.278, seed 1 +0.262, seed 2 +0.289 dB.
