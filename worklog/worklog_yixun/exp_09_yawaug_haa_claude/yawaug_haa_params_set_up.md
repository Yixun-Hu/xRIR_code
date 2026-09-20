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
