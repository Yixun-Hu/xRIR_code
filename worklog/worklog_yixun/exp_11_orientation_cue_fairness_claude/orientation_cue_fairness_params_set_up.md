# Parameters and set-up — orientation_cue_fairness (exp_11)

## Arms (see plan §2.2)
- A `control`: SimpleViT, exp_01 same-budget control `ckpt/xRIR_simple_8_shot/epoch_12.pth` (sha256 in `tools/exp04_profiles.CONTROL`), room frame — exp_02 completions `ckpt/sim2real/control/`.
- A′ `released`: authors' `checkpoints/xRIR_unseen.pth`, room frame — exp_02 `ckpt/sim2real/released/` (admission through the legacy receipt to be confirmed by the Coder/Reviewer; otherwise an external row from exp_02's canonical JSON).
- B `cyl`: CylindricalViT, exp_01 `ckpt/xRIR_cyl_8_shot/epoch_12.pth`, room frame — exp_02 `ckpt/sim2real/cyl/`.
- C `cyl_or`: `cylindrical_oriented`, exp_06 pretraining `ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth` (sha256 3df2ceb5…), heading frame — exp_06 `ckpt/exp06/sim2real/cyl_or/`.
- D `control_hf`: SimpleViT (A's checkpoint), heading frame — exp_06 `ckpt/exp06/sim2real/control_hf/`.
- E `yawaug`: yaw-augmented SimpleViT, exp_04 `ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth` (sha256 f8e64052…), room frame — exp_09 `ckpt/exp09/sim2real/yawaug/`.
- F `cyl_hf`: CylindricalViT (B's checkpoint), heading frame — exp_06 `ckpt/exp06/sim2real/cyl_hf/`.
- **G `yawaug_hf`** (new, Phase 1): E's checkpoint, **heading frame** (k = 128 in every room from `ckpt/exp06/heading/*.json`) — `ckpt/exp11/sim2real/yawaug_hf/{seed0,seed1,seed2,zeroshot}`.
- **H `simple_or`**, **I `simple_or_yaw`** (Phase 2, optional): new backbone `simple_oriented` (SimpleViT with the two cos/sin azimuth planes, `channels = 5`), pretrained from scratch — H with exp_01's recipe, I with exp_04's yaw-augmented recipe — attempt roots `ckpt/exp11/pretrain/xRIR_simpor_8_shot`, `…/xRIR_simpor_yawaug_8_shot`, promoted `final/epoch_012.pth`; HAA in the heading frame → `ckpt/exp11/sim2real/{simple_or,simple_or_yaw}/`.

## Pretraining recipes (Phase 2; frozen)
- Arm H = exp_01's control recipe (`ckpt/xRIR_simple_8_shot/args.json`): `num_shot 8, max_len 9600, lr 1e-3, weight_decay 1e-4, decay_epochs 3, lr_gamma 0.1, epochs 12, batch_size 32, accum_steps 2, seed 0, tf32 true, num_workers 12`, `vit_dim 512, vit_depth 12, vit_heads 8, vit_mlp_dim 512`, `yaw_aug 0`; `XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=8`.
- Arm I = exp_04's yaw-augmented recipe (`ckpt/xRIR_simple_yawaug_8_shot/final/args.json`): the same plus `yaw_aug 1, yaw_aug_seed 0, yaw_aug_width 512`, `save_every 0`, `epoch_ckpt_every 1` (exp_04 ran with `OMP_NUM_THREADS=2`; operational, not recipe).
- Arm checkpoint = `epoch_012.pth`, completion contract as exp_06 §5 (child exit 0, log closed, revalidation, schema, `epoch_012.pth == last.pth["model"]`).

## Fine-tuning and evaluation recipe (frozen; = exp_02 = exp_06 §6.2 = exp_09)
- Stage 1: class_room + hallway + complex_room, 1000 epochs, val every 10, TF32, full batch (`batch_size 0`), AdamW lr 1e-4, wd 1e-4, ×0.1 per 50 epochs, selection on the DiffRIR validation split; stage 2: per room, 200 epochs, val every 2, init = stage-1 `best.pth`.
- Evaluation: DiffRIR test split, K = 8, `eval_seed 0`, `max_len 9600`, `depth_variant default`, unseeded Griffin-Lim (primary phase policy), EDT / C50 / T60 (T60 skipped for dampened_room); test sizes 463 / 198 / 423 / 198 (classroom / dampened / hallway / complex).
- Seeds 0, 1, 2 + zero-shot evaluation of the init in the arm's frame.
- Pipeline: `tools/exp06_haa_pipeline.sh <gpu> <init>:0 <init>:1 <init>:2 <init>:zeroshot` with `EXP06_HAA_OUT=ckpt/exp11/sim2real`, `EXP06_HAA_RECORD=<this record>`, `EXP06_HEADING_DIR=ckpt/exp06/heading`, `EXP09_YAWAUG_CKPT=ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth`; each child finalised by `tools/exp06_finalize.py`.

## Statistics (exp_06 §7 machinery)
- `tools/exp06_bootstrap.py`: two-way and query-cluster percentile bootstraps, `n_boot` 10 000 (50 000 adjusted), seeds 0/1, convergence tolerance 0.10 with one quadrupling, void rule 1 % / 99 %, alpha 0.05, Bonferroni-11 screens, margin +0.23 dB (N2/P3 non-inferiority; P1/P2 equivalence).

## Schedule (owner's ETAs, 2026-09-26)
- First exclusive card ≈ Sep 29 16:00 (40 k ≈ 12:30 + final watcher evals), second ≈ Oct 3 (FLAC exp_13 tier-L runs to 40 k steps); Phase 1 ≈ 3 h on the first card; Phase 2 pretrainings ≈ 31 h + 28 h.
