# oriented_cyl (exp_06) — parameters and configuration

**Status:** PREPARED 2026-09-15 15:11 at main `53307b6` (approvals fill). To be finalised at launch (GPU 1 after the peer's hand-over, ≈ 2026-09-16 13:00) with the attempt directory, pid, and the ladder rungs 4–5 results.

## Confirmatory pretraining (arm C, plan §5)
- Entry point: `tools/exp06_launch.sh full --gpu 1 --reviewed-commit <launch commit> --attempt-root ckpt/exp06/pretrain/xRIR_cylor_8_shot` → child `tools/exp06_train.py --backbone cylindrical_oriented --save-dir <attempt> --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 500 --epoch-ckpt-every 1 --run-type full --approved worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json --reviewed-commit <launch commit>`
- Environment: `CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; xRIR conda env (Python 3.8.20, torch 2.0.1+cu117, numpy 1.23.5); one RTX A6000 (48 GB).
- Recipe identical to exp_01 (`ckpt/xRIR_{simple,cyl}_8_shot/args.json`) except `backbone`, `save_dir`, `epoch_ckpt_every 1` (declared operational difference) and the exp_06 provenance fields; M-tier ViT dims 512/12/8/512; expected ≈ 156 min/epoch → ≈ 31 h.
- Approvals: `code` section filled at `53307b6` (see notebook 2026-09-15); `reused`/`artifacts` null until their artefacts exist; `artifacts.heading` = sha256 of the four heading JSONs (c37053c3…, 2eddf047…, 53641c37…, c3495910… for class/dampened/hallway/complex) to be committed before HAA execution.
- Heading: `ckpt/exp06/heading/*.json` — all rooms estimated −90°, k = 128, confirmatory (rung 3, HEAD 53307b6).
- Arm checkpoint: `<attempt>/epoch_012.pth`; completion by `tools/exp06_finalize.py --run-type full`; promotion symlink `ckpt/exp06/pretrain/xRIR_cylor_8_shot/final`.

## Ladder before `full` (plan §9)
- Rung 4 smokes (GPU 1, ≤ 3 GB, 300 s each): `tools/exp06_launch.sh smoke --gpu 1 --reviewed-commit <launch commit>` (trainer parity, oriented smoke, fixture) + HAA smokes via `tools/exp06_smoke.py --entry exp06_haa_finetune / exp06_haa_eval` with the fixture and `--heading-json-dir ckpt/exp06/heading`.
- Rung 5 probe: `tools/exp06_launch.sh probe --gpu 1 --reviewed-commit <launch commit>` (200 batches at 32 × 2, TF32, `--no-save`, budget 2400 s / 46 GB) → receipt; full launch only if extrapolated epoch time ≤ 1.15 × 156 min and peak memory fits.

## HAA stage (plan §6.2; after G1)
- `tools/exp06_haa_pipeline.sh <gpu> cyl_or:0 cyl_or:1 cyl_or:2 control_hf:0 control_hf:1 control_hf:2 cyl_hf:0 cyl_hf:1 cyl_hf:2 zeroshot` with `EXP06_HEADING_DIR=ckpt/exp06/heading`, `EXP06_CYLOR_CKPT=ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth`; stage 1 1000 epochs / val every 10; stage 2 200 / 2; K = 8, eval_seed 0, unseeded Griffin-Lim (primary); S1 sensitivity (per-query seeded) afterwards.

## Simulated evaluation (plan §6.3)
- `tools/exp06_eval_launch.py` per seed 42–46 with `ckpt/yaw_aug/reference_manifest_k8_seed*.json`, `--conditions P`, k = 0, TF32 off, batch 16, checkpoint role arm / epoch 12; arm A from exp_04's control runs; arm B evaluated in exp_06 unless exp_05's M-tier runs are admissible.
