# oriented_cyl (exp_06) — parameters and configuration

**Status:** LAUNCHING 2026-09-16 12:07 at main `6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113` (gate OK; rung 4 passed as A6 exploratory diagnostics; rung 5 probe running). Attempt directory and pid appended at launch.

## Confirmatory pretraining (arm C, plan §5)
- Entry point: `tools/exp06_launch.sh full --gpu 1 --reviewed-commit 6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113 --attempt-root ckpt/exp06/pretrain/xRIR_cylor_8_shot` → child `tools/exp06_train.py --backbone cylindrical_oriented --save-dir <attempt> --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 500 --epoch-ckpt-every 1 --run-type full --approved worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json --reviewed-commit 6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113`
- Environment: `CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; xRIR conda env (Python 3.8.20, torch 2.0.1+cu117, numpy 1.23.5); one RTX A6000 (48 GB).
- Recipe identical to exp_01 (`ckpt/xRIR_{simple,cyl}_8_shot/args.json`) except `backbone`, `save_dir`, `epoch_ckpt_every 1` (declared operational difference) and the exp_06 provenance fields; M-tier ViT dims 512/12/8/512; expected ≈ 156 min/epoch → ≈ 31 h.
- Approvals: `code` section filled at `53307b6` (see notebook 2026-09-15); `reused`/`artifacts` null until their artefacts exist; `artifacts.heading` = sha256 of the four heading JSONs (c37053c3…, 2eddf047…, 53641c37…, c3495910… for class/dampened/hallway/complex) to be committed before HAA execution.
- Heading: `ckpt/exp06/heading/*.json` — all rooms estimated −90°, k = 128, confirmatory (rung 3, HEAD 53307b6).
- Arm checkpoint: `<attempt>/epoch_012.pth`; completion by `tools/exp06_finalize.py --run-type full`; promotion symlink `ckpt/exp06/pretrain/xRIR_cylor_8_shot/final`.

## Ladder before `full` (plan §9)
- Rung 4 smokes (GPU 1, ≤ 3 GB, 300 s each): `tools/exp06_launch.sh smoke --gpu 1 --reviewed-commit 6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113` (trainer parity, oriented smoke, fixture) + HAA smokes via `tools/exp06_smoke.py --entry exp06_haa_finetune / exp06_haa_eval` with the fixture and `--heading-json-dir ckpt/exp06/heading`.
- Rung 5 probe: `tools/exp06_launch.sh probe --gpu 1 --reviewed-commit 6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113` (200 batches at 32 × 2, TF32, `--no-save`, budget 2400 s / 46 GB) → receipt; full launch only if extrapolated epoch time ≤ 1.15 × 156 min and peak memory fits.

## HAA stage (plan §6.2; after G1)
- `tools/exp06_haa_pipeline.sh <gpu> cyl_or:0 cyl_or:1 cyl_or:2 control_hf:0 control_hf:1 control_hf:2 cyl_hf:0 cyl_hf:1 cyl_hf:2 zeroshot` with `EXP06_HEADING_DIR=ckpt/exp06/heading`, `EXP06_CYLOR_CKPT=ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth`; stage 1 1000 epochs / val every 10; stage 2 200 / 2; K = 8, eval_seed 0, unseeded Griffin-Lim (primary); S1 sensitivity (per-query seeded) afterwards.

## Simulated evaluation (plan §6.3)
- `tools/exp06_eval_launch.py` per seed 42–46 with `ckpt/yaw_aug/reference_manifest_k8_seed*.json`, `--conditions P`, k = 0, TF32 off, batch 16, checkpoint role arm / epoch 12; arm A from exp_04's control runs; arm B evaluated in exp_06 unless exp_05's M-tier runs are admissible.

## Launch record (appended at launch)
- **Launched** 2026-09-16 12:14:15 (local) at main `6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113`; launcher pid 2631653 (`ckpt/exp06/pretrain/xRIR_cylor_8_shot/attempt_20260916T161415/launch.pid`), trainer pid 2633255 (`ckpt/exp06/pretrain/xRIR_cylor_8_shot/attempt_20260916T161415/child.pid`); attempt `ckpt/exp06/pretrain/xRIR_cylor_8_shot/attempt_20260916T161415`; logs `oriented_cyl_2026-09-16_12:14:15_launchday_full.log` (launcher) and `oriented_cyl_20260916T161415_train_full.log` (trainer, tee'd).
- **Rung 5 probe** (`probe_20260916T160828`): 200 micro-batches at 32 × 2 TF32, outcome ok, 221.7 s wall, peak 31.28 GB, steady 31.8–32.6 samples/s = 0.99 s per micro-batch (ceiling 1.16) ⇒ projected epoch ≈ 153 min, 12 epochs ≈ 30.6 h ⇒ end ≈ 2026-09-17 19:00–20:30 (+ finalisation). Completion `passed: true`.
- **Rung 4** — five smokes as exploratory diagnostics (A6), all ok within 6 GB / 300 s; trainer parity identical.
- **Epoch 1** (14:50): 156.2 min wall (train 154.8 min; ≤ 179-min criterion PASSED despite a 2.5-min foreign co-tenant 13:57–13:59); avg train loss 0.01584 (stft 0.00485, decay 0.01100), test loss 0.01683 (best); exp_01 references at epoch 1: cylindrical 0.01592 / 0.01790 (157.4 min), simple 0.01584 / 0.01835 (138.9 min). Epoch 2 running at 29–32 samples/s (host load 35 from foreign CPU jobs) ⇒ completion ≈ 2026-09-17 19:30–20:30.
