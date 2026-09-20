# exp_07 seen_protocol — parameters and set-up (retrospective, 2026-09-20)

Reconstructed from `effective_args.json`, `train_manifest.json`, probe receipts and completions after the runs.

| Arm | Backbone / yaw | Recipe | Reviewed commit | GPU | Probe T_run (h) / peak (GiB) | Certified training wall (h, completion) | Epoch-12 test loss |
|---|---|---|---|---|---|---|---|
| seen_simple | simple / yaw_aug 0 | bs 32×2, lr 0.001, wd 0.0001, decay 3/0.1, 12 ep, seed 0, TF32 True, workers 12, protocol seen | 076cd40 | 0 | 27.0 / 30.8 | 29.80 | 0.0155264 |
| seen_cyl | cylindrical / yaw_aug 0 | bs 32×2, lr 0.001, wd 0.0001, decay 3/0.1, 12 ep, seed 0, TF32 True, workers 12, protocol seen | b793a96 | 0 | 31.8 / 30.8 | 30.70 | 0.0153038 |
| seen_aug | simple / yaw_aug 1 | bs 32×2, lr 0.001, wd 0.0001, decay 3/0.1, 12 ep, seed 0, TF32 True, workers 12, protocol seen | b14dc3b | 1 | 27.3 / 30.8 | 27.55 | 0.0157880 |

Environment: `OMP_NUM_THREADS=2 PYTHONHASHSEED=0 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; torch 2.0.1+cu117, Python 3.8.20, RTX A6000 48 GB; seen training set 296 454 files (inventory `ckpt/exp07/train_inventory_seen.json`), 9 265 micro-batches per epoch. Evaluation: seen split 6 217 queries (`treble_multi_room_dataset/seen_test_split.pkl`), K = 8 and K = 1, seeds 42–46 (`ckpt/exp07/reference_manifest_seen_k{8,1}_seed*.json`), batch 16, TF32 off, condition P at k = 0; entry point `tools/exp07_eval_launch.py --split seen`. Released reference `checkpoints/xRIR_seen.pth` (sha `1762c702…`). Approval pins filled at `0e47341`.


Architecture (all three arms, M tier): ViT dim 512 / depth 12 / heads 8 / mlp 512, patch (16, 32) on the 256×512 panorama, `intermediate_ch` 256; training `num_shot` 8, `max_len` 9600, sample rate 22 050 Hz; seen_aug additionally `yaw_aug 1`, `yaw_aug_seed` and `yaw_aug_width` as recorded below. Receipts/ledgers: seen_simple: vit 512/12/8/512, num_shot 8, max_len 9600, yaw_aug 0 (seed 0, width 512), probe receipt `ckpt/exp07/seen_simple/_probe_20260917T034540probe_seen_simple.json`, ledger `ckpt/exp07/seen_simple/cumulative_hours.json`; seen_cyl: vit 512/12/8/512, num_shot 8, max_len 9600, yaw_aug 0 (seed 0, width 512), probe receipt `ckpt/exp07/seen_cyl/_probe_20260918T151139probe_seen_cyl.json`, ledger `ckpt/exp07/seen_cyl/cumulative_hours.json`; seen_aug: vit 512/12/8/512, num_shot 8, max_len 9600, yaw_aug 1 (seed 0, width 512), probe receipt `ckpt/exp07/seen_aug/_probe_20260918T205229probe_seen_aug.json`, ledger `ckpt/exp07/seen_aug/cumulative_hours.json`.

This file is retrospective (reconstructed on 2026-09-20 from the artefacts); the SOP expects it before launch — recorded as a documentation deviation.
