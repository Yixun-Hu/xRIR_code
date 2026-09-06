# Parameters — cylvit_vs_simplevit

## Training runs (identical except `--backbone` / `--save-dir`)

| arg | value |
|---|---|
| `backbone` | `cylindrical` / `simple`, `ckpt/xRIR_simple_8_shot` |
| `save_dir` | `ckpt/xRIR_cyl_8_shot` / `simple`, `ckpt/xRIR_simple_8_shot` |
| `num_shot` | `8` |
| `max_len` | `9600` |
| `lr` | `0.001` |
| `weight_decay` | `0.0001` |
| `decay_epochs` | `3` |
| `lr_gamma` | `0.1` |
| `epochs` | `12` |
| `batch_size` | `32` |
| `accum_steps` | `2` |
| `num_workers` | `12` |
| `seed` | `0` |
| `tf32` | `True` |
| `log_interval` | `50` |
| `save_every` | `500` |
| `epoch_ckpt_every` | `5` |
| `resume` | `None` |
| `max_train_batches` | `0` |
| `max_test_batches` | `0` |
| `test_subset` | `0` |

Derived: 296,334 training RIRs (AcousticRooms unseen-split train set, 244 rooms), 9,261 batches of 32 per epoch, 4,630 optimizer steps per epoch at effective batch 64, 12 epochs = 55,560 steps (3.56 M samples). LR schedule `ExponentialLR`: lr = 1e-3 × 0.1^(epoch/3) → 4.6e-4 (ep 1) … 1e-7 (ep 12). Loss = `stft_l1_loss` (log-magnitude L1) + `compute_spect_energy_decay_losses` (Schroeder decay L1, weight 0.01). Reference RIRs: K=8 other sources at the same receiver, re-drawn every call. Precision: FP32 weights, TF32 matmul/cuDNN. Environment: conda `xRIR` (Python 3.8.20, torch 2.0.1+cu117, numpy 1.23.5); one NVIDIA RTX A6000 (48 GB) per run; data from the local mirror `XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`.

## Model configuration (both)

`xRIR(num_channels=8, n_bins=310, dim=512, intermediate_ch=256, image_size=(256,512), patch_size=(16,32), depth=12, mlp_dim=512, heads=8)`. Cylindrical variant: `CylindricalViT(in_channels=3, image_size=(256,512), patch_size=(16,32), dim=512, depth=12, heads=8, dim_head=64, mlp_dim=512)` in place of `SimpleViT`; 32.12 M vs 32.08 M parameters (the extra 8×496 are the circular relative-position bias tables). STFT `fft 124 / hop 31 / win 62` → 63×310; panorama coord scale 5.0; sample rate 22050; `max_len` 9600.

## Evaluation configuration

`eval_xRIR_backbone.py --split unseen --num-shot 8 --seed 0 --num-workers 6 --threads 4` (no `--shuffle`, no `--max-samples`): full unseen test split (6337 queries, 17 held-out rooms), references drawn once per query by the seeded loader and therefore identical across models and epochs; Griffin-Lim (n_fft 124, 32 iters) inversion; EDT / C50 / T60 on the first 8000 samples exactly as in `eval_unseen.py`. Seen split once at epoch 9 (`--split seen`, 6217 queries). Baseline reproduction used the untouched `eval_unseen.py` / `eval_seen.py` (batch 1, `shuffle=True`, random references).

## Deviations from the authors' script, with reasons

| Item | Authors (`train_xRIR_unseen.py`) | This experiment | Reason |
|---|---|---|---|
| epochs / decay_epochs | 200 / 50 | 12 / 3 | 2.3–2.6 h per epoch on one A6000 (~20 days for 200); same LR endpoints and ratio |
| batch | 64 | 32 × 2 accumulation | 64 OOMs at 48 GB (BatchNorm in the audio encoder sees batch 32) |
| precision | FP32 | TF32 | speed; applied to both models |
| checkpoint selection | best test loss | evaluate every epoch; judge on averaged paired CIs | avoid selection on the test split |
| loader | no workers, CIFS share | 12 workers on a local mirror | throughput (873 vs 1.7 samples/s) |
