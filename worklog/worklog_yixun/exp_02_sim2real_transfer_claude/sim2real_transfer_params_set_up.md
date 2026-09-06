# Parameters — sim2real_transfer

## Data cache (`sim_to_real/prepare_haa.py`, 2026-09-05 16:05, `~/data_cache/HAA_xrir/<room>/`)
| room | scene | n | train | valid | test | scale (train peak) | depth map |
|---|---|---|---|---|---|---|---|
| class_room | classroomBase | 630 | 12 | 155 | 463 | 0.0742 | repo `sim_to_real/depth_map/class_room.npy` |
| dampened_room | dampenedBase | 276 | 12 | 66 | 198 | 0.1620 | HAA_processed render (released map misaligned) |
| hallway | hallwayBase | 576 | 12 | 141 | 423 | 0.1094 | repo `hallway.npy` |
| complex_room | complexBase | 408 | 12 | 198 | 198 | 0.0363 | HAA_processed render (released map is a copy of hallway) |

Source: `/media/diskstation/yixunhu/HAA_processed/<scene>/mono_rirs_22050Hz/<i>.wav` (22.05 kHz mono, propagation delay preserved: onset vs distance correlation 1.00), `metadata/{scenes,poses}_metadata.json` (speaker_xyz, DiffRIR train/valid indices, mic xyz). Stored: first 22050 samples per RIR as float32; model uses the first 9600.

## Fine-tuning (`sim_to_real/finetune_haa.py`), both stages
AdamW lr 1e-4, weight_decay 1e-4, `ExponentialLR(decay_epochs=50, gamma=0.1)` stepped per epoch, batch = all training targets (36 in stage 1, 12 in stage 2; one optimizer step per epoch), K=8 references re-drawn per step from the other 11 training RIRs, TF32 on, loss = STFT log-mag L1 + 0.01 × energy-decay L1 (same `compute_loss` as exp_01), `max_len` 9600. Stage 1: `--rooms class_room hallway complex_room --epochs 1000 --val-every 10`; val set = union of the three rooms' DiffRIR valid splits (494 samples), references fixed by `--eval-seed 0`. Stage 2: `--rooms <room> --epochs 200 --val-every 2`, init = stage-1 `best.pth`, val = that room's valid split. Selection: lowest val loss (epoch 0 = init evaluated too). Seeds 0/1/2 seed torch/numpy for the training reference draws and data order.

Inits: `released` = `checkpoints/xRIR_unseen.pth` (backbone simple); `control` = `ckpt/xRIR_simple_8_shot/epoch_12.pth`; `cyl` = `ckpt/xRIR_cyl_8_shot/epoch_12.pth` (backbone cylindrical); `released_repomaps` = released init with `--depth-variant repo` (as-released maps for all rooms).

## Evaluation (`sim_to_real/eval_haa.py`)
Per room, DiffRIR test split, batch 1, `--eval-seed 0` (identical references for every model/seed), Griffin-Lim (n_fft 124, 32 iters); EDT / C50 / T60 / envelope error on the full 9600-sample waveforms (as `eval_classroom.py`); T60 skipped for `dampened_room`; per-sample JSON per room. Zero-shot rows: the three inits evaluated without fine-tuning.

## Job queues (`sim_to_real/run_haa_pipeline.sh`)
GPU 0: `zeroshot:0 cyl:0 control:0 released:0 cyl:2 released_repomaps:0`; GPU 1: `control:1 cyl:1 released:1 control:2 released:2`. Environment as exp_01 (conda `xRIR`, A6000s).
