# Commands — cylvit_vs_simplevit

All from the repo root with `conda activate xRIR` (or `~/miniconda3/envs/xRIR/bin/python`) and `export PYTHONPATH=$PYTHONPATH:$(pwd)`.

## Baseline reproduction (2026-09-04 12:19, both on the CIFS share; ~85 min each)
```bash
CUDA_VISIBLE_DEVICES=1 nohup python eval_unseen.py > ckpt/baseline_eval_logs/eval_unseen.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval_seen.py   > ckpt/baseline_eval_logs/eval_seen.log   2>&1 &
```

## Verification before training (2026-09-04 12:29–12:33)
```bash
CUDA_VISIBLE_DEVICES=1 python model/xRIR_cyl.py                  # forward both backbones: params / shapes / finite
python train_xRIR_backbone.py --backbone cylindrical --save-dir ckpt/_smoke --epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --num-workers 4 --save-every 0
python eval_xRIR_backbone.py --backbone simple --split unseen --checkpoint checkpoints/xRIR_unseen.pth --max-samples 5 --num-workers 2
# equivariance check (scratchpad script, roll by one azimuth patch): cyl err 1.9e-6, simple err 2.0
```

## FAILED first training launch (2026-09-04 12:34, superseded — stalled on CIFS, killed 12:40)
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 nohup python train_xRIR_backbone.py --backbone cylindrical --save-dir ckpt/xRIR_cyl_8_shot --num-shot 8 --batch-size 32 --accum-steps 2 --tf32 --epochs 12 --decay-epochs 3 --num-workers 16 --log-interval 50 --save-every 500 --seed 0 &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 nohup python train_xRIR_backbone.py --backbone simple      --save-dir ckpt/xRIR_simple_8_shot ...same flags... &
```

## Local data mirror (2026-09-04 12:41–13:24) and verification
```bash
# 30 rsync jobs, 6 in parallel: /media/diskstation/yixunhu/FLAC/AcousticRooms/{single_channel_ir_1,depth_map,metadata}/<category>/ -> ~/data_cache/AcousticRooms/{single_channel_ir,depth_map,metadata}/<category>/
rsync -rlt --no-perms --no-owner --no-group SRC/ DST/          # per (dir, category)
rsync -n -rlti --no-perms --no-owner --no-group --stats SRC/ DST/   # dry run: 0 files to transfer (only a .DS_Store)
XRIR_DATA_PATH=~/data_cache/AcousticRooms python -c "...xRIR_Dataset split sizes: 296334/6337 unseen, 6217 seen; loader 873 samples/s"
```

## Training (2026-09-04 13:25:01, the runs reported in results)
```bash
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
COMMON="--num-shot 8 --batch-size 32 --accum-steps 2 --tf32 --epochs 12 --decay-epochs 3 --num-workers 12 --log-interval 50 --save-every 500 --seed 0"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 nohup python train_xRIR_backbone.py --backbone cylindrical --save-dir ckpt/xRIR_cyl_8_shot    $COMMON > ckpt/xRIR_cyl_8_shot/train.log    2>&1 &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 nohup python train_xRIR_backbone.py --backbone simple      --save-dir ckpt/xRIR_simple_8_shot $COMMON > ckpt/xRIR_simple_8_shot/train.log 2>&1 &
```

## Per-epoch paired evaluation (run automatically after each epoch, N = 01..12)
```bash
# snapshot: extract that epoch's weights from last.pth as soon as history.jsonl logs the epoch -> ckpt/xRIR_{cyl,simple}_8_shot/epoch_NN.pth
CUDA_VISIBLE_DEVICES=0 python eval_xRIR_backbone.py --backbone cylindrical --split unseen --checkpoint ckpt/xRIR_cyl_8_shot/epoch_NN.pth    --num-workers 6 --threads 4 --log-interval 1000 --save-metrics ckpt/xRIR_cyl_8_shot/eval_unseen_epochNN.json    --save-per-sample ckpt/xRIR_cyl_8_shot/per_sample_unseen_epochNN.json
CUDA_VISIBLE_DEVICES=1 python eval_xRIR_backbone.py --backbone simple      --split unseen --checkpoint ckpt/xRIR_simple_8_shot/epoch_NN.pth --num-workers 6 --threads 4 --log-interval 1000 --save-metrics ckpt/xRIR_simple_8_shot/eval_unseen_epochNN.json --save-per-sample ckpt/xRIR_simple_8_shot/per_sample_unseen_epochNN.json
python tools/compare_eval.py --a ckpt/xRIR_cyl_8_shot/per_sample_unseen_epochNN.json --b ckpt/xRIR_simple_8_shot/per_sample_unseen_epochNN.json
```
Note: epoch 01 used `best_epoch01.pth` (= epoch-1 weights); epoch 02's control snapshot was recovered from `best.pth` (its epoch-2 weights) after the first pipeline version missed the `last.pth` window; epoch 08's compare step was re-run by hand after the pipeline script was edited while running.

## Seen split, epoch 9 (2026-09-05 14:59)
```bash
CUDA_VISIBLE_DEVICES=0 python eval_xRIR_backbone.py --backbone cylindrical --split seen --checkpoint ckpt/xRIR_cyl_8_shot/epoch_09.pth    --num-workers 6 --threads 4 --save-metrics ckpt/xRIR_cyl_8_shot/eval_seen_epoch09.json    --save-per-sample ckpt/xRIR_cyl_8_shot/per_sample_seen_epoch09.json
CUDA_VISIBLE_DEVICES=1 python eval_xRIR_backbone.py --backbone simple      --split seen --checkpoint ckpt/xRIR_simple_8_shot/epoch_09.pth --num-workers 6 --threads 4 --save-metrics ckpt/xRIR_simple_8_shot/eval_seen_epoch09.json --save-per-sample ckpt/xRIR_simple_8_shot/per_sample_seen_epoch09.json
python tools/compare_eval.py --a ckpt/xRIR_cyl_8_shot/per_sample_seen_epoch09.json --b ckpt/xRIR_simple_8_shot/per_sample_seen_epoch09.json --by-test-rooms
```

## Final summary (2026-09-05 21:05)
```bash
python tools/summarize_epochs.py --avg-epochs 5-12 --json ckpt/backbone_comparison_stats.json --summary ckpt/backbone_comparison_summary.txt   # canonical stats + hash-bound summary (2026-09-06 00:0x rerun after review round 3; earlier runs: 21:05 without --json, 23:4x with --json only)
python worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py           # results page from the canonical JSON
```
