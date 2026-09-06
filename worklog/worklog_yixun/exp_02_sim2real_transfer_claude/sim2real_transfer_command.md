# Commands — sim2real_transfer

From the repo root, `export PYTHONPATH=$PYTHONPATH:$(pwd)`, conda `xRIR`.

## Cache build and checks (2026-09-05 16:05–16:12)
```bash
python sim_to_real/prepare_haa.py --out ~/data_cache/HAA_xrir      # writes rirs/xyzs/speaker_xyz/depth{,_repo,_local}.npy + meta.json per room
python sim_to_real/haa_dataset.py                                   # split sizes 48 / 560 / 1282, shapes, deterministic refs
# one-off checks (scratch): onset-vs-distance correlation per room; repo-vs-local depth horizon profiles; array_equal(complex_room.npy, hallway.npy) == True
```

## Smoke tests (2026-09-05 16:15, shared GPU 1)
```bash
python sim_to_real/finetune_haa.py --backbone simple --init checkpoints/xRIR_unseen.pth --rooms class_room hallway complex_room --save-dir ckpt/_s2r_smoke/stage1 --epochs 2 --val-every 1 --batch-size 4 --accum-steps 9 --val-batch-size 4 --tf32
python sim_to_real/finetune_haa.py --backbone cylindrical --init ckpt/xRIR_cyl_8_shot/epoch_09.pth --rooms dampened_room --save-dir ckpt/_s2r_smoke/stage2 --epochs 2 --val-every 1 --batch-size 4 --accum-steps 3 --val-batch-size 4 --tf32
python sim_to_real/eval_haa.py --backbone simple --checkpoint ckpt/_s2r_smoke/stage1/best.pth --rooms class_room dampened_room --max-samples 5 --save-dir ckpt/_s2r_smoke/evalA
python tools/compare_eval.py --a ckpt/_s2r_smoke/evalB/per_sample_dampened_room.json --b ckpt/_s2r_smoke/evalA/per_sample_dampened_room.json --n-boot 500
python sim_to_real/summarize_haa.py --root ckpt/_s2r_smoke_root --n-boot 300
```

## Launch (staged 2026-09-05 16:23; started automatically 21:01:39 after both exp_01 epoch-12 evals finished)
```bash
# launcher (scratchpad launch_sim2real.sh) waits for ckpt/xRIR_{cyl,simple}_8_shot/{per_sample_unseen_epoch12.json,epoch_12.pth}, then:
nohup sim_to_real/run_haa_pipeline.sh 0 zeroshot:0 cyl:0 control:0 released:0 cyl:2 released_repomaps:0 > ckpt/sim2real/queue_gpu0.log 2>&1 &
nohup sim_to_real/run_haa_pipeline.sh 1 control:1 cyl:1 released:1 control:2 released:2                > ckpt/sim2real/queue_gpu1.log 2>&1 &
# each job: stage1 -> stage2 x 4 rooms -> eval x 4 rooms, e.g. for cyl seed 0:
python sim_to_real/finetune_haa.py --backbone cylindrical --init ckpt/xRIR_cyl_8_shot/epoch_12.pth --rooms class_room hallway complex_room --save-dir ckpt/sim2real/cyl/seed0/stage1 --seed 0 --epochs 1000 --val-every 10 --tf32
python sim_to_real/finetune_haa.py --backbone cylindrical --init ckpt/sim2real/cyl/seed0/stage1/best.pth --rooms class_room --save-dir ckpt/sim2real/cyl/seed0/stage2_class_room --seed 0 --epochs 200 --val-every 2 --tf32
python sim_to_real/eval_haa.py --backbone cylindrical --checkpoint ckpt/sim2real/cyl/seed0/stage2_class_room/best.pth --rooms class_room --save-dir ckpt/sim2real/cyl/seed0/eval
# zero-shot: python sim_to_real/eval_haa.py --backbone <bb> --checkpoint <init> --rooms class_room dampened_room hallway complex_room --save-dir ckpt/sim2real/<init>/zeroshot
# summary when both queues finish:
python sim_to_real/summarize_haa.py --root ckpt/sim2real --json ckpt/sim2real/stats.json --summary ckpt/sim2real/summary.txt   # final; refuses a partial run set
python sim_to_real/summarize_haa.py --root ckpt/sim2real --allow-partial --json ckpt/sim2real/stats.json --summary ckpt/sim2real/summary.txt   # interim DRAFT artefacts (2026-09-05 23:4x–00:0x, 2 seeds)
python worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py   # page from stats.json (DRAFT-watermarked while incomplete)
```
