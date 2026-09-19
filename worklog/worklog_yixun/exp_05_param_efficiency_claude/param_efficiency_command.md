
## Confirmatory trainings (2026-09-13 12:39 →; GPU 1 queue, reviewed commit f19b9b6)
```bash
# sequential queue: S_simple → S_cylindrical → L_cylindrical (scratchpad exp05_train_queue.sh); per arm:
nohup setsid tools/exp04_launch.sh full --tier <T> --backbone <B> --gpu 1 --reviewed-commit f19b9b6 --probe-json ckpt/exp05/<T>_<B>/_probe_<TS>_<T>_<B>.json --timestamp <TS> > worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_<TS>_launcher_<T>_<B>.log 2>&1
# L_simple on GPU 0 after exp_04's evaluations
```

## Record (2026-09-18 19:48–20:03, main 73e90d4, `exp05_record_run.sh`)
```
python $A/make_results_md.py   $FLAGS --attempt $ATTEMPTS --out $W/param_efficiency_results.md
python $A/make_results_html.py $FLAGS --attempt $ATTEMPTS --out $W/param_efficiency_01_results.html
python $A/make_figures.py --curve $R/CURVE_K8.json $R/CURVE_K1.json --outdir $W/param_efficiency_figures
python $A/bind_provenance.py --runs ckpt/exp05/eval/* --attempt $ATTEMPTS --results $JSONS --rendered <md> <html> --figures $W/param_efficiency_figures/* --out ckpt/exp05/reports
python $A/check_record.py ckpt/exp05/reports    # exit 0
```
($A, $W, $R, $ATTEMPTS, $JSONS, $FLAGS as in the assets README; binding report `binding_report_20260918T234858147838Z.json`, sha256 60810502078f4bcc0a287e626d88182a1e33da50796595bb822b80847eec29ef)


# Retrospective reconstruction (2026-09-18, Planner; Codex record review finding 6)

The entries below were reconstructed AFTER the runs from the launcher-written manifests, probe receipts, queue logs and notebook, not recorded at launch time. Sources are named per entry.


## S_simple — full training (attempt `attempt_20260913T123905`, reviewed commit `f19b9b6`, GPU 1, started 2026-09-13T12:39:07.917917-04:00, ended 2026-09-14T00:02:55.088081-04:00, 11.44 h cumulative incl. probes)

Launcher-recorded trainer command (`train_manifest.json: command`), env `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONHASHSEED=0 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`:
```
/home/yixunhu/miniconda3/envs/xRIR/bin/python train_xRIR_backbone.py --backbone simple --save-dir ckpt/exp05/S_simple/attempt_20260913T123905 --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1 --vit-dim 256 --vit-depth 6 --vit-heads 4 --vit-mlp-dim 256
```

Probe receipt `_probe_20260913T102216_S_simple.json`: reviewed `f19b9b6`, passed True, T_run 11.0 h, peak reserved 11.3 GiB, GPU 1.


## S_cylindrical — full training (attempt `attempt_20260914T000333`, reviewed commit `f19b9b6`, GPU 1, started 2026-09-14T00:03:35.620926-04:00, ended 2026-09-14T12:02:44.182857-04:00, 12.03 h cumulative incl. probes)

Launcher-recorded trainer command (`train_manifest.json: command`), env `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONHASHSEED=0 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`:
```
/home/yixunhu/miniconda3/envs/xRIR/bin/python train_xRIR_backbone.py --backbone cylindrical --save-dir ckpt/exp05/S_cylindrical/attempt_20260914T000333 --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1 --vit-dim 256 --vit-depth 6 --vit-heads 4 --vit-mlp-dim 256
```

Probe receipt `_probe_20260913T102428_S_cylindrical.json`: reviewed `f19b9b6`, passed True, T_run 11.9 h, peak reserved 11.3 GiB, GPU 1.


## L_simple — full training (attempt `attempt_20260915T060543`, reviewed commit `862923e`, GPU 0, started 2026-09-15T06:05:45.894027-04:00, ended 2026-09-16T23:45:20.024645-04:00, 41.81 h cumulative incl. probes)

Launcher-recorded trainer command (`train_manifest.json: command`), env `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONHASHSEED=0 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`:
```
/home/yixunhu/miniconda3/envs/xRIR/bin/python train_xRIR_backbone.py --backbone simple --save-dir ckpt/exp05/L_simple/attempt_20260915T060543 --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1 --vit-dim 768 --vit-depth 12 --vit-heads 12 --vit-mlp-dim 768
```

Probe receipt `_probe_20260913T102644_L_simple.json`: reviewed `f19b9b6`, passed True, T_run 41.3 h, peak reserved 44.4 GiB, GPU 1.

Probe receipt `_probe_20260915T060057_L_simple.json`: reviewed `862923e`, passed True, T_run 41.4 h, peak reserved 44.4 GiB, GPU 0.


## L_cylindrical — full training (attempt `attempt_20260914T120322`, reviewed commit `f19b9b6`, GPU 1, started 2026-09-14T12:03:25.965411-04:00, ended 2026-09-16T10:08:28.400639-04:00, 46.17 h cumulative incl. probes)

Launcher-recorded trainer command (`train_manifest.json: command`), env `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONHASHSEED=0 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`:
```
/home/yixunhu/miniconda3/envs/xRIR/bin/python train_xRIR_backbone.py --backbone cylindrical --save-dir ckpt/exp05/L_cylindrical/attempt_20260914T120322 --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1 --vit-dim 768 --vit-depth 12 --vit-heads 12 --vit-mlp-dim 768
```

Probe receipt `_probe_20260913T103057_L_cylindrical.json`: reviewed `f19b9b6`, passed True, T_run 46.0 h, peak reserved 44.4 GiB, GPU 1.


Launcher invocations (notebook, 2026-09-13/15): `nohup setsid tools/exp04_launch.sh full --tier {S|L} --backbone {simple|cylindrical} --gpu {0|1} --reviewed-commit {f19b9b6|862923e} --probe-json <receipt> --timestamp <ts>` after `tools/exp04_launch.sh probe --tier … --backbone … --gpu … --reviewed-commit …`; goldens `param_efficiency_results_assets/argv_golden_{S,L}_{simple,cylindrical}.txt`.


## Evaluations (66 runs: six arms × {K = 8, K = 1} × seeds 42–46 + six seed-42 K = 8 yaw blocks)

Queue script `exp05_eval_queue.sh <gpu> <reviewed_commit> <arms…>` (scratchpad; one `tools.exp04_eval_launch --tier` child per run; binds `train_manifest`/`train_completion` for S/L, none for M). Executed: L_cylindrical + L_simple on GPU 0 2026-09-16 23:46 → 2026-09-17 03:45 (reviewed `3e2c28b`; 20 k0 runs + 2 yaw); S_simple, S_cylindrical, M_simple, M_cylindrical on GPU 1 2026-09-17 22:20 → 2026-09-18 04:30 (reviewed `ddfcb66`; 40 k0 runs + 4 yaw). One recorded child command (`eval_manifest.json: command`, S_simple K = 8 seed 42):
```
/home/yixunhu/miniconda3/envs/xRIR/bin/python /home/yixunhu/codespace/xRIR_code/tools/exp05_eval.py --backbone simple --checkpoint /home/yixunhu/codespace/xRIR_code/ckpt/exp05/S_simple/attempt_20260913T123905/epoch_012.pth --manifest /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/reference_manifest_k8_seed42.json --manifest-hash 9cf5c8a236afa4e38aa111d8658769767543389b169d6b16f7042c092d3faa10 --out-dir /home/yixunhu/codespace/xRIR_code/ckpt/exp05/eval/S_simple_k8_seed42_k0 --eval-manifest /home/yixunhu/codespace/xRIR_code/ckpt/exp05/eval/S_simple_k8_seed42_k0/eval_manifest.json --conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --batch-size 16 --num-workers 6 --max-samples 0 --gl-seed 42 --threads 4 --log-interval 10 --decomposition-batches 0 --tier S
```


## Producers and record
See the "Record" section above (2026-09-18 11:00 producers via `exp05_producers.sh`; 19:48 generators → figures → bind → check via `exp05_record_run.sh`).
