# Commands — yaw_aug_xrir (exp_04)

All from the repo root with `export PYTHONPATH=$PYTHONPATH:$(pwd); export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; `python` = `/home/yixunhu/miniconda3/envs/xRIR/bin/python`. Base commit `8cb87d1e0e8cbd85913911fdefd64aa69ced931d`.

## Reference manifests for the five evaluation seeds (2026-09-12 15:07, base `8cb87d1`)
```bash
python - <<'PY'
from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
from tools.reference_manifest import build_manifest, save_manifest, manifest_hash
for K in (8, 1):
    ds = xRIR_Dataset(split='test', max_len=9600, num_shot=K)
    for seed in (42, 43, 44, 45, 46):
        m = build_manifest(ds, seed=seed, num_shot=K); save_manifest(m, f'ckpt/yaw_aug/reference_manifest_k{K}_seed{seed}.json'); print(K, seed, manifest_hash(m))
PY
# index with semantic hashes and file sha256: ckpt/yaw_aug/reference_manifests_index.json
```

## Coder round 1 (2026-09-12 15:05:25, Codex gpt-6-astra)
```bash
codex exec -s workspace-write --skip-git-repo-check "$(cat worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/coder_prompts/round1_prompt.md)" < /dev/null   # log yaw_aug_xrir_2026-09-12_15:05:25_coder_round1.log
```
codex exec … round1b_prompt.md   # log yaw_aug_xrir_2026-09-12_15:19:31_coder_round1b.log
codex exec … coder_prompts/round2_prompt.md   # log yaw_aug_xrir_2026-09-12_15:43:21_coder_round2.log

## Coder round 2 bounded audit smoke (2026-09-12 15:50)
```bash
export PYTHONPATH=$PYTHONPATH:/home/yixunhu/codespace/xRIR_code
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=2 NUMBA_CACHE_DIR=/tmp/xrir_numba_cache
/home/yixunhu/miniconda3/envs/xRIR/bin/python -m tools.yaw_aug --audit --n-batches 3 --batch-size 32 --seed 0 --out ckpt/yaw_aug/_audit_smoke.json
```
Log: `yaw_aug_xrir_2026-09-12_15:50:01_audit_smoke.log`.

## Coder round 2 full test suite
Same exports as the bounded audit above.
```bash
/home/yixunhu/miniconda3/envs/xRIR/bin/python -m pytest tests -q -p no:cacheprovider
```
Log: `/tmp/exp04_round2_full_suite.log`.

## 2026-09-12T16:34:11-04:00 — round-2 close-out rung-5 smoke (GPU 1)

```bash
export PYTHONPATH=$PYTHONPATH:/home/yixunhu/codespace/xRIR_code
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=2 NUMBA_CACHE_DIR=/tmp/xrir_numba_cache
/home/yixunhu/miniconda3/envs/xRIR/bin/python train_xRIR_backbone.py --backbone simple --save-dir /tmp/exp04_closeout_rung5/smoke --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --accum-steps 2 --num-workers 4 --seed 0 --tf32 --log-interval 1 --save-every 0 --epoch-ckpt-every 0 --no-save --yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512
```

Log: `/tmp/exp04_closeout_rung5.log` (also copied to this record folder after completion).

## 2026-09-12T16:38:57-04:00 — close-out final full suite

```bash
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=$PYTHONPATH:/home/yixunhu/codespace/xRIR_code:/tmp/exp04_closeout_gpu1
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export PYTHONHASHSEED=0 OMP_NUM_THREADS=2 NUMBA_CACHE_DIR=/tmp/xrir_numba_cache
/home/yixunhu/miniconda3/envs/xRIR/bin/python -m pytest tests -q -p no:cacheprovider
```

`/tmp/exp04_closeout_gpu1/sitecustomize.py` selects `torch.cuda.set_device(1)` while retaining both visible GPUs for the exp_03 record check. Log: `/tmp/exp04_closeout_full_suite.log`.
