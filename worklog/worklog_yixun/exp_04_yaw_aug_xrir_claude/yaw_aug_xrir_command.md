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
