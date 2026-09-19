# Commands — seen_protocol (exp_07)

All from the repo (or worktree) root; `python` = `/home/yixunhu/miniconda3/envs/xRIR/bin/python`; `export PYTHONPATH=$PYTHONPATH:$(pwd) XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`.

## Plan reviews (Codex, read-only) — 2026-09-15 09:59 and 10:07
```bash
codex exec -s read-only -C $(pwd) --skip-git-repo-check "$(cat worklog/worklog_yixun/exp_07_seen_protocol_claude/review_prompts/plan_prompt.md)" < /dev/null    # round 1; round 2 with plan_round2_prompt.md
```

## Coder rounds (Claude Opus 5 subagent, Agent tool model `opus`, worktree /home/yixunhu/codespace/xRIR_code_wt07, branch exp07-window) and Codex code reviews (read-only in the worktree)
```bash
git worktree add -b exp07-window /home/yixunhu/codespace/xRIR_code_wt07 HEAD; ln -s $(pwd)/ckpt /home/yixunhu/codespace/xRIR_code_wt07/ckpt; ln -s $(pwd)/checkpoints /home/yixunhu/codespace/xRIR_code_wt07/checkpoints
# Coder: Agent(model=opus) reading coder_prompts/round<N>_prompt.md; Reviewer:
codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07 --skip-git-repo-check "$(cat worklog/worklog_yixun/exp_07_seen_protocol_claude/review_prompts/code_round<N>_prompt.md)" < /dev/null   # log seen_protocol_<ts>_codex_code_round<N>.log
```

## Seen training inventory (Planner, 2026-09-15 11:45, worktree code at a035d7f)
```bash
cd /home/yixunhu/codespace/xRIR_code_wt07 && PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES='' python -c "from tools import provenance as p; p.train_data_identity('/home/yixunhu/data_cache/AcousticRooms', protocol='seen', cache_path='ckpt/exp07/train_inventory_seen.json')"
```

## Seen reference manifests (Planner, 2026-09-15 13:22, worktree code at 7b3ad6c)
```bash
cd /home/yixunhu/codespace/xRIR_code_wt07 && PYTHONPATH=$(pwd) XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED=0 python -m tools.exp07_manifests --data-root $XRIR_DATA_PATH --out-dir ckpt/exp07 --seeds 42 43 44 45 46 --num-shots 8 1   # log seen_protocol_2026-09-15_13:22:*_manifests.log
```

## seen_simple full (chain, 2026-09-17 03:45–03:50, GPU 0, reviewed 076cd40)
```
tools/exp07_launch.sh smoke --backbone simple --yaw-aug 0 --gpu 0 --reviewed-commit 076cd4072bc3d5349914cb0eff1775cb6730f829 --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260917T034540smoke
tools/exp07_launch.sh probe --backbone simple --yaw-aug 0 --gpu 0 --reviewed-commit 076cd4072bc3d5349914cb0eff1775cb6730f829 --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260917T034540probe
tools/exp07_launch.sh full  --backbone simple --yaw-aug 0 --gpu 0 --reviewed-commit 076cd4072bc3d5349914cb0eff1775cb6730f829 --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260917T034540full --probe-json ckpt/exp07/seen_simple/_probe_20260917T034540probe_seen_simple.json
```
(run inside the detached chain `gpu0_chain.sh`, env PYTHONHASHSEED=0 OMP_NUM_THREADS=2 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms)

## seen_cyl full (chain `exp07_arm_chain.sh 0 seen_cyl cylindrical 0 '2026-09-18 14:45'`, 2026-09-18 15:11–15:17, GPU 0, reviewed b793a96)
```
tools/exp07_launch.sh smoke --backbone cylindrical --yaw-aug 0 --gpu 0 --reviewed-commit b793a96d80c4ecc68df1fad2f6f3cb555bcbd27b --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260918T151139smoke
tools/exp07_launch.sh probe --backbone cylindrical --yaw-aug 0 --gpu 0 --reviewed-commit b793a96d80c4ecc68df1fad2f6f3cb555bcbd27b --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260918T151139probe
tools/exp07_launch.sh full  --backbone cylindrical --yaw-aug 0 --gpu 0 --reviewed-commit b793a96d80c4ecc68df1fad2f6f3cb555bcbd27b --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260918T151139full --probe-json ckpt/exp07/seen_cyl/_probe_20260918T151139probe_seen_cyl.json
```

## seen_aug full (chain `exp07_arm_chain.sh 1 seen_aug simple 1 '2026-09-18 20:47'`, 2026-09-18 20:52–20:57, GPU 1, reviewed b14dc3b)
```
tools/exp07_launch.sh smoke --backbone simple --yaw-aug 1 --gpu 1 --reviewed-commit b14dc3b0c56ec3c10bd5664cd828edac8c757135 --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260918T205229smoke
tools/exp07_launch.sh probe --backbone simple --yaw-aug 1 --gpu 1 --reviewed-commit b14dc3b0c56ec3c10bd5664cd828edac8c757135 --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260918T205229probe
tools/exp07_launch.sh full  --backbone simple --yaw-aug 1 --gpu 1 --reviewed-commit b14dc3b0c56ec3c10bd5664cd828edac8c757135 --log-dir worklog/worklog_yixun/exp_07_seen_protocol_claude --timestamp 20260918T205229full --probe-json ckpt/exp07/seen_aug/_probe_20260918T205229probe_seen_aug.json
```
