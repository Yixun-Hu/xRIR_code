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
