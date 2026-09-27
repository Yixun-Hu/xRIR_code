# Commands — orientation_cue_fairness (exp_11)

(Every command run for this experiment, in order; logs live next to this file.)
- 2026-09-26 17:31 Codex plan review v1: `CUDA_VISIBLE_DEVICES='' codex exec -s read-only -c 'model_reasoning_effort="xhigh"' -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check "$(cat review_prompts/plan_prompt.md)" < /dev/null` → `orientation_cue_fairness_2026-09-26_17:31:52_codex_plan_review.log` (pid 2942814).
- 2026-09-26 17:44 Probe: `codex exec -s read-only -c 'model_reasoning_effort="ultra"' … "Reply with exactly the word OK"` → header "reasoning effort: ultra", reply OK (scratchpad `codex_ultra_probe.log`): the CLI accepts ultra.
- 2026-09-26 17:44 Codex plan review round 2 at ULTRA: `CUDA_VISIBLE_DEVICES='' codex exec -s read-only -c 'model_reasoning_effort="ultra"' -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check "$(cat review_prompts/plan_round2_prompt.md)" < /dev/null` → `orientation_cue_fairness_2026-09-26_17:44:43_codex_plan_round2_review.log`.
- 2026-09-26 17:50 Coder round 1: Agent tool, model opus, prompt `coder_prompts/round1_opus_prompt.md`, worktree /home/yixunhu/codespace/xRIR_code_wt branch exp11-cue (from main c6233e3).
- 2026-09-26 20:25 Codex code review round 1 at ULTRA on the worktree (tip 5db0c82): codex exec -s read-only -c 'model_reasoning_effort="ultra"' -C /home/yixunhu/codespace/xRIR_code_wt --skip-git-repo-check "$(cat review_prompts/codex_code_round1_prompt.md)" < /dev/null → orientation_cue_fairness_2026-09-26_20:24:52_codex_code_round1_review.log (pid 3364697).
