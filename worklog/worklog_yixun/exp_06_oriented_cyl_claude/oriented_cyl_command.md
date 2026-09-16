# oriented_cyl (exp_06) — reproduction commands (appended at launch time)

## Planner probes (CPU, informational; superseded by reviewed tools)
- 2026-09-14 20:4x `OMP_NUM_THREADS=4 ~/miniconda3/envs/xRIR/bin/python oriented_cyl_results_assets/planner_probes/heading_rule_check.py` — §4 discrete heading rule on the four HAA rooms (training rows only).

## Plan reviews (Codex, read-only)
- 2026-09-14 20:27 `codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check "$(cat review_prompts/plan_prompt.md)" < /dev/null` → `oriented_cyl_2026-09-14_20:27:02_codex_plan_review.log`, review `oriented_cyl_codex_plan_review.md`.
- 2026-09-14 20:39 same with `review_prompts/plan_round2_prompt.md` → `oriented_cyl_2026-09-14_20:39:27_codex_plan_review_round2.log`.
- 2026-09-14 20:51 same with `review_prompts/plan_round3_prompt.md` → `oriented_cyl_2026-09-14_20:51:46_codex_plan_review_round3.log`.

## Coder rounds (Codex gpt-6-astra, workspace-write, worktree /home/yixunhu/codespace/xRIR_code_wt, branch exp06-window)
- 2026-09-14 20:58 round 1: `cd /home/yixunhu/codespace/xRIR_code_wt && codex exec -s workspace-write -C /home/yixunhu/codespace/xRIR_code_wt --skip-git-repo-check "$(cat coder_prompts/round1_prompt.md)" < /dev/null` → `oriented_cyl_2026-09-14_20:58:54_coder_round1.log`.
- 2026-09-15 00:04 round 2a: same form with `coder_prompts/round2a_prompt.md` → `oriented_cyl_2026-09-15_00:04:38_coder_round2a.log`.
- 2026-09-15 00:13 round 2a (Codex) STOPPED at 10 commits by the Coder role change; archived → `worklog/worklog_yixun/archive_codex_round2a_superseded_2026-09-15/`, branch `archive/codex-round2a-2026-09-15`.
- 2026-09-15 00:13 round 2a (Opus 5): Agent tool, model `opus`, prompt `coder_prompts/round2a_opus_prompt.md` (worktree branch `exp06-window` from `a950605`); report → `oriented_cyl_<ts>_coder_round2a_opus_report.md`.
- 2026-09-15 01:00 Codex code review of round 2a: `codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code_wt …` with the briefing + `review_prompts/codex_code_round2a_prompt.md` → `oriented_cyl_2026-09-15_01:00:10_codex_code_round2a_review.log`.
- 2026-09-15 02:14 Codex close review of round 2a fixes (4f92f54..2fe25bf) → `oriented_cyl_2026-09-15_02:14:34_codex_code_round2a_close_review.log`.
- 2026-09-15 03:40 Codex close review 2 of round 2a (2fe25bf..bf51484) → `oriented_cyl_2026-09-15_03:40:08_codex_code_round2a_close2_review.log`.
- 2026-09-15 04:38 Codex close review 3 of round 2a (bf51484..a977a68) → `oriented_cyl_2026-09-15_04:38:18_codex_code_round2a_close3_review.log`.
- 2026-09-15 05:09 Codex close review 4 of round 2a (a977a68..12e06fc) → `oriented_cyl_2026-09-15_05:09:41_codex_code_round2a_close4_review.log`.
- 2026-09-15 05:19 Codex `full_train` integrative review (rounds 1 + 2a, tip 12e06fc) → `oriented_cyl_2026-09-15_05:19:04_codex_code_full_train_review.log`.
- 2026-09-15 05:20 round 2b (Opus 5): Agent tool, model `opus`, prompt `coder_prompts/round2b_opus_prompt.md` (worktree /home/yixunhu/codespace/xRIR_code_wt2b, branch `exp06-round2b` from `12e06fc`); report → `oriented_cyl_<ts>_coder_round2b_opus_report.md`.
- 2026-09-15 05:40 training-gate fix (Opus 5): Agent tool, prompt `coder_prompts/round2a_gate_fix_opus_prompt.md` (worktree /home/yixunhu/codespace/xRIR_code_wt, branch `exp06-window` from `12e06fc`); report → `oriented_cyl_<ts>_coder_round2a_gate_fix_opus_report.md`.
- 2026-09-15 06:28 Codex code review of round 2b (12e06fc..d6a4f42, worktree wt2b) → `oriented_cyl_2026-09-15_06:28:41_codex_code_round2b_review.log`.
- 2026-09-15 07:31 Codex close review of round 2b fixes (d6a4f42..3df20b8, wt2b) → `oriented_cyl_2026-09-15_07:31:17_codex_code_round2b_close_review.log`.
- 2026-09-15 07:32 Codex `full_train` review 2 (rounds 1 + 2a + gate fix, tip f18383a) → `oriented_cyl_2026-09-15_07:32:23_codex_code_full_train2_review.log`.
- 2026-09-15 07:49 round 3a (Opus 5): Agent tool, prompt `coder_prompts/round3a_opus_prompt.md` (wt2b, branch `exp06-round2b` from `3df20b8`); report → `oriented_cyl_<ts>_coder_round3a_opus_report.md`.
- 2026-09-15 08:43 Codex `full_train` review 3 (tip a313253) → `oriented_cyl_2026-09-15_08:43:17_codex_code_full_train3_review.log`.
- 2026-09-15 08:53 Codex code review of round 3a (3df20b8..895f512, wt2b) → `oriented_cyl_2026-09-15_08:53:08_codex_code_round3a_review.log`.
- 2026-09-15 09:09 round 3a fix (Opus 5): Agent tool, prompt `coder_prompts/round3a_fix_opus_prompt.md` (wt2b, `exp06-round2b` from `895f512`).
- 2026-09-15 09:59 Codex close review of round 3a fixes (895f512..7f29877, wt2b) → `oriented_cyl_2026-09-15_09:59:28_codex_code_round3a_close_review.log`.
- 2026-09-15 10:09 round 3b (Opus 5): Agent tool, prompt `coder_prompts/round3b_opus_prompt.md` (wt2b, `exp06-round2b` from `7f29877`).
- 2026-09-15 11:09 Codex code review of round 3b (7f29877..0ca806c, wt2b) → `oriented_cyl_2026-09-15_11:08:57_codex_code_round3b_review.log`.
- 2026-09-15 11:11 Codex pre-merge review of `exp06-window` at a40f81b → `oriented_cyl_2026-09-15_11:11:12_codex_code_premerge_review.log`.
- 2026-09-15 12:10 Codex pre-merge CLOSE review (a40f81b..5091f6c) → `oriented_cyl_2026-09-15_12:10:10_codex_code_premerge_close_review.log`.
- 2026-09-15 12:11 round 3b fix (Opus 5): Agent tool, prompt `coder_prompts/round3b_fix_opus_prompt.md` (wt2b; cycle −1 merges `exp06-window` into `exp06-round2b`).
- 2026-09-15 14:00 Codex close review of round 3b fixes (0ca806c..df7ea24 incl. merge 90ce3ba, wt2b) → `oriented_cyl_2026-09-15_14:00:27_codex_code_round3b_close_review.log`.
- 2026-09-15 15:09 heading JSONs (rung 3): for r in class_room dampened_room hallway complex_room: `python tools/exp06_heading.py --room-dir ~/data_cache/HAA_xrir/$r --out ckpt/exp06/heading/$r.json` (HEAD 53307b6, CPU).
- 2026-09-15 15:10 Codex verification of the approvals fill + heading records (main, 53307b6) → `oriented_cyl_2026-09-15_15:10:13_codex_approvals_fill_review.log`.
- 2026-09-15 15:42 Codex close review 2 of round 3b (df7ea24..07001a0, wt2b) → `oriented_cyl_2026-09-15_15:42:25_codex_code_round3b_close2_review.log`.
- 2026-09-15 15:48 Codex review of the launch-gate test fix (4dd37ec on exp06-launch-gate) → `oriented_cyl_2026-09-15_15:48:20_codex_launch_gate_test_review.log`.
- 2026-09-15 16:29 Codex close review of the launch-gate test fix (4dd37ec..4ea1b2b) → `oriented_cyl_2026-09-15_16:29:03_codex_launch_gate_test_close_review.log`.
- 2026-09-15 17:47 Codex close review 3 of round 3b (07001a0..6ac832c, wt2b) → `oriented_cyl_2026-09-15_17:47:15_codex_code_round3b_close3_review.log`.
- 2026-09-16 08:05 Codex verification of the approvals re-fill at b9f6ebe → `oriented_cyl_2026-09-16_08:05:36_codex_approvals_refill_review.log`.
- 2026-09-16 08:17 Codex review of the launch-day runbook (Planner one-off) → `oriented_cyl_2026-09-16_08:17:08_codex_runbook_review.log`.
- 2026-09-16 08:26 Codex close review of the runbook v2 + A4 → `oriented_cyl_2026-09-16_08:26:45_codex_runbook_close_review.log`.
- 2026-09-16 08:34 Codex close review 2 of the runbook (v3) → `oriented_cyl_2026-09-16_08:34:06_codex_runbook_close2_review.log`.
- 2026-09-16 11:23 runbook `smoke` step (GPU 1, co-tenant with FLAC exp_23): `GPU=1 bash oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh smoke`.
- 2026-09-16 11:26 pre-launch round (Opus 5): prompt `coder_prompts/prelaunch_round_opus_prompt.md` (wt, branch `exp06-prelaunch` from 515136f).
- 2026-09-16_11:51:09 co-tenant measurement (exploratory, no-save, GPU 1 shared with FLAC exp_23 pid 2407803): tools/exp06_smoke.py --entry exp06_train --run-type probe --exploratory --alarm-seconds 2400 --max-gb 42 -- --backbone cylindrical_oriented --epochs 1 --max-train-batches 100 --max-test-batches 5 --batch-size 32 --accum-steps 2 --tf32 --num-workers 12 --log-interval 10 --no-save --run-type probe → `oriented_cyl_2026-09-16_11:51:09_cotenant_probe_gpu1.log`
- 2026-09-16 11:59 rung-4 smokes as exploratory diagnostics (A6): `GPU=1 bash oriented_cyl_results_assets/planner_probes/rung4_exploratory.sh`.
- 2026-09-16 12:08 rung 5: `GPU=1 bash oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh probe` (main 6dc0b8e).
- 2026-09-16 12:15 rung 7 FULL: `GPU=1 bash oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh full` → launcher `tools/exp06_launch.sh full --gpu 1 --reviewed-commit 6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113`; attempt `attempt_20260916T161415`.
- 2026-09-16 12:57 Codex review of the pre-launch round (515136f..8bc7cf8, wt, branch exp06-prelaunch) → `oriented_cyl_2026-09-16_12:57:28_codex_code_prelaunch_review.log`.
- 2026-09-16 13:53 Codex close review of the pre-launch round fix (8bc7cf8..c867e24) → `oriented_cyl_2026-09-16_13:53:07_codex_code_prelaunch_close_review.log`.
- 2026-09-16 14:56 training watch (Planner Monitor, 60-s cadence, 30-min re-arm): trainer pid 2633255 alive → `history.jsonl` rows → failure signatures (`Traceback|Error|error:|OOM|out of memory|Killed|loss nan|loss inf|EXP06_CHILD_EXIT|ABORT`, excluding the `UserWarning`/`nan samp/s` false positives) → foreign compute processes on GPU 1 (uuid …684b9600).
- 2026-09-16 15:06 post-training runbook (new, one step per invocation): `bash oriented_cyl_results_assets/planner_probes/posttrain_runbook.sh <status|merge|refill|receipt|g1|haa [jobs]|sim|summarize>` (`GPU=1` default; `merge` needs `PEER_ACK=1`); logs `oriented_cyl_<ts>_posttrain_<step>.log`. `status` exercised live at 15:06 → exit 3 (launcher alive, epoch 2 at 750/9261).
- Planned Codex reviews after the merge (read-only, `codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check "$(cat review_prompts/codex_code_review_briefing.md) $(cat review_prompts/<prompt>)" < /dev/null`): `codex_approvals_posttrain_prompt.md` (the two approvals commits) and `codex_code_full_prompt.md` (A1's integrative `full` review over 77ef292..HEAD, gating G1/HAA/sim). Placeholders `<SHA_MERGE>`, `<SHA_REFILL>`, `<SHA_RECEIPT>`, `<SHA_FULL>` are filled at run time.
