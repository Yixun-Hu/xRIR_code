# exp_10 yaw_pilot — commands (added at launch time)

## 2026-09-26T20:18:06-04:00 — released_k8 chain on CPU (probe + controls → check-online → parity vs exp_03 → gate → full)
Reviewed/merged commit: `2408fb9` (tools), launched from main `fc644f5`.
```bash
nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_arm_chain.sh \
  released_k8 simple checkpoints/xRIR_unseen.pth ckpt/yaw_rotation/reference_manifest.json \
  47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d 8 cpu 1 \
  ckpt/yaw_rotation/sweep_released/per_sample_yaw.json > /dev/null 2>&1 &
```
Outputs: `ckpt/exp10/released_k8_probe/` (+ `check_online.json`, `parity_exp03.json`, `summary/`), then `ckpt/exp10/released_k8_all/`. Log: `yaw_pilot_<timestamp>_released_k8_chain_cpu.log`.

## 2026-09-26T22:37:57-04:00 — GPU 1 queue: released_k1 (controls only) → control_k8 → cyl_k8, each probe → gate → full
Launched from main `99abf77` (tools at `2408fb9`), GPU 1 co-tenant (cylindrical-dinov3 ping at 21:4x: 21 160 MiB free).
```bash
nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_gpu_queue.sh 1 > /dev/null 2>&1 &
```
Per-arm commands are those of `exp10_arm_chain.sh` with `cuda 1`; K = 1 manifest hash `f6d71f86d5d313f2a982fe6f8cb801116a20ea1c37b0a65d290c088934fd73e3`. Logs: `yaw_pilot_<ts>_gpu1_queue.log` + one `_<arm>_chain_cuda.log` per arm.
- 22:39 re-queue of `released_k1` after the queue (v1 queue bug: empty K = 1 hash): `nohup setsid …/exp10_released_k1_after_queue.sh 1 <queue log> &` → runs the arm chain with `cuda 1` and the hard-coded K = 1 hash once the queue log says QUEUE DONE.

## 2026-09-27T03:28:05-04:00 — released_k8 re-run on GPU 1 (primary backend); CPU run moved to `ckpt/exp10/cpu_protocol/released_k8_{probe,all}` (meta stores no path; identity is by hashes)
Launched from main `b47ff59`.
```bash
nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_arm_chain.sh \
  released_k8 simple checkpoints/xRIR_unseen.pth ckpt/yaw_rotation/reference_manifest.json \
  47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d 8 cuda 1 \
  ckpt/yaw_rotation/sweep_released/per_sample_yaw.json > /dev/null 2>&1 &
```
- 04:38 CPU-protocol derived reports regenerated in place after the directory move (old files kept as `*_premove.json`): `python tools/exp10_compare.py check-online ckpt/exp10/cpu_protocol/released_k8_all --json …/check_online.json`; `python tools/exp10_compare.py parity-exp03 ckpt/exp10/cpu_protocol/released_k8_all ckpt/yaw_rotation/sweep_released/per_sample_yaw.json --json …/parity_exp03.json` (log `yaw_pilot_2026-09-27_04:38:18_cpu_protocol_reports_regen.log`).

## 2026-09-27T07:49:17-04:00 — end-game (tooling merged at `415268d`)
```bash
nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_finish.sh > /dev/null 2>&1 &
```

## Complete command register (added 2026-09-27 after the record review; entries marked **[R]** were reconstructed from the worklog and the run logs, not captured at launch time — see the worklog for why)

Common environment for every command below unless stated: `cd /home/yixunhu/codespace/xRIR_code`; `conda activate xRIR` (Python 3.8.20, torch 2.0.1+cu117, numpy 1.23.5); `export PYTHONPATH=${PYTHONPATH:-}:$(pwd)`; `export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; `CUDA_VISIBLE_DEVICES=""` for CPU runs, `=1` for GPU 1 runs (set inside `exp10_arm_chain.sh` from its `device`/`gpu` arguments). Tools: `tools/exp10_{yaw_pilot,compare,summarize}.py` as merged at `2408fb9` (unchanged for every run; `tool_sha256` in each `meta.json`). Chain/queue scripts: `yaw_pilot_results_assets/scripts/` at the main HEAD named per entry.

1. **[R] 2026-09-26 20:20 — first released_k8 CPU chain launch, FAILED before its first log line.** Same command as entry 2, chain script v1 (`git` HEAD `9a9de52`). Cause: `set -u` with `PYTHONPATH` unset in the detached shell — to reproduce the failure, launch with `PYTHONPATH` UNSET (the common environment above does not apply to this entry). Its stdout/stderr were redirected to `/dev/null` by the launch wrapper, so **no terminal output of this attempt exists**; the only evidence is the absence of any log/artefact and the worklog entry at 20:20. Not backdated; recorded here as unrecoverable.
2. **2026-09-26 20:21 — released_k8 CPU chain (retry, chain script v1.1, HEAD `ee31cbc`):**
   `nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_arm_chain.sh released_k8 simple checkpoints/xRIR_unseen.pth ckpt/yaw_rotation/reference_manifest.json 47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d 8 cpu 1 ckpt/yaw_rotation/sweep_released/per_sample_yaw.json > /dev/null 2>&1 &`
   → `ckpt/exp10/released_k8_probe` (gate passed 20:40) → `ckpt/exp10/released_k8_all` (done 03:26); log `yaw_pilot_2026-09-26_20:21:22_released_k8_chain_cpu.log`. Both directories were later moved to `ckpt/exp10/cpu_protocol/` (entry 9).
3. **2026-09-26 22:37 — GPU 1 queue (queue script v1, HEAD `0a770d1`):**
   `nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_gpu_queue.sh 1 > /dev/null 2>&1 &`
   → arm `released_k1` **FAILED at 22:38** (empty K = 1 manifest hash: the queue computed it with the system `python3`; log `yaw_pilot_2026-09-26_22:38:03_released_k1_chain_cuda_ABORTED_k1_hash_bug.log`; no artefact written) → `control_k8` probe/full (22:38–01:04, log `…_22:38:09_control_k8_chain_cuda.log`) → `cyl_k8` probe/full (01:04–02:24, log `…_01:04:35_cyl_k8_chain_cuda.log`); queue log `yaw_pilot_2026-09-26_22:37:58_gpu1_queue.log`.
4. **2026-09-26 22:4x — queue script v1.1 (hard-coded K = 1 hash) + released_k1 re-queue (HEAD `c737b94`):**
   `nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_released_k1_after_queue.sh 1 worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_2026-09-26_22:37:58_gpu1_queue.log > /dev/null 2>&1 &`
   → waits for `QUEUE DONE`, then the arm chain `released_k1 simple checkpoints/xRIR_unseen.pth ckpt/yaw_rotation/reference_manifest_k1.json f6d71f86d5d313f2a982fe6f8cb801116a20ea1c37b0a65d290c088934fd73e3 1 cuda 1 none` (02:25–03:14; log `yaw_pilot_2026-09-27_02:25:07_released_k1_chain_cuda.log`).
5. **[R] Coder-round smokes (developmental; NOT the production implementation; run by the Coder in the worktree; reconstructed from the Coder reports and the retained scratch artefacts).** Environment for every line of this entry (overrides the common environment): `cd /home/yixunhu/codespace/xRIR_code_wt10 && conda activate xRIR && export PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt10 && export CUDA_VISIBLE_DEVICES="" && export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; `SCRATCH` = `/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad`.
   - Round 1 (worktree commit `222f1fe`, dirty tree, evaluator sha256 `1618de6daa1648afd8f2b8b8ad20a250a3b93817d8452b3e0db071180a19b341`, execution `20260926T223309764572Z-3ace4bc5181644ff8e07414bb9ca9e81`):
     `python tools/exp10_yaw_pilot.py --arm smoke --backbone simple --checkpoint checkpoints/xRIR_unseen.pth --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash 47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d --num-shot 8 --ks 0,64,128,256,384 --batches 0 --controls --device cpu --threads 16 --out-dir SCRATCH/exp10_smoke` → `SCRATCH/exp10_smoke/{meta,metrics,per_sample}.json`, `wav_k*.npy`;
     `python tools/exp10_compare.py check-online SCRATCH/exp10_smoke --json SCRATCH/check_online.json` → `/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad/check_online.json`;
     `python tools/exp10_compare.py parity-exp03 SCRATCH/exp10_smoke ckpt/yaw_rotation/sweep_released/per_sample_yaw.json --json SCRATCH/parity.json` → `/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad/parity.json`;
     `python tools/exp10_summarize.py --runs SCRATCH/exp10_smoke --out SCRATCH/exp10_smoke_summary` (default 10 000 draws) → `SCRATCH/exp10_smoke_summary/yaw_pilot_summary.json` (n_boot 10 000).
   - Round 2 (worktree commit `d3e927b`, dirty tree, evaluator sha256 `e727cba016dcc2d78c7ba193fca85b75116c40f9e9d567be784970a2974c8823` = the production evaluator, execution `20260926T233632447114Z-98921ae86a804928af8e3b11dc57c819`):
     the same evaluator command with `--out-dir SCRATCH/exp10_smoke_r2` → `SCRATCH/exp10_smoke_r2/{meta,metrics,per_sample}.json`, `wav_k*.npy`;
     `python tools/exp10_compare.py check-online SCRATCH/exp10_smoke_r2 --json SCRATCH/exp10_smoke_r2/check_online.json`;
     `python tools/exp10_compare.py parity-exp03 SCRATCH/exp10_smoke_r2 ckpt/yaw_rotation/sweep_released/per_sample_yaw.json --json SCRATCH/exp10_smoke_r2/parity.json`;
     `python tools/exp10_summarize.py --runs SCRATCH/exp10_smoke_r2 --out SCRATCH/exp10_smoke_r2/summary --n-boot 2000` → `SCRATCH/exp10_smoke_r2/summary/yaw_pilot_summary.json` (n_boot 2 000).
   - Round 3 (worktree commit `1ebb87d`; figure code only): `python tools/exp10_summarize.py --runs SCRATCH/exp10_smoke_r2 --out SCRATCH/exp10_smoke_r3_summary --n-boot 2000` → `SCRATCH/exp10_smoke_r3_summary/yaw_pilot_summary.json` (n_boot 2 000).
   The numbers of these smokes are quoted in the Codex code reviews; the "unchanged for every run" implementation statement of the common environment applies to the production runs (entries 2–4, 7, 10, 11), not to these developmental smokes.
6. **[R] 2026-09-27 03:30 — interim summaries (superseded by the finish; main HEAD `b47ff59`, tools at `2408fb9`):** `nice -n 5 python tools/exp10_summarize.py --runs ckpt/exp10/cpu_protocol/released_k8_all --out ckpt/exp10/cpu_protocol/summary` (log `…/scratchpad/e10_sum_cpu.log`, exit 0) and `nice -n 5 python tools/exp10_summarize.py --runs ckpt/exp10/control_k8_all ckpt/exp10/cyl_k8_all ckpt/exp10/released_k1_all --out ckpt/exp10/summary_interim` (log `…/scratchpad/e10_sum_interim.log`, exit 0); both CPU, default 10 000 draws. `ckpt/exp10/cpu_protocol/summary` was rebuilt by the finish (entries 10–11).
7. **2026-09-27 03:28 — released_k8 GPU re-run (HEAD `5016b63`):** command as in the 03:28 entry above (`… 8 cuda 1 ckpt/yaw_rotation/sweep_released/per_sample_yaw.json`); log `yaw_pilot_2026-09-27_03:28:06_released_k8_chain_cuda.log`.
8. **[R] 2026-09-27 04:43 — four-arm preview summary (scratch; main HEAD `28fb4f1`, tools at `2408fb9`):** `nice -n 5 python tools/exp10_summarize.py --runs ckpt/exp10/released_k8_all ckpt/exp10/released_k1_all ckpt/exp10/control_k8_all ckpt/exp10/cyl_k8_all --out /tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad/e10_summary_preview`, run detached by `…/scratchpad/run_e10_preview.sh` (log `…/scratchpad/e10_sum_preview.log`, `PREVIEW_EXIT=0`; a first attempt as a harness background job was killed by the harness's false low-memory signal). Used only to draft the analysis; verified identical to the canonical summary apart from `generated_at`/`run_dir`.
9. **2026-09-27 03:27 — CPU-run relocation:** `mv ckpt/exp10/released_k8_probe ckpt/exp10/cpu_protocol/released_k8_probe; mv ckpt/exp10/released_k8_all ckpt/exp10/cpu_protocol/released_k8_all` (meta stores no path); **04:38 — derived reports regenerated in place (main HEAD `e9ae3c4`, tools at `2408fb9`):** `python tools/exp10_compare.py check-online ckpt/exp10/cpu_protocol/released_k8_all --json ckpt/exp10/cpu_protocol/released_k8_all/check_online.json` and `python tools/exp10_compare.py parity-exp03 ckpt/exp10/cpu_protocol/released_k8_all ckpt/yaw_rotation/sweep_released/per_sample_yaw.json --json ckpt/exp10/cpu_protocol/released_k8_all/parity_exp03.json` (log `yaw_pilot_2026-09-27_04:38:18_cpu_protocol_reports_regen.log`; old reports kept as `*_premove.json`).
10. **2026-09-27 07:49 — finish (reviewed tooling merge `415268d`; process HEAD at run time `97ede19` = that merge plus bookkeeping commits):** `nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_finish.sh > /dev/null 2>&1 &`; log `yaw_pilot_2026-09-27_07:49:18_finish.log`; outputs `ckpt/exp10/summary`, `ckpt/exp10/cpu_protocol/summary`, `yaw_pilot_results.md`, `yaw_pilot_01_results.html`, `yaw_pilot_results_assets/generated/` (46 assets, `SHA256SUMS`).
11. **2026-09-27 08:42 — finish re-run after round 8 (reviewed tooling merge `9327a4d`; process HEAD at run time `f01116f`):** `nohup setsid worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_finish.sh > /dev/null 2>&1 &` (republishes figures/pages; statistics unchanged).
