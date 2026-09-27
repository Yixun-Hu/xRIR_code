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
