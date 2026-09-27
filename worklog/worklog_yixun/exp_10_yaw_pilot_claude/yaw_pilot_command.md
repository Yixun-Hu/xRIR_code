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
