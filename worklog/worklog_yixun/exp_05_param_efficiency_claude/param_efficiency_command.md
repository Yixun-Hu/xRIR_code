
## Confirmatory trainings (2026-09-13 12:39 →; GPU 1 queue, reviewed commit f19b9b6)
```bash
# sequential queue: S_simple → S_cylindrical → L_cylindrical (scratchpad exp05_train_queue.sh); per arm:
nohup setsid tools/exp04_launch.sh full --tier <T> --backbone <B> --gpu 1 --reviewed-commit f19b9b6 --probe-json ckpt/exp05/<T>_<B>/_probe_<TS>_<T>_<B>.json --timestamp <TS> > worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_<TS>_launcher_<T>_<B>.log 2>&1
# L_simple on GPU 0 after exp_04's evaluations
```
