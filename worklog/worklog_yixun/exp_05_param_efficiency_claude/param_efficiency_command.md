
## Confirmatory trainings (2026-09-13 12:39 →; GPU 1 queue, reviewed commit f19b9b6)
```bash
# sequential queue: S_simple → S_cylindrical → L_cylindrical (scratchpad exp05_train_queue.sh); per arm:
nohup setsid tools/exp04_launch.sh full --tier <T> --backbone <B> --gpu 1 --reviewed-commit f19b9b6 --probe-json ckpt/exp05/<T>_<B>/_probe_<TS>_<T>_<B>.json --timestamp <TS> > worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_<TS>_launcher_<T>_<B>.log 2>&1
# L_simple on GPU 0 after exp_04's evaluations
```

## Record (2026-09-18 19:48–20:03, main 73e90d4, `exp05_record_run.sh`)
```
python $A/make_results_md.py   $FLAGS --attempt $ATTEMPTS --out $W/param_efficiency_results.md
python $A/make_results_html.py $FLAGS --attempt $ATTEMPTS --out $W/param_efficiency_01_results.html
python $A/make_figures.py --curve $R/CURVE_K8.json $R/CURVE_K1.json --outdir $W/param_efficiency_figures
python $A/bind_provenance.py --runs ckpt/exp05/eval/* --attempt $ATTEMPTS --results $JSONS --rendered <md> <html> --figures $W/param_efficiency_figures/* --out ckpt/exp05/reports
python $A/check_record.py ckpt/exp05/reports    # exit 0
```
($A, $W, $R, $ATTEMPTS, $JSONS, $FLAGS as in the assets README; binding report `binding_report_20260918T234858147838Z.json`, sha256 60810502078f4bcc0a287e626d88182a1e33da50796595bb822b80847eec29ef)
