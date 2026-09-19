#!/bin/bash
# exp_05 record: generators -> figures -> bind -> check, from the MAIN checkout (README §2–5). Re-runnable; commit the outputs afterwards.
set -uo pipefail
cd /home/yixunhu/codespace/xRIR_code
export PYTHONPATH=$PWD XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
A=worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets
W=worklog/worklog_yixun/exp_05_param_efficiency_claude
R=ckpt/exp05/results
ATTEMPTS="ckpt/exp05/S_simple/final ckpt/exp05/S_cylindrical/final ckpt/exp05/L_simple/final ckpt/exp05/L_cylindrical/final"
JSONS="$R/CURVE_K8.json $R/CURVE_K1.json $R/TARGETS_K8.json $R/TARGETS_K1.json $R/YAW_K8_SEED42.json"
FLAGS="--curve-k8 $R/CURVE_K8.json --curve-k1 $R/CURVE_K1.json --targets-k8 $R/TARGETS_K8.json --targets-k1 $R/TARGETS_K1.json --yaw-k8 $R/YAW_K8_SEED42.json"
LOG=$W/param_efficiency_$(date +%Y%m%dT%H%M%S)_record_run.log
step() { echo "$(date -Is) STEP $1" >> "$LOG"; }
step "generators start HEAD=$(git rev-parse HEAD)"
$PY $A/make_results_md.py   $FLAGS --attempt $ATTEMPTS --out $W/param_efficiency_results.md >> "$LOG" 2>&1 || { step "ABORT md"; exit 2; }
$PY $A/make_results_html.py $FLAGS --attempt $ATTEMPTS --out $W/param_efficiency_01_results.html >> "$LOG" 2>&1 || { step "ABORT html"; exit 3; }
mkdir -p $W/param_efficiency_figures
$PY $A/make_figures.py --curve $R/CURVE_K8.json $R/CURVE_K1.json --outdir $W/param_efficiency_figures >> "$LOG" 2>&1 || { step "ABORT figures"; exit 4; }
step "bind start"; mkdir -p ckpt/exp05/reports
$PY $A/bind_provenance.py --runs ckpt/exp05/eval/* --attempt $ATTEMPTS --results $JSONS \
  --rendered $W/param_efficiency_results.md $W/param_efficiency_01_results.html \
  --figures $W/param_efficiency_figures/* --out ckpt/exp05/reports >> "$LOG" 2>&1 || { step "ABORT bind"; exit 5; }
step "check start"
$PY $A/check_record.py ckpt/exp05/reports >> "$LOG" 2>&1; rc=$?; step "check exit=$rc"
step "RECORD RUN DONE check_exit=$rc"; echo "$LOG"
