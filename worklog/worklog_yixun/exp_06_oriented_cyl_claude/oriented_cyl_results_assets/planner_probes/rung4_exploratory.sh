#!/bin/bash
# rung-4 smokes as exploratory diagnostics (plan A6): 6 GB / 300 s each, receipts + provenance under ckpt/exp06/_smoke
set -uo pipefail
R=/home/yixunhu/codespace/xRIR_code; cd $R
export PYTHONPATH=$R XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-1}
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python; SHA=$(git rev-parse HEAD); TS=$(date +%Y-%m-%d_%H:%M:%S)
E=$R/worklog/worklog_yixun/exp_06_oriented_cyl_claude; A=worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json
B=ckpt/exp06/_smoke/rung4_$TS; mkdir -p $B; LOG=$E/oriented_cyl_${TS}_rung4_exploratory.log
run() { # name entry argv...
  local name=$1 entry=$2; shift 2; mkdir -p $B/$name
  echo "=== $name ($entry) $(date +%T) ===" | tee -a $LOG
  $PY tools/exp06_smoke.py --entry $entry --receipt $B/$name/receipt.json --run-type smoke --exploratory --provenance-out $B/$name/provenance.json --approved $A --reviewed-commit $SHA --alarm-seconds 300 --max-gb 6 -- "$@" 2>&1 | grep -v -i -E "warn|_VF|view_as_real" | tee -a $LOG | grep -E "Train Epoch|Avg|EXP06_SMOKE|loss|Error|Traceback" | cut -c1-200 | tail -6
  $PY -c "import json; d=json.load(open('$B/$name/receipt.json')); print('$name RECEIPT', d['outcome'], 'exit', d['exit_status'], 'wall_s', round(d['wall_s'],1), 'peak_GB', round(d['peak_bytes']/2**30,2))" | tee -a $LOG
}
run a_trainer trainer --backbone simple --save-dir $B/a_trainer/run --epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --num-workers 4 --save-every 0 --no-save
run a_exp06_train exp06_train --backbone simple --save-dir $B/a_exp06_train/run --epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --num-workers 4 --save-every 0 --no-save --run-type smoke --exploratory
run b_oriented exp06_train --backbone cylindrical_oriented --save-dir $B/b_oriented/run --epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --num-workers 4 --save-every 0 --no-save --run-type smoke --exploratory
FX=ckpt/exp06/_smoke/fixture_cylor.pth; [ -f $FX ] || $PY tools/exp06_smoke.py --make-fixture $FX 2>&1 | tail -1 | tee -a $LOG
run c_haa_finetune exp06_haa_finetune --backbone cylindrical_oriented --init $FX --rooms class_room --heading-json-dir ckpt/exp06/heading --save-dir $B/c_haa_finetune/run --epochs 2 --val-every 1 --batch-size 4 --val-batch-size 4 --seed 0
run d_haa_eval exp06_haa_eval --backbone cylindrical_oriented --checkpoint $B/c_haa_finetune/run/best.pth --heading-json-dir ckpt/exp06/heading --rooms hallway --max-samples 4 --save-dir $B/d_haa_eval/run --seed 0
echo "=== parity: printed losses a_trainer vs a_exp06_train ===" | tee -a $LOG
diff <(grep -E "Train Epoch: 1 \[|Avg Train Loss" $LOG | sed -n '1,4p') <(grep -E "Train Epoch: 1 \[|Avg Train Loss" $LOG | sed -n '5,8p') && echo "PARITY OK (identical printed lines)" | tee -a $LOG || echo "PARITY DIFF (inspect)" | tee -a $LOG
echo "RUNG4 DONE $(date +%T) dir $B" | tee -a $LOG
