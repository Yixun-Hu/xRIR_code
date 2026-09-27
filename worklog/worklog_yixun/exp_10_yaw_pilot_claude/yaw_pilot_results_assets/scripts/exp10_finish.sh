#!/bin/bash
# exp_10 yaw_pilot — end-game after all four GPU arms are complete: canonical summary (4 arms), CPU-protocol summary,
# backend-sensitivity table, results Markdown + HTML + assets, SHA256SUMS. Idempotent (overwrites generated outputs).
set -uo pipefail
cd /home/yixunhu/codespace/xRIR_code
source ~/miniconda3/etc/profile.d/conda.sh; conda activate xRIR
export PYTHONPATH=${PYTHONPATH:-}:$(pwd); export CUDA_VISIBLE_DEVICES=""
E=worklog/worklog_yixun/exp_10_yaw_pilot_claude; A=$E/yaw_pilot_results_assets; G=$A/generated
LOG=$E/yaw_pilot_$(date +%Y-%m-%d_%H:%M:%S)_finish.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ARMS="released_k8 released_k1 control_k8 cyl_k8"
for a in $ARMS; do
  python - "$a" <<'PY' || { say "REFUSED: $a incomplete"; exit 2; }
import json, sys
a = sys.argv[1]
m = json.load(open("ckpt/exp10/%s_all/meta.json" % a)); assert m["complete"] and m["n_queries"] == 6337 and m["device"] == "cuda", (a, m["complete"], m["n_queries"], m["device"])
assert json.load(open("ckpt/exp10/%s_all/check_online.json" % a))["ok"], a
PY
done
say "HEAD $(git rev-parse HEAD); all four GPU arms complete"
python tools/exp10_summarize.py --runs ckpt/exp10/released_k8_all ckpt/exp10/released_k1_all ckpt/exp10/control_k8_all ckpt/exp10/cyl_k8_all --out ckpt/exp10/summary 2>&1 | tee -a "$LOG" || exit 3
python tools/exp10_summarize.py --runs ckpt/exp10/cpu_protocol/released_k8_all --out ckpt/exp10/cpu_protocol/summary 2>&1 | tee -a "$LOG" || exit 4
mkdir -p "$G/runs" "$G/cpu_protocol"
python $A/make_backend_table.py --gpu ckpt/exp10/summary/yaw_pilot_summary.json --cpu ckpt/exp10/cpu_protocol/summary/yaw_pilot_summary.json --arm released_k8 --out-md $G/backend_sensitivity_released_k8.md --out-json $G/backend_sensitivity_released_k8.json 2>&1 | tee -a "$LOG" || exit 5
PAR=""; ONL=""; PRB=""
for a in $ARMS; do
  mkdir -p "$G/runs/$a"; cp ckpt/exp10/${a}_all/meta.json "$G/runs/$a/meta_all.json"; cp ckpt/exp10/${a}_all/check_online.json "$G/runs/$a/check_online_all.json"; cp ckpt/exp10/${a}_probe/meta.json "$G/runs/$a/meta_probe.json"; cp ckpt/exp10/${a}_probe/summary/yaw_pilot_summary.json "$G/runs/$a/probe_summary.json"
  [ -f ckpt/exp10/${a}_all/parity_exp03.json ] && { cp ckpt/exp10/${a}_all/parity_exp03.json "$G/runs/$a/parity_exp03_all.json"; cp ckpt/exp10/${a}_probe/parity_exp03.json "$G/runs/$a/parity_exp03_probe.json"; PAR="$PAR $a=ckpt/exp10/${a}_all/parity_exp03.json"; }
  ONL="$ONL $a=ckpt/exp10/${a}_all/check_online.json"; PRB="$PRB $a=ckpt/exp10/${a}_probe/summary/yaw_pilot_summary.json"
done
cp ckpt/exp10/cpu_protocol/released_k8_all/meta.json "$G/cpu_protocol/meta_all.json"; cp ckpt/exp10/cpu_protocol/released_k8_all/parity_exp03.json "$G/cpu_protocol/parity_exp03_all.json"; cp ckpt/exp10/cpu_protocol/released_k8_all/check_online.json "$G/cpu_protocol/check_online_all.json"
cp ckpt/exp10/cpu_protocol/summary/yaw_pilot_summary.json ckpt/exp10/cpu_protocol/summary/yaw_pilot_tables.md ckpt/exp10/cpu_protocol/summary/yaw_pilot_gaps.csv "$G/cpu_protocol/"; cp ckpt/exp10/cpu_protocol/summary/yaw_pilot_gaps_released_k8.png "$G/cpu_protocol/yaw_pilot_gaps_released_k8_cpu.png"; cp ckpt/exp10/cpu_protocol/summary/yaw_pilot_gaps_released_k8.pdf "$G/cpu_protocol/yaw_pilot_gaps_released_k8_cpu.pdf"
python $A/make_results_md.py --summary ckpt/exp10/summary/yaw_pilot_summary.json --out $E/yaw_pilot_results.md --parity $PAR --check-online $ONL --probe $PRB 2>&1 | tee -a "$LOG" || exit 6
python $A/make_results_html.py --summary ckpt/exp10/summary/yaw_pilot_summary.json --out $E/yaw_pilot_01_results.html --assets $G --parity $PAR --check-online $ONL --probe $PRB 2>&1 | tee -a "$LOG" || exit 7
( cd "$G" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
say "FINISH DONE: $(wc -l < $G/SHA256SUMS) assets"
