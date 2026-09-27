#!/bin/bash
# exp_10 yaw_pilot — one arm end to end: probe (+controls) → check-online → parity vs exp_03 → gate → full run → check-online → parity (full).
# usage: exp10_arm_chain.sh <arm> <backbone> <checkpoint> <manifest> <manifest_hash> <num_shot> <device cpu|cuda> [gpu_index] [exp03_per_sample|none]
# Gate = comparator `ok` on the probe parity (all §7 criteria + A1 masks) AND probe controls ok (summariser controls table).
# For an arm without an exp_03 predecessor (released_k1) pass `none`: controls only.
set -uo pipefail
ARM=$1; BB=$2; CK=$3; MAN=$4; HASH=$5; K=$6; DEV=$7; GPU=${8:-1}; EXP03=${9:-none}
cd /home/yixunhu/codespace/xRIR_code
source ~/miniconda3/etc/profile.d/conda.sh; conda activate xRIR
export PYTHONPATH=${PYTHONPATH:-}:$(pwd); export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
if [ "$DEV" = "cuda" ]; then export CUDA_VISIBLE_DEVICES=$GPU; else export CUDA_VISIBLE_DEVICES=""; fi
E=worklog/worklog_yixun/exp_10_yaw_pilot_claude
LOG=$E/yaw_pilot_$(date +%Y-%m-%d_%H:%M:%S)_${ARM}_chain_${DEV}.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
[ -n "$(git status --porcelain | grep -v '^ M worklog/\|^?? worklog/')" ] && { say "REFUSED: tree dirty outside worklog/"; exit 3; }
say "HEAD $(git rev-parse HEAD) arm=$ARM backbone=$BB ckpt=$CK manifest=$MAN K=$K device=$DEV gpu=$GPU exp03=$EXP03"
run_stage() {  # <batches> <controls-flag>
  local B=$1 CF=$2 OUT=ckpt/exp10/${ARM}_$1
  [ -e "$OUT/meta.json" ] && { say "REFUSED: $OUT exists"; return 2; }
  say "stage $B → $OUT"
  python tools/exp10_yaw_pilot.py --arm "$ARM" --backbone "$BB" --checkpoint "$CK" --manifest "$MAN" --manifest-hash "$HASH" --num-shot "$K" \
    --ks 0,64,128,256,384 --batches "$B" $CF --device "$DEV" --threads 16 --out-dir "$OUT" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && { say "stage $B FAILED rc=$rc"; return $rc; }
  python tools/exp10_compare.py check-online "$OUT" --json "$OUT/check_online.json" 2>&1 | tee -a "$LOG" || { say "check-online FAILED"; return 4; }
  if [ "$EXP03" != "none" ]; then
    python tools/exp10_compare.py parity-exp03 "$OUT" "$EXP03" --json "$OUT/parity_exp03.json" 2>&1 | tee -a "$LOG" || { say "parity FAILED to run"; return 5; }
  fi
  return 0
}
run_stage probe --controls || exit $?
PROBE=ckpt/exp10/${ARM}_probe
python tools/exp10_summarize.py --runs "$PROBE" --out "$PROBE/summary" --n-boot 2000 2>&1 | tee -a "$LOG" || { say "probe summary FAILED"; exit 6; }
CTRL_OK=$(python -c "import json,sys; s=json.load(open('$PROBE/summary/yaw_pilot_summary.json')); a=s['arms'][0] if isinstance(s.get('arms'),list) else list(s['arms'].values())[0]; print(a['controls']['ok'])" 2>/dev/null || echo unknown)
if [ "$EXP03" != "none" ]; then PAR_OK=$(python -c "import json; print(json.load(open('$PROBE/parity_exp03.json'))['ok'])"); else PAR_OK=n/a; fi
say "GATE: controls_ok=$CTRL_OK parity_ok=$PAR_OK"
if [ "$CTRL_OK" != "True" ] || { [ "$EXP03" != "none" ] && [ "$PAR_OK" != "True" ]; }; then say "GATE FAILED — full run NOT launched"; exit 7; fi
say "GATE PASSED — launching full run"
run_stage all "" || exit $?
say "ARM DONE"
