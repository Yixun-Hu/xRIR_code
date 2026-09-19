#!/bin/bash
# usage: exp05_eval_queue.sh <gpu> <reviewed_commit> <arm> [<arm> ...]   arm ∈ S_simple S_cylindrical L_simple L_cylindrical M_simple M_cylindrical
# Per arm: 10 standalone k=0 runs (K=8, K=1 × seeds 42..46) + the seed-42 K=8 yaw block {0,32,64,448,480}; all through tools.exp04_eval_launch --tier (entry exp05).
cd /home/yixunhu/codespace/xRIR_code
GPU=$1; SHA=$2; shift 2
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONPATH=/home/yixunhu/codespace/xRIR_code
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python; P=worklog/worklog_yixun/exp_05_param_efficiency_claude; IDX=ckpt/yaw_aug/reference_manifests_index.json
QLOG=$P/param_efficiency_$(date +%Y%m%dT%H%M%S)_eval_queue_gpu${GPU}.log; mkdir -p ckpt/exp05/eval
hash_of() { $PY -c "import json,sys; print(json.load(open('$IDX'))['k%s_seed%s' % (sys.argv[1], sys.argv[2])]['hash'])" "$1" "$2"; }
run_eval() { # label tier backbone ckpt K seed cols bind...
  local label=$1 tier=$2 bb=$3 ck=$4 K=$5 seed=$6 cols=$7; shift 7
  local man=ckpt/yaw_aug/reference_manifest_k${K}_seed${seed}.json out=ckpt/exp05/eval/${label}
  [ -e "$out" ] && { echo "$(date -Is) SKIP existing $out" >> "$QLOG"; return 0; }
  echo "$(date -Is) START $label" >> "$QLOG"
  nohup setsid $PY -m tools.exp04_eval_launch --tier $tier --backbone $bb --checkpoint $ck --manifest $man --manifest-hash $(hash_of $K $seed) \
    --out-dir $out --yaw-cols $cols --acoustic-cols $cols --e-acoustic-cols --conditions P --batch-size 16 --num-workers 6 --threads 4 \
    --decomposition-batches 0 --gl-seed $seed --max-samples 0 --log-interval 10 --num-shot $K --run-label $label --reviewed-commit $SHA \
    --data-root $XRIR_DATA_PATH --log-dir $P --gpu $GPU "$@" > "$P/param_efficiency_$(date +%Y%m%dT%H%M%S)_evallauncher_${label}.log" 2>&1
  echo "$(date -Is) END $label exit=$?" >> "$QLOG"
}
for arm in "$@"; do
  tier=${arm%%_*}; bb=${arm#*_}
  case $arm in
    M_simple) ck=ckpt/xRIR_simple_8_shot/epoch_12.pth; B="";;
    M_cylindrical) ck=ckpt/xRIR_cyl_8_shot/epoch_12.pth; B="";;
    *) att=$(readlink -f ckpt/exp05/${arm}/final); ck=$att/epoch_012.pth; B="--bind-input train_manifest=$att/train_manifest.json --bind-input train_completion=$att/completion.json";;
  esac
  for K in 8 1; do for seed in 42 43 44 45 46; do run_eval ${arm}_k${K}_seed${seed}_k0 $tier $bb $ck $K $seed "0" $B; done; done
  run_eval ${arm}_k8_seed42_yaw $tier $bb $ck 8 42 "0 32 64 448 480" $B
done
echo "EVAL_QUEUE_DONE gpu=$GPU $(date -Is)" >> "$QLOG"; echo "$QLOG"
