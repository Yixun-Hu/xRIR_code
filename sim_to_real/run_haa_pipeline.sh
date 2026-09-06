#!/bin/bash
# Two-stage sim-to-real fine-tuning + evaluation for one queue of jobs on one GPU.
# Usage: run_haa_pipeline.sh <gpu> <job> [<job> ...]   with job = <init_name>:<seed>
#   init_name in {released, control, cyl, released_repomaps, zeroshot}
# Outputs under ckpt/sim2real/<init_name>/seed<seed>/{stage1, stage2_<room>, eval}.
set -u
GPU=$1; shift
cd /home/yixunhu/codespace/xRIR_code
export PYTHONPATH=$PYTHONPATH:$(pwd)
P=~/miniconda3/envs/xRIR/bin/python
OUT=ckpt/sim2real
ROOMS="class_room dampened_room hallway complex_room"
S1_ROOMS="class_room hallway complex_room"   # dampened excluded from stage 1, as in the repo
S1="--epochs 1000 --val-every 10 --tf32"
S2="--epochs 200 --val-every 2 --tf32"

init_of() { case $1 in
  released|released_repomaps) echo "simple checkpoints/xRIR_unseen.pth";;
  control) echo "simple ckpt/xRIR_simple_8_shot/epoch_12.pth";;
  cyl) echo "cylindrical ckpt/xRIR_cyl_8_shot/epoch_12.pth";;
  *) echo "unknown init $1" >&2; return 1;; esac; }
depth_of() { [ "$1" = released_repomaps ] && echo "--depth-variant repo" || echo ""; }

for job in "$@"; do
  name=${job%%:*}; seed=${job##*:}
  if [ "$name" = zeroshot ]; then   # zero-shot evals of all three inits (no fine-tuning)
    for n in released control cyl; do
      set -- $(init_of $n); bb=$1; ck=$2; d=$OUT/$n/zeroshot
      [ -f $d/metrics_all.json ] && { echo "skip $d"; continue; }
      mkdir -p $d; echo "=== zero-shot $n ($(date +%T)) ==="
      CUDA_VISIBLE_DEVICES=$GPU $P sim_to_real/eval_haa.py --backbone $bb --checkpoint $ck --rooms $ROOMS --save-dir $d > $d/eval.log 2>&1 || echo "FAILED zero-shot $n"
      grep -v "Warning\|warn\|_VF\|view_as_real" $d/eval.log | tail -4
    done
    continue
  fi
  set -- $(init_of $name); bb=$1; ck=$2; dv=$(depth_of $name); base=$OUT/$name/seed$seed
  echo "=== job $name seed $seed on GPU $GPU ($(date +%T)) ==="
  if [ ! -f $base/stage1/summary.json ]; then
    mkdir -p $base/stage1
    CUDA_VISIBLE_DEVICES=$GPU $P sim_to_real/finetune_haa.py --backbone $bb --init $ck --rooms $S1_ROOMS --save-dir $base/stage1 --seed $seed $S1 $dv > $base/stage1/train.log 2>&1 || { echo "FAILED stage1 $name seed $seed"; continue; }
  fi
  echo "  stage1: $(grep '^done' $base/stage1/train.log | cut -c1-120)"
  for room in $ROOMS; do
    if [ ! -f $base/stage2_$room/summary.json ]; then
      mkdir -p $base/stage2_$room
      CUDA_VISIBLE_DEVICES=$GPU $P sim_to_real/finetune_haa.py --backbone $bb --init $base/stage1/best.pth --rooms $room --save-dir $base/stage2_$room --seed $seed $S2 $dv > $base/stage2_$room/train.log 2>&1 || { echo "FAILED stage2 $name seed $seed $room"; continue; }
    fi
    if [ ! -f $base/eval/metrics_$room.json ]; then
      mkdir -p $base/eval
      CUDA_VISIBLE_DEVICES=$GPU $P sim_to_real/eval_haa.py --backbone $bb --checkpoint $base/stage2_$room/best.pth --rooms $room --save-dir $base/eval $dv > $base/eval/eval_$room.log 2>&1 || { echo "FAILED eval $name seed $seed $room"; continue; }
    fi
    echo "  $(grep -v 'Warning\|warn\|_VF\|view_as_real' $base/eval/eval_$room.log | tail -1)"
  done
done
echo "QUEUE_DONE GPU $GPU ($(date +%T))"
