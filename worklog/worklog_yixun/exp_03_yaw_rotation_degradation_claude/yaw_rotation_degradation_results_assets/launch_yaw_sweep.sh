#!/bin/bash
# exp_03 launch: (1) k=0 gate runs for the three models, (2) full sweep. Usage: launch_yaw_sweep.sh gate|sweep
set -u
cd /home/yixunhu/codespace/xRIR_code
export PYTHONPATH=${PYTHONPATH:-}:$(pwd); export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
P=~/miniconda3/envs/xRIR/bin/python
E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude
MAN=ckpt/yaw_rotation/reference_manifest.json
HASH=47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d
MODE=$1
declare -A BB=( [control]=simple [cyl]=cylindrical [released]=simple )
declare -A CK=( [control]=ckpt/xRIR_simple_8_shot/epoch_12.pth [cyl]=ckpt/xRIR_cyl_8_shot/epoch_12.pth [released]=checkpoints/xRIR_unseen.pth )
run_one() {  # name gpu manifest hash tag extra-args...
  local name=$1 gpu=$2 man=$3 hash=$4 tag=$5; shift 5
  local out=ckpt/yaw_rotation/${MODE}_${tag}; mkdir -p $out
  local ts=$(date +%Y-%m-%d_%H:%M:%S)
  CUDA_VISIBLE_DEVICES=$gpu $P eval_yaw_rotation.py --backbone ${BB[$name]} --checkpoint ${CK[$name]} --manifest $man --manifest-hash $hash --out-dir $out --batch-size 16 --num-workers 6 --threads 4 --log-interval 10 "$@" > $E/yaw_rotation_degradation_${ts}_${MODE}_${tag}.log 2>&1
  echo "$(date +%T) done $MODE $tag exit $?"
}
MAN1=ckpt/yaw_rotation/reference_manifest_seed1.json; HASH1=6ca8164e5727d5f4db84569d5effa840b2cb6e6f77819a69bbb37bb30c6ab11e
MAN2=ckpt/yaw_rotation/reference_manifest_seed2.json; HASH2=757e5a966ee2d069a07accecbc43e46bd564ea7c788a28c219773df64e986234
K0="--yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols"
if [ "$MODE" = gate ]; then
  ( run_one control 0 $MAN $HASH control $K0 ; run_one released 0 $MAN $HASH released $K0 ; run_one control 0 $MAN1 $HASH1 noise_seed1 $K0 ) &
  ( run_one cyl 1 $MAN $HASH cyl $K0 ; run_one control 1 $MAN2 $HASH2 noise_seed2 $K0 ) &
  wait; echo "GATE_DONE $(date +%T)"
elif [ "$MODE" = sweep ]; then
  ( run_one control 0 $MAN $HASH control ; run_one released 0 $MAN $HASH released ) &
  ( run_one cyl 1 $MAN $HASH cyl ) &
  wait; echo "SWEEP_DONE $(date +%T)"
fi
