#!/bin/bash
# exp_03 launcher v2 (Planner; after the full review). Usage: launch_yaw_sweep_v2.sh gate|sweep
# - every evaluator exit status is propagated; the mode fails if any run fails
# - output directories must not pre-exist (no stale artefacts); --decomposition-batches 0 for gate runs
# - sweep refuses to start unless ckpt/yaw_rotation/gate_stats.json exists with gate_pass true and matches its summary hash
# - a provenance.json sidecar (git SHA, checkpoint sha256, manifest hash, per-sample sha256, command) is written next to each run
set -u
cd /home/yixunhu/codespace/xRIR_code
export PYTHONPATH=${PYTHONPATH:-}:$(pwd); export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
P=~/miniconda3/envs/xRIR/bin/python
E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude
MAN=ckpt/yaw_rotation/reference_manifest.json;       HASH=47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d
MAN1=ckpt/yaw_rotation/reference_manifest_seed1.json; HASH1=6ca8164e5727d5f4db84569d5effa840b2cb6e6f77819a69bbb37bb30c6ab11e
MAN2=ckpt/yaw_rotation/reference_manifest_seed2.json; HASH2=757e5a966ee2d069a07accecbc43e46bd564ea7c788a28c219773df64e986234
GIT_SHA=$(git rev-parse HEAD)
MODE=${1:?usage: gate|sweep}
declare -A BB=( [control]=simple [cyl]=cylindrical [released]=simple )
declare -A CK=( [control]=ckpt/xRIR_simple_8_shot/epoch_12.pth [cyl]=ckpt/xRIR_cyl_8_shot/epoch_12.pth [released]=checkpoints/xRIR_unseen.pth )
STATUS_FILE=$(mktemp); : > $STATUS_FILE.codes
run_one() {  # name gpu manifest hash tag extra-args...
  local name=$1 gpu=$2 man=$3 hash=$4 tag=$5; shift 5
  local out=ckpt/yaw_rotation/${MODE}_${tag}
  if [ -e "$out" ]; then echo "$(date +%T) REFUSED $MODE $tag: $out already exists" | tee -a $STATUS_FILE; echo "1" >> $STATUS_FILE.codes; return 1; fi
  mkdir -p $out
  local ts
  ts=$(date +%Y-%m-%d_%H:%M:%S)
  local log=$E/yaw_rotation_degradation_${ts}_${MODE}_${tag}.log
  local cmd="CUDA_VISIBLE_DEVICES=$gpu $P eval_yaw_rotation.py --backbone ${BB[$name]} --checkpoint ${CK[$name]} --manifest $man --manifest-hash $hash --out-dir $out --batch-size 16 --num-workers 6 --threads 4 --log-interval 10 $*"
  CUDA_VISIBLE_DEVICES=$gpu $P eval_yaw_rotation.py --backbone ${BB[$name]} --checkpoint ${CK[$name]} --manifest $man --manifest-hash $hash --out-dir $out --batch-size 16 --num-workers 6 --threads 4 --log-interval 10 "$@" > $log 2>&1
  local rc=$?
  echo "$rc" >> $STATUS_FILE.codes
  if [ $rc -eq 0 ]; then
    $P - "$out" "${CK[$name]}" "$hash" "$GIT_SHA" "$cmd" "$log" <<'PY'
import hashlib, json, sys, datetime
out, ck, mh, sha, cmd, log = sys.argv[1:7]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"git_sha": sha, "checkpoint": ck, "checkpoint_sha256": h(ck), "manifest_hash": mh,
           "per_sample_sha256": h(f"{out}/per_sample_yaw.json"), "metrics_sha256": h(f"{out}/metrics_yaw.json"),
           "command": cmd, "log": log, "written": datetime.datetime.now().astimezone().isoformat()},
          open(f"{out}/provenance.json", "w"), indent=1)
PY
  fi
  echo "$(date +%T) done $MODE $tag exit $rc" | tee -a $STATUS_FILE
  return $rc
}
K0="--yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --decomposition-batches 0"
if [ "$MODE" = gate ]; then
  ( run_one control 0 $MAN $HASH control $K0 && run_one released 0 $MAN $HASH released $K0 && run_one control 0 $MAN1 $HASH1 noise_seed1 $K0 ) & A=$!
  ( run_one cyl 1 $MAN $HASH cyl $K0 && run_one control 1 $MAN2 $HASH2 noise_seed2 $K0 ) & B=$!
  wait $A; ra=$?; wait $B; rb=$?
  if [ $ra -eq 0 ] && [ $rb -eq 0 ] && [ "$(sort -u $STATUS_FILE.codes | tr -d '\n')" = "0" ]; then echo "GATE_DONE $(date +%T)"; else echo "GATE_FAILED $(date +%T) codes: $(tr '\n' ' ' < $STATUS_FILE.codes)"; exit 1; fi
elif [ "$MODE" = sweep ]; then
  G=ckpt/yaw_rotation/gate_stats.json
  $P - "$G" <<'PY' || { echo "SWEEP_REFUSED: gate not passed or unbound"; exit 1; }
import hashlib, json, sys
g = json.load(open(sys.argv[1]))
assert g.get("mode") == "k0-gate" and g.get("gate_pass") is True, ("gate_pass", g.get("gate_pass"))
sp = g.get("summary_path"); assert sp and hashlib.sha256(open(sp).read().encode()).hexdigest() == g["summary_sha256"], "gate summary hash mismatch"
print("gate ok:", sp)
PY
  ( run_one control 0 $MAN $HASH control && run_one released 0 $MAN $HASH released ) & A=$!
  ( run_one cyl 1 $MAN $HASH cyl ) & B=$!
  wait $A; ra=$?; wait $B; rb=$?
  if [ $ra -eq 0 ] && [ $rb -eq 0 ] && [ "$(sort -u $STATUS_FILE.codes | tr -d '\n')" = "0" ]; then echo "SWEEP_DONE $(date +%T)"; else echo "SWEEP_FAILED $(date +%T) codes: $(tr '\n' ' ' < $STATUS_FILE.codes)"; exit 1; fi
fi
