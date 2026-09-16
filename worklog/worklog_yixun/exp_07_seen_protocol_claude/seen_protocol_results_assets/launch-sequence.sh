#!/bin/bash
# Review artifact: command sequence for the Planner, not executed by this review.
# Run each stage only after the prior gate passes. Coordinate the merge window and
# select a genuinely free GPU under the agreed ordering; record every launch.
set -euo pipefail
cd /home/yixunhu/codespace/xRIR_code
git switch main
git merge --no-ff f39a441 -m 'Merge exp_07 seen protocol'
MERGE=$(git rev-parse HEAD)
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
REC=worklog/worklog_yixun/exp_07_seen_protocol_claude
ASSETS=$REC/seen_protocol_results_assets
OUT=ckpt/exp07/results
export PYTHONPATH="$PWD" XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export PYTHONHASHSEED=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
: "${GPU:?Set GPU to a free physical device, 0 or 1, under the agreed schedule}"
mkdir -p "$OUT"

PARITY="$REC/gpu_parity_${MERGE}.log"
"$PY" tools/exp07_parity.py --log "$PARITY" --reviewed-commit "$MERGE" --gpu "$GPU"
PARITY_RECEIPT="${PARITY%.log}.receipt.json"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/exp07_audit.py --protocol seen \
  --n-batches 3 --batch-size 32 --seed 0 --num-workers 12 \
  --out ckpt/exp07/alignment_audit_seen.json

# The ten registered manifests have already been built and pinned.
eval07() {
  local role=$1 backbone=$2 checkpoint=$3 shot=$4 seed=$5
  local manifest="ckpt/exp07/reference_manifest_seen_k${shot}_seed${seed}.json"
  local digest
  digest=$("$PY" -c 'import sys; from tools.reference_manifest import load_manifest, manifest_hash; print(manifest_hash(load_manifest(sys.argv[1])))' "$manifest")
  local binds=()
  if [[ "$role" != released_seen ]]; then
    local attempt="ckpt/exp07/$role/final"
    binds=(--bind-input "train_manifest=$attempt/train_manifest.json"
           --bind-input "train_completion=$attempt/completion.json"
           --bind-input "train_args=$attempt/args.json")
  fi
  "$PY" tools/exp07_eval_launch.py --split seen --backbone "$backbone" \
    --checkpoint "$checkpoint" --num-shot "$shot" --manifest "$manifest" \
    --manifest-hash "$digest" --gl-seed "$seed" --batch-size 16 \
    --conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols \
    --decomposition-batches 0 --max-samples 0 --num-workers 12 --threads 2 \
    --out-dir "ckpt/exp07/eval/${role}_k${shot}_seed${seed}_k0" \
    --run-label "${role}_k${shot}_seed${seed}_k0" --reviewed-commit "$MERGE" \
    --data-root "$XRIR_DATA_PATH" --log-dir "$REC" --gpu "$GPU" "${binds[@]}"
}
for seed in 42 43 44 45 46; do
  eval07 released_seen simple checkpoints/xRIR_seen.pth 8 "$seed"
done
"$PY" tools/exp07_calibration.py --reviewed-commit "$MERGE" \
  --runs ckpt/exp07/eval/released_seen_k8_seed{42..46}_k0 \
  --json "$OUT/CALIBRATION_SEEN_V1.json"
# A failed calibration stops here; investigate under the plan before proceeding.

train07() {
  local role=$1 backbone=$2 yaw=$3 stamp
  stamp=$(date +%Y%m%dT%H%M%S%N)
  local flags=(--backbone "$backbone" --yaw-aug "$yaw" --gpu "$GPU"
               --reviewed-commit "$MERGE" --log-dir "$REC")
  "$PY" tools/exp07_launcher.py smoke "${flags[@]}" --timestamp "${stamp}_smoke"
  "$PY" tools/exp07_launcher.py probe "${flags[@]}" --timestamp "${stamp}_probe"
  nohup setsid tools/exp07_launch.sh full "${flags[@]}" --timestamp "${stamp}_full" \
    --probe-json "ckpt/exp07/$role/_probe_${stamp}_probe_${role}.json" \
    > "$REC/${role}_${stamp}_launcher.log" 2>&1 &
  wait "$!"
}
# Execute these per available GPU under ordering B, recording each command at launch.
# Use the project's detached-job wrapper for long full/evaluation jobs.
train07 seen_cyl cylindrical 0
train07 seen_simple simple 0
train07 seen_aug simple 1

for shot in 8 1; do
  for seed in 42 43 44 45 46; do
    eval07 seen_simple simple ckpt/exp07/seen_simple/final/epoch_012.pth "$shot" "$seed"
    eval07 seen_cyl cylindrical ckpt/exp07/seen_cyl/final/epoch_012.pth "$shot" "$seed"
    eval07 seen_aug simple ckpt/exp07/seen_aug/final/epoch_012.pth "$shot" "$seed"
  done
done
for seed in 42 43 44 45 46; do
  eval07 released_seen simple checkpoints/xRIR_seen.pth 1 "$seed"
done

# Fill every pin together after the three certified trainings and evaluations.
"$PY" - <<'PY'
import json
import subprocess
from pathlib import Path
from tools import provenance as p
from tools.exp07_profiles import ARMS, APPROVED_DIGESTS_PATH
repo=Path.cwd(); head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
modules=dict(evaluator='tools.exp07_eval',writer='tools.exp07_eval_launch',
             training_launcher='tools.exp07_launcher',training='tools.exp07_train',
             producer_table='tools.exp07_table',producer_pairs='tools.exp07_pairs')
closures={}
for key,module in modules.items():
    files=p.source_closure(module,repo)
    if key=='training_launcher':
        files+=['tools/exp07_probe.py','tools/exp07_launch.sh']
    records,digest=p.closure_record(files,head,repo)
    assert records and all(r['reviewed_blob_sha256']==r['working_tree_sha256'] and
                           not r['commits_after_reviewed'] for r in records)
    closures[key]=[digest] if key=='training_launcher' else digest
checkpoints={}
for arm in ARMS:
    if arm['reference']: continue
    path=repo/arm['checkpoint']; digest=p.sha256_file(path)
    manifest=json.loads((path.parent/'train_manifest.json').read_text())
    completion=json.loads((path.parent/'completion.json').read_text())
    assert completion['outputs']['epoch_012.pth']==digest
    assert manifest['source_closures']['training']['sha256']==closures['training']
    assert manifest['source_closures']['launcher']['sha256'] in closures['training_launcher']
    checkpoints[arm['role']]=dict(path=arm['checkpoint'],epoch=12,sha256=digest)
APPROVED_DIGESTS_PATH.write_text(json.dumps(dict(schema_version=1,closures=closures,
                                               checkpoints=checkpoints),sort_keys=True,indent=2)+'\n')
PY
# Review the concrete pin fill, then commit before invoking any publication producer.
git diff -- "$ASSETS/approved_digests.json"
git add "$ASSETS/approved_digests.json"
git commit -m 'exp_07: fill reviewed closure and certified checkpoint pins'
"$PY" -c 'from tools.exp07_profiles import load_approved_digests; assert load_approved_digests()[0]["schema_version"] == 1'

RUNS=(ckpt/exp07/eval/{seen_simple,seen_cyl,seen_aug,released_seen}_k{8,1}_seed{42..46}_k0)
"$PY" tools/exp07_table.py --profile TABLE_SEEN_V1 --runs "${RUNS[@]}" \
  --json "$OUT/TABLE_SEEN_V1.json" --md worklog/worklog_yixun/model_comparison_seen.md
for role in seen_cyl seen_aug released_seen; do
  "$PY" tools/exp07_pairs.py --profile PAIRS_SEEN_V1 \
    --runs-a ckpt/exp07/eval/"${role}"_k{8,1}_seed{42..46}_k0 \
    --runs-b ckpt/exp07/eval/seen_simple_k{8,1}_seed{42..46}_k0 \
    --reconverge --json "$OUT/PAIRS_SEEN_V1_${role}.json" \
    --summary "$OUT/PAIRS_SEEN_V1_${role}.txt"
done
UNSEEN=ckpt/yaw_aug/results/TABLE_V1.json
UNSEEN_BINDING=$("$PY" -c 'from pathlib import Path; print(sorted(Path("ckpt/yaw_aug").glob("binding_report_*.json"))[-1])')
PAIRS=("$OUT"/PAIRS_SEEN_V1_{seen_cyl,seen_aug,released_seen}.json)
COMMON=(--table "$OUT/TABLE_SEEN_V1.json" --pairs "${PAIRS[@]}"
        --unseen-table "$UNSEEN" --unseen-binding "$UNSEEN_BINDING")
"$PY" "$ASSETS/make_results_md.py" "${COMMON[@]}" --out "$REC/seen_protocol_results.md"
"$PY" "$ASSETS/make_results_html.py" "${COMMON[@]}" --out "$REC/seen_protocol_01_results.html"
"$PY" "$ASSETS/make_latex.py" "${COMMON[@]}" --out "$ASSETS/table_seen_unseen.tex"
"$PY" "$ASSETS/bind_provenance.py" --runs "${RUNS[@]}" \
  --attempt ckpt/exp07/{seen_simple,seen_cyl,seen_aug}/final \
  --audit ckpt/exp07/alignment_audit_seen.json \
  --evidence "gpu_parity=$PARITY_RECEIPT" \
  --evidence "calibration=$OUT/CALIBRATION_SEEN_V1.json" \
  --results "$OUT/TABLE_SEEN_V1.json" "${PAIRS[@]}" \
  --rendered "$REC/seen_protocol_results.md" "$REC/seen_protocol_01_results.html" \
    "$ASSETS/table_seen_unseen.tex" \
  --unseen-table "$UNSEEN" --unseen-binding "$UNSEEN_BINDING" --out ckpt/exp07
"$PY" "$ASSETS/check_record.py" ckpt/exp07
