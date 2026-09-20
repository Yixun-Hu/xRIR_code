#!/bin/bash
# exp_07 end-game (Codex-reviewed sequence, round 9/12): wait for seen_cyl + seen_aug certification and their 10 evaluations each ->
# released K=1 evaluations (5, GPU 0) -> pin fill + commit -> producers -> generators -> bind (with evidence) -> check.
set -uo pipefail
cd /home/yixunhu/codespace/xRIR_code
S=/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python; REC=worklog/worklog_yixun/exp_07_seen_protocol_claude; ASSETS=$REC/seen_protocol_results_assets; OUT=ckpt/exp07/results
export PYTHONPATH="$PWD" XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONHASHSEED=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
LOG=$REC/seen_protocol_$(date +%Y%m%dT%H%M%S)_finish.log; say() { echo "$(date -Is) $*" >> "$LOG"; }
say "FINISH START pid $$"
evals_done() { local n=0; for K in 8 1; do for s in 42 43 44 45 46; do [ -f ckpt/exp07/eval/${1}_k${K}_seed${s}_k0/completion.json ] && n=$((n+1)); done; done; [ $n -eq 10 ]; }
until [ -L ckpt/exp07/seen_cyl/final ] && [ -L ckpt/exp07/seen_aug/final ] && evals_done seen_cyl && evals_done seen_aug; do sleep 120; done
say "STEP both arms certified and evaluated (seen_cyl -> $(readlink ckpt/exp07/seen_cyl/final); seen_aug -> $(readlink ckpt/exp07/seen_aug/final))"
until [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep 9c72ac69)" ]; do sleep 60; done
if git status --short --untracked-files=all | grep -v '^.. worklog/' | grep -q .; then say "FINISH ABORT: tree dirty outside worklog"; exit 2; fi
SHA=$(git rev-parse HEAD); say "STEP released K=1 evaluations on GPU 0 reviewed=$SHA"
for seed in 42 43 44 45 46; do
  out=ckpt/exp07/eval/released_seen_k1_seed${seed}_k0; [ -e "$out" ] && { say "SKIP $out"; continue; }
  man=ckpt/exp07/reference_manifest_seen_k1_seed${seed}.json; digest=$("$PY" -c 'import sys; from tools.reference_manifest import load_manifest, manifest_hash; print(manifest_hash(load_manifest(sys.argv[1])))' "$man")
  CUDA_VISIBLE_DEVICES=0 "$PY" tools/exp07_eval_launch.py --split seen --backbone simple --checkpoint checkpoints/xRIR_seen.pth --num-shot 1 \
    --manifest "$man" --manifest-hash "$digest" --gl-seed $seed --batch-size 16 --conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols \
    --decomposition-batches 0 --max-samples 0 --num-workers 12 --threads 2 --out-dir "$out" --run-label "released_seen_k1_seed${seed}_k0" \
    --reviewed-commit "$SHA" --data-root "$XRIR_DATA_PATH" --log-dir "$REC" --gpu 0 > "$REC/seen_protocol_$(date +%Y%m%dT%H%M%S)_evallauncher_released_seen_k1_seed${seed}_k0.log" 2>&1
  say "STEP released k1 seed$seed exit=$?"
done
say "STEP pin fill"
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import json, subprocess
from pathlib import Path
from tools import provenance as p
from tools.exp07_profiles import ARMS, APPROVED_DIGESTS_PATH
repo=Path.cwd(); head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
modules=dict(evaluator='tools.exp07_eval',writer='tools.exp07_eval_launch',training_launcher='tools.exp07_launcher',training='tools.exp07_train',producer_table='tools.exp07_table',producer_pairs='tools.exp07_pairs')
closures={}
for key,module in modules.items():
    files=p.source_closure(module,repo)
    if key=='training_launcher': files+=['tools/exp07_probe.py','tools/exp07_launch.sh']
    records,digest=p.closure_record(files,head,repo)
    assert records and all(r['reviewed_blob_sha256']==r['working_tree_sha256'] and not r['commits_after_reviewed'] for r in records), key
    closures[key]=[digest] if key=='training_launcher' else digest
checkpoints={}
for arm in ARMS:
    if arm['reference']: continue
    path=repo/arm['checkpoint']; digest=p.sha256_file(path)
    manifest=json.loads((path.parent/'train_manifest.json').read_text()); completion=json.loads((path.parent/'completion.json').read_text())
    assert completion['outputs']['epoch_012.pth']==digest, arm['role']
    assert manifest['source_closures']['training']['sha256']==closures['training'], (arm['role'],'training closure')
    assert manifest['source_closures']['launcher']['sha256'] in closures['training_launcher'], (arm['role'],'launcher closure')
    checkpoints[arm['role']]=dict(path=arm['checkpoint'],epoch=12,sha256=digest)
APPROVED_DIGESTS_PATH.write_text(json.dumps(dict(schema_version=1,closures=closures,checkpoints=checkpoints),sort_keys=True,indent=2)+'\n')
print('PINS WRITTEN', {k:(v if isinstance(v,str) else v) for k,v in closures.items()}, {k:v['sha256'][:12] for k,v in checkpoints.items()})
PYEOF
[ $? -eq 0 ] || { say "FINISH ABORT: pin fill failed"; exit 3; }
git add "$ASSETS/approved_digests.json" && git commit -q -m "exp_07: fill reviewed closure and certified checkpoint pins

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && say "STEP pins committed $(git rev-parse --short HEAD)" || { say "FINISH ABORT: pin commit failed"; exit 3; }
"$PY" -c 'from tools.exp07_profiles import load_approved_digests; assert load_approved_digests()[0]["schema_version"] == 1' >> "$LOG" 2>&1 || { say "FINISH ABORT: approvals load"; exit 3; }
mkdir -p "$OUT"; RUNS=(ckpt/exp07/eval/{seen_simple,seen_cyl,seen_aug,released_seen}_k{8,1}_seed{42..46}_k0)
say "STEP producers"
"$PY" tools/exp07_table.py --profile TABLE_SEEN_V1 --runs "${RUNS[@]}" --json "$OUT/TABLE_SEEN_V1.json" --md worklog/worklog_yixun/model_comparison_seen.md >> "$LOG" 2>&1 || { say "FINISH ABORT: table"; exit 4; }
for role in seen_cyl seen_aug released_seen; do
  "$PY" tools/exp07_pairs.py --profile PAIRS_SEEN_V1 --runs-a ckpt/exp07/eval/"${role}"_k{8,1}_seed{42..46}_k0 --runs-b ckpt/exp07/eval/seen_simple_k{8,1}_seed{42..46}_k0 --reconverge --json "$OUT/PAIRS_SEEN_V1_${role}.json" --summary "$OUT/PAIRS_SEEN_V1_${role}.txt" >> "$LOG" 2>&1 || { say "FINISH ABORT: pairs $role"; exit 4; }
done
UNSEEN=ckpt/yaw_aug/results/TABLE_V1.json; UNSEEN_BINDING=$("$PY" -c 'from pathlib import Path; print(sorted(Path("ckpt/yaw_aug").glob("binding_report_*.json"))[-1])')
PAIRS=("$OUT"/PAIRS_SEEN_V1_{seen_cyl,seen_aug,released_seen}.json); COMMON=(--table "$OUT/TABLE_SEEN_V1.json" --pairs "${PAIRS[@]}" --unseen-table "$UNSEEN" --unseen-binding "$UNSEEN_BINDING")
say "STEP generators"
"$PY" "$ASSETS/make_results_md.py" "${COMMON[@]}" --out "$REC/seen_protocol_results.md" >> "$LOG" 2>&1 || { say "FINISH ABORT: md"; exit 5; }
"$PY" "$ASSETS/make_results_html.py" "${COMMON[@]}" --out "$REC/seen_protocol_01_results.html" >> "$LOG" 2>&1 || { say "FINISH ABORT: html"; exit 5; }
"$PY" "$ASSETS/make_latex.py" "${COMMON[@]}" --out "$ASSETS/table_seen_unseen.tex" >> "$LOG" 2>&1 || { say "FINISH ABORT: latex"; exit 5; }
PARITY_RECEIPT=$(ls $REC/gpu_parity_*.receipt.json | head -1)
say "STEP bind (parity receipt $PARITY_RECEIPT)"
"$PY" "$ASSETS/bind_provenance.py" --runs "${RUNS[@]}" --attempt ckpt/exp07/{seen_simple,seen_cyl,seen_aug}/final --audit ckpt/exp07/alignment_audit_seen.json \
  --evidence "gpu_parity=$PARITY_RECEIPT" --evidence "calibration=$OUT/CALIBRATION_SEEN_V1.json" --results "$OUT/TABLE_SEEN_V1.json" "${PAIRS[@]}" \
  --rendered "$REC/seen_protocol_results.md" "$REC/seen_protocol_01_results.html" "$ASSETS/table_seen_unseen.tex" \
  --unseen-table "$UNSEEN" --unseen-binding "$UNSEEN_BINDING" --out ckpt/exp07 >> "$LOG" 2>&1 || { say "FINISH ABORT: bind"; exit 6; }
say "STEP check"; "$PY" "$ASSETS/check_record.py" ckpt/exp07 >> "$LOG" 2>&1; rc=$?; say "STEP check exit=$rc"; say "FINISH DONE check_exit=$rc"
