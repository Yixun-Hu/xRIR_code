# exp_07 seen_protocol — Codex code review, round 9 close + FINAL integrative verdict (`full`): APPROVED

**Reviewer:** OpenAI Codex `gpt-6-astra` (Ultra), `codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07`, prompt `review_prompts/code_round9_close_prompt.md`, log `seen_protocol_2026-09-16_01:05:57_codex_code_round9_close.log`. **Scope:** exp07-window `fd6e521..f39a441` (round 9) and the whole branch `f492613..f39a441` for the integrative verdict. **Date:** 2026-09-16 01:06–01:51 EDT.

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-16

**Round-9 verdict: approve. Both round-8 blockers are closed.**  
**FINAL INTEGRATIVE verdict (`full`): approved for merge and GPU work.**  
**Exact blocking findings: none (`[]`).**

Reviewed every changed line in `fd6e521..f39a441`, replayed the prior reproductions, and repeated integration and preservation checks.

1. **Nit — optional constant rename.** [bind_provenance.py:60](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:60): `TRAINING_DEPENDENCIES` names three required output entries; `training_dependencies()` correctly assembles all seven named entries. The name can suggest the complete set. **Minimal fix:** optionally rename it `TRAINING_OUTPUT_DEPENDENCIES` and clarify the comment. **I do not require the rename for closure or merge.**

## Verified

- **Exact dependencies:** rebuilt matrix: **880 mutations, unexpected `[]`, extra allowed 0**. TABLE/cyl/aug/released fixture maps contain **239/131/131/125 inputs**, with **40/20/20/20 contracts**. Each completed dependency map equals its sidecar inputs.
- **Additional attacks:** **81 end-to-end product mutations** refused by both `collect` and `check_record`, with outer hashes repaired. These cover ledger/receipt omissions and every requested addition: unused epoch checkpoints, history, effective arguments, probe manifests/completions, aborted-attempt files, foreign producer closures, unused-role inventory files, and the released checkpoint in cyl/simple.
- **Attempt preservation:** all **70 fixture attempt files and logs** remain bound. Every independent byte edit makes the saved report fail checking.
- **Calibration:** **75 checks** across `build_table`, `build_calibration`, and `measure`, using the full synthetic **6,217-query** fixture. For every metric, **6,215 versus 6,217 is accepted; 6,214 and 6,212 are refused**. Zero finite queries and an entirely NaN seed are refused. Five additional tests reject overflowed seed means. Approval-loader traps were never called; real closure derivation succeeds without the approval JSON. EDT remains in seconds.
- **Numerical preservation:** table `rows` **and full table JSON are byte-identical** against `fd6e521` with identical fixture and producer metadata.
- **Prior reproductions:** round 6 **17 passed / 6 obsolete failures**; round 7 **27/7**; round 8 **58/13**, including carried replay copies. Round 8’s failures are twelve repaired-hole expectations and one reference to the deleted `bound` API. **Refusal-asserting regressions: none.** Updated reproductions and new tests pass: **79 unique reviewer cases** after correcting two harness setup issues.

| Check | Result |
|---|---|
| Clone at `f39a441`, local `main` created first; `python -m pytest tests -q -p no:cacheprovider` | **2,606 passed / 46 skipped / 0 failed**, exit 0; 2,318 s |
| `static_checks.sh` from this worktree | **Exit 0; 318 passed** |
| Record tools in both collection orders | **221 passed / 1 skipped each** |
| Python 3.8.20 parsing/compilation, shell syntax, all three `git diff --check` scopes | Passed |
| `git cat-file -e main:<path>` for every changed non-worklog path | All **33 paths are new** |
| Dry-run `git merge --no-ff --no-commit f39a441` into main `464b031`—a descendant of `968e46c` | No conflicts |
| Merged exp_04 `check_record.py ckpt/yaw_aug` | Exit 0 |
| Merged exp_06 parser-contract test | 1 passed |

Direct comparison also confirms **15 exp_04 closure files across 46 evaluations** retain their recorded bytes. The released checkpoint digest matches its registered value. No tracked files or prior reviewer files were edited; review writes stayed inside this worktree, and CUDA remained hidden.

Authoritative replay counts use explicit filenames. Initial directory collection also traversed clone symlinks; clean/dirty clone separation and literal-NaN fixture construction were corrected before accepting results.

Evidence: [saved review](/home/yixunhu/codespace/xRIR_code_wt07/.review-round9/seen_protocol_codex_code_round9_close_review.md), [full suite](/home/yixunhu/codespace/xRIR_code_wt07/.review-round9/full-suite.log), [static checks](/home/yixunhu/codespace/xRIR_code_wt07/.review-round9/static-root.log), [mutation evidence index](/home/yixunhu/codespace/xRIR_code_wt07/.review-round9/review-evidence.json), [calibration boundaries](/home/yixunhu/codespace/xRIR_code_wt07/.review-round9/corrected-boundaries.log).

## Planner launch sequence

Record closure, coordinate the merge window, and select a free GPU under the agreed schedule. The nine real GPU parity cases remain deferred and must pass first. The complete command reference is also saved in [launch-sequence.sh](/home/yixunhu/codespace/xRIR_code_wt07/.review-round9/launch-sequence.sh). These commands were reviewed, not executed.

**Merge → parity → audit → released evaluations → calibration:**

```bash
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
: "${GPU:?Set GPU to a free physical device under the agreed schedule}"
mkdir -p "$OUT"

PARITY="$REC/gpu_parity_${MERGE}.log"
"$PY" tools/exp07_parity.py \
  --log "$PARITY" --reviewed-commit "$MERGE" --gpu "$GPU"
PARITY_RECEIPT="${PARITY%.log}.receipt.json"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/exp07_audit.py \
  --protocol seen --n-batches 3 --batch-size 32 --seed 0 --num-workers 12 \
  --out ckpt/exp07/alignment_audit_seen.json

eval07() {
  local role=$1 backbone=$2 checkpoint=$3 shot=$4 seed=$5
  local manifest="ckpt/exp07/reference_manifest_seen_k${shot}_seed${seed}.json"
  local digest
  local binds=()
  digest=$("$PY" -c \
    'import sys; from tools.reference_manifest import load_manifest,manifest_hash; print(manifest_hash(load_manifest(sys.argv[1])))' \
    "$manifest")
  if [[ "$role" != released_seen ]]; then
    local attempt="ckpt/exp07/$role/final"
    binds=(--bind-input "train_manifest=$attempt/train_manifest.json"
           --bind-input "train_completion=$attempt/completion.json"
           --bind-input "train_args=$attempt/args.json")
  fi
  "$PY" tools/exp07_eval_launch.py --split seen --backbone "$backbone" \
    --checkpoint "$checkpoint" --num-shot "$shot" \
    --manifest "$manifest" --manifest-hash "$digest" --gl-seed "$seed" \
    --batch-size 16 --conditions P --yaw-cols 0 --acoustic-cols 0 \
    --e-acoustic-cols --decomposition-batches 0 --max-samples 0 \
    --num-workers 12 --threads 2 \
    --out-dir "ckpt/exp07/eval/${role}_k${shot}_seed${seed}_k0" \
    --run-label "${role}_k${shot}_seed${seed}_k0" \
    --reviewed-commit "$MERGE" --data-root "$XRIR_DATA_PATH" \
    --log-dir "$REC" --gpu "$GPU" "${binds[@]}"
}

for seed in 42 43 44 45 46; do
  eval07 released_seen simple checkpoints/xRIR_seen.pth 8 "$seed"
done
"$PY" tools/exp07_calibration.py --reviewed-commit "$MERGE" \
  --runs ckpt/exp07/eval/released_seen_k8_seed{42..46}_k0 \
  --json "$OUT/CALIBRATION_SEEN_V1.json"
```

A failed calibration stops advancement and invokes the plan’s investigation procedure.

**Per-arm smoke → probe → full**, selecting `GPU` for each scheduled arm and recording each launch:

```bash
train07() {
  local role=$1 backbone=$2 yaw=$3 stamp
  stamp=$(date +%Y%m%dT%H%M%S%N)
  local flags=(--backbone "$backbone" --yaw-aug "$yaw" --gpu "$GPU"
               --reviewed-commit "$MERGE" --log-dir "$REC")
  "$PY" tools/exp07_launcher.py smoke "${flags[@]}" \
    --timestamp "${stamp}_smoke"
  "$PY" tools/exp07_launcher.py probe "${flags[@]}" \
    --timestamp "${stamp}_probe"
  nohup setsid tools/exp07_launch.sh full "${flags[@]}" \
    --timestamp "${stamp}_full" \
    --probe-json "ckpt/exp07/$role/_probe_${stamp}_probe_${role}.json" \
    > "$REC/${role}_${stamp}_launcher.log" 2>&1 &
  wait "$!"
}

train07 seen_cyl cylindrical 0
train07 seen_simple simple 0
train07 seen_aug simple 1
```

**Remaining evaluations**, after certified training completion:

```bash
for role in seen_simple seen_cyl seen_aug; do
  backbone=simple
  if [[ "$role" == seen_cyl ]]; then backbone=cylindrical; fi
  for shot in 8 1; do
    for seed in 42 43 44 45 46; do
      eval07 "$role" "$backbone" \
        "ckpt/exp07/$role/final/epoch_012.pth" "$shot" "$seed"
    done
  done
done
for seed in 42 43 44 45 46; do
  eval07 released_seen simple checkpoints/xRIR_seen.pth 1 "$seed"
done
```

**Fill every pin together from committed reviewed code.** The launcher closure includes the probe and shell, matching `build_fields`:

```bash
"$PY" - <<'PY'
import json
import subprocess
from pathlib import Path
from tools import provenance as p
from tools.exp07_profiles import ARMS, APPROVED_DIGESTS_PATH

repo = Path.cwd()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
modules = dict(
    evaluator='tools.exp07_eval', writer='tools.exp07_eval_launch',
    training_launcher='tools.exp07_launcher', training='tools.exp07_train',
    producer_table='tools.exp07_table', producer_pairs='tools.exp07_pairs')
closures = {}
for key, module in modules.items():
    files = p.source_closure(module, repo)
    if key == 'training_launcher':
        files += ['tools/exp07_probe.py', 'tools/exp07_launch.sh']
    records, digest = p.closure_record(files, head, repo)
    assert records and all(
        r['reviewed_blob_sha256'] == r['working_tree_sha256']
        and not r['commits_after_reviewed'] for r in records)
    closures[key] = [digest] if key == 'training_launcher' else digest

checkpoints = {}
for arm in ARMS:
    if arm['reference']:
        continue
    path = repo / arm['checkpoint']
    digest = p.sha256_file(path)
    manifest = json.loads((path.parent / 'train_manifest.json').read_text())
    completion = json.loads((path.parent / 'completion.json').read_text())
    assert completion['outputs']['epoch_012.pth'] == digest
    assert manifest['source_closures']['training']['sha256'] == closures['training']
    assert manifest['source_closures']['launcher']['sha256'] in closures['training_launcher']
    checkpoints[arm['role']] = dict(path=arm['checkpoint'], epoch=12, sha256=digest)

APPROVED_DIGESTS_PATH.write_text(json.dumps(
    dict(schema_version=1, closures=closures, checkpoints=checkpoints),
    sort_keys=True, indent=2) + '\n')
PY
git diff -- "$ASSETS/approved_digests.json"
```

Review the concrete pin fill, then commit before publication:

```bash
git add "$ASSETS/approved_digests.json"
git commit -m 'exp_07: fill reviewed closure and certified checkpoint pins'
"$PY" -c 'from tools.exp07_profiles import load_approved_digests; assert load_approved_digests()[0]["schema_version"] == 1'
```

**Producers → generators → bind with evidence → check:**

```bash
RUNS=(ckpt/exp07/eval/{seen_simple,seen_cyl,seen_aug,released_seen}_k{8,1}_seed{42..46}_k0)
"$PY" tools/exp07_table.py --profile TABLE_SEEN_V1 --runs "${RUNS[@]}" \
  --json "$OUT/TABLE_SEEN_V1.json" \
  --md worklog/worklog_yixun/model_comparison_seen.md

for role in seen_cyl seen_aug released_seen; do
  "$PY" tools/exp07_pairs.py --profile PAIRS_SEEN_V1 \
    --runs-a ckpt/exp07/eval/"${role}"_k{8,1}_seed{42..46}_k0 \
    --runs-b ckpt/exp07/eval/seen_simple_k{8,1}_seed{42..46}_k0 \
    --reconverge --json "$OUT/PAIRS_SEEN_V1_${role}.json" \
    --summary "$OUT/PAIRS_SEEN_V1_${role}.txt"
done

UNSEEN=ckpt/yaw_aug/results/TABLE_V1.json
UNSEEN_BINDING=$("$PY" -c \
  'from pathlib import Path; print(sorted(Path("ckpt/yaw_aug").glob("binding_report_*.json"))[-1])')
PAIRS=("$OUT"/PAIRS_SEEN_V1_{seen_cyl,seen_aug,released_seen}.json)
COMMON=(--table "$OUT/TABLE_SEEN_V1.json" --pairs "${PAIRS[@]}"
        --unseen-table "$UNSEEN" --unseen-binding "$UNSEEN_BINDING")

"$PY" "$ASSETS/make_results_md.py" "${COMMON[@]}" \
  --out "$REC/seen_protocol_results.md"
"$PY" "$ASSETS/make_results_html.py" "${COMMON[@]}" \
  --out "$REC/seen_protocol_01_results.html"
"$PY" "$ASSETS/make_latex.py" "${COMMON[@]}" \
  --out "$ASSETS/table_seen_unseen.tex"

"$PY" "$ASSETS/bind_provenance.py" --runs "${RUNS[@]}" \
  --attempt ckpt/exp07/{seen_simple,seen_cyl,seen_aug}/final \
  --audit ckpt/exp07/alignment_audit_seen.json \
  --evidence "gpu_parity=$PARITY_RECEIPT" \
  --evidence "calibration=$OUT/CALIBRATION_SEEN_V1.json" \
  --results "$OUT/TABLE_SEEN_V1.json" "${PAIRS[@]}" \
  --rendered "$REC/seen_protocol_results.md" \
    "$REC/seen_protocol_01_results.html" "$ASSETS/table_seen_unseen.tex" \
  --unseen-table "$UNSEEN" --unseen-binding "$UNSEEN_BINDING" \
  --out ckpt/exp07
"$PY" "$ASSETS/check_record.py" ckpt/exp07
```
