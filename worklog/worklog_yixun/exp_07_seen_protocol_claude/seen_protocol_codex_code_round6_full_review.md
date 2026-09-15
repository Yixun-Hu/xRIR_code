# exp_07 seen_protocol — Codex code review, round 6 + repeated integrative review (`full`)

**Reviewer:** OpenAI Codex `gpt-6-astra` (Ultra), `codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07`, prompt `review_prompts/code_round6_full_prompt.md`, log `seen_protocol_2026-09-15_16:49:12_codex_code_round6_full.log`. **Scope:** exp07-window `2a27144..9c3b458` (Part A) and `f492613..9c3b458` (Part B). **Date:** 2026-09-15 16:49–17:26 EDT.

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

# PART A — Coder round 6

**Verdict: request changes. Blocking findings: 1–4.**

Reviewed the 19 exp_07 non-merge commits in `2a27144..9c3b458`, excluding the two main merges.

Amendment A1’s relocation is structurally complete. The previous merge blockers are resolved. However, publication admission still accepts incomplete or contradictory evidence, and relocation removed one launcher refusal.

## Findings

### 1. Blocker — binder coverage still permits whole-run omissions and unchecked dependencies

[bind_provenance.py:225](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:225)

Coverage is checked only for runs remaining in `side['run_flags']`. Other declared inputs are compared with a partial `known_digests` map; unknown dependencies generally escape validation.

**Independently reproduced, repairing outer hashes and retaining the fixture’s approval pins:**

- Remove one table run from `run_flags`, contracts, and inputs: accepted.
- Remove a required `train_inventory.json` dependency: accepted.
- Substitute the declared reference-manifest digest: accepted.
- Add a nonexistent dependency with an invented digest: accepted.

Thus, complete table rows do not establish complete provenance.

**Minimal fix:** derive each product’s expected run set from the bound role/K/seed identities; require exact coverage and required training dependencies. Validate every declared dependency against its bound artifact, rejecting unknown or orphaned inputs.

### 2. Blocker — calibration has no producer, and gating evidence can describe failed or unrelated checks

[bind_provenance.py:64](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:64), [README.md:33](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/README.md:33)

There is **no calibration producer**. The schema requires hand-supplied means and SDs without binding or deriving them from the five released-checkpoint K=8 evaluations.

A fabricated, finite summary with `mean = 1000000` and `sd = 1000000` for all three metrics was accepted and serialized successfully. The formula is recomputed, but its operands are unverified.

Likewise, a parity log containing **`9 failed, 0 passed`** was accepted: the only content check is non-emptiness.

**Minimal fix:**

- Add a reviewed calibration producer for the five validated released-checkpoint runs, seeds 42–46. Derive finite seed means, sample SDs, and the existing acceptance rule; bind the evaluations, checkpoint, manifests, and reviewed commit. This must work before new-arm approval pins exist.
- Validate a parity receipt recording the exact required tests, successful exit, zero failures/skips, reviewed identity, and log digest.

### 3. Blocker — aborted/probe attempt logs remain outside the binding

[bind_provenance.py:42](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:42)

`other_attempts()` hashes files beneath each directory. The real launcher deliberately keeps logs **outside** those directories.

**Reproduced:**

- Changing an aborted attempt’s referenced external log leaves the report identical.
- Changing a probe completion’s referenced external log leaves the report identical.
- Removing `abort.json` while leaving other attempt files is accepted.

Changing `abort.json` itself is now detected, so that exact previous mutation is fixed.

**Minimal fix:** validate each ledger-listed attempt’s terminal state and bind its referenced external evidence. Require the appropriate completion or abort receipt, account for legitimate setup failures, and validate probe-to-receipt linkage.

### 4. Blocker — launcher-side split refusal was lost, and the binder accepts contradictory output metas

[tools/exp07_eval_launch.py:110](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_eval_launch.py:110), [bind_provenance.py:155](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:155)

The relocated launcher delegates completion to shared `execute_run`, which does not compare `split` or `seen_split_sha256`. The previous `test_a_child_that_evaluated_another_split_is_not_finalized` was removed.

Using a stub child, **`exp07_eval_launch.main()` certified a seen manifest whose output metas said unseen and carried another split digest**.

Separately, changing both output metas to unseen in the bound fixture, then repairing completion/product hashes, was accepted by the binder. Table/pairs admission correctly rejected that fixture.

Keeping split fields out of `completion.json` is sound: its manifest/output hashes bind them. The missing requirement is agreement among those bound fields.

**Minimal fix:** restore independent split/digest validation before launcher certification and in binder admission, using exp_07-owned code. Restore mutation tests for both output metas and both fields.

### 5. Should-fix — four GPU trainer tests bypass the relocated entry point

[tests/test_exp07_trainer.py:133](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_trainer.py:133)

Both sides call the same `trainer.build_model` and `trainer.compute_loss`; only the parsed namespace differs. Neither side invokes `exp07_train.run_trainer`.

These tests would miss an entry-point regression involving its scoped parser/dataset patches or RNG use.

**Minimal fix:** exercise the actual entry point for a bounded real-batch step and compare initialization, loss, gradients, update, and RNG state against the unchanged trainer for both backbones and yaw settings.

### 6. Should-fix — a hardlink output alias can overwrite an input

[make_results_md.py:190](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_results_md.py:190)

`Path.resolve()` catches ordinary aliases and symlinks, but not hardlinks. Creating a hardlink from a synthetic exp_04 binding report to the requested Markdown output caused the generator to overwrite that input.

**Minimal fix:** compare existing destinations with protected inputs using filesystem identity (`samefile` or device/inode), and test this across all three generators.

### 7. Should-fix — documented repeated `--evidence` drops the first value

[bind_provenance.py:300](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:300)

The parser uses `nargs='+'` with ordinary store behavior. Therefore:

```text
--evidence gpu_parity=g.log --evidence calibration=c.json
```

retains only `calibration=c.json`.

**Minimal fix:** accumulate repeated occurrences and flatten them, or consistently document and test the current working syntax:

```text
--evidence gpu_parity=g.log calibration=c.json
```

### 8. Nit — sufficiently small positive SDs still print as zero

[make_latex.py:68](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_latex.py:68)

`sd_number(1e-13, 3)` returns `0.000000000000`. The twelve-decimal cap contradicts the stated positive-SD invariant.

**Minimal fix:** use scientific notation when the decimal cap would still produce zero.

### 9. Nit — the advertised removal of `--entry` is incomplete

[tools/exp07_eval_launch.py:44](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_eval_launch.py:44)

The public parser still exposes `--entry {exp07}`. It cannot select another evaluator, but “no `--entry` anywhere” is inaccurate.

**Minimal fix:** remove the redundant option and update the corresponding documentation/tests.

## Verified

### Relocation, protocol seam, and closures

- All nine named restored modules and **55 pre-existing test paths** match main’s bytes.
- Relative to `fc80c49`: **43 new files, 7,450 insertions, zero deletions**.
- `--protocol unseen` produces the trainer’s exact namespace plus `protocol`.
- `protocol` reaches both `args.json` and `XRIR_RUNTIME_ARGS`.
- Independent checks found no Python, NumPy, or Torch RNG consumption by the seam; parser, dataset, and `sys.argv` bindings restore on exceptions.
- `source_closure('tools.exp07_train')` contains **13 files**: the unchanged trainer’s 11-file closure plus the entry point and seen dataset module. Both dataset modules are present.
- Training manifests derive their training closure from **`tools.exp07_train`**, consistently with launcher minimum checks and producer admission.
- The launcher closure is derived from `tools.exp07_launcher` **plus `tools/exp07_probe.py` and `tools/exp07_launch.sh`**, deduplicated by `closure_record`. The shell must be included when filling its approval.
- The real Planner cache loads through `exp07_provenance.train_data_identity`: **296,454 files**, inventory digest `b18e9ec3488c1c15a6e1d074ab0d99169b6ccee5c8724eadc12adbb0e7832631`; cache bytes unchanged.

### `seen_overrides()`

The actual override set is:

```text
REPO, TRAIN_MINIMUM, tier_gates, child_environment,
check_golden, check_runtime, compare_control,
complete_attempt, recovery_evidence, LogGuard
```

The reused functions’ module-global lookups were enumerated:

| Function | Module globals |
|---|---|
| `execute_attempt` | `LauncherFailure`, `LogGuard`, `Path`, `abort_attempt`, `abort_log`, `check_budget`, `complete_attempt`, `create_attempt`, `datetime`, `diagnostic_value`, `full_hours`, `p`, `resource_gate`, `tier_gates`, `time` |
| `complete_attempt` | `_complete_attempt`, `p` |
| `_complete_attempt` | `LauncherFailure`, `Path`, `REQUIRED_INPUTS`, `account_hours`, `datetime`, `directory_listing`, `json`, `os`, `p`, `promote`, `tier_gates`, `uuid`, `validate_outputs` |
| `finalize_attempt` | `LogGuard`, `PYTHON`, `Path`, `assert_quiescent`, `complete_attempt`, `datetime`, `fcntl`, `hours_record`, `json`, `math`, `p`, `recovery_evidence`, `sys`, `tier_gates` |
| `run_child` | `LauncherFailure`, `REPO`, `child_environment`, `datetime`, `json`, `os`, `p`, `subprocess`, `time` |

Transitive checks cover `validate_outputs`’s `check_runtime` and the inherited guard’s globals. Its unpatched `tier_probe` branch applies to non-M tiers; exp_07 enforces M.

**No missing reachable experiment-specific override was found.** The set is somewhat broader than necessary: exp_07 calls its own golden/control checks directly.

Importing exp_04 first, nested contexts, and inner/outer exceptions all restored the complete exp_04 module namespace **by identity**. No lingering monkeypatch was found. The bound default `runner=run_child` still resolves patched globals correctly.

### Retained behavior and mutation replay

- Probe/evaluator/writer restatements retain the reviewed numerical and transaction paths; GPU numerical parity remains deferred.
- Seen pickle bytes remain bound through manifest → completion → binder. The semantic disagreement hole is finding 4.
- Round-4 closure-record and execution-contract mutations now refuse.
- Round-5 per-cell convergence/decision/pairing mutations, empty tables, non-final pairs, single missing run files, wrong known inventory digests, and false approval paths now refuse.
- Checkpoint substitution on disk is refused.
- Registered 20,000 → 40,000 reconvergence exists; a second failure remains non-final.
- Exploratory disclosure, required spectral metrics, approval lifecycle/immutability, K-labelled footnotes, and producer → generator → binder → checker ordering are retained.

The identified lost/weakened checks are findings **4 and 5**. Remaining incomplete fixes are detailed above.

Independent reproductions are retained in [test_review_adversarial.py](/home/yixunhu/codespace/xRIR_code_wt07/.review-round6/test_review_adversarial.py), [adversarial.log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round6/adversarial.log), and [additional-adversarial.log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round6/additional-adversarial.log). These comprise 23 reviewer test cases, including assertions confirming the counterexamples.

---

# PART B — FINAL INTEGRATIVE REVIEW (`full`)

**Verdict: not approved. Blocking findings: Part A 1–4.**

The previous review’s blockers **9 and 10 are resolved**. Merge preservation checks pass; the remaining blockers concern exp_07’s own certification/publication contract.

## Plan §3 coverage under A1

| Planned round | Delivery |
|---|---|
| 1 — protocol, inventory, audit | Relocated and CPU-verified. GPU entry-point coverage needs finding 5; real GPU evidence remains pending. |
| 2 — launcher, probes, gates, recovery | Relocated; goldens, timing limits, retry rules, scoped dispatch, and finalization checks pass. |
| 3 — seen evaluator and launcher | Numerical loop and split bindings delivered. Launcher refusal regressed: finding 4. Calibration producer missing: finding 2. |
| 4 — profiles, approvals, table | Delivered; previous training-admission defects and checkpoint-byte check fixed. |
| 5 — paired producer | Delivered, including registered reconvergence. |
| 6 — renderers, binder, checker | Present, but incomplete under findings 1–4 and 6–8. |

## Verified

### Closure rule and merge safety

- `git diff --name-only f492613..9c3b458`: **241 paths**.
- Every changed non-worklog path is absent from `f492613`.
- **Closure-rule exceptions: none.**
- Dry-run `--no-ff --no-commit` merge of `9c3b458` into main **`6f14b3b`** in an isolated clone: **exit 0, no conflicts**.
- The exp_06 parser-contract test passed on that merged tree.

The real exp_04 command also completed with **exit 0** from the branch clone:

```bash
python worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_results_assets/check_record.py ckpt/yaw_aug
```

Because recorded paths can point back to MAIN, I additionally compared the recorded closures against **branch-clone bytes**: all **15 source files across 46 evaluations** matched. Exp_06’s filled approval also matches the current branch closure. See [structure.log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round6/structure.log).

### Definitive CPU suite

Exact HEAD: **`9c3b458`**, isolated clone, prescribed Python 3.8 interpreter, empty `CUDA_VISIBLE_DEVICES`, bytecode disabled, and `-p no:cacheprovider`.

```text
python -m pytest tests -q -p no:cacheprovider
2481 passed, 46 skipped, 211 warnings in 1878.85s
```

**Zero failures.** [Full log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round6/full-suite.log)

The requested historical suites were included:

| Suite | Passed | Skipped |
|---|---:|---:|
| `test_exp04*` | 368 | 1 |
| `test_exp05*` | 250 | 13 |
| `test_exp06*` | 816 | 3 |
| `test_exp07*` | 377 | 9 |
| `test_provenance.py` | 32 | 0 |
| `test_yaw_aug.py` | 106 | 4 |
| `test_results_table.py` | 27 | 0 |
| `test_paired_compare.py` | 75 | 0 |

`static_checks.sh`: **exit 0**:

- Publication subset: **200 passed**.
- Exp07 → exp04 → exp03 collection order: **156 passed, 1 skipped**.
- Reverse order: **156 passed, 1 skipped**.

[Static-check log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round6/static-checks.log)

### Hygiene

- No GPU work or tracked source edits were performed. Review writes stayed under `.review-round6` in this worktree.
- `git diff --check` and `git diff --check fc80c49..9c3b458`: **exit 0**.
- Whole-range `git diff --check f492613..9c3b458`: **exit 2**, with **5,084 whitespace warnings, all in inherited worklog records**.
- Process-record correction: **eight**, rather than six, commits exceed 200 total changed lines: `9d13581`, `1dcc5e5`, `f217d43`, `2074100`, `5db051f`, `ea3ce0b`, `4c6e842`, `7ffea95`. Their coherent relocation/fix scope is reviewable; correct the deviation count.

## Deferred GPU tests

Exact current node IDs:

```text
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_the_trainer[False-simple-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_the_trainer[False-cylindrical-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_the_trainer[True-simple-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_the_trainer[True-cylindrical-32]
tests/test_exp07_eval_gpu.py::test_unseen_32_query_run_is_bit_identical_to_exp04
tests/test_exp07_eval_gpu.py::test_seen_final_batch_matches_a_direct_frozen_call[8]
tests/test_exp07_eval_gpu.py::test_seen_final_batch_matches_a_direct_frozen_call[1]
tests/test_exp07_eval_gpu.py::test_the_launcher_runs_the_seen_split_and_matches_the_frozen_functions[8]
tests/test_exp07_eval_gpu.py::test_the_launcher_runs_the_seen_split_and_matches_the_frozen_functions[1]
```

Required artifacts:

| Tests | Requirements |
|---|---|
| Four trainer cases | Free GPU and real unseen AcousticRooms K=8 batch; no historical checkpoint required. Strengthen these under finding 5. |
| Unseen evaluator parity | `ckpt/xRIR_simple_8_shot/epoch_12.pth`, `ckpt/yaw_aug/reference_manifest_k8_seed42.json`, referenced data. |
| Direct seen final-batch cases | Same checkpoint, frozen seen pickle, seen data; tests construct the final 25-query manifests. |
| Seen launcher cases | Same checkpoint, seen seed-42 manifests at K=1 and K=8, referenced data, reviewed checkout. |

## Planner launch sequence

**Use this sequence after the blocking fixes and their review.** These commands reflect the relocated interfaces.

### 1. Merge, parity, audit

From the MAIN checkout, set `GPU` to an available physical device, `0` or `1`:

```bash
git switch main
git merge --no-ff exp07-window
MERGE=$(git rev-parse HEAD)

PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
REC=worklog/worklog_yixun/exp_07_seen_protocol_claude
ASSETS="$REC/seen_protocol_results_assets"

: "${GPU:?Set GPU to an available physical device}"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export PYTHONHASHSEED=0 OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="$GPU"
set -o pipefail

PARITY="$REC/gpu_parity_${MERGE}.log"
"$PY" -m pytest \
  tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_the_trainer \
  tests/test_exp07_eval_gpu.py -q -p no:cacheprovider 2>&1 | tee "$PARITY"
```

Require all nine cases to pass, with no skips, before continuing.

```bash
"$PY" tools/exp07_audit.py --protocol seen \
  --out ckpt/exp07/alignment_audit_seen.json \
  --n-batches 3 --batch-size 32 --num-workers 12 --seed 0
```

The ten reference manifests already exist. If constructing them in a fresh layout, the command is:

```bash
"$PY" tools/exp07_manifests.py --data-root "$XRIR_DATA_PATH" \
  --out-dir ckpt/exp07 --num-shots 8 1 --seeds 42 43 44 45 46
```

### 2. Released-checkpoint calibration

Use this helper for calibration and later evaluations:

```bash
mkdir -p ckpt/exp07/eval ckpt/exp07/results

eval07() {
  local role=$1 backbone=$2 checkpoint=$3 k=$4 seed=$5
  local ref="ckpt/exp07/reference_manifest_seen_k${k}_seed${seed}.json"
  local hash
  local binds=()

  hash=$("$PY" -c \
    'import sys; from tools.reference_manifest import load_manifest,manifest_hash; print(manifest_hash(load_manifest(sys.argv[1])))' \
    "$ref")

  if [ "$role" != released_seen ]; then
    local final="ckpt/exp07/$role/final"
    binds=(--bind-input "train_args=$final/args.json"
           --bind-input "train_manifest=$final/train_manifest.json"
           --bind-input "train_completion=$final/completion.json")
  fi

  "$PY" tools/exp07_eval_launch.py --split seen \
    --backbone "$backbone" --checkpoint "$checkpoint" --num-shot "$k" \
    --manifest "$ref" --manifest-hash "$hash" --gl-seed "$seed" \
    --conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols \
    --decomposition-batches 0 --batch-size 16 --max-samples 0 \
    --num-workers 12 --threads 2 --data-root "$XRIR_DATA_PATH" \
    --gpu "$GPU" --reviewed-commit "$MERGE" --log-dir "$REC" \
    --run-label "${role}_k${k}_seed${seed}_k0" \
    --out-dir "ckpt/exp07/eval/${role}_k${k}_seed${seed}_k0" \
    "${binds[@]}"
}

for seed in 42 43 44 45 46; do
  eval07 released_seen simple checkpoints/xRIR_seen.pth 8 "$seed"
done
```

**The next command is missing from the branch:** nothing produces the required calibration JSON from these five runs. Finding 2 must supply that producer. The ordinary table producer cannot serve this pre-training gate because production admission requires the new-arm pins.

Apply the registered historical comparison in EDT seconds, C50 dB, and T60 percent. If calibration fails, follow the plan’s investigation procedure before proceeding.

### 3. Per-arm smoke, probe, full

Use these settings, one scheduled arm at a time:

| `ROLE` | `BACKBONE` | `YAW` |
|---|---|---:|
| `seen_simple` | `simple` | 0 |
| `seen_cyl` | `cylindrical` | 0 |
| `seen_aug` | `simple` | 1 |

```bash
STAMP=$(date +%Y%m%dT%H%M%S%N)

tools/exp07_launch.sh smoke --backbone "$BACKBONE" --yaw-aug "$YAW" \
  --gpu "$GPU" --reviewed-commit "$MERGE" --log-dir "$REC" \
  --timestamp "$STAMP"

tools/exp07_launch.sh probe --backbone "$BACKBONE" --yaw-aug "$YAW" \
  --gpu "$GPU" --reviewed-commit "$MERGE" --log-dir "$REC" \
  --timestamp "$STAMP"

RECEIPT="ckpt/exp07/$ROLE/_probe_${STAMP}_${ROLE}.json"

nohup setsid tools/exp07_launch.sh full \
  --backbone "$BACKBONE" --yaw-aug "$YAW" --gpu "$GPU" \
  --reviewed-commit "$MERGE" --probe-json "$RECEIPT" \
  --log-dir "$REC" --timestamp "$STAMP" \
  > "$REC/${ROLE}_${STAMP}_launcher.log" 2>&1 &
```

### 4. Remaining evaluations and pin finalization

After certified training completion:

```bash
for role in seen_simple seen_cyl seen_aug; do
  backbone=simple
  if [ "$role" = seen_cyl ]; then backbone=cylindrical; fi
  for k in 8 1; do
    for seed in 42 43 44 45 46; do
      eval07 "$role" "$backbone" \
        "ckpt/exp07/$role/final/epoch_012.pth" "$k" "$seed"
    done
  done
done

for seed in 42 43 44 45 46; do
  eval07 released_seen simple checkpoints/xRIR_seen.pth 1 "$seed"
done
```

Retain the five K=8 calibration runs, giving 40 evaluations total.

Fill and review the approval JSON together: three certified checkpoint hashes; training closure from `tools.exp07_train`; launcher closure including the shell as specified above; evaluator/writer from `tools.exp07_eval`/`tools.exp07_eval_launch`; producer closures from `tools.exp07_table`/`tools.exp07_pairs`. Compute from committed reviewed code and require agreement with the recorded runs.

```bash
git add "$ASSETS/approved_digests.json"
git commit -m "exp_07: finalize checkpoint and closure pins"
```

### 5. Producers → generators → bind → check

```bash
RUNS=(ckpt/exp07/eval/*_k[18]_seed4[2-6]_k0)
TABLE=ckpt/exp07/results/TABLE_SEEN_V1.json

"$PY" tools/exp07_table.py --profile TABLE_SEEN_V1 \
  --runs "${RUNS[@]}" --json "$TABLE" \
  --md worklog/worklog_yixun/model_comparison_seen.md

for role in seen_cyl seen_aug released_seen; do
  "$PY" tools/exp07_pairs.py --profile PAIRS_SEEN_V1 \
    --runs-a ckpt/exp07/eval/"${role}"_k{1,8}_seed{42..46}_k0 \
    --runs-b ckpt/exp07/eval/seen_simple_k{1,8}_seed{42..46}_k0 \
    --reconverge \
    --json "ckpt/exp07/results/PAIRS_SEEN_V1_${role}.json" \
    --summary "ckpt/exp07/results/PAIRS_SEEN_V1_${role}.txt"
done
```

Require all pairings to be final. A later retry invocation needs fresh output paths.

Set `UNSEEN_BINDING` to the verified exp_04 report and `CALIBRATION` to the validated report produced by the finding-2 fix:

```bash
PAIRS=(ckpt/exp07/results/PAIRS_SEEN_V1_{seen_cyl,seen_aug,released_seen}.json)
UNSEEN=ckpt/yaw_aug/results/TABLE_V1.json
COMMON=(--table "$TABLE" --pairs "${PAIRS[@]}"
        --unseen-table "$UNSEEN" --unseen-binding "$UNSEEN_BINDING")

"$PY" "$ASSETS/make_results_md.py" "${COMMON[@]}" \
  --out "$REC/seen_protocol_results.md"
"$PY" "$ASSETS/make_results_html.py" "${COMMON[@]}" \
  --out "$REC/seen_protocol_01_results.html"
"$PY" "$ASSETS/make_latex.py" "${COMMON[@]}" \
  --out "$REC/table_seen_unseen.tex"

"$PY" "$ASSETS/bind_provenance.py" \
  --runs "${RUNS[@]}" \
  --attempt ckpt/exp07/{seen_simple,seen_cyl,seen_aug}/final \
  --audit ckpt/exp07/alignment_audit_seen.json \
  --evidence "gpu_parity=$PARITY" "calibration=$CALIBRATION" \
  --results "$TABLE" "${PAIRS[@]}" \
  --rendered "$REC/seen_protocol_results.md" \
    "$REC/seen_protocol_01_results.html" "$REC/table_seen_unseen.tex" \
  --unseen-table "$UNSEEN" --unseen-binding "$UNSEEN_BINDING" \
  --out ckpt/exp07

"$PY" "$ASSETS/check_record.py" ckpt/exp07
```

**Final verdicts:** Part A **request changes**; Part B **not approved**.  
**Exact blocking list: findings 1, 2, 3, and 4.**
