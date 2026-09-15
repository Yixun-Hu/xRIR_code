**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15 · log `seen_protocol_2026-09-15_14:59:08_codex_code_round5_full.log`

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

# PART A — Coder round 5

**Scope:** `7b3ad6c..2a27144`, all 16 commits; **1,724 insertions / 46 deletions across 15 files**.

**Verdict: request changes. Blocking findings: 1, 2, 3.**

The round-4 blockers are closed in producer admission. The new publication and record-binding layers still admit inconsistent or incomplete evidence.

## Findings

### 1. Blocker — generators accept unconverged and contradictory paired cells

[make_results_md.py:87](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_results_md.py:87)

`check_pairs()` validates aggregate finality and cell coverage, but omits the corresponding checks inside each cell.

**Reproduced:** with the canonical JSON’s sidecar digest repaired, shared generator admission accepts each independent mutation:

- `statistics.absolute.reconverge_required = true`
- `statistics.relative.convergence.passed = false`
- `cell.decision_driving = true`
- A cell pairing inconsistent with its parent pairing

Markdown, HTML and LaTeX all use this admission path. Thus an interval explicitly requiring reconvergence can reach publication.

**Minimal fix:** validate both statistics’ convergence/finality, every cell’s descriptive status and pairing identity, and their consistency with aggregate flags. Share this validation with the binder.

### 2. Blocker — the binder accepts incomplete products and mismatched producer inputs

[bind_provenance.py:93](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:93)

The binder calls `md.load()` without table/pair completeness validation. Its input check filters selected filenames and accepts a **subset** of bound run inputs. Exp_04’s additional completeness predicate was dropped.

**Reproduced, retaining the genuine approval pins:**

- Nonfinal pairs with a nonempty reconvergence list
- A table with `rows = []`
- Removal of one run’s `metrics_yaw.json` provenance input
- A substituted training-inventory input digest
- An approval path replaced with `/nonexistent/approved.json`

The altered JSON and sidecar were kept internally consistent; current JSON digests were cited in the synthetic rendered documents. `collect()` accepted all five cases.

**Why:** the resulting report certifies products whose claimed evidence is incomplete or differs from the evidence actually bound. `check_record.py` repeats the same acceptance.

**Minimal fix:** apply complete canonical validation, require exact per-product run/input coverage, and verify every producer dependency and approval identity against its bound artifact. Restore the omitted completeness predicate.

### 3. Blocker — the final record omits required execution evidence

[bind_provenance.py:43](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:43), [bind_provenance.py:146](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:146)

Plan §3 requires parity/calibration/audit evidence and **every attempt, including aborted attempts**. The implementation binds three successful final attempts, receipts and ledger bytes. It does not traverse earlier attempts or probe-attempt evidence, and has no interface for parity/calibration evidence.

**Reproduced:**

- Added a ledger-listed aborted full attempt, bound the record, changed its `abort.json`: recomputation returned an **identical report**.
- An unrelated audit containing only `{"protocol":"unseen","passed":false}` was accepted.

The maximum-two-full-attempts assertion works on the supplied ledger, but does not establish the promised complete attempt history.

**Minimal fix:** bind lifecycle evidence for every ledger-listed attempt, including aborted/probe attempts; add parity and calibration evidence; validate the actual seen-audit schema, cohort and execution identity.

### 4. Should-fix — the registered 40,000-resample rerun remains unavailable

[tools/exp07_pairs.py:196](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_pairs.py:196)

The CLI always uses the fixed 20,000-resample profile. It tells users to rerun flagged cells with a larger count but supplies no reproducible registered path for doing so.

**Why:** plan §2’s publication procedure cannot be completed through the approved interface.

**Minimal fix:** implement the prescribed 40,000-resample retry, retaining the original diagnostic, retry count and provenance. Continued failure must remain nonfinal.

### 5. Should-fix — generator output can overwrite the exp_04 binding report

[make_results_md.py:142](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_results_md.py:142)

Output-overlap protection includes canonical products and sidecars but omits `--unseen-binding`.

**Reproduced:** setting `--out` equal to the synthetic exp_04 binding-report path replaced that JSON with Markdown.

**Minimal fix:** protect the resolved binding-report path alongside every other input, including aliases; verify refusal preserves its bytes.

### 6. Should-fix — runtime checkpoint agreement is still not tested

[tests/test_exp07_profiles.py:244](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_profiles.py:244)

New-arm profile digests remain `None` intentionally. Therefore this assertion:

```python
assert not path.exists() or prov.sha256_file(path)
```

accepts any readable checkpoint. It never compares against the filled runtime approval.

**Minimal fix:** load the authoritative runtime pin and compare available checkpoint bytes and completion linkage against it. Exercise matching and substituted checkpoints using isolated fixtures.

Round-4’s lifecycle and inventory-agreement requests are closed; its checkpoint-agreement request remains incomplete.

### 7. Should-fix — LaTeX SD footnotes omit K and can print positive SDs as zero

[make_latex.py:68](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_latex.py:68)

`note(name, shot, rows)` ignores `shot`, producing two indistinguishably labelled SD entries for each recipe method. Fixed precision also violates the plan’s nonzero-display requirement: an SD of `0.000001` prints as zero in all three columns.

**Minimal fix:** label every footnote with K and use sufficient precision for positive SDs, while preserving genuine zeros.

### 8. Nit — README publication order cannot complete its first binding

[README.md:53](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/README.md:53)

The instructions run bind/check before generating documents, although binding requires nonempty `--rendered` files containing the canonical digests.

**Minimal fix:** document **producers → generators → bind → check**.

## Verified

### Round-4 closure

| Previous item | Result |
|---|---|
| Blocker 1: unchecked training/launcher closure records | Closed. Empty records, unreviewed records and incorrect digest labels refuse. |
| Blocker 2: training execution/data contract | Closed for every listed mutation: inventory count/digest, smoke mode, dirty flag, effective epochs/backbone/yaw/save directory, and 1,000-hour projection. |
| Required spectral metrics | Closed; all five registered metrics required. |
| Exploratory Markdown disclosure | Closed; status and deviations displayed. |
| Approval lifecycle and inventory agreement | Closed. |
| Runtime checkpoint agreement | Finding 6 remains. |
| Recursive approval immutability | Closed. |

The fixture now creates a real reviewed Git commit and represents inventory evidence, receipt-derived limits, normalized args and twelve checkpoint entries. All **40 named admission refusals** pass. This verifies current behavior; it does not independently establish the claimed historical red-before-fix ordering.

### Renderers, statistics and binding

- Table aggregation preserves five-seed means, sample SD (`ddof=1`), EDT units and finite-count tolerance.
- Paired statistics preserve shared query resampling, whole-room query weighting, paired cohorts, absolute/relative statistics and descriptive status.
- Normal Markdown/HTML values trace to canonical JSON; generation is deterministic.
- Top-level exploratory, reconvergence, incomplete-coverage and unbound-unseen refusals work.
- Combined tables validate exp_04’s JSON through its sidecar and binding-report digest.
- LaTeX matches the reference’s seven-column seen/unseen structure; mean rounding, unique-minimum bolding, unbolded ties, reference-row exclusion and canonical-JSON caption pointers work. Finding 7 affects its SD footnotes.
- Valid synthetic binding/checking covers three final attempts, the released checkpoint, forty evaluations, outputs, companions, approval bytes and exp_04 inputs. Findings 2–3 limit what that certification proves.
- The default exp_04 `write_outputs()` path produced **byte-identical JSON, Markdown and sidecar** against the prechange implementation with time/output location held fixed. The diff contains two behavioral edits plus a docstring: **3 insertions / 2 deletions**.

Dataset pins agree with available artifacts: ten manifest hashes, 6,217-query identity `be5095a8…`, and inventory `251b7d7b…`. Both K values reference the same complete file set. The new literals `296454`, `60`, and `1.5` agree with launcher contracts. Checkpoint/closure approval pins correctly remain unfilled at this stage.

Independent counterexamples and round-4 closure replays are retained in [the review script](/tmp/exp07-review-edgge526/test_review_adversarial.py) and [its log](/tmp/exp07-review-edgge526/adversarial.log).

---

# PART B — FINAL INTEGRATIVE REVIEW (`full`)

**Scope:** `f492613..2a27144`; **5,483 insertions / 116 deletions across 45 files**.

**Verdict: not approved. Blocking findings: 1, 2, 3, 9, 10.**

## Additional blocking findings

### 9. Blocker — merging breaks exp_04’s existing record check through live source validation

Change site: [tools/exp04_eval.py:129](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp04_eval.py:129).

Verification path: [exp_04 check_record.py:20](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_results_assets/check_record.py:20) → [snapshot():57](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_results_assets/bind_provenance.py:57) → [provenance.revalidate():385](/home/yixunhu/codespace/xRIR_code_wt07/tools/provenance.py:385).

**The precise answer to the producer-pin question:** the checker does **not** recompute the producer closure at HEAD against `6d32a5e2…`. It compares recorded producer/approval identities. Nevertheless, the merge breaks checking because evaluation snapshots rehash recorded evaluator/writer source paths against the **live repository**.

Using an actual bound exp_04 evaluation manifest:

- Source revalidation against main’s premerge files: **no mismatches**.
- Against the dry-run merged files: mismatches for:
  - Entry-point closure: `tools/exp04_eval.py`, `tools/provenance.py`
  - Writer closure: those two files plus `tools/exp04_eval_launch.py`

Passing the report’s recorded HEAD controls ancestry/report identity; it does not restore historical source files.

Separately, fresh `results_table` production changes closure from approved `6d32a5e2…` to `1856f447…`; both `tools/results_table.py` and `tools/provenance.py` contribute.

**Minimal fix:** preserve the historical shared evaluation closures through exp_07-specific implementations, or review a historical-source verification migration that checks complete recorded closures against immutable reviewed Git blobs while retaining strict artifact/data checks. Moving only the renderer hook or repinning only the producer does **not** repair the checker failure. Historical exp_05 M-run reuse needs the same treatment, or new evaluations under the reviewed evaluator.

### 10. Blocker — current main’s exp_06 integration fails, and filled approvals become stale

[train_xRIR_backbone.py:61](/home/yixunhu/codespace/xRIR_code_wt07/train_xRIR_backbone.py:61), [exp_06 parser contract:50](/home/yixunhu/codespace/xRIR_code/tests/test_exp06_train.py:50).

Main advanced during review and now includes exp_06. On main **`53307b6`**, this exact test passes before merging and fails afterward:

```text
tests/test_exp06_train.py::test_shared_flags_and_defaults_equal_the_trainers
```

Failure: the shared trainer has the additional `protocol` argument.

Moreover, hashing the merged source files changes these nine filled exp_06 code approvals:

```text
trainer, finalize, smoke, heading, profiles,
eval, eval_launch, haa_finetune, haa_eval
```

For example, its trainer closure changes from `5a2c6df0…` to `cbe84cfb…`. Exp_06’s admission explicitly rejects mismatched code approvals.

**Minimal fix:** review the parser contract’s treatment of exp_06’s fixed unseen protocol, rerun integration checks, and review/refill affected approvals for subsequent exp_06 execution. Preserve any already-started attempt’s original provenance.

Directly disjoint source edits do not establish dependency compatibility.

## Verified

### Plan §3 delivery

| Planned round | Assessment |
|---|---|
| 1 — protocol, inventory, audit | Implemented; real split sizes/loading verified. GPU parity, real seen training smoke and audit evidence remain deferred. |
| 2 — launcher, probes, timing/retry gates | Implemented; CPU contracts and goldens pass. Actual receipts and training evidence await GPU execution. |
| 3 — evaluator and launcher | Implemented; CPU refusals pass. Five evaluation GPU tests remain deferred. |
| 4 — profiles and table producer | Implemented; round-4 blockers closed. Finding 6 remains. |
| 5 — paired producer | Implemented except the prescribed 40,000-resample rerun path. |
| 6 — renderers and bound record | Implemented but incomplete: findings 1–3 and 5–8. |

Thus **all six implementation areas exist, but the plan is not fully delivered**.

### Frozen closure rule

This command returns **no paths**:

```bash
git diff --name-only f492613..2a27144 -- \
  eval_yaw_rotation.py eval_unseen.py eval_xRIR_backbone.py \
  'model/*.py' tools/yaw_rotation.py tools/reference_manifest.py \
  tools/per_sample_metrics.py 'treble_multi_room_dataset/*.py' \
  utils/spec_utils.py tools/paired_stats.py tools/summarize_yaw.py
```

The exp_03 frozen evaluator closure also retains its approved digest in the latest merged clone.

### Shared-file consequences

All eight specifically named files changed:

| File | Consequence |
|---|---|
| `train_xRIR_backbone.py` | Adds protocol dispatch and the seen module to training closure membership. |
| `tools/provenance.py` | Adds protocol-aware training identity; preserves legacy unseen cache handling, but changes dependent closure digests. |
| `tools/exp04_launcher.py` | Changes launcher identity; existing attempts retain their recorded spawn identity. |
| `tools/exp05_gates.py` | Extends receipt/timing validation and seen retry enforcement; historical receipt/launcher identities remain authoritative. |
| `tools/exp05_probe.py` | Adds M/seen probing; existing unseen receipt schema remains supported. |
| `tools/exp04_eval.py` | Factory hook changes evaluator/writer identities and triggers finding 9. |
| `tools/exp04_eval_launch.py` | Seen entry/bindings change writer identity and trigger finding 9. |
| `tools/results_table.py` | Default output behavior is byte-identical; producer source identity changes. |

`tools/yaw_aug.py` and `tools/exp04_launch.sh` also change shared training/launcher dependencies.

### Exp_05: retain the historical training pin

Fresh imports establish:

| Training closure | Files | Digest |
|---|---:|---|
| `f19b9b6` | 11; seen module absent | `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13` |
| `2a27144` | 12; seen module included | `dcd3eb1bf8239b709b859aae6ce34721c843cfd071623e9e309adc5628d74c45` |

All four actual exp_05 training manifests record the **old `5b2da250…` closure**:

- S_simple, S_cylindrical and L_cylindrical: reviewed commit `f19b9b6`
- L_simple: reviewed commit `862923e`

Their recorded launcher digests are:

```text
f19b9b6 attempts:
0bff0d4fe270740c1adc411fd3e5d945742c2dcb545290d19e663d610fd8df20

862923e L_simple:
9dadbef0786a404654f01005fc3c143ba3d114210d41e3cf9f02e0779db657a6
```

**Exp_05 A9 `closures.training` must therefore remain the old `5b2da250…` value.** [tools/param_curve.py:221](/home/yixunhu/codespace/xRIR_code_wt07/tools/param_curve.py:221) compares the bound training manifests’ recorded launcher/training digests with approval pins; it does not recompute training identity from HEAD.

Consequently:

- Running L attempts retain their spawn closures; A7 records subsequent source drift.
- Already-certified S artifacts can be admitted into newly reviewed exp_05 evaluations without retraining.
- New evaluations use `--reviewed-commit <merge>` and their actual new evaluator/writer closures; the producer gets its actual new closure at approval fill.
- Historical exp_04 M evaluations still encounter finding 9’s live-source validation issue.

### Merge safety

Performed `git merge --no-ff --no-commit 2a27144` in isolated clones of main at `ec7e649`, `4d7a6cb`, and finally **`53307b6`**.

**All three merges were conflict-free.** Main’s notebook/reviews/prompts coexist with the branch’s approval template, six goldens and code. Exp_06’s promised direct edits are disjoint.

The merged tree nevertheless fails the exp_06 parser test described in finding 10. A conflict-free merge is therefore not an integration approval.

### CPU suite and hygiene

All execution used the prescribed interpreter, empty `CUDA_VISIBLE_DEVICES`, disabled bytecode/cache-provider writes, and isolated temporary clones.

| Check | Independent result |
|---|---|
| Full branch `pytest tests -q -p no:cacheprovider` | **1,592 passed, 43 skipped**, 600.79 s |
| Exp_07 → exp_04 → exp_03 collection order | **378 passed, 10 skipped** |
| Reverse experiment collection order | **378 passed, 10 skipped** |
| `static_checks.sh` | **Exit 0**: 150 passed; then 118 passed/1 skipped in each record order |
| Compilation of all changed Python files | **35 passed** |
| Changed shell scripts, `bash -n` | **2 passed** |
| `git diff --check f492613..2a27144` | **Exit 0** |
| Latest-main parser test before/after merge | **1 passed / 1 failed** |

The full-suite result is one pass higher than the Coder’s reported 1,591. [Full log](/tmp/exp07-review-edgge526/full-suite.log), [static log](/tmp/exp07-review-edgge526/static-checks.log), [merged regression log](/tmp/exp07-review-edgge526/final-main-merged.log).

All round-5 co-author trailers are present. The three oversized commits are confirmed at **314, 311 and 208 changed lines**; they are coherent recorded process deviations.

No GPU work or live-artifact mutation occurred. The original worktree retains only its pre-existing untracked `checkpoints` entry.

## Deferred GPU gate — exact nine node IDs

Run on the first free GPU **before any exp_07 probe**:

```text
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange[False-simple-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange[False-cylindrical-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange[True-simple-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange[True-cylindrical-32]
tests/test_exp07_eval_gpu.py::test_unseen_32_query_run_is_bit_identical_to_exp04
tests/test_exp07_eval_gpu.py::test_seen_final_batch_matches_a_direct_frozen_call[8]
tests/test_exp07_eval_gpu.py::test_seen_final_batch_matches_a_direct_frozen_call[1]
tests/test_exp07_eval_gpu.py::test_the_launcher_runs_the_seen_split_and_matches_the_frozen_functions[8]
tests/test_exp07_eval_gpu.py::test_the_launcher_runs_the_seen_split_and_matches_the_frozen_functions[1]
```

Required artifacts:

- Trainer parity: real AcousticRooms unseen batch of 32, K=8, both model constructors, and Git object `f492613`; no trained checkpoint required.
- Unseen evaluation parity: `ckpt/xRIR_simple_8_shot/epoch_12.pth` and `ckpt/yaw_aug/reference_manifest_k8_seed42.json`.
- Seen direct/final-batch tests: that checkpoint, seen split pickle and referenced data; the direct tests retain the actual final nine queries.
- Seen launcher tests: seed-42 seen manifests at K=8/K=1, referenced data, and a clean reviewed merge checkout.

These nine tests were collected, **not executed**.

## Planner launch sequence after the findings are resolved

Use the final reviewed merge SHA throughout. Commands below describe the existing interfaces; findings 3–4 require additional implemented interfaces before publication.

### 1. Merge, then GPU parity

From the merged repository root:

```bash
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
REC=worklog/worklog_yixun/exp_07_seen_protocol_claude
ASSETS="$REC/seen_protocol_results_assets"
export PYTHONPATH="$PWD"
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=2
```

Set `MERGE` to the reviewed merge SHA and `GPU` to the first free physical GPU.

```bash
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m pytest \
  tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange \
  tests/test_exp07_eval_gpu.py -q -p no:cacheprovider
```

Require **9 passed, 0 skipped**.

### 2. Seen alignment audit

```bash
CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/yaw_aug.py \
  --audit --protocol seen \
  --out ckpt/exp07/alignment_audit_seen.json --n-batches 3
```

Retain the audit’s arguments, environment, counts and cohort digest, which incorporates query paths and offsets.

### 3. Released-checkpoint calibration

Run K=8, seeds 42–46 with:

```text
ROLE=released_seen
BACKBONE=simple
CHECKPOINT=checkpoints/xRIR_seen.pth
```

Use this evaluation command for calibration and subsequent evaluations, substituting the recorded role/K/seed values:

```bash
MANIFEST="ckpt/exp07/reference_manifest_seen_k${K}_seed${SEED}.json"
HASH=$("$PY" -c \
  'import sys; from tools.reference_manifest import load_manifest, manifest_hash; print(manifest_hash(load_manifest(sys.argv[1])))' \
  "$MANIFEST")

"$PY" tools/exp04_eval_launch.py \
  --entry exp07 --split seen --backbone "$BACKBONE" \
  --checkpoint "$CHECKPOINT" --num-shot "$K" \
  --manifest "$MANIFEST" --manifest-hash "$HASH" --gl-seed "$SEED" \
  --conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols \
  --decomposition-batches 0 --batch-size 16 --max-samples 0 \
  --num-workers 12 --threads 2 \
  --data-root "$XRIR_DATA_PATH" --gpu "$GPU" \
  --reviewed-commit "$MERGE" --log-dir "$REC" \
  --run-label "${ROLE}_k${K}_seed${SEED}_k0" \
  --out-dir "ckpt/exp07/eval/${ROLE}_k${K}_seed${SEED}_k0" \
  "${TRAIN_BINDS[@]}"
```

`TRAIN_BINDS=()` for the released checkpoint. Omit `--tf32`; evaluation is TF32 off.

For **each** EDT/C50/T60 metric, require:

```text
|five-seed mean − historical value|
    ≤ 3 × five-seed sample SD + 0.02 × historical value
```

Historical values: **0.0389 s, 1.029 dB, 7.27%**.

If any fails, perform the prescribed matched-reference 200-query investigation within two GPU-hours and record the cause before admitting new-arm evaluations.

### 4. Smoke, probe, then full training for each arm

| Role | Backbone | Yaw |
|---|---|---:|
| `seen_simple` | `simple` | 0 |
| `seen_cyl` | `cylindrical` | 0 |
| `seen_aug` | `simple` | 1 |

```bash
tools/exp04_launch.sh smoke --protocol seen \
  --backbone "$BACKBONE" --yaw-aug "$YAW" \
  --gpu "$GPU" --reviewed-commit "$MERGE" --log-dir "$REC"

tools/exp04_launch.sh probe --protocol seen \
  --backbone "$BACKBONE" --yaw-aug "$YAW" \
  --gpu "$GPU" --reviewed-commit "$MERGE" \
  --log-dir "$REC" --timestamp "$STAMP"
```

The receipt is:

```text
ckpt/exp07/<ROLE>/_probe_<STAMP>_<ROLE>.json
```

Then launch with that exact receipt:

```bash
nohup setsid tools/exp04_launch.sh full --protocol seen \
  --backbone "$BACKBONE" --yaw-aug "$YAW" \
  --gpu "$GPU" --reviewed-commit "$MERGE" \
  --probe-json "$RECEIPT" --log-dir "$REC" \
  > "$REC/${ROLE}_${STAMP}_launcher.log" 2>&1 &
```

Require projection ≤60 hours, cumulative ceiling `1.5 × T_run`, epoch-1 ≤`1.05 × T_epoch`, twelve checkpoints, certified completion, no resume and at most one retry.

### 5. Complete the forty evaluations

Evaluate each new arm at K=1/8 and seeds 42–46 using its `final/epoch_012.pth`, with:

```bash
TRAIN_BINDS=(
  --bind-input "train_args=ckpt/exp07/$ROLE/final/args.json"
  --bind-input "train_manifest=ckpt/exp07/$ROLE/final/train_manifest.json"
  --bind-input "train_completion=ckpt/exp07/$ROLE/final/completion.json"
)
```

Use the evaluation command above. Add the released checkpoint’s five K=1 evaluations; retain its five K=8 calibration runs. Total: **40**.

### 6. Pin-finalisation commit

Fill the approval atomically with actual epoch-12 checkpoint identities, recorded training/launcher identities, evaluation/writer identities used, and final producer closures. Commit the filled approval before producing canonical results.

### 7. Producers → generators → bind → check

```bash
"$PY" tools/exp07_table.py --profile TABLE_SEEN_V1 \
  --runs ckpt/exp07/eval/{seen_simple,seen_cyl,seen_aug,released_seen}_k{1,8}_seed{42..46}_k0 \
  --json ckpt/exp07/results/TABLE_SEEN_V1.json \
  --md worklog/worklog_yixun/model_comparison_seen.md

for role in seen_cyl seen_aug released_seen; do
  "$PY" tools/exp07_pairs.py --profile PAIRS_SEEN_V1 \
    --runs-a ckpt/exp07/eval/${role}_k{1,8}_seed{42..46}_k0 \
    --runs-b ckpt/exp07/eval/seen_simple_k{1,8}_seed{42..46}_k0 \
    --json "ckpt/exp07/results/PAIRS_SEEN_V1_${role}.json" \
    --summary "ckpt/exp07/results/PAIRS_SEEN_V1_${role}.summary.txt"
done
```

Require all 30 paired cells final, including the corrected retry procedure where necessary.

Each generator takes:

```text
--table <TABLE_SEEN_V1.json>
--pairs <cyl JSON> <aug JSON> <released JSON>
--unseen-table ckpt/yaw_aug/results/TABLE_V1.json
--unseen-binding <verified exp_04 binding report>
--out <document>
```

Generate with `make_results_md.py`, `make_results_html.py`, and `make_latex.py`. Then:

```bash
"$PY" "$ASSETS/bind_provenance.py" \
  --runs ckpt/exp07/eval/{seen_simple,seen_cyl,seen_aug,released_seen}_k{1,8}_seed{42..46}_k0 \
  --attempt ckpt/exp07/{seen_simple,seen_cyl,seen_aug}/final \
  --audit ckpt/exp07/alignment_audit_seen.json \
  --results <table JSON and three pairing JSONs> \
  --rendered <Markdown HTML LaTeX> \
  --unseen-table ckpt/yaw_aug/results/TABLE_V1.json \
  --unseen-binding <verified exp_04 binding report> \
  --out ckpt/exp07

"$PY" "$ASSETS/check_record.py" ckpt/exp07
```

The corrected binder must additionally cover the evidence required by finding 3; those arguments do not exist yet.

## Final verdicts and exact blocking list

- **Part A: request changes.**
- **Part B: not approved.**

**Blocking findings:**

1. Generators accept unconverged/contradictory paired cells.
2. Binder accepts incomplete products and mismatched evidence.
3. Final record omits required attempt, parity and calibration evidence.
9. Merge invalidates exp_04’s existing record check.
10. Merge fails exp_06’s parser contract and changes nine filled closure approvals.
375,403
**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

# PART A — Coder round 5

**Scope:** `7b3ad6c..2a27144`, all 16 commits; **1,724 insertions / 46 deletions across 15 files**.

**Verdict: request changes. Blocking findings: 1, 2, 3.**

The round-4 blockers are closed in producer admission. The new publication and record-binding layers still admit inconsistent or incomplete evidence.

## Findings

### 1. Blocker — generators accept unconverged and contradictory paired cells

[make_results_md.py:87](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_results_md.py:87)

`check_pairs()` validates aggregate finality and cell coverage, but omits the corresponding checks inside each cell.

**Reproduced:** with the canonical JSON’s sidecar digest repaired, shared generator admission accepts each independent mutation:

- `statistics.absolute.reconverge_required = true`
- `statistics.relative.convergence.passed = false`
- `cell.decision_driving = true`
- A cell pairing inconsistent with its parent pairing

Markdown, HTML and LaTeX all use this admission path. Thus an interval explicitly requiring reconvergence can reach publication.

**Minimal fix:** validate both statistics’ convergence/finality, every cell’s descriptive status and pairing identity, and their consistency with aggregate flags. Share this validation with the binder.

### 2. Blocker — the binder accepts incomplete products and mismatched producer inputs

[bind_provenance.py:93](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:93)

The binder calls `md.load()` without table/pair completeness validation. Its input check filters selected filenames and accepts a **subset** of bound run inputs. Exp_04’s additional completeness predicate was dropped.

**Reproduced, retaining the genuine approval pins:**

- Nonfinal pairs with a nonempty reconvergence list
- A table with `rows = []`
- Removal of one run’s `metrics_yaw.json` provenance input
- A substituted training-inventory input digest
- An approval path replaced with `/nonexistent/approved.json`

The altered JSON and sidecar were kept internally consistent; current JSON digests were cited in the synthetic rendered documents. `collect()` accepted all five cases.

**Why:** the resulting report certifies products whose claimed evidence is incomplete or differs from the evidence actually bound. `check_record.py` repeats the same acceptance.

**Minimal fix:** apply complete canonical validation, require exact per-product run/input coverage, and verify every producer dependency and approval identity against its bound artifact. Restore the omitted completeness predicate.

### 3. Blocker — the final record omits required execution evidence

[bind_provenance.py:43](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:43), [bind_provenance.py:146](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:146)

Plan §3 requires parity/calibration/audit evidence and **every attempt, including aborted attempts**. The implementation binds three successful final attempts, receipts and ledger bytes. It does not traverse earlier attempts or probe-attempt evidence, and has no interface for parity/calibration evidence.

**Reproduced:**

- Added a ledger-listed aborted full attempt, bound the record, changed its `abort.json`: recomputation returned an **identical report**.
- An unrelated audit containing only `{"protocol":"unseen","passed":false}` was accepted.

The maximum-two-full-attempts assertion works on the supplied ledger, but does not establish the promised complete attempt history.

**Minimal fix:** bind lifecycle evidence for every ledger-listed attempt, including aborted/probe attempts; add parity and calibration evidence; validate the actual seen-audit schema, cohort and execution identity.

### 4. Should-fix — the registered 40,000-resample rerun remains unavailable

[tools/exp07_pairs.py:196](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_pairs.py:196)

The CLI always uses the fixed 20,000-resample profile. It tells users to rerun flagged cells with a larger count but supplies no reproducible registered path for doing so.

**Why:** plan §2’s publication procedure cannot be completed through the approved interface.

**Minimal fix:** implement the prescribed 40,000-resample retry, retaining the original diagnostic, retry count and provenance. Continued failure must remain nonfinal.

### 5. Should-fix — generator output can overwrite the exp_04 binding report

[make_results_md.py:142](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_results_md.py:142)

Output-overlap protection includes canonical products and sidecars but omits `--unseen-binding`.

**Reproduced:** setting `--out` equal to the synthetic exp_04 binding-report path replaced that JSON with Markdown.

**Minimal fix:** protect the resolved binding-report path alongside every other input, including aliases; verify refusal preserves its bytes.

### 6. Should-fix — runtime checkpoint agreement is still not tested

[tests/test_exp07_profiles.py:244](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_profiles.py:244)

New-arm profile digests remain `None` intentionally. Therefore this assertion:

```python
assert not path.exists() or prov.sha256_file(path)
```

accepts any readable checkpoint. It never compares against the filled runtime approval.

**Minimal fix:** load the authoritative runtime pin and compare available checkpoint bytes and completion linkage against it. Exercise matching and substituted checkpoints using isolated fixtures.

Round-4’s lifecycle and inventory-agreement requests are closed; its checkpoint-agreement request remains incomplete.

### 7. Should-fix — LaTeX SD footnotes omit K and can print positive SDs as zero

[make_latex.py:68](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_latex.py:68)

`note(name, shot, rows)` ignores `shot`, producing two indistinguishably labelled SD entries for each recipe method. Fixed precision also violates the plan’s nonzero-display requirement: an SD of `0.000001` prints as zero in all three columns.

**Minimal fix:** label every footnote with K and use sufficient precision for positive SDs, while preserving genuine zeros.

### 8. Nit — README publication order cannot complete its first binding

[README.md:53](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/README.md:53)

The instructions run bind/check before generating documents, although binding requires nonempty `--rendered` files containing the canonical digests.

**Minimal fix:** document **producers → generators → bind → check**.

## Verified

### Round-4 closure

| Previous item | Result |
|---|---|
| Blocker 1: unchecked training/launcher closure records | Closed. Empty records, unreviewed records and incorrect digest labels refuse. |
| Blocker 2: training execution/data contract | Closed for every listed mutation: inventory count/digest, smoke mode, dirty flag, effective epochs/backbone/yaw/save directory, and 1,000-hour projection. |
| Required spectral metrics | Closed; all five registered metrics required. |
| Exploratory Markdown disclosure | Closed; status and deviations displayed. |
| Approval lifecycle and inventory agreement | Closed. |
| Runtime checkpoint agreement | Finding 6 remains. |
| Recursive approval immutability | Closed. |

The fixture now creates a real reviewed Git commit and represents inventory evidence, receipt-derived limits, normalized args and twelve checkpoint entries. All **40 named admission refusals** pass. This verifies current behavior; it does not independently establish the claimed historical red-before-fix ordering.

### Renderers, statistics and binding

- Table aggregation preserves five-seed means, sample SD (`ddof=1`), EDT units and finite-count tolerance.
- Paired statistics preserve shared query resampling, whole-room query weighting, paired cohorts, absolute/relative statistics and descriptive status.
- Normal Markdown/HTML values trace to canonical JSON; generation is deterministic.
- Top-level exploratory, reconvergence, incomplete-coverage and unbound-unseen refusals work.
- Combined tables validate exp_04’s JSON through its sidecar and binding-report digest.
- LaTeX matches the reference’s seven-column seen/unseen structure; mean rounding, unique-minimum bolding, unbolded ties, reference-row exclusion and canonical-JSON caption pointers work. Finding 7 affects its SD footnotes.
- Valid synthetic binding/checking covers three final attempts, the released checkpoint, forty evaluations, outputs, companions, approval bytes and exp_04 inputs. Findings 2–3 limit what that certification proves.
- The default exp_04 `write_outputs()` path produced **byte-identical JSON, Markdown and sidecar** against the prechange implementation with time/output location held fixed. The diff contains two behavioral edits plus a docstring: **3 insertions / 2 deletions**.

Dataset pins agree with available artifacts: ten manifest hashes, 6,217-query identity `be5095a8…`, and inventory `251b7d7b…`. Both K values reference the same complete file set. The new literals `296454`, `60`, and `1.5` agree with launcher contracts. Checkpoint/closure approval pins correctly remain unfilled at this stage.

Independent counterexamples and round-4 closure replays are retained in [the review script](/tmp/exp07-review-edgge526/test_review_adversarial.py) and [its log](/tmp/exp07-review-edgge526/adversarial.log).

---

# PART B — FINAL INTEGRATIVE REVIEW (`full`)

**Scope:** `f492613..2a27144`; **5,483 insertions / 116 deletions across 45 files**.

**Verdict: not approved. Blocking findings: 1, 2, 3, 9, 10.**

## Additional blocking findings

### 9. Blocker — merging breaks exp_04’s existing record check through live source validation

Change site: [tools/exp04_eval.py:129](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp04_eval.py:129).

Verification path: [exp_04 check_record.py:20](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_results_assets/check_record.py:20) → [snapshot():57](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_results_assets/bind_provenance.py:57) → [provenance.revalidate():385](/home/yixunhu/codespace/xRIR_code_wt07/tools/provenance.py:385).

**The precise answer to the producer-pin question:** the checker does **not** recompute the producer closure at HEAD against `6d32a5e2…`. It compares recorded producer/approval identities. Nevertheless, the merge breaks checking because evaluation snapshots rehash recorded evaluator/writer source paths against the **live repository**.

Using an actual bound exp_04 evaluation manifest:

- Source revalidation against main’s premerge files: **no mismatches**.
- Against the dry-run merged files: mismatches for:
  - Entry-point closure: `tools/exp04_eval.py`, `tools/provenance.py`
  - Writer closure: those two files plus `tools/exp04_eval_launch.py`

Passing the report’s recorded HEAD controls ancestry/report identity; it does not restore historical source files.

Separately, fresh `results_table` production changes closure from approved `6d32a5e2…` to `1856f447…`; both `tools/results_table.py` and `tools/provenance.py` contribute.

**Minimal fix:** preserve the historical shared evaluation closures through exp_07-specific implementations, or review a historical-source verification migration that checks complete recorded closures against immutable reviewed Git blobs while retaining strict artifact/data checks. Moving only the renderer hook or repinning only the producer does **not** repair the checker failure. Historical exp_05 M-run reuse needs the same treatment, or new evaluations under the reviewed evaluator.

### 10. Blocker — current main’s exp_06 integration fails, and filled approvals become stale

[train_xRIR_backbone.py:61](/home/yixunhu/codespace/xRIR_code_wt07/train_xRIR_backbone.py:61), [exp_06 parser contract:50](/home/yixunhu/codespace/xRIR_code/tests/test_exp06_train.py:50).

Main advanced during review and now includes exp_06. On main **`53307b6`**, this exact test passes before merging and fails afterward:

```text
tests/test_exp06_train.py::test_shared_flags_and_defaults_equal_the_trainers
```

Failure: the shared trainer has the additional `protocol` argument.

Moreover, hashing the merged source files changes these nine filled exp_06 code approvals:

```text
trainer, finalize, smoke, heading, profiles,
eval, eval_launch, haa_finetune, haa_eval
```

For example, its trainer closure changes from `5a2c6df0…` to `cbe84cfb…`. Exp_06’s admission explicitly rejects mismatched code approvals.

**Minimal fix:** review the parser contract’s treatment of exp_06’s fixed unseen protocol, rerun integration checks, and review/refill affected approvals for subsequent exp_06 execution. Preserve any already-started attempt’s original provenance.

Directly disjoint source edits do not establish dependency compatibility.

## Verified

### Plan §3 delivery

| Planned round | Assessment |
|---|---|
| 1 — protocol, inventory, audit | Implemented; real split sizes/loading verified. GPU parity, real seen training smoke and audit evidence remain deferred. |
| 2 — launcher, probes, timing/retry gates | Implemented; CPU contracts and goldens pass. Actual receipts and training evidence await GPU execution. |
| 3 — evaluator and launcher | Implemented; CPU refusals pass. Five evaluation GPU tests remain deferred. |
| 4 — profiles and table producer | Implemented; round-4 blockers closed. Finding 6 remains. |
| 5 — paired producer | Implemented except the prescribed 40,000-resample rerun path. |
| 6 — renderers and bound record | Implemented but incomplete: findings 1–3 and 5–8. |

Thus **all six implementation areas exist, but the plan is not fully delivered**.

### Frozen closure rule

This command returns **no paths**:

```bash
git diff --name-only f492613..2a27144 -- \
  eval_yaw_rotation.py eval_unseen.py eval_xRIR_backbone.py \
  'model/*.py' tools/yaw_rotation.py tools/reference_manifest.py \
  tools/per_sample_metrics.py 'treble_multi_room_dataset/*.py' \
  utils/spec_utils.py tools/paired_stats.py tools/summarize_yaw.py
```

The exp_03 frozen evaluator closure also retains its approved digest in the latest merged clone.

### Shared-file consequences

All eight specifically named files changed:

| File | Consequence |
|---|---|
| `train_xRIR_backbone.py` | Adds protocol dispatch and the seen module to training closure membership. |
| `tools/provenance.py` | Adds protocol-aware training identity; preserves legacy unseen cache handling, but changes dependent closure digests. |
| `tools/exp04_launcher.py` | Changes launcher identity; existing attempts retain their recorded spawn identity. |
| `tools/exp05_gates.py` | Extends receipt/timing validation and seen retry enforcement; historical receipt/launcher identities remain authoritative. |
| `tools/exp05_probe.py` | Adds M/seen probing; existing unseen receipt schema remains supported. |
| `tools/exp04_eval.py` | Factory hook changes evaluator/writer identities and triggers finding 9. |
| `tools/exp04_eval_launch.py` | Seen entry/bindings change writer identity and trigger finding 9. |
| `tools/results_table.py` | Default output behavior is byte-identical; producer source identity changes. |

`tools/yaw_aug.py` and `tools/exp04_launch.sh` also change shared training/launcher dependencies.

### Exp_05: retain the historical training pin

Fresh imports establish:

| Training closure | Files | Digest |
|---|---:|---|
| `f19b9b6` | 11; seen module absent | `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13` |
| `2a27144` | 12; seen module included | `dcd3eb1bf8239b709b859aae6ce34721c843cfd071623e9e309adc5628d74c45` |

All four actual exp_05 training manifests record the **old `5b2da250…` closure**:

- S_simple, S_cylindrical and L_cylindrical: reviewed commit `f19b9b6`
- L_simple: reviewed commit `862923e`

Their recorded launcher digests are:

```text
f19b9b6 attempts:
0bff0d4fe270740c1adc411fd3e5d945742c2dcb545290d19e663d610fd8df20

862923e L_simple:
9dadbef0786a404654f01005fc3c143ba3d114210d41e3cf9f02e0779db657a6
```

**Exp_05 A9 `closures.training` must therefore remain the old `5b2da250…` value.** [tools/param_curve.py:221](/home/yixunhu/codespace/xRIR_code_wt07/tools/param_curve.py:221) compares the bound training manifests’ recorded launcher/training digests with approval pins; it does not recompute training identity from HEAD.

Consequently:

- Running L attempts retain their spawn closures; A7 records subsequent source drift.
- Already-certified S artifacts can be admitted into newly reviewed exp_05 evaluations without retraining.
- New evaluations use `--reviewed-commit <merge>` and their actual new evaluator/writer closures; the producer gets its actual new closure at approval fill.
- Historical exp_04 M evaluations still encounter finding 9’s live-source validation issue.

### Merge safety

Performed `git merge --no-ff --no-commit 2a27144` in isolated clones of main at `ec7e649`, `4d7a6cb`, and finally **`53307b6`**.

**All three merges were conflict-free.** Main’s notebook/reviews/prompts coexist with the branch’s approval template, six goldens and code. Exp_06’s promised direct edits are disjoint.

The merged tree nevertheless fails the exp_06 parser test described in finding 10. A conflict-free merge is therefore not an integration approval.

### CPU suite and hygiene

All execution used the prescribed interpreter, empty `CUDA_VISIBLE_DEVICES`, disabled bytecode/cache-provider writes, and isolated temporary clones.

| Check | Independent result |
|---|---|
| Full branch `pytest tests -q -p no:cacheprovider` | **1,592 passed, 43 skipped**, 600.79 s |
| Exp_07 → exp_04 → exp_03 collection order | **378 passed, 10 skipped** |
| Reverse experiment collection order | **378 passed, 10 skipped** |
| `static_checks.sh` | **Exit 0**: 150 passed; then 118 passed/1 skipped in each record order |
| Compilation of all changed Python files | **35 passed** |
| Changed shell scripts, `bash -n` | **2 passed** |
| `git diff --check f492613..2a27144` | **Exit 0** |
| Latest-main parser test before/after merge | **1 passed / 1 failed** |

The full-suite result is one pass higher than the Coder’s reported 1,591. [Full log](/tmp/exp07-review-edgge526/full-suite.log), [static log](/tmp/exp07-review-edgge526/static-checks.log), [merged regression log](/tmp/exp07-review-edgge526/final-main-merged.log).

All round-5 co-author trailers are present. The three oversized commits are confirmed at **314, 311 and 208 changed lines**; they are coherent recorded process deviations.

No GPU work or live-artifact mutation occurred. The original worktree retains only its pre-existing untracked `checkpoints` entry.

## Deferred GPU gate — exact nine node IDs

Run on the first free GPU **before any exp_07 probe**:

```text
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange[False-simple-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange[False-cylindrical-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange[True-simple-32]
tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange[True-cylindrical-32]
tests/test_exp07_eval_gpu.py::test_unseen_32_query_run_is_bit_identical_to_exp04
tests/test_exp07_eval_gpu.py::test_seen_final_batch_matches_a_direct_frozen_call[8]
tests/test_exp07_eval_gpu.py::test_seen_final_batch_matches_a_direct_frozen_call[1]
tests/test_exp07_eval_gpu.py::test_the_launcher_runs_the_seen_split_and_matches_the_frozen_functions[8]
tests/test_exp07_eval_gpu.py::test_the_launcher_runs_the_seen_split_and_matches_the_frozen_functions[1]
```

Required artifacts:

- Trainer parity: real AcousticRooms unseen batch of 32, K=8, both model constructors, and Git object `f492613`; no trained checkpoint required.
- Unseen evaluation parity: `ckpt/xRIR_simple_8_shot/epoch_12.pth` and `ckpt/yaw_aug/reference_manifest_k8_seed42.json`.
- Seen direct/final-batch tests: that checkpoint, seen split pickle and referenced data; the direct tests retain the actual final nine queries.
- Seen launcher tests: seed-42 seen manifests at K=8/K=1, referenced data, and a clean reviewed merge checkout.

These nine tests were collected, **not executed**.

## Planner launch sequence after the findings are resolved

Use the final reviewed merge SHA throughout. Commands below describe the existing interfaces; findings 3–4 require additional implemented interfaces before publication.

### 1. Merge, then GPU parity

From the merged repository root:

```bash
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
REC=worklog/worklog_yixun/exp_07_seen_protocol_claude
ASSETS="$REC/seen_protocol_results_assets"
export PYTHONPATH="$PWD"
export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=2
```

Set `MERGE` to the reviewed merge SHA and `GPU` to the first free physical GPU.

```bash
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m pytest \
  tests/test_exp07_trainer.py::test_unseen_step_bit_identical_to_prechange \
  tests/test_exp07_eval_gpu.py -q -p no:cacheprovider
```

Require **9 passed, 0 skipped**.

### 2. Seen alignment audit

```bash
CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/yaw_aug.py \
  --audit --protocol seen \
  --out ckpt/exp07/alignment_audit_seen.json --n-batches 3
```

Retain the audit’s arguments, environment, counts and cohort digest, which incorporates query paths and offsets.

### 3. Released-checkpoint calibration

Run K=8, seeds 42–46 with:

```text
ROLE=released_seen
BACKBONE=simple
CHECKPOINT=checkpoints/xRIR_seen.pth
```

Use this evaluation command for calibration and subsequent evaluations, substituting the recorded role/K/seed values:

```bash
MANIFEST="ckpt/exp07/reference_manifest_seen_k${K}_seed${SEED}.json"
HASH=$("$PY" -c \
  'import sys; from tools.reference_manifest import load_manifest, manifest_hash; print(manifest_hash(load_manifest(sys.argv[1])))' \
  "$MANIFEST")

"$PY" tools/exp04_eval_launch.py \
  --entry exp07 --split seen --backbone "$BACKBONE" \
  --checkpoint "$CHECKPOINT" --num-shot "$K" \
  --manifest "$MANIFEST" --manifest-hash "$HASH" --gl-seed "$SEED" \
  --conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols \
  --decomposition-batches 0 --batch-size 16 --max-samples 0 \
  --num-workers 12 --threads 2 \
  --data-root "$XRIR_DATA_PATH" --gpu "$GPU" \
  --reviewed-commit "$MERGE" --log-dir "$REC" \
  --run-label "${ROLE}_k${K}_seed${SEED}_k0" \
  --out-dir "ckpt/exp07/eval/${ROLE}_k${K}_seed${SEED}_k0" \
  "${TRAIN_BINDS[@]}"
```

`TRAIN_BINDS=()` for the released checkpoint. Omit `--tf32`; evaluation is TF32 off.

For **each** EDT/C50/T60 metric, require:

```text
|five-seed mean − historical value|
    ≤ 3 × five-seed sample SD + 0.02 × historical value
```

Historical values: **0.0389 s, 1.029 dB, 7.27%**.

If any fails, perform the prescribed matched-reference 200-query investigation within two GPU-hours and record the cause before admitting new-arm evaluations.

### 4. Smoke, probe, then full training for each arm

| Role | Backbone | Yaw |
|---|---|---:|
| `seen_simple` | `simple` | 0 |
| `seen_cyl` | `cylindrical` | 0 |
| `seen_aug` | `simple` | 1 |

```bash
tools/exp04_launch.sh smoke --protocol seen \
  --backbone "$BACKBONE" --yaw-aug "$YAW" \
  --gpu "$GPU" --reviewed-commit "$MERGE" --log-dir "$REC"

tools/exp04_launch.sh probe --protocol seen \
  --backbone "$BACKBONE" --yaw-aug "$YAW" \
  --gpu "$GPU" --reviewed-commit "$MERGE" \
  --log-dir "$REC" --timestamp "$STAMP"
```

The receipt is:

```text
ckpt/exp07/<ROLE>/_probe_<STAMP>_<ROLE>.json
```

Then launch with that exact receipt:

```bash
nohup setsid tools/exp04_launch.sh full --protocol seen \
  --backbone "$BACKBONE" --yaw-aug "$YAW" \
  --gpu "$GPU" --reviewed-commit "$MERGE" \
  --probe-json "$RECEIPT" --log-dir "$REC" \
  > "$REC/${ROLE}_${STAMP}_launcher.log" 2>&1 &
```

Require projection ≤60 hours, cumulative ceiling `1.5 × T_run`, epoch-1 ≤`1.05 × T_epoch`, twelve checkpoints, certified completion, no resume and at most one retry.

### 5. Complete the forty evaluations

Evaluate each new arm at K=1/8 and seeds 42–46 using its `final/epoch_012.pth`, with:

```bash
TRAIN_BINDS=(
  --bind-input "train_args=ckpt/exp07/$ROLE/final/args.json"
  --bind-input "train_manifest=ckpt/exp07/$ROLE/final/train_manifest.json"
  --bind-input "train_completion=ckpt/exp07/$ROLE/final/completion.json"
)
```

Use the evaluation command above. Add the released checkpoint’s five K=1 evaluations; retain its five K=8 calibration runs. Total: **40**.

### 6. Pin-finalisation commit

Fill the approval atomically with actual epoch-12 checkpoint identities, recorded training/launcher identities, evaluation/writer identities used, and final producer closures. Commit the filled approval before producing canonical results.

### 7. Producers → generators → bind → check

```bash
"$PY" tools/exp07_table.py --profile TABLE_SEEN_V1 \
  --runs ckpt/exp07/eval/{seen_simple,seen_cyl,seen_aug,released_seen}_k{1,8}_seed{42..46}_k0 \
  --json ckpt/exp07/results/TABLE_SEEN_V1.json \
  --md worklog/worklog_yixun/model_comparison_seen.md

for role in seen_cyl seen_aug released_seen; do
  "$PY" tools/exp07_pairs.py --profile PAIRS_SEEN_V1 \
    --runs-a ckpt/exp07/eval/${role}_k{1,8}_seed{42..46}_k0 \
    --runs-b ckpt/exp07/eval/seen_simple_k{1,8}_seed{42..46}_k0 \
    --json "ckpt/exp07/results/PAIRS_SEEN_V1_${role}.json" \
    --summary "ckpt/exp07/results/PAIRS_SEEN_V1_${role}.summary.txt"
done
```

Require all 30 paired cells final, including the corrected retry procedure where necessary.

Each generator takes:

```text
--table <TABLE_SEEN_V1.json>
--pairs <cyl JSON> <aug JSON> <released JSON>
--unseen-table ckpt/yaw_aug/results/TABLE_V1.json
--unseen-binding <verified exp_04 binding report>
--out <document>
```

Generate with `make_results_md.py`, `make_results_html.py`, and `make_latex.py`. Then:

```bash
"$PY" "$ASSETS/bind_provenance.py" \
  --runs ckpt/exp07/eval/{seen_simple,seen_cyl,seen_aug,released_seen}_k{1,8}_seed{42..46}_k0 \
  --attempt ckpt/exp07/{seen_simple,seen_cyl,seen_aug}/final \
  --audit ckpt/exp07/alignment_audit_seen.json \
  --results <table JSON and three pairing JSONs> \
  --rendered <Markdown HTML LaTeX> \
  --unseen-table ckpt/yaw_aug/results/TABLE_V1.json \
  --unseen-binding <verified exp_04 binding report> \
  --out ckpt/exp07

"$PY" "$ASSETS/check_record.py" ckpt/exp07
```

The corrected binder must additionally cover the evidence required by finding 3; those arguments do not exist yet.

## Final verdicts and exact blocking list

- **Part A: request changes.**
- **Part B: not approved.**

**Blocking findings:**

1. Generators accept unconverged/contradictory paired cells.
2. Binder accepts incomplete products and mismatched evidence.
3. Final record omits required attempt, parity and calibration evidence.
9. Merge invalidates exp_04’s existing record check.
10. Merge fails exp_06’s parser contract and changes nine filled closure approvals.
