# Coder report — exp_06 post-review fix round (Codex full review F2–F5)

- **Coder:** Claude Opus 5 (max effort), Agent subagent of the Planner session.
- **Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-fullfix`, base `main` = `ddfcb66`, tip = `ac05dd9`.
- **Date:** 2026-09-17, 21:56 → 23:35 local (−04:00).
- **Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, always with
  `PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''`.
- **Round input:** `oriented_cyl_codex_code_full_review.md` at `08a03e7`, findings **F2, F3, F4, F5**.
  F1 (parent directories of `ckpt/exp06/sim_eval/<arm>/`) is the Planner's and was not touched.
- **No GPU was used.** No process this round did not start was signalled; `pkill -f` / `pgrep -f`
  were never used. The only process stopped was this round's own first full-suite run (pid 176140,
  recorded before it was started), superseded by the run at the final commit.
- **Not merged.** `main` was not modified; nothing outside the worktree was written except this file.

---

## 1. Commits

| SHA | Subject | changed lines (total / source / tests) |
|---|---|---|
| `69f2f14` | exp06 F5: H3 withholds a verdict that rests on a zero-width or unconverged interval | 141 / 69 / 72 |
| `bd4d112` | exp06 F4: arm B is admitted only through its two registered routes | 106 / 29 / 77 |
| `35c035a` | exp06 F2: a reconstructed training receipt for arm B's exp_01 checkpoint | 516 / 278 / 238 |
| `1168bb2` | exp06 F2: the launcher validates the training evidence it binds, before execute_run | 361 / 193 / 168 |
| `fc9748d` | exp06 F2: the comparer re-validates the training evidence, and the runbook argv is tested | 343 / 27 / 316 |
| `fd00c2b` | exp06 F3: enforce_producer compares the approvals with the identities about to run | 221 / 89 / 132 |
| `d496b48` | exp06 F3: the mirror probe enforces 6.4's approvals before it loads a model | 119 / 52 / 67 |
| `e1a0266` | exp06 F3: the simulated-evaluation launcher enforces 6.4 before it creates a run | 107 / 35 / 72 |
| `3f1e214` | exp06 F3: the HAA queue refuses an unapproved job before it acquires the root | 94 / 40 / 54 |
| `206aa4e` | exp06 F3: an HAA child's completion binds the approvals its entry point was approved by | 144 / 43 / 101 |
| `ac05dd9` | exp06 F3: state the heading-coverage rule enforce_producer actually applies | 5 / 5 / 0 |

Whole round: 16 files, **2092 insertions / 53 deletions**.

**Commit-size disclosure (SOP "generally < 200 changed lines").** Nine of eleven commits are under
200 source lines. `35c035a` adds a whole new module (278 source lines, `tools/exp06_legacy_train_receipt.py`)
and its 238 test lines in one commit, because the module has no useful smaller unit — enumeration,
writer and verifier are one contract. `1168bb2` is 193 source lines across the new
`tools/exp06_train_evidence.py` (155) and the launcher wiring (38).

Every commit carries the two required trailers
(`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` / `Claude-Session: …`). TDD was followed
per finding: the tests of each finding were written and run **red** before the implementation
(F5's four tests, F4's five, F2's launcher/comparer/integration sets, F3's per-producer sets);
tests and implementation then landed in one commit, as the SOP permits.

---

## 2. What each finding became

### F2 — training-evidence bindings for the simulated evaluations

Three commits (`35c035a`, `1168bb2`, `fc9748d`) and two new modules.

**`tools/exp06_legacy_train_receipt.py`** (new, gated CLI) — arm B's historical-evidence branch:

```
python tools/exp06_legacy_train_receipt.py --train-dir ckpt/xRIR_cyl_8_shot \
    --checkpoint ckpt/xRIR_cyl_8_shot/epoch_12.pth --out ckpt/exp06/train_receipt_cyl.json
```

It writes a `label: "reconstructed"` receipt, exclusive-create (`provenance.write_manifest`),
enumerating `args.json`, `history.jsonl`, `train.log`, **every** `epoch_NN.pth` of the directory
(ordered by epoch, so exp_01's `epoch_005.pth`/`epoch_010.pth` are enumerated beside `epoch_05.pth`/
`epoch_10.pth`) and `best.pth`/`last.pth` when present — **18 artefacts** for the real
`ckpt/xRIR_cyl_8_shot` — with `files_sha256` over the enumeration, the checkpoint's own sha256 and
epoch, the recorded epoch count from `history.jsonl`, `provenance.git_state`, and the tool's own
source closure (`provenance.source_closure` + `closure_record`, strict against HEAD).
It refuses: a checkpoint outside the enumerated directory, a checkpoint that is not an
`epoch_NN.pth`, a checkpoint that is not the **last** epoch the history records, a history that
disagrees with `args.json['epochs']` or is not exactly `1..N`, a missing required file, a backbone
that is not the registered arm's, and — the gate — weights that hash to no registered exp_01 arm of
`tools.exp04_profiles` (`CONTROL`, `CYL`). `verify()` re-reads the receipt and re-hashes every
enumerated artefact through a caller-supplied `hasher`.

**`tools/exp06_train_evidence.py`** (new) — the semantics both the launcher and the comparer use, so
one rule decides launch and admission:

- role `arm`: `train_completion` must be a `completion.json` with `run_type == 'full'`,
  `admissible_arm is True`, `diagnostic is False`, `exploratory is False`, `epochs == 12`,
  `artifacts['epoch_012.pth'] == sha256(--checkpoint)`, and a `checkpoint` block
  `{path: 'epoch_012.pth', epoch: 12, sha256: <same>}`; it must sit in the `run_dir` it names.
  `train_manifest` must be that run's `provenance.json`, hashing to
  `artifacts['provenance.json']`, recording the same `run_type` and a `reviewed_commit` equal to the
  completion's `approvals.committed_at`. A `best.pth` or any other epoch offered as `--checkpoint`
  fails on the artifact hash (tested).
- role `baseline`: `train_completion` is the reconstructed receipt, `train_manifest` the
  `args.json` the receipt enumerates (see the design decision in §4).
- role `diagnostic`: binds nothing and is never an arm.

**`tools/exp06_eval_launch.py`** — `training_evidence(args)` runs in `main` **before**
`launcher.execute_run`, so a run directory is never created for an unbound arm; `build_fields`
computes it when a caller does not pass it, and the record is published in the manifest as
`training_evidence`. `__main__` now prints `refusing: …` and exits non-zero.

**`tools/exp06_compare.py`** — `check_evidence` re-runs the same rule at admission for every
`exp06`-route run, with `epoch_sha256` = the approved `artifacts.epoch_012.sha256` for C and
`registered_sha256` = `exp04_profiles.CYL['sha256']` for B, and routes the receipt's artefact reads
through the comparer's own `bind`, so the enumerated artefacts are bound like every other input
(read once across the five seeds; a contradictory second read is a refusal).

**Fixtures corrected.** `tests/test_exp06_compare.py::write_run` used to satisfy both training names
with the run's own evaluation log — exactly the masking the review named. It now builds a real
arm-C attempt (`provenance.json` + `completion.json`) and a real reconstructed receipt over a
synthetic exp_01 directory, and a dedicated test asserts that binding the log is refused.

**Integration test (`tests/test_exp06_sim_evidence.py`, new).** Builds the exact argv the runbook's
`sim` step writes for both arms plus the two `--bind-input` flags, and drives
`parse_args → split_bindings → training_evidence → build_fields → check_fields` with no GPU and no
child, then writes the run directory the way the comparer fixtures do and requires
`exp06_compare.admit_run` to admit it for C and for B. Refusals covered: either binding missing,
both missing (the runbook's original argv), the two arms' evidence swapped, a non-admissible
completion, and a receipt whose enumerated artefact moved.

### F3 — producer-side approval enforcement

**`tools/exp06_approvals_api.py`** gains `enforce_producer(producer, repo, commit,
approved_path=None, checkpoint=None, headings=None, exploratory=False)`: it loads the approvals
bound to the bytes committed at `commit`, keeps `require_producer`'s "every required leaf is filled"
deviations, recomputes the producer's own `code` keys at `commit` via
`exp06_profiles.compute_code_digests(repo, commit, keys=…)` and names every mismatching key,
compares `sha256(checkpoint)` with `artifacts.epoch_012.sha256` when a checkpoint is given and each
heading JSON with `artifacts.heading[room]` when headings are given, and returns
`{producer, keys_checked, approvals: {path, sha256, committed_at}, artifacts, deviations,
exploratory, admissibility}`. Production raises naming every deviation; `exploratory=True` records
them and labels the output `diagnostic`. `approved_path_default()` names the record's committed
approvals file, so `None` can never silently mean the all-null template.

The **producer → code-key map** is explicit (`PRODUCER_CODE_KEYS`):

| producer | code keys required to equal the approvals |
|---|---|
| `heading` | `heading` |
| `legacy_receipt` | `summarize_haa` |
| `mirror_probe` | `mirror_probe`, `probe_align`, `encoder`, `factory`, `heading` |
| `haa_children` | `haa_finetune`, `haa_eval`, `haa_pipeline_sh`, `finalize`, `encoder`, `factory`, `heading`, `recipe` |
| `summarize_haa` | `summarize_haa` |
| `sim_eval` | `eval`, `eval_launch`, `encoder`, `factory`, `evaluator_exp03` |
| `compare` | `compare` |
| `pages` | (none) |

`HEADING_COMPLETE = ('haa_children',)`: the HAA queue runs all four rooms and must offer all four
headings; a one-room producer (the hallway mirror probe) offers that room's heading and is checked
on it. Every room offered must be a registered one either way.

Wired into four producers:

1. **`tools/exp06_mirror_probe.py`** — `enforce_approvals(args)` in `main`, **before any checkpoint
   blob is read** (a test intercepts `state_from_blob` and asserts nothing was loaded). New flags
   `--approved` (default: the record asset), `--approved-commit` (default: this checkout's HEAD),
   `--exploratory`. The receipt is written into `gate_g1.json` as `approvals`, and the record now
   carries `admissibility` (`confirmatory` / `diagnostic`); both are echoed on stdout.
2. **`tools/exp06_eval_launch.py`** — `approvals_receipt(args, repo)` in `main`, before
   `execute_run`; recorded in the manifest as `approvals`. Arm C's checkpoint is checked against
   `artifacts.epoch_012`; arm B evaluates exp_01's weights, which are not that artifact, so its
   identity stays the registered sha (comparer) + the reconstructed receipt (F2).
   `--approved` defaults to the record asset, `--approved-commit` to the run's `--reviewed-commit`.
3. **`tools/exp06_haa_pipeline.sh::prepare_job`** — `approvals_ok <init> <checkpoint>` is the
   **first** gate, before `open_job`, so a refusal creates, adopts and renames nothing; it runs on
   the CPU (`CUDA_VISIBLE_DEVICES=""`) at the queue's HEAD for `haa_children` with the four heading
   JSONs and the init: `artifacts.epoch_012` for `cyl_or`, the registered `exp04_profiles`
   `CONTROL`/`CYL` sha for `control_hf`/`cyl_hf`. A refusal is `prepare_failed <root> approvals`.
   The dry run prints `APPROVALS <init> checkpoint=… heading=… producer=haa_children` (the golden
   queue in `tests/test_exp06_haa_pipeline.py` was updated accordingly).
4. **`tools/exp06_finalize.py`** — `haa_approvals(closure, run_type, commit, repo)` loads the
   approvals of the repository being finalised, bound to the blob committed at the **child's own**
   `reviewed_commit`, and requires the child's recorded entry-point closure digest to equal the
   approved `code.haa_finetune` / `code.haa_eval`; the binding is recorded in the completion as
   `approvals` + `code_digests`, exactly as `full_evidence` records it. A missing approvals file, a
   null leaf and a well-formed but different digest are each a named refusal.

`tools/exp06_summarize_haa.py` and `tools/exp06_compare.py` already compared their own producer
closure with `code.summarize_haa` / `code.compare` (`check_producer_identity`), so they were left
alone — the F3 gap was the four producers above.

### F4 — arm B's admission route

`tools/exp06_compare.py`: each role declares the closed set of routes §6.3 registers for it
(`C: ('exp06',)`, `A: ('exp04',)`, `B: ('exp05', 'exp06')`); the route is now established **before**
the checkpoint block that depends on it and a run outside its role's registered routes is refused by
name. The historical missing-epoch exception is the reused routes': an `exp06`-route evaluation must
record its own `checkpoint_epoch`. On the exp_05 route (whose evaluator records no
`checkpoint_epoch`), `check_exp05_tier` now requires the registered arm to be the twelfth-epoch one,
so the epoch is established from the registry rather than defaulted.

Regression: the real `ckpt/yaw_aug/eval/cyl_k8_seed42_k0` — the review's own counterexample, which
had been admitted with all 6337 queries — is refused as B with `…is admitted only through the
registered routes exp05/exp06, not exp04` (skipped if the directory is absent). The five real
exp_04 control runs remain admitted as A (existing test, still green).

### F5 — H3's convergence rule

`tools/exp06_compare.py`: `convergence_attempt` records one attempt (`n_boot`, both companion
intervals, the endpoint movement, width and ratio, and the reason it failed), and `converged_draws`
applies §7's policy around the inherited resampling — a zero-width interval is refused, the draws
are quadrupled **once** on any failure, and a second failure withholds the verdict. The strict
endpoint rule is `tools.exp06_bootstrap.convergence_endpoints`, which already refuses zero width;
`tools/paired_compare.py` is untouched. The cell publishes `convergence.attempts`,
`convergence.status` and the `n_boot` of the attempt the rule stopped at; `analyse` keeps turning a
failed cell into `verdict: "not converged"` and a deviation.

The review's counterexample — five seeds of identical per-query values, which used to publish
`rho = 0`, `[0, 0]` and **"non-inferior"** — is a test: both contrasts report `not converged`, two
attempts (`n_boot` 100 then 400), and the run records `convergence gate failed`.

---

## 3. New source-closure digests

`from tools import exp06_profiles as ap; ap.compute_code_digests(repo, 'HEAD')` in the worktree at
`ac05dd9`, CUDA hidden. **Six of the twenty `code` keys changed** and must be re-approved before any
confirmatory launch:

| key | approved (committed) | this branch (`ac05dd9`) |
|---|---|---|
| `compare` | `2e711c967c7bcbe27f07cf552584e06e3f30f8472e1247a7d81a0eb46c44bfd7` | `1de772a0690fb01b8b045c25a1b144f32973f995e36204a4b1bd9ac3f5202e93` |
| `eval_launch` | `b6c72faef9bd3cd1afe3d4a986fee7f24df1d846d24ecd9abec619e5116fe24c` | `e1d685870818df3b2a3fe6aa60ee90db16b75a7f961f4015d46ab540dfa5f067` |
| `finalize` | `5ef318e8da734e6076f89237014d1e2366078692b1da83ef93b304d1dc609399` | `1edbaf2caaffc6ba4e65d6a7fc6bda119f7e4597523f0072bb323c0af7af6b18` |
| `haa_pipeline_sh` | `92b077d8eb8a4f55decf3bc86a8cb186109a6139869d098f9b73c2d0ba9b6636` | `2ad3e562c1e322d3a035e2aed3bb28d55c940d20e0e9516d88c242a7df0055a7` |
| `mirror_probe` | `13ab5ff6f9d7af435971c8c96cac3900399a69f42270afd16dccf8e467068d54` | `df41d6b6903125bbd35fb377dc44b160f9023fd761e60a66167bf0c8931cd91f` |
| `summarize_haa` | `2ad63b6dea015f996f1568a4e6e8d1513d356c83246b96dd19cc98c3fa4cefb4` | `4538e1129d6f5e6da3037f1e6586f899a08d4c1848d1d138548d9929bb10e595` |

Unchanged: `bootstrap`, `encoder`, `eval`, `evaluator_exp03`, `factory`, `haa_eval`, `haa_finetune`,
`heading`, `launch_sh`, `probe_align`, `profiles`, `recipe`, `smoke`, `trainer`.

Causes: `compare` and `eval_launch` import the two new modules (and `compare` now imports
`exp06_bootstrap`); `mirror_probe`, `eval_launch`, `compare` and `summarize_haa` all contain the
edited `tools/exp06_approvals_api.py` in their import closures (`summarize_haa` changes by closure
propagation only — its own bytes and `exp06_finalize`'s edits); `finalize` and `haa_pipeline_sh` are
their own edits. `evaluator_exp03` still recomputes to the pinned
`5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`.

**The two new modules need no new approvals key**: `tools/exp06_train_evidence.py` and
`tools/exp06_legacy_train_receipt.py` are inside the import closures of both `code.compare` and
`code.eval_launch` (verified), so their bytes are already covered by keys the Planner re-fills.

---

## 4. Design decisions

**(a) Arm B's receipt design — `train_completion` = receipt, `train_manifest` = the enumerated
`args.json`.** Arm C binds the training run's *startup* record (`provenance.json`) as
`train_manifest` and its *certification* (`completion.json`) as `train_completion`. Arm B keeps the
same meanings: the reconstructed receipt is the certification, and exp_01's own
`ckpt/xRIR_cyl_8_shot/args.json` — which the receipt enumerates and hashes — is the startup record.
The launcher and the comparer require `train_manifest` to resolve to exactly
`<receipt.train_dir>/args.json` and to hash to the value the receipt enumerates, so the two names
stay distinct and each carries evidence, rather than binding the same file twice.

**(b) No approvals-schema change.** The B receipt is *not* approved by hash, unlike the HAA legacy
receipt. It does not need to be: its authority comes from constants in reviewed code rather than
from the approvals file — the writer refuses weights that hash to no registered exp_01 arm, and both
the launcher and the comparer pin the receipt's checkpoint to `exp04_profiles.CYL['sha256']` and
re-hash every artefact it enumerates at every admission. Adding a `reused.legacy_train_receipt` key
would have nulled every `reused`-requiring producer (the comparer and the summariser) until a
further bookkeeping commit, for no authority the registered constant does not already supply.
`tools/exp06_approved_digests_template.json` and the record's `approved_digests.json` are therefore
**unmodified**, and the only approvals work left is re-filling the six `code` digests above.

**(c) Producer → code-key map.** See the table in §2 (F3). It is the keys each producer *executes*;
the existing `PRODUCER_REQUIREMENTS` still says which leaves must be filled. No producer's map
contains its own output.

**(d) `enforce_producer` keeps `require_producer`'s deviations** rather than replacing them, so a
null leaf and a drifted digest are reported together and both refuse.

**(e) The HAA finalizer reads the approvals of the repository it is finalising**
(`Path(repo)/exp06_profiles.APPROVED_RELATIVE`), not of this checkout, because the finalizer is
given the child's repository and the child's `reviewed_commit`. The stub repository the HAA
finalizer fixtures use now carries its own committed approvals (`stub_approvals` in
`tests/test_exp06_finalize.py`), so the gate is exercised by all ~110 HAA finalizer tests instead of
being stubbed past.

---

## 5. Validation

- `git diff main --stat` — 16 files, 2092 insertions / 53 deletions. **No pinned or forbidden file
  was touched**: the diff against `main` over `eval_yaw_rotation.py`, `eval_unseen.py`,
  `eval_xRIR_backbone.py`, `model/`, `tools/yaw_rotation.py`, `tools/reference_manifest.py`,
  `tools/per_sample_metrics.py`, `treble_multi_room_dataset/`, `utils/spec_utils.py`,
  `train_xRIR_backbone.py`, `tools/exp04_*.py`, `tools/exp05_*.py`, `tools/paired_compare.py`,
  `tools/provenance.py`, `sim_to_real/` and `worklog/` is **empty**.
- `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh` — passes.
- `python -m py_compile` of every changed `.py` (source and tests) — passes.
- `git diff main --check` — clean.
- Per-file exp_06 runs, all green at the final commit or the commit that introduced them:
  `test_exp06_compare.py` 95 passed · `test_exp06_eval_launch.py` 71 passed ·
  `test_exp06_sim_evidence.py` 11 passed · `test_exp06_legacy_train_receipt.py` 22 passed ·
  `test_exp06_approvals_api.py` 45 passed · `test_exp06_mirror_probe.py` 62 passed, 2 skipped ·
  `test_exp06_haa_pipeline.py` 47 passed · `test_exp06_finalize.py` 311 passed ·
  `test_exp06_finalize_haa_consumer.py` + `test_exp06_summarize_haa.py` + `test_exp06_launch.py` +
  `test_exp06_haa.py` 169 passed.
- **Real-data readback of the new tool (ladder rung 3), read-only on `ckpt/`:** the CLI wrote a
  receipt for the real `ckpt/xRIR_cyl_8_shot` in 5.4 s — 18 artefacts, `epochs = 12`, checkpoint
  `8ba344ad25d4a68f2ed7f22b4fb90e72b354fafe78b4d28d3eeeb6c1f8eab48e` (= `exp04_profiles.CYL`),
  receipt sha256 `ccf05f2d7a42cfa66429069daeebe8731103243e1f02086ad86ca1ebbceae64d` — and
  `exp06_train_evidence.check('baseline', …)` admitted it against that registered sha with
  `train_manifest = ckpt/xRIR_cyl_8_shot/args.json`, binding 19 inputs (18 artefacts + the
  receipt). Written to the scratchpad, **not** to `ckpt/`; the Planner writes the real one.
- **Full suite** (`python -m pytest tests -q -p no:cacheprovider`, CUDA hidden, at `ac05dd9`,
  41 min 27 s): **1 failed, 3066 passed, 50 skipped**. The one failure is the expected
  approvals-drift guard named below; `grep -E '^(FAILED|ERROR)'` over the whole log returns
  that single line and nothing else.

**Expected failure.** `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`
fails, and only that test, naming exactly the six keys of §3: `compare`, `eval_launch`, `finalize`,
`haa_pipeline_sh`, `mirror_probe`, `summarize_haa`. This is the approvals-drift guard doing its job;
it goes green in the Planner's approvals re-fill commit. The other 23 tests in that file pass.

---

## 6. Open items for the Planner

1. **Re-fill the six `code` digests** of §3 in
   `worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/approved_digests.json`
   in a reviewed bookkeeping commit. Until then **every** producer refuses — which is the intended
   post-fix behaviour, but it does block G1, HAA and sim.
2. **The runbook's `sim` step** needs the two `--bind-input` flags added to each generated command
   (`train_manifest=<attempt>/provenance.json`, `train_completion=<attempt>/completion.json` for
   `cyl_or`; `train_manifest=ckpt/xRIR_cyl_8_shot/args.json`,
   `train_completion=<receipt>` for `cyl`) and `EXP06_F2_LANDED=1`. The runbook lives under
   `worklog/`, which this round may not modify.
3. **A new runbook step is required before `sim`:** write arm B's receipt once with
   `python tools/exp06_legacy_train_receipt.py --train-dir ckpt/xRIR_cyl_8_shot --checkpoint
   ckpt/xRIR_cyl_8_shot/epoch_12.pth --out ckpt/exp06/train_receipt_cyl.json` (exclusive-create; it
   enumerates 16 artefacts of ~2 GB and takes a few seconds). It must run from a clean tree, since
   the writer records its own closure strictly against HEAD.
4. **`producer_ok` in the runbook still calls `require_producer`.** It is no longer the only gate —
   each producer now enforces `enforce_producer` itself — but upgrading the helper would make the
   refusal happen one step earlier, before `nohup`. Planner's call.
5. **F1 is still open** (parent directories of `ckpt/exp06/sim_eval/<arm>/`); the runbook already
   carries the `mkdir -p`, and no code change was made for it this round.
6. **A note on the real inputs:** while writing the pipeline gate I observed that, at `main`, the
   real `cyl_or` HAA job *passes* `enforce_producer('haa_children', …)` — the four heading JSONs and
   `ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth` all match the committed approvals.
   That is evidence the artefacts section is correctly filled; it will refuse again only until the
   six `code` digests are re-approved.
