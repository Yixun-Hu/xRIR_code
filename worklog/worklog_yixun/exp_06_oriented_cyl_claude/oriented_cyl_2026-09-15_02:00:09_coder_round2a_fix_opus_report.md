**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 oriented_cyl — round 2a FIX cycle report

Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`, from the reviewed
round-2a tip `4f92f54`. CPU only (`CUDA_VISIBLE_DEVICES=''`), `OMP_NUM_THREADS=4`,
`PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt`,
`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest with `-p no:cacheprovider`. Python
`/home/yixunhu/miniconda3/envs/xRIR/bin/python` (3.8). No GPU work started, no process
signalled that this session did not start, nothing written under `ckpt/`, and the only file
written under the main tree is this report. The superseded archive folder and the
`archive/codex-round2a-2026-09-15` branch were not read.

Files changed since `4f92f54` (exactly the round-2a set, nothing else):
`tools/exp06_{recipe,train,finalize,smoke}.py`, `tools/exp06_launch.sh`,
`tests/test_exp06_{recipe,train,finalize,smoke,launch}.py`.
`git diff --name-only main...HEAD --diff-filter=M` outside those paths is empty, so no
pinned or composed file is modified.

## 1. Commits

| # | SHA | changed lines (ins/del) | subject |
|---|---|---|---|
| 1 | `975e999` | 28 / 3 | count parameters inside a forked RNG (review nit 8) |
| 2 | `8da3354` | 108 / 23 | check smoke budgets, entry statuses and the live memory ceiling (review 7) |
| 3 | `7914be7` | 99 / 10 | name every malformed finalizer input as a refusal (review 9) |
| 4 | `e9dd0a5` | 196 / 7 | bind startup, retained and checkpoint arguments type-strictly (blocker 2) |
| 5 | `6d1e870` | 230 / 43 | revalidate the closure, the reviewed blobs and the data inventory (blocker 1) |
| 6 | `0ce5d13` | 192 / 25 | bind the child-exit receipt, the live pid and a stable log (blocker 4a) |
| 7 | `b822695` | 104 / 21 | drain the child pipe to EOF before closing the log (blocker 4b) |
| 8 | `3721cc8` | 330 / 76 | enforce provenance, heading and artefact contracts on HAA training (blocker 3a) |
| 9 | `cb9a6f3` | 134 / 39 | bind the evaluation child to its checkpoint, heading and side labels (blocker 3b) |
| 10 | `e9e6e6f` | 207 / 39 | derive job admission from child roles, hashes and lineage (blocker 3c) |
| 11 | `373b39f` | 94 / 20 | require a valid diagnostic receipt and gate recovery finalization (review 5a) |
| 12 | `4c66184` | 44 / 7 | resolve one data root through the dataset module and pin it (review 6) |
| 13 | `1d5e116` | 115 / 27 | share the child lifecycle with smoke, probe and recovery (review 5b) |
| 14 | `62655ae` | 2 / 2 | import the backbone registry directly in the finalizer |
| 15 | `2fe25bf` | 73 / 23 | require each child's role artefacts and refresh the module docstrings |

Every commit carries both required trailer lines (verified by grep over all bodies). No
amend, rebase, reset or push. Round totals since `4f92f54`: 10 files changed,
1 907 insertions, 316 deletions.

## 2. Per-finding fix and its adversarial regression test

### Blocker 1 — full completion revalidates execution inputs (`6d1e870`)

`full_evidence` now calls three new functions instead of hashing reviewed blobs only.
`load_provenance` requires the whole record (`run_type, repo, reviewed_commit,
source_closures, registry_sha256, git_state, environment, command, effective_args`) plus a
data identity with a real inventory. `verify_source_closure` re-imports the recorded entry
module hermetically (`closure_paths` → `provenance.source_closure`), requires identical
membership, requires a fresh `closure_record` digest equal to the recorded one, requires the
per-file three-way agreement *working tree now == bytes at spawn == reviewed blob at the
recorded commit*, and refuses an empty closure. `revalidate_inputs` delegates to
`tools.provenance.revalidate` with `required=('train_data_identity','source_closures')`.

Regression tests (bytes actually change):
`test_a_modified_closure_file_is_refused_after_provenance_was_written` finalizes a run
successfully, then appends a comment to `tools/exp06_recipe.py` **in a hardlinked clone** and
requires a refusal naming that file (the reviewed-blob digest is untouched by the edit);
`test_changed_training_data_bytes_are_refused` rewrites and then deletes a file in the
recorded data root; `test_incomplete_provenance_is_refused` drops each of seven required keys;
`test_source_closure_and_identity_shapes_are_refused` covers an empty closure carrying its own
correct empty-list digest, a membership change, a missing/foreign `entry_module`, and an
identity without an inventory.

### Blocker 2 — type-strict, three-source argument validation (`e9dd0a5`)

`tools/exp06_recipe.py` gained `strict_equal` (recursive, `True` ≠ `1`, `0` ≠ `False`,
`12` ≠ `12.0`, list ≠ tuple), `compare_sources`, `check_presence` (every OPERATIONAL and
EXP06 field required) and `check_all(..., historical=False)`; `check_derived` now refuses
`param_counts` whose values are not native ints before it compares them.
`exp06_finalize.check_argument_sources` validates `args.json`, `last.pth["args"]` and
`provenance.effective_args` separately against `check_all`, runs the budget check once, and
then requires recursive type-strict agreement between all three.

Regression tests: `test_argument_source_refusals_are_named` parametrises provenance
`max_train_batches=3` against `0` in the other two; `tf32=1` and `no_save=0` inside
`last.pth`; float `param_counts`; a missing operational field (`num_workers`); a missing
`exp06_git_head`; provenance without `effective_args`; `last.pth` without `args`. Recipe-level:
`test_strict_equality_separates_types_recursively` (13 pairs),
`test_compare_sources_names_the_disagreeing_field_and_sources`,
`test_current_runs_require_every_operational_and_exp06_field` (each field dropped in turn,
and shown to be tolerated only on the explicit `historical=True` path),
`test_param_counts_must_be_native_integers`.

### Blocker 3 — HAA branches enforce the contracts (`3721cc8`, `cb9a6f3`, `e9e6e6f`)

*Fine-tuning child*: `haa_child_arguments` loads and verifies provenance, run type and the
three-way closure of the module named in the record (`tools.exp06_haa_finetune`), revalidates
inputs, and requires `args.json` to agree type-strictly with `provenance.effective_args`.
`_heading_binding` now requires per room `phi_deg` (finite), `k` (int in [0,512)) with
`k == exp06_heading.heading_roll_k(phi_deg)`, `decision ∈ {estimated, override}`, `sha256`
and `path`; it hashes the JSON at that path, parses it with `read_heading_json` and requires
room/φ/k/decision to agree. `haa_history` requires the declared validation cadence with finite
losses and a `summary.json` whose `best_epoch` is a validated epoch. `best.pth`/`last.pth` are
CPU-loaded and their key set must equal `build_xrir_exp06(args.backbone, args.num_shot)`'s.
`init_sha256` must be the hash of the recorded `init` file.

*Evaluation child*: same provenance/closure/argument gate, then `meta.frame == args.frame`,
`meta.backbone == args.backbone`, `strict_equal(meta.heading, args.heading)`,
`meta.checkpoint_sha256 == sha256(<checkpoint named in args>)` (the checkpoint must exist and
carry this arm's parameter names), and a `side_label` array of the same length as `index`
whose values are `-1`/`1` (booleans rejected explicitly, since `True == 1`).

*Job*: `child_role` maps each expected path to its required run type; `child_completion`
schema-validates each child completion (13 common keys plus the per-role keys), requires
`diagnostic is False` and `child_exit == 0`, and **re-hashes every artefact the child recorded**;
`job_identity` requires one backbone, one frame, one heading roll per room across children and
the exp_02 lineage (`stage2_*.init_sha256` == `stage1.artifacts['best.pth']`,
`eval/<room>.checkpoint_sha256` == `stage2_<room>.artifacts['best.pth']`; zero-shot: one shared
checkpoint hash). `admissible_arm` is never read from a child.

Regression tests: the review's three admissible counterexamples are refused —
`test_haa_train_refusals_are_named` (24 cases) covers a missing `provenance.json`, a wrong
`run_type`, closure drift introduced by editing the stub entry module, an unresolvable entry
module, a missing `phi_deg`, `k` ≠ roll(φ), a heading JSON whose bytes changed, a heading JSON
belonging to another room, a wrong `init_sha256`, malformed history epochs and a non-finite
loss, `summary.json` missing a key, and a checkpoint with the wrong parameter set;
`test_haa_eval_refusals_are_named` (18 cases) covers `meta` heading `k` differing from `args`,
a conflicting `checkpoint_sha256`, a missing/short/zero/boolean `side_label`, a missing
`index`, a frame mismatch and a missing provenance;
`test_a_job_of_bare_admissible_claims_is_refused` is exactly the nine-child
`{"run_type":"haa_eval","admissible_arm":true}` job; `test_job_refusals_are_named[stale_hash]`
is the "valid schema but a hash that no longer matches" case (a child artefact rewritten after
its completion was published), alongside role, diagnostic, non-zero-exit, backbone, frame,
heading-`k` and both lineage mismatches.

### Blocker 4 — closed-writer / log evidence (`0ce5d13`, `b822695`)

Launcher: `run_child` creates a FIFO under the run directory, starts `cat >> "$log"` on its
read end, runs the child with `> "$pipe" 2>&1`, writes `launch.pid`, then waits for **both**
the child and the sink — the sink sees EOF only when every process holding the write end
(descendants included) has exited. `close_child` then calls the new
`tools/exp06_finalize.py child-exit` subcommand, which appends `EXP06_CHILD_EXIT <status>
<iso>` (fsynced) and exclusively writes `child_exit.json` with
`{child_pid, status, ended_at, log, log_sha256_after_marker}`. The `tail -f` viewer is gone —
the sink is the only path into the log.

Finalizer: `refuse_live_launch` refuses whenever `launch.pid` names a live process, in every
mode including recovery; `closed_log` reads the bytes **once**, hashes exactly those bytes and
validates the marker on them; `child_exit_receipt` requires the receipt, its four keys, a
type-strict `status` equal to `--child-exit`, and `log_sha256_after_marker` equal to the
validated digest; after all evidence is gathered the file is re-hashed and any change refuses
with `stale log`.

Regression tests: `test_a_surviving_descendant_cannot_write_past_the_end_marker` drives the
launcher's lifecycle in library mode with a stub child that backgrounds
`( sleep 2; echo late ) &` on the inherited descriptor — `late` must appear in the log and the
marker must still be the last line, with the receipt binding the post-marker hash and the pid
file naming the child; `test_a_failing_child_reports_its_status_through_the_lifecycle`;
`test_a_live_launch_pid_refuses_finalization_in_every_mode` (own pid in `launch.pid`, for both
`full` and `probe`); `test_the_child_exit_receipt_must_bind_the_hashed_log` (missing, wrong
status, wrong hash, malformed JSON, missing key);
`test_a_log_that_changes_during_validation_is_refused` (a monkeypatched evidence step appends
to the log mid-validation); `test_child_exit_subcommand_appends_the_marker_and_writes_the_receipt`.

### Should-fix 5 — shared lifecycle, diagnostic receipts, recovery (`373b39f`, `1d5e116`)

`diagnostic_evidence` now requires a receipt: it must exist, be JSON, be marked
`diagnostic: true`, have an argv containing `--no-save` and an integer `exit_status`;
`passed = exit_status == 0 and child_exit == 0`, so a failed diagnostic still gets a completion
marked `passed: false` and never `admissible_arm`. `preflight` accepts a `finalize` mode
(reviewed commit, clean tree, live-pid checks; no GPU query). The launcher gained
`diagnostic()`, `promote()` and `abort()`: smoke and probe now run through `run_child` /
`close_child` (own run dir, `launch.pid`, drained pipe, marker, `child_exit.json`) and are
finalized with `--run-type smoke|probe --receipt <path>`; recovery `finalize` runs the
preflight with `--reviewed-commit`/`--gpu`, promotes `final` on success and, on failure,
renames **both** the attempt directory and the log with `_ABORTED_<reason>`.

Regression tests: `test_diagnostic_runs_need_a_receipt_but_no_artifacts`,
`test_a_failed_diagnostic_is_recorded_and_never_admissible`,
`test_invalid_diagnostic_receipts_are_refused` (8 cases: absent, no `--receipt`, malformed
JSON, `diagnostic: false`, argv without `--no-save`, a saving argv, no `exit_status`, a float
`exit_status`), `test_diagnostic_modes_run_the_same_child_lifecycle`,
`test_recovery_finalize_gates_promotes_and_aborts`,
`test_abort_renames_the_attempt_and_its_log`,
`test_preflight_supports_the_recovery_mode_without_a_gpu_query`.

### Should-fix 6 — one registered data root (`4c66184`)

`tools/exp06_train.DATA_ROOT` is now `treble_xRIR_dataset.BASE_DATA_PATH` itself (no second
fallback); `resolve_data_root()` realpaths it and refuses a missing directory, is used for the
inventory, and is recorded as `data_root` in `provenance.json`. The launcher exports
`XRIR_DATA_PATH` (default `/home/yixunhu/data_cache/AcousticRooms`, overridable), refuses a
non-directory, and prints it on the `ENV` line, which the dry-run goldens now assert.

Regression tests: `test_the_data_root_is_the_dataset_modules_own_resolution`,
`test_an_absent_data_root_is_refused`, `test_provenance_records_the_resolved_data_root`, plus
the updated `ENV` golden in the three launcher dry-run tests.

### Should-fix 7 — smoke budgets, statuses and a live ceiling (`8da3354`)

`check_budget` refuses an `alarm_seconds` or `max_gb` that is not a finite positive number
(booleans and strings included) before the entry is imported. The receipt's `exit_status` is
now numeric (0, the integer an entry returns, 3 for an alarm/memory abort, 1 for an exception)
with a new `outcome` string, an `entry_status` field and `aborted_memory`. A periodic
`setitimer(ITIMER_REAL, tick, tick)` watchdog reads `torch.cuda.max_memory_allocated()` (0
without CUDA) and aborts **during** execution with status 3 and `aborted_memory: true`; the
retrospective peak check is kept for the post-run path.

Regression tests: `test_non_positive_or_non_finite_budgets_are_refused` (11 cases),
`test_entry_returning_a_nonzero_status_fails_the_smoke` (a stub returning 7),
`test_memory_ceiling_aborts_during_execution` (a stub sleeping 60 s with the peak
monkeypatched over budget must abort in ≪ 30 s with `aborted_memory: true`),
`test_a_successful_smoke_records_a_zero_status`.

### Should-fix 9 — named refusals for malformed inputs (`7914be7`)

`_read_json` requires a JSON object, `_load_torch` converts every `torch.load` failure and any
non-mapping container into a named `ValueError`, `_state_dict` requires a name→tensor mapping,
and `_history_rows` names the first malformed line. `last.pth` without `model` is now an
explicit refusal instead of a `KeyError`.

Regression tests: `test_malformed_inputs_raise_named_refusals` (10 cases: non-dict `last.pth`,
`last.pth` without `model`, a non-state-dict `model`, a non-dict and an unpicklable
`epoch_012.pth`, invalid and non-object `args.json`, invalid and non-object `provenance.json`,
a non-JSON history line) and `test_cli_exits_two_on_a_malformed_checkpoint` (exit 2,
`EXP06_FINALIZE_REFUSED`, nothing written).

### Nit 8 — parameter counting does not move the global RNG (`975e999`)

`expected_param_counts` builds its model inside `torch.random.fork_rng(devices=[])`.
`test_param_count_validation_leaves_the_global_rng_untouched` clears the cache, seeds, captures
`torch.random.get_rng_state()`, calls the cold path and requires the state to be byte-identical,
then checks that the next `torch.randn(3)` draw is unchanged.

### Nit 10 — red before green

Every cycle obtained a genuinely failing run on the new tests before the implementation edit
(counts in §3). The single exception is disclosed in §4 (D-2).

## 3. Validation

Red runs before each implementation (new tests only, on the touched files):

| cycle | red |
|---|---|
| nit 8 | 1 failed, 62 deselected |
| review 7 | 18 failed, 4 passed |
| review 9 | 6 failed, 5 passed, 49 deselected |
| blocker 2 (recipe) | 25 failed, 63 passed |
| blocker 2 (finalizer) | 4 failed, 5 passed, 60 deselected |
| blocker 1 | 13 failed, 2 passed, 79 deselected |
| blocker 4a | 8 failed, 83 passed |
| blocker 4b | 2 failed, 10 deselected |
| blocker 3a | 17 failed, 9 passed, 80 deselected |
| blocker 3b | 12 failed, 8 passed, 99 deselected |
| blocker 3c | 14 failed, 5 passed, 110 deselected |
| review 5a | 11 failed, 8 passed, 132 deselected |
| review 6 | 5 failed, 22 passed (retrospective, see D-2) |
| review 5b | 3 failed, 13 deselected |
| child role artefacts (`2fe25bf`) | 3 failed, 19 passed, 119 deselected |

Static checks at the end:

```
git diff --check 4f92f54..HEAD                                  clean
python -m py_compile tools/exp06_{recipe,train,finalize,smoke}.py
                     tests/test_exp06_{recipe,train,finalize,smoke,launch}.py   OK
bash -n tools/exp06_launch.sh                                    OK
git diff --name-only main...HEAD --diff-filter=M outside exp06   (none)
```

Final suite: see §5.

## 4. Discrepancies and deviations (nothing deviated silently)

- **D-1 Commit sizes.** Counting insertions only, three commits exceed 200 (`6d1e870` 230,
  `3721cc8` 330, `e9e6e6f` 207). Counting insertions + deletions, five exceed 200 (203, 273,
  217, 406, 246). The findings were split as far as the code allowed (blocker 3 into three
  commits, blocker 4 into two, should-fix 5 into two); the HAA commits are large because the
  fixtures they replace had to be rebuilt in the same change to keep the suite green.
- **D-2 One retrospective red.** For should-fix 6 the tests and the implementation were written
  in one step; red was then demonstrated by `git stash push -- tools/exp06_train.py
  tools/exp06_launch.sh`, re-running (5 failed / 22 passed) and `git stash pop`. Every other
  cycle was red-first.
- **D-3 Smoke receipt schema.** `exit_status` changed from a string (`'ok'`, `'alarm'`,
  `'memory'`, `'error: X'`) to an integer, because should-fix 5 requires the finalizer to check
  `exit_status == 0`. The former wording now lives in `outcome`; `entry_status` records what an
  entry returned. Existing round-2a assertions were updated.
- **D-4 Watchdog period.** The prompt asked for `setitimer(ITIMER_REAL, 1.0, 1.0)`; implemented
  as `tick = min(1.0, alarm_seconds)` for both the initial and the repeat interval, so a
  sub-second alarm budget still fires on its deadline. For every real budget (300 s, 2400 s)
  this is exactly 1.0/1.0. The handler checks memory first, then the wall-clock deadline.
- **D-5 HAA history "epochs contiguous from 1".** exp_02's protocol validates every `val_every`
  epochs (10 in stage 1, 2 in stage 2), so literal contiguity would refuse every real run.
  Implemented as the stronger, argument-derived rule: the recorded epochs must equal
  `range(val_every, epochs+1, val_every)` exactly, with `val_every`/`epochs` required as
  positive ints and `summary.json.best_epoch` required to be one of them. With `val_every == 1`
  this reduces to contiguous-from-1.
- **D-6 Heading binding schema.** Verifying "the sha256 matches the heading JSON at the recorded
  path" needs a path, which plan §6.2's `{room: φ, k, json sha256}` does not name. The frozen
  binding is therefore `{phi_deg, k, decision, sha256, path}`; round 2b's entry points must
  write `decision` and `path`. Relative paths resolve against `--repo`.
- **D-7 Job lineage frozen to exp_02's pipeline.** Round 2b does not exist, so
  "init/checkpoint lineage" was frozen to `sim_to_real/run_haa_pipeline.sh`: stage 2 starts from
  `stage1/best.pth`, evaluation runs `stage2_<room>/best.pth`, and a zero-shot job's four
  evaluations share one checkpoint hash. If round 2b selects on `last.pth` instead, this check
  must be revisited.
- **D-8 Zero-shot child paths.** `expected_children('zeroshot')` still returns `eval/<room>`;
  `haa_job_evidence` additionally strips a leading `zeroshot/` so both
  `<init>/zeroshot` (job dir) and `<init>` (children `zeroshot/eval/<room>`) layouts are
  accepted, and `child_role` maps both spellings to `haa_eval`.
- **D-9 Provenance key names.** The prompt's `source_closure` / `data_identity` / `argv` are
  called `source_closures.<name>` / `train_data_identity` / `command` in the record
  `tools/exp06_train.py` actually writes; the finalizer requires those actual names. HAA
  children may record `data_identity` instead (they do not read the AcousticRooms train split).
  A closure entry now also records `entry_module`, without which the finalizer cannot recompute
  the right closure; `tools/exp06_train.py` writes `'tools.exp06_train'`.
- **D-10 `closure_paths` memoisation.** The hermetic re-import costs 5–15 s. Production
  finalization calls it once; the `lru_cache` only spares repeated calls inside one process
  (the test suite). Documented in the docstring. The per-file hashes are *not* cached.
- **D-11 `child_exit.json` is required for every run type**, `haa_job` included, so the
  closed-writer evidence is uniform. It is written by the new `child-exit` subcommand, which
  refuses to overwrite an existing receipt.
- **D-12 Launcher library mode.** `EXP06_LAUNCH_LIB=1 source tools/exp06_launch.sh` defines the
  functions and returns without parsing a mode, so `run_child`/`close_child`/`abort` are
  testable on CPU with a stub child. No new tool file was added, since this cycle may only edit
  the round-2a files.
- **D-13 `tail -f` removed** from the `full` mode: the pipe sink is now the only writer path
  into the log, and a second reader adds nothing.
- **D-14 Full-branch tests run against a clone.** The three-way source check requires
  working tree == HEAD, which is false while a closure file is edited but uncommitted. The
  fixture therefore makes one hardlinked `git clone --local --single-branch` of the worktree per
  session (0.3 s, no extra disk) and finalizes against it; that clone also carries the
  "modify a tracked closure file after provenance was written" adversarial case. HAA tests use a
  second tiny stub repo on which `tools.exp06_haa_finetune` / `tools.exp06_haa_eval` resolve.
- **D-15 Diagnostic validity vs. pass.** The prompt lists `exit_status == 0` as a validity
  requirement *and* asks for `passed: false` on failed diagnostics. Implemented as: validity =
  exists + JSON + `diagnostic: true` + `--no-save` in argv + integer `exit_status`;
  `passed = exit_status == 0 and child_exit == 0`. A failed diagnostic is recorded, not refused,
  and is never `admissible_arm`.
- **D-16 `check_all` gained `historical=False`.** The narrower exp_01 normalisation path is only
  reached by passing `historical=True` explicitly; `check_derived` still names a missing
  `backbone` on both paths, which the presence test excludes for that one field.
- **D-17 Closure growth.** `tools/exp06_finalize.py` now imports `tools.exp06_heading` and
  `model.xRIR_cyl_oriented`, and `tools/exp06_recipe.py` imports `torch` directly. The §6.4
  `code` approvals must be filled from the digests as they stand after this cycle.
- **D-18 GPU ladder untouched.** Rungs 4–7 of plan §9 (the GPU smokes, the fit/timing probe and
  the pre-launch acceptance criteria) are still unrun; every change here was validated on CPU.
- **D-19 Two follow-up commits after the thirteen findings.** `62655ae` replaces a
  `build_xrir_exp06.__globals__['BACKBONES_EXP06']` lookup with a direct import; `2fe25bf` adds
  the per-role artefact-name guard on the job branch (without it, a hand-written child
  completion lacking `best.pth` would raise `KeyError` inside `job_identity`, violating the
  should-fix 9 interface) and refreshes the `exp06_finalize` / `exp06_smoke` module docstrings,
  which no longer described the new evidence.
- **D-20 The `full` branch does not check `epoch_012.pth`'s parameter names** against
  `build_xrir_exp06`, only that it equals `last.pth['model']` tensor-wise; the name check is
  applied to the HAA checkpoints, where the review asked for it. Adding it to the pretraining
  branch would be a one-line change but was left out of scope for this fix cycle.

## 5. Final suite

```
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_exp06_*.py tests/test_provenance.py \
    -q -p no:cacheprovider
```

```
SKIPPED [3] tests/test_exp06_factory.py:47: full pinned forward requires CUDA
404 passed, 3 skipped, 31 warnings in 255.08s (0:04:15)
```

The three skips are the pre-existing CUDA-only pinned-forward checks of round 1. The round-2a
tip `4f92f54` ran 262 passed / 3 skipped; this cycle adds 142 tests, all of them adversarial
regression tests for the review's findings. `tests/test_provenance.py` is inside that total and
green. The exp_03 pinned-closure suite was run separately and is untouched:

```
python -m pytest tests/test_exp03_record_tools.py -q -p no:cacheprovider
31 passed, 1 skipped in 3.47s
```

Final tip: `2fe25bf` on `exp06-window`; the worktree is clean.
