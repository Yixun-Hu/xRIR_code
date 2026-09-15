**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 oriented_cyl — round 2a, fix cycle 2 (Codex close review, findings 1, 2, 3a/3b/3c, 4, 5, 9)

Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`, from `2fe25bf` to
`bf51484`. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), no GPU work, no process
signalled, nothing written under `ckpt/` or the main tree except this report. Only round-2a
files changed: `tools/exp06_finalize.py`, `tools/exp06_launch.sh`,
`tests/test_exp06_finalize.py`, `tests/test_exp06_launch.py`
(`git diff main...HEAD --diff-filter=M --name-only` is empty: no file that exists on `main`
was modified).

## 1. Commits (changed lines = insertions + deletions; every one ≤ 200)

| # | Commit | Changed | Subject |
|---|---|---:|---|
| 1 | `22250f9` | 119 | red tests for a complete training-data identity (blocker 1) |
| 2 | `e79fc0f` | 85 | derive and require the whole training inventory (blocker 1) |
| 3 | `dddbba3` | 59 | red tests binding the exp06_* fields to the record (blocker 2) |
| 4 | `2ac4fee` | 32 | bind the exp06_* arguments to the execution record (blocker 2) |
| 5 | `aead393` | 32 | red tests for malformed nested inputs (review 9) |
| 6 | `13da458` | 25 | type every nested provenance container (review 9) |
| 7 | `1bcc7e9` | 70 | red tests for a failing sink and the receipt schema (blocker 4) |
| 8 | `846e5b4` | 63 | enforce the sink status and the receipt schema (blocker 4) |
| 9 | `7ac020b` | 74 | red tests for launch ownership and recovery (review 5) |
| 10 | `d3eeba2` | 121 | launcher ownership, every launch location, real reasons (review 5) |
| 11 | `1b54f37` | 113 | red tests for the real HAA history format (blocker 3a) |
| 12 | `906b694` | 60 | validate the pipeline's real HAA history (blocker 3a) |
| 13 | `6b15002` | 101 | red tests for parsed evaluation metrics (blocker 3b) |
| 14 | `eaba017` | 48 | parse and cross-check the evaluation metrics (blocker 3b) |
| 15 | `4d793c7` | 133 | real job children and a declared job spec (blocker 3c, tests) |
| 16 | `84596ca` | 137 | red job-admission counterexamples (blocker 3c, tests) |
| 17 | `8300de6` | 182 | re-run every child validator inside the job (blocker 3c) |
| 18 | `1571715` | 27 | red tests for a child log that was never closed (blocker 3c) |
| 19 | `21f7a2c` | 13 | re-validate each child's closed log and receipt (blocker 3c) |
| 20 | `bf51484` | 56 | document the round-2a contracts in the module headers |

Every commit carries the two trailer lines. Nothing was amended, rebased, reset or pushed.

## 2. Red before green, per cycle (no retrospective reds)

Each implementation commit is preceded by its own test commit; the failing run below was
observed **before** the implementation was written, on the tree of that test commit.

| Cycle | Red run (command) | Result |
|---|---|---|
| 1 (blocker 1) | `pytest tests/test_exp06_finalize.py` | **9 failed**, 143 passed |
| 2 (blocker 2) | `pytest tests/test_exp06_finalize.py` | **7 failed**, 152 passed |
| 9 | `pytest tests/test_exp06_finalize.py -k malformed` | **7 failed**, 11 passed |
| 4 (blocker 4) | `... -k "receipt or child_exit_subcommand"` / `tests/test_exp06_launch.py` | **9 failed** + **1 failed** |
| 5 | `pytest tests/test_exp06_launch.py` | **8 failed**, 13 passed |
| 3a | `... -k "haa_train or exp02 or off_cadence or unimproved"` | **18 failed**, 22 passed |
| 3b | `... -k "haa_eval or metrics or dampened or room_directory"` | **11 failed**, 21 passed |
| 3c (job) | `pytest tests/test_exp06_finalize.py -k job` | **37 failed** (every job test) |
| 3c (closed log) | `... -k never_closed` | **2 failed** |

## 3. What each finding's fix is, and the regression that pins it

**Finding 1 — training-data identity.** `check_identity_schema` (called by `load_provenance`
for every identity record on every run type) refuses a non-object, an **empty** inventory, an
entry without a str path / 64-hex sha256 / int size / int mtime_ns, an `inventory_files` that
is not the entry count, an `inventory_bytes` that is not their size, and an
`inventory_sha256` that is not `inventory_digest(entries)`. For a `full` run,
`verify_train_identity` additionally requires `provenance.json` to record the resolved
`data_root`, requires the identity's `data_root` to resolve to it, and derives the expected
membership from the pinned helper: `train_inventory_paths` calls
`tools.provenance.train_data_identity` for the recorded root, reusing exp_04's cache
(`ckpt/yaw_aug/train_inventory.json`) **only when that cache already describes this root** and
otherwise recomputing into a `TemporaryDirectory` — it never writes under `ckpt/`. Any
training file missing from the recorded inventory is refused.
*Tests:* the tiny data root is now a real AcousticRooms layout (ten category directories, two
train IRs in `Apartments_idx_1`, one IR in the unseen room `Bathrooms_idx_18`) and the fixture
records what the pinned helper produces for it, so the helper is exercised end to end.
`test_the_expected_membership_comes_from_the_pinned_helper` (membership is the train split,
the unseen room's file is not in it, an unreachable root refuses),
`test_an_incomplete_training_inventory_is_refused` (empty + correct empty digest, one file
short, root mismatch, missing root, wrong file count, wrong byte total, wrong digest, broken
entry, non-list inventory), `test_the_recorded_data_root_must_be_the_one_the_run_resolved`.
`tools/provenance.py` is untouched.

**Finding 2 — exp_06 bindings.** `check_exp06_bindings` recomputes the registry digest from
`model.xRIR_cyl_oriented.BACKBONES_EXP06` at finalisation and requires *both*
`provenance.registry_sha256` and `args.exp06_registry_sha256` to equal it;
`exp06_source_closure_sha256` must equal the closure digest `verify_source_closure` has just
recomputed from the reviewed blobs (and therefore the provenance value);
`exp06_git_head` must equal `provenance.git_state.HEAD`; `exp06_run_type` must equal the run
type the finalizer was invoked with; `exp06_provenance_path` must resolve to the
`provenance.json` that was validated.
*Tests:* `test_false_exp06_bindings_are_refused_even_when_all_copies_agree` writes the same
false value into all three recorded copies (run type `probe`, false HEAD, false registry and
closure digests, `/wrong/provenance.json`) and each is refused;
`test_the_registry_digest_is_recomputed_at_finalisation`;
`test_the_provenance_path_must_be_the_validated_record` (a byte-identical copy elsewhere is
refused, a repo-relative path to the real record is accepted). The fixture now records the
true bindings, so the success path proves the binding, not a placeholder.

**Finding 3a — HAA history.** `haa_history` implements exactly what
`sim_to_real/finetune_haa.py` lines 95-140 write: `epochs + 1` rows, row *i* recording epoch
*i*; epoch 0 with `train_loss: null` and a finite `val_loss`; every later epoch with a finite
`train_loss` and `lr`; `val_loss` present and finite **exactly** on `epoch % val_every == 0`
or `epoch == epochs` and absent otherwise; `is_best` only on validated rows and only `true`.
`summary.json` must record `best_epoch`, `best_val_loss`, `init_val_loss`, `epochs`, with
`best_val_loss` equal to the minimum validated loss (epoch 0 included), `best_epoch` a
validated epoch that achieved it (**0 is admissible**), `init_val_loss` equal to the epoch-0
row and `epochs` equal to `args.json`.
*Tests:* `pipeline_history()` writes the real format from the writer's own logic and is used
by every `haa_train` fixture; `test_an_off_cadence_final_epoch_is_the_pipeline_format`
(epochs 7, val_every 2 → validated 0, 2, 4, 6, 7);
`test_an_unimproved_initialisation_may_remain_the_best_checkpoint` (best_epoch 0);
`test_the_retained_exp02_history_is_accepted` runs the validator over the retained
`ckpt/sim2real/control/seed0/stage1/{history.jsonl,summary.json,args.json}` (1001 rows, 101
validated epochs, best_epoch 930) and is skipped only if that record is absent — it **ran and
passed** here. Refusals: missing epoch-0 row, a trained epoch 0, a missing epoch, an
off-cadence `val_loss` or `is_best`, a NaN train loss, a missing `lr`, a non-finite val loss,
and each of `best_val_loss` / `best_epoch` / `init_val_loss` / `epochs` disagreeing with the
history.

**Finding 3b — evaluation metrics.** `haa_metrics` parses `metrics_<room>.json`, requires
every field `sim_to_real/eval_haa.py` writes (`backbone`, `checkpoint`, `room`, `split`,
`num_shot`, `eval_seed`, `n_samples`, the six metric summaries and the three counts), requires
`room` to equal the room being evaluated, `backbone`/`checkpoint`/`num_shot`/`eval_seed`/
`split` to equal `args.json` type-strictly, `n_samples == len(index)`, and for each metric a
`{mean, median, n}` summary whose `n` is the number of finite per-sample values and whose
`mean` equals their recomputed mean (`math.isclose`, rel/abs tol 1e-9); `t60_error_pct` must
be `null` exactly for `dampened_room`. `haa_eval_evidence` also requires the child's directory
to be named for its room.
*Tests:* the fixtures write the real summary recomputed from the per-sample values (with one
non-finite C50 to exercise the finite filter); `test_the_metrics_summary_is_parsed_and_cross_checked`
covers `not JSON`, a missing key, a wrong `n_samples`, a wrong mean, a wrong `n`, a foreign
room/backbone/seed/checkpoint and a null T60 outside the dampened room;
`test_the_dampened_room_records_no_t60`;
`test_an_evaluation_child_must_sit_in_its_own_room_directory`.

**Finding 3c — job admission.** `haa_job_evidence` now requires `--job-spec` and, for every
child, `verify_child` **re-runs the child's own role validator** (`haa_train_evidence` /
`haa_eval_evidence`) over the child directory — so the full per-role artefact set is required
again (training: `args.json`, `provenance.json`, `history.jsonl`, `summary.json`, `best.pth`,
`last.pth`; evaluation: `args.json`, `provenance.json`, `metrics_<room>.json`,
`per_sample_<room>.json`) — and then binds the completion to that result: `schema_version`,
`run_dir`, run type by path, non-diagnostic, `child_exit == 0`, every artefact hash equal to
the re-run's, and `backbone`/`frame`/`heading` plus the role's `rooms`/`init_sha256`/
`best_epoch`/`seed` or `room`/`checkpoint_sha256`/`samples`/`seed` equal to the re-run's.
`rehash_bound_evidence` rehashes the bound log and `child_exit.json` **and** re-validates them
(`closed_log` + `child_exit_receipt`), so a child whose log was never closed is refused even
when its digests agree. `check_job_spec` requires every child to agree with the declared job
(backbone, frame, seed, rooms ⊆ the job's rooms, `stage2_<room>`/`eval/<room>` covering
exactly their path's room, and each heading roll equal to the spec's `k`), and `job_lineage`
derives the chain from the spec: stage1 from the job's init sha256, each `stage2_<room>` from
`sha256(stage1/best.pth)`, each `eval/<room>` from `sha256(stage2_<room>/best.pth)`, and every
zero-shot evaluation from the job's init. `admissible_arm` is still derived here.
*Tests:* the job fixture builds **nine real children** that this finalizer itself certified
(real provenance and closures, heading bindings verified against real heading JSONs, real
histories, metrics and per-sample files, and checkpoints with distinct hashes), and the
refusals now include the review's four counterexamples plus the new ones: removed evaluation
artefacts, a stale log hash, a stale `child_exit.json`, a wrong room claim, a wrong `run_dir`,
a wrong `schema_version`, a child run at another seed, a stale (valid but different)
checkpoint, a missing artefact, a bare-claims job, broken stage-1 and stage-2 lineage, and a
child whose log carries no end marker or whose receipt disagrees with its status.
`test_the_job_spec_is_required_and_validated` covers an absent, unparseable, incomplete or
internally invalid spec.

**Finding 4 — sink failure and the receipt.** `run_child` captures the sink's status
(`wait "$sink" || sink_status=$?`) and, when it is non-zero, says `SINK_FAILED`, aborts to
`<attempt>_ABORTED_sink_failed` / `<log>_ABORTED_sink_failed` and exits 3 **before**
`close_child` runs — no end marker, no receipt, no completion. The receipt now records the
`started_at` the launcher takes at spawn, and `child_exit_receipt` requires a positive integer
`child_pid`, an integer `status` equal to the reported one, ISO-8601 `started_at`/`ended_at`
with offsets, `ended_at >= started_at`, `ended_at` equal to the log marker's timestamp, and a
64-hex `log_sha256_after_marker` equal to the digest of the validated bytes; the `child-exit`
subcommand refuses a malformed `--started-at`, a non-positive pid or a backwards clock.
*Tests:* `test_a_failing_log_sink_aborts_the_launch` is a genuine CPU fault — the harness sets
`ulimit -f 1`, the child writes 8 KiB, and the real `cat` sink dies of SIGXFSZ **after
consuming output**; the test asserts exit 3, both `_ABORTED_sink_failed` renames, no marker in
the log and no `child_exit.json`. No production seam was added for this.
`test_the_child_exit_receipt_must_bind_the_hashed_log` gained pid, timestamp, ordering, marker
and digest cases.

**Finding 5 — ownership and recovery.** `own_launch <dir>` writes the **launcher's** pid to
`launch.pid` (full and diagnostic modes alike) and `run_child` writes the child's to
`child.pid`. `refuse_live_launch(run_dir, owner_pid)` refuses any live pid except a
`launch.pid` that equals the `--owner-pid` the launcher passes to its own finalisation; a live
`child.pid` is never admissible, and recovery (`finalize` mode) passes no owner at all.
`preflight` takes a repeatable `--attempt-root` and the launcher passes both the attempt root
and `ckpt/exp06/_smoke`; `live_launches` scans `*/launch.pid` and `*/child.pid` under every
root. Diagnostic directories are created exclusively (`mkdir -p` the parent, then plain
`mkdir`). The recovery path aborts with the real reason `finalize_refused`, and the `full`
dry-run prints `child_exit_<code>`; no `<reason>` placeholder remains anywhere.
*Tests:* `test_the_launcher_owns_launch_pid_while_the_child_has_its_own`,
`test_a_live_owner_may_finalize_its_own_attempt`, `test_preflight_scans_every_launch_location`,
`test_a_diagnostic_directory_is_created_exclusively`,
`test_recovery_finalize_gates_promotes_and_aborts` (and the full dry-run golden) assert that
no line contains `<reason>`.

**Finding 9 — malformed nested structures.** `_mapping(value, label)` refuses a non-object or
a container with non-string keys before anything traverses or sorts it, and is applied to
`git_state` (which must also carry `HEAD`, `dirty`, `untracked`, `dirty_outside_worklog`,
`diff_sha256`), `environment`, `mutable_inputs` (each entry a record with a str `path`), both
identity records, the job spec, its heading and each child's `artifacts`. `_state_dict` sorts
`repr(key)` instead of the keys themselves.
*Tests:* the three review cases (`{1: tensor, 'bad': 'not a tensor'}` in `last.pth['model']`,
`git_state=[]`, `mutable_inputs=null`) plus an incomplete `git_state`, a pathless
`mutable_inputs`, a list `data_identity` and a list `environment` — all named `ValueError`s,
and the CLI exits 2 (`test_cli_exits_two_on_a_malformed_checkpoint` still covers the exit).

## 4. Validation

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH=<wt> XRIR_DATA_PATH=~/data_cache/AcousticRooms
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/exp06_opus_numba
```

| Check | Result |
|---|---|
| `python -m py_compile tools/exp06_{recipe,train,finalize,smoke,heading}.py tests/test_exp06_{finalize,launch}.py` | OK |
| `bash -n tools/exp06_launch.sh` | OK |
| `git diff --check 2fe25bf..HEAD` | clean |
| `git diff main...HEAD --diff-filter=M --name-only` | empty (no pinned file modified) |
| `python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider` | **488 passed, 3 skipped** in 409 s (exit 0). The only skips are the three `tests/test_exp06_factory.py:47` cases marked "full pinned forward requires CUDA"; the retained exp_02 history test **ran and passed**. |
| `python -m pytest tests/test_provenance.py -q` | **32 passed** (green throughout) |

## 5. Discrepancies, deviations and choices (nothing silent)

1. **`--job-spec` is mandatory for `haa_job`** (fail-closed). The prompt says the pipeline
   writes it; that pipeline is round 2b, so this round defines and enforces the schema:
   `{init, backbone, frame, init_sha256, seed, rooms, expect, heading: {room: k}}`, with
   `rooms` required to be exactly the four `sim_to_real.haa_dataset.ROOMS` and `heading`
   required (and only allowed) in the heading frame. Round 2b's pipeline must write this file.
2. **Every HAA child's `args.json` must record an integer `seed`.** The job spec declares the
   seed and each child must agree with it, so `haa_child_arguments` now requires the field on
   both child roles. Round 2b's `exp06_haa_finetune` / `exp06_haa_eval` must write it (the
   frozen `sim_to_real/finetune_haa.py` already does; `eval_haa.py` records only `eval_seed`,
   which is why the exp_06 entries — not the frozen ones — own this key).
3. **Directory/room binding.** An evaluation child must live in a directory named for its
   room, and in a job `stage2_<room>` and `eval/<room>` must cover exactly that room. This is
   how "wrong room claim" is refused at the child level as well as the job level.
4. **Sink-failure exit code 3.** The prompt asked for the abort path; the launcher exits 3 so
   a failed sink is distinguishable from a child's own status. The CPU fault test uses
   `ulimit -f 1` (the sink dies of SIGXFSZ after consuming output) rather than adding an
   injectable sink command, so no test seam exists in the production path.
5. **`--owner-pid`.** Making `launch.pid` name the launcher means the launcher is alive while
   it finalizes its own attempt, so the finalizer needs to know which live pid is the owner.
   The launcher passes `--owner-pid $$` (printed as `<pid>` in `--dry-run`); recovery passes
   none, so a foreign live launcher still refuses. `preflight`'s record field `attempt_root`
   is now a list of the roots that were scanned.
6. **Receipts written before this commit would be refused** (no `started_at`). Nothing has run
   yet, so there is no artefact to migrate.
7. **Three test expectations were corrected while going green on 3c** (the refusals were
   real, the expected cause strings were wrong): `stale_hash` now overwrites `best.pth` with a
   *valid but different* checkpoint (garbage bytes were refused earlier, as "unreadable
   best.pth"); `lineage_init` also rebuilds the dependent `eval/hallway` child so the only
   remaining break is the stage-2 lineage; the absent-spec regex is `job-spec`.
8. **Runtime.** The job tests now build nine real children each, so
   `tests/test_exp06_finalize.py` takes about 3.5 minutes (218+ tests). This is the price of
   deriving job admission from re-validated evidence rather than from child claims.
9. **Still open from the close review (not in the mandatory list):** finding 10's process nit
   is addressed going forward — all 20 commits here are ≤ 200 changed lines and every
   implementation commit has a recorded red run on its own test commit — but the earlier
   history is unchanged, as requested.
