**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 round 2a — training-gate fix cycle (Codex `full_train` review, findings 1–6, nits 8–9)

Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`, from `12e06fc` to
**`f18383a32c90985a197e5fb82a4584f6e1529794`**. CPU only (`CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=4`), Python 3.8 (`/home/yixunhu/miniconda3/envs/xRIR/bin/python`).
No existing tracked source file was modified: `git diff main...exp06-window --diff-filter=M
--name-only` is **empty**; the 20 changed paths are all `model/`, `tools/` and `tests/`
additions of rounds 1–2a plus the new `tools/exp06_profiles.py` and
`tools/exp06_approved_digests_template.json`. Finding 7 and the omitted-T60 nit were left
to the round-2b Coder; no hunk of mine falls in `_heading_binding` (line 631 at `12e06fc`)
or `haa_metrics` (line 804) — my finalizer hunks are at ≤ 576 and ≥ 1164 in that numbering.

## Commits (23: 11 red, 12 implementation; every one ≤ 200 changed lines)

| sha | ± lines | subject |
|---|---:|---|
| `4fb3f7c1` | 33 | red tests for checkpoint dtype/shape and the probe's recipe flags (nits 8, 9) |
| `cf7df40f` | 11 | bind checkpoint dtype/shape and give the probe the registered flags (nits 8, 9) |
| `ccb06b52` | 54 | red tests for type-strict, bounded operational arguments (finding 4) |
| `4120094f` | 63 | validate operational argument types, bounds and cadence (finding 4) |
| `439236f3` | 44 | red tests for the exit receipt's child pid liveness (finding 3) |
| `56dc9451` | 37 | refuse an exit receipt whose child pid is alive or disagrees (finding 3) |
| `3154c3bb` | 132 | red tests for the approvals module and its null template (finding 1) |
| `04c44c79` | 189 | the approvals schema and its null template, both copies (finding 1) |
| `7abc0c60` | 76 | recompute code digests and admit runs fail-closed (finding 1) |
| `54ad12de` | 78 | red tests for the trainer's orchestration and approvals bindings (finding 1) |
| `1e04b8b7` | 82 | record the launcher, finalizer and approvals identities at spawn (finding 1) |
| `18e8b5cc` | 103 | red tests for approvals and orchestration at finalisation (finding 1) |
| `73157775` | 105 | verify approvals, code digests and orchestration at finalisation (finding 1) |
| `eb0a1c99` | 65 | red tests for the launcher's approval gate (finding 1) |
| `4885e175` | 89 | gate every launch on the approved code digests (finding 1) |
| `265be9c0` | 66 | red tests for the geometry-input inventory (finding 2) |
| `d5ff641e` | 100 | inventory and rehash every geometry input of the run (finding 2) |
| `815f58f8` | 66 | red tests for the diagnostic receipt's identity and budgets (finding 6) |
| `3834d04b` | 84 | diagnostics record their runner, budgets, timing and provenance (finding 6) |
| `090ab7f8` | 154 | red tests for validated diagnostic evidence at finalisation (finding 6) |
| `78523b6d` | 120 | require validated diagnostic evidence and its provenance (finding 6) |
| `163f8181` | 87 | red tests for a failed diagnostic's status and abort naming (finding 5) |
| `f18383a3` | 36 | a failed diagnostic aborts the launch with the child's status (finding 5) |

Every implementation commit is preceded by a commit whose tests genuinely fail. Red counts
in order: 3 failed · 26 failed · 2 failed · 1 collection error (module absent) · 5 failed ·
10 failed · 5 failed · 21 errors · 7 failed · 23 failed · 4 failed.

## Per finding

**1 — training approvals and orchestration bindings.** New `tools/exp06_profiles.py`
(schema 1; `code` / `reused` / `artifacts` sections; `load_approved_digests`,
`code_digest`, `closure_of`, `file_digest`, `compute_code_digests`, `present_keys`,
`require`, `TRAINING_KEYS`) and the all-null template written byte-identically to
`tools/exp06_approved_digests_template.json` (worktree, committed) and
`oriented_cyl_results_assets/approved_digests.json` (main tree, uncommitted —
sha256 `0ec10825bce5f9156537b5eb42934ab5d987b218d439f15e94e971501f3e018e` for both).
`evaluator_exp03` is the only filled value and is *recomputable*: a test recomputes
`closure_record(source_closure('eval_yaw_rotation'), HEAD)` and reproduces
`5ba818d8…1be48`. `exp06_train` now records `orchestration_closures`
(`launcher` = `tools/exp06_launch.sh` bound as a file, `finalizer` =
`tools.exp06_finalize` + its import closure, both with working-tree and reviewed-blob
hashes), `code_digests` for the eight `TRAINING_KEYS`, `approvals` = {path, sha256,
schema_version} and `exploratory`; it takes `--approved`, `--reviewed-commit` and
`--exploratory`, and `main()` refuses a `--run-type full` run without `--approved` at
spawn. The launcher preflight (`exp06_finalize.approval_gate`) recomputes the
`TRAINING_KEYS` digests at the reviewed commit and requires equality with the approved
`code` section — null refuses every mode, `--exploratory` turns deviations into a recorded
list and is itself refused for mode `full` — and `tools/exp06_launch.sh` passes
`--approved`/`--reviewed-commit` to both the training child and the smoke runner.
`exp06_finalize.verify_approvals` recomputes the same digests at finalisation and requires
three-way agreement (recomputed ↔ recorded in provenance ↔ the approvals file re-read and
hash-checked against the value bound at spawn), and `verify_orchestration` compares the
launcher and finalizer bytes working-tree ↔ recorded ↔ reviewed.
*Regression*: `tests/test_exp06_profiles.py` (15 cases) and, in
`tests/test_exp06_finalize_gate.py`, the review's own reproduction — appending a byte to
`tools/exp06_launch.sh` or `tools/exp06_finalize.py` in the temp clone after provenance is
refused with `drift` — plus null-approvals refusal, a rehashed approvals file, missing
`code_digests`/`orchestration_closures`, and `exploratory: true` on a full run.

**2 — geometry-input inventory.** `exp06_train.geometry_paths(root, splits)` derives, from
the pinned dataset's own `file_list` for the train **and** test splits, the metadata JSON of
every IR and the depth map of every receiver — which is exactly the reference geometry too,
because the references are other sources at the *same* receiver and so are IRs of the same
room directory. `geometry_identity` hashes them through `tools.provenance._inventory`
(imported, the helper unchanged) and the trainer records it in `provenance.json`.
`exp06_finalize.verify_geometry_identity` requires it, derives membership again from the
dataset (never from the record), requires an exact membership match, rehashes every file
and requires the inventory digest to agree.
*Regression*: an independently enumerated membership equals the dataset's list; a changed
metadata JSON and a changed depth map are each refused; an absent, truncated or
wrong-rooted inventory is refused.

**3 — receipt child pid.** `child_exit_receipt` now checks the receipt's `child_pid` for
liveness (new `_alive`, which `_pid_of` also uses) and requires it to equal the `child.pid`
sidecar when that exists. The owner exception stays on `launch.pid` only.
*Regression*: a receipt naming the test's own pid is refused; a disagreeing `child.pid` is
refused, an agreeing one is admitted. The test helper now seals receipts with
`/proc/sys/kernel/pid_max`, a pid that can never be live.

**4 — operational validation.** `exp06_recipe.check_operational` is type-strict and
bounded: `backbone ∈ BACKBONES_EXP06`, non-empty `save_dir` string, `num_workers ≥ 0`,
`log_interval > 0`, `save_every ≥ 0`, `epoch_ckpt_every == 1` (module constant
`EPOCH_CKPT_EVERY`), `env` a mapping of str → str|None; `True` never passes for `1` and
`1.0` never for `1`. `check_all` calls it except on the historical path, which is unchanged.
*Regression*: the review's four reproductions (`epoch_ckpt_every = 2`, `= True`,
`save_every = None`, `num_workers = "twelve"`) plus 13 more, each named; every missing
operational field named; and an explicit case that the historical path still admits
exp_01's `epoch_ckpt_every = 5`.

**5 — failed diagnostics.** `diagnostic()` now records the completion first (keeping
`passed: false`), then on a refused finalisation renames the run and log
`_ABORTED_finalize_refused` and exits 2, and on a non-zero child status renames
`_ABORTED_child_failed_<status>` and exits with that status — so `smoke` stops at the first
failed rung and `probe` cannot look successful.
*Regression*: shell tests through a stub interpreter on `PATH`-free dispatch (`PYTHON`
overridden, real interpreter used for `child-exit`): child status 3 → exit 3, both renames,
`--child-exit 3` still passed to the finalizer, and `LAUNCHER_CONTINUED` not printed;
finalizer status 2 → exit 2 and the `finalize_refused` renames; the success path continues.

**6 — diagnostic evidence.** The smoke runner records `runner`, `runner_closure_sha256`,
`entry`, `module`, `argv`, `run_type`, `started_at`/`ended_at`, `wall_s`, `peak_bytes`,
`alarm_seconds`, `max_gb`, `outcome ∈ {ok, failed, aborted_alarm, aborted_memory}` (with the
exception type in the new `outcome_detail`), `exit_status`, `exploratory`, `git_head` and,
with `--provenance-out`, a `provenance.json` (`run_type` smoke|probe, its own closure,
orchestration closures, code digests, approvals, exploratory) which the launcher now always
requests. `diagnostic_evidence` loads and verifies that provenance (closure three ways,
approvals with the exploratory tolerance) and `diagnostic_receipt` requires and types every
field, binding `git_head`, the runner closure digest, the run type and the exploratory flag
to the provenance.
*Regression*: the review's minimal receipt `{diagnostic, argv, exit_status}` is refused;
each of the 15 required fields is refused when missing; wrong runner, wrong closure digest,
wrong HEAD, wrong run type, an outcome outside the set, a negative `wall_s` or
`peak_bytes`, and a missing `provenance.json` are each refused; a complete one is accepted
and its budgets, wall time and peak reach `completion.json`.

**Nit 8** — `epoch_012.pth` must match `last.pth["model"]` in dtype and shape per key
before `torch.equal`; a float64 copy and a reshaped copy are each refused by name.
**Nit 9** — the probe passes `--decay-epochs 3 --log-interval 50` explicitly; the dry-run
golden was updated.

## Validation

| command | outcome |
|---|---|
| `python -m py_compile tools/exp06_*.py tests/test_exp06_*.py` | pass |
| `bash -n tools/exp06_launch.sh` | pass |
| `git diff --check main...exp06-window` | pass |
| `git diff main...exp06-window --diff-filter=M --name-only` | empty |
| `CUDA_VISIBLE_DEVICES='' pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider` | **639 passed, 4 skipped** (11 m 24 s) |
| `pytest tests/test_exp03_record_tools.py -q -p no:cacheprovider` | 31 passed, 1 skipped |
| `bash tools/exp06_launch.sh {full,probe,smoke,finalize} --dry-run …` | exit 0, argv as the goldens |

Baseline at `12e06fc` was 540 passed / 3 skipped; the fourth skip is the new
`test_the_record_copy_is_byte_identical_when_it_is_present`, which skips because the record
asset lives in the main tree and not on this branch (it will run after the merge).

### Real-mirror geometry-inventory timing (read-only, 4 threads, `XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`)

```
enumeration   3.1 s   312 321 files (302 671 metadata JSON + 9 650 depth maps)
                      train 296 334 IRs, test 6 337 IRs
hashing      56.4 s   9.48 GB, digest aae056d749366e9f…
```

**Well under the 10-minute threshold**, so full hashing stands and no sampled-plus-manifest
scheme is proposed. Production uses `GEOMETRY_WORKERS = 8`, so the finalizer's rehash costs
at most this ≈ 1 minute; the trainer pays it once at spawn as well.

## Discrepancies and deliberate choices (none silent)

1. **`launch_sh` / `haa_pipeline_sh` have no Python entry.** The prompt's formula reads
   `closure_record(source_closure(module) [+ shell files])`. A shell file is not importable,
   so these two keys are specified as `(module=None, extra=(path,))` and their digest is the
   closure over that one file. Binding them to some Python module's closure instead would
   have made `launch_sh ⊇ finalize` and hidden a shell edit behind a Python one.
2. **`tools.exp06_probe_align` is not a key.** Plan §6.4 lists it; the gate-fix prompt's key
   list does not. I followed the prompt; round 3b should add it (and the other producers)
   when it extends the module.
3. **`--approved` is required for `--run-type full` in `main()`, not in the parser.** A
   parser-level `required` would have broken every existing test that parses the registered
   recipe argv without an approvals file. The refusal still happens at spawn, before the
   dataset is built, and the finalizer refuses a `full` provenance with no approvals
   binding, so the path stays fail-closed.
4. **`load_approved_digests` does not require git-committed bytes** (exp_04's
   `tools/exp04_profiles.py` does). The main-tree record asset is not in this worktree and
   the default path is the worktree copy, so a `git cat-file` check would be against the
   wrong repository. Committed-ness is enforced instead by the preflight, which already
   requires a clean tree outside `worklog/` at the reviewed commit, and mid-run edits are
   caught by the sha256 the run binds and the finalizer re-reads.
5. **`closure_of` caches per process, stat-validated.** `source_closure` spawns an
   interpreter per module (≈ 6 s), and a producer asks for eight keys. `_paths_of` is
   `lru_cache`d, and `closure_record` is cached on `(name, repo, commit, (path, size,
   mtime_ns)…)` — the same stat validation `provenance.train_data_identity` uses for data —
   so an edited file always misses the cache. A producer process computes each key once, so
   in production the cache is cold at every decision point anyway.
6. **The geometry rehash wall time is printed, not recorded.** `completion.json` must be
   byte-identical on a re-run and a measured duration never is; the finalizer prints
   `EXP06_GEOMETRY_REHASH <files> files in <s> s` and records only `geometry_files`,
   `geometry_bytes`, `geometry_splits` and `geometry_sha256`.
7. **`exp06_finalize` now imports `exp06_train`** (for `geometry_paths` and
   `orchestration_closures`). That is deliberate: the finalizer's independent membership
   derivation depends on the dataset's split logic, and importing it puts the dataset module
   inside the finalizer's own bound closure. It enlarges the `finalize` code digest, which is
   recomputed everywhere and pinned nowhere yet.
8. **Provenance fields remain a fifth schema class** in `tools/exp06_recipe.py` (plan §5
   words four). Folding `exp06_*` into OPERATIONAL would make a provenance digest look like a
   declared operational difference; keeping them apart is what lets
   `check_exp06_bindings` require each one to equal the execution record. Recorded here as
   the textual deviation the review already noted.
9. **Test-contract updates forced by the fixes** (all in the round-2a test files): the
   diagnostic fixtures now build a run directory with `provenance.json`, the receipt
   assertion is a superset check, `test_changed_training_data_bytes_are_refused` accepts
   `membership` as the cause when a deleted IR takes its geometry out of the split with it,
   `test_preflight_cli_exits_two_on_refusal`'s success case is now an `--exploratory smoke`
   (a checkout without the exp_06 modules can admit nothing else), and the job-owner test
   writes the receipt's own dead pid into `child.pid`.
10. **Not addressed, by instruction:** finding 7 (the heading consumer contract) and the
    omitted-T60 NaN-only nit belong to the concurrent round-2b Coder.
