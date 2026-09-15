**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# Round 2b — evaluation entry point, evaluation launcher, HAA wrappers, HAA pipeline

Worktree `/home/yixunhu/codespace/xRIR_code_wt2b`, branch `exp06-round2b` from `12e06fc`
(rounds 1 and 2a closed). CPU only (`CUDA_VISIBLE_DEVICES=''`), `OMP_NUM_THREADS=4`,
`PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt2b`,
`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest with `-p no:cacheprovider`, interpreter
`/home/yixunhu/miniconda3/envs/xRIR/bin/python`. Nothing outside the worktree was written
except this report; no process I did not start was signalled; `ckpt/` was never written;
the superseded archive and `archive/codex-round2a-2026-09-15` were not read.

A scope update arrived mid-round from the Planner (a seventh cycle for the finalizer's
heading consumer contract, and the instruction to keep finalizer edits confined and to put
new finalizer tests in a new file). It is implemented as cycle 6 below.

## Files

New: `tools/exp06_eval.py`, `tools/exp06_eval_launch.py`, `tools/exp06_haa_finetune.py`,
`tools/exp06_haa_eval.py`, `tools/exp06_haa_pipeline.sh`, `tests/test_exp06_eval.py`,
`tests/test_exp06_eval_launch.py`, `tests/test_exp06_haa.py`,
`tests/test_exp06_haa_pipeline.py`, `tests/test_exp06_finalize_haa_consumer.py` (authorised
by the scope update).

Modified: `tools/exp06_finalize.py` (only the omitted-T60 check and `_heading_binding`) and
`tests/test_exp06_finalize.py` (only the `heading_jsons` fixture and the two HAA argument
helpers — see discrepancy 12).

`git diff main...HEAD --diff-filter=M --name-only` is **empty**: no file that exists on
`main` was modified, so every pinned/composed source (`train_xRIR_backbone.py` and its
closure, `tools/exp04_*.py`, `tools/exp05_*.py`, `tools/provenance.py`,
`tools/paired_compare.py`, `sim_to_real/*.py`, exp_03's twelve pinned files,
`tests/test_provenance.py`, `tests/test_exp03_record_tools.py`) is byte-identical.

## Commits (18, oldest first; red test commit then implementation, per cycle)

| SHA | changed lines | what |
|---|---|---|
| `9e727af` | 38 (+38) | **red** — infinity in an omitted room's T60, standalone and job admission |
| `f350d51` | 6 (+4/−2) | cycle 0 — every omitted-room T60 must be NaN (close-4 finding 1) |
| `ae7ea78` | 136 (+136) | **red** — `tools/exp06_eval.py` |
| `5f6d49b` | 102 (+102) | cycle 1 — simulated-room evaluation entry point |
| `c5c81c6` | 198 (+198) | **red** — `tools/exp06_eval_launch.py` |
| `65e099f` | 143 (+143) | cycle 2 — evaluation launcher adapter |
| `483287b` | 196 (+196) | **red** — HAA fine-tuning wrapper |
| `aa61fc4` | 294 (+294) | cycle 3 — `tools/exp06_haa_finetune.py` (over the 200-line guidance, see 11) |
| `cc77266` | 71 (+71) | **red** — HAA evaluation parser, datasets, records |
| `d82b126` | 104 (+101/−3) | cycle 4a — `tools/exp06_haa_eval.py` part 1 |
| `6262141` | 84 (+84) | **red** — per-query Griffin-Lim seeding, side labels, room summary |
| `855ec9a` | 151 (+149/−2) | cycle 4b — the evaluation loop and the output writers |
| `4957053` | 163 (+163) | **red** — pipeline dry-run goldens |
| `140c3c9` | 212 (+212) | cycle 5a — `tools/exp06_haa_pipeline.sh` (over the guidance, see 11) |
| `2fce9ef` | 174 (+174) | **red** — end-to-end HAA job contract on written artefacts |
| `e5461a6` | 9 (+5/−4) | cycle 5b — a HAA child records the repository it was launched from |
| `4b30735` | 227 (+183/−44) | **red** — heading consumer contract; cycle-0 tests moved to the new file |
| `d6a4f42` | 18 (+16/−2) | cycle 6 — re-verify every bound heading, require `confirmatory` |

Every commit carries the two trailer lines (`Co-Authored-By: Claude Opus 5` and
`Claude-Session: …`), verified for all 18. No amend, rebase, reset or push.

Red counts actually reproduced before each implementation:

| red commit | failures |
|---|---|
| `9e727af` | 4 (`DID NOT RAISE`, both signs × standalone/job) |
| `ae7ea78` | collection error: `cannot import name 'exp06_eval' from 'tools'` (19 tests) |
| `c5c81c6` | collection error: `cannot import name 'exp06_eval_launch'` (16 tests) |
| `483287b` | collection error: `cannot import name 'exp06_haa_finetune'` (11 tests) |
| `cc77266` | 4 failed |
| `6262141` | 7 failed (the 8th, `…inversion_reproducible`, was outside the `-k` filter) |
| `4957053` | 11 failed |
| `2fce9ef` | 6 failed (`TypeError: prepare() got an unexpected keyword argument 'repo'`) |
| `4b30735` | 6 failed, 6 passed (the 6 passing are the moved T60 regressions) |

## What each cycle produced

**Cycle 0 — `tools/exp06_finalize.py` (close-4 finding 1).** In `haa_metrics`, a room whose
T60 the paper omits must now have `math.isnan` for **every** per-sample T60; the previous
`not finite` accepted ±infinity. Regressions for both signs, standalone and at job
admission with the child's artefact hash refreshed.

**Cycle 1 — `tools/exp06_eval.py`.** `build_parser(require_eval_manifest)` = exp_04's parser
with the `--backbone` choices replaced by `sorted(BACKBONES_EXP06)` on the fresh instance
(exp_04's own parser is asserted unchanged), plus `--checkpoint-role {arm,baseline,diagnostic}`
(default `arm`) and `--checkpoint-epoch` (int). `exp06_metadata` = `{model_class,
registry_sha256, checkpoint_role, checkpoint_epoch, heading: None, frame: 'room'}`;
`validate_manifest` adds those to exp_04's handshake and treats an **absent** key as a
mismatch; `build_fields` delegates to `exp04_eval_launch.build_fields(module=sys.modules[__name__])`
and updates the metadata; `run_exp06` passes `model_factory`, the metadata and the validator
into `run_exp04`. Tests: registry choices, factory routing by class identity for all three
names, unknown name refused, metadata keys and values (registry digest equals the
finalizer's), manifest round-trip, per-field mismatch **and** per-field omission refused,
another `checkpoint_role`/`checkpoint_epoch` refused, and a `child_command`-compatible
parser (every action has option strings, no positionals).

**Cycle 2 — `tools/exp06_eval_launch.py`.** Own `parse_args`; `split_bindings` accepts
`MUTABLE_INPUTS ∪ {heading}`, validates the heading record with `read_heading_json`, refuses
a refused/missing/malformed one, hashes it, and hands only exp_04's names to exp_04;
`child_command` mirrors the shared launcher's serialiser against `tools/exp06_eval.py`;
`build_fields` delegates with the stripped bindings, appends
`source_closures['writer_exp06']` (refusing closure drift), merges the exp_06 bindings and
then re-runs the manifest consistency check (`check_fields`) — all inside the
`fields_factory`, i.e. before `execute_run` writes the manifest from those fields; `main`
runs the child through the unchanged `execute_run`. Tests: golden child argv, the
launcher-created `--eval-manifest` path, malformed/unlisted/duplicate bindings, the three
unusable-heading cases, both writer closures plus the entrypoint closure in the fields,
drifted writer closure refused, and a dirty-tree refusal against a real throwaway git repo
with a modified tracked file.

**Cycle 3 — `tools/exp06_haa_finetune.py`.** The pinned flags and defaults (asserted equal
by introspection against `sim_to_real.finetune_haa.parse_args`), `--backbone` over
`BACKBONES_EXP06`, plus `--root`, `--heading-json-dir` and a fixed `--run-type haa_train`.
`load_headings` requires one decided, `confirmatory` record per room, re-read against
`<root>/<room>` so the four cache inputs are rehashed. `prepare` (no CUDA, no writes)
resolves the root, selects the frame (oriented backbone without a heading directory is
refused), builds `HeadingFrameDataset` or the pinned `HAADataset`, builds the model,
computes the cache `data_identity`, the provenance fields and the `args.json` mapping
(`frame`, `heading`, `init_sha256`, `haa_root`, `git_head`, `exp06_registry_sha256`,
`exp06_source_closure_sha256`, `run_type`, integer `seed`). `write_records` writes
`provenance.json` exclusively and `args.json` as the same mapping. `main` is
`finetune_haa.main` step for step with the imported `evaluate`, `compute_loss`,
`seed_everything`, `load_model_state` and `ExponentialLR`.

**Cycle 4 — `tools/exp06_haa_eval.py`.** The pinned evaluation flags and defaults (asserted
by introspection against `sim_to_real.eval_haa.parse_args`), plus `--root`,
`--heading-json-dir`, `--seed`, `--gl-seed-per-query` and a fixed `--run-type haa_eval`;
the same fail-closed heading rule; one dataset per room with the pinned `.items` order and
reference draw (asserted item-for-item against `HAADataset`); `per_sample_meta` = the
pinned meta keys plus `room`, `frame`, `heading`, `checkpoint_sha256`, `gl_seed_per_query`,
`registry_sha256`, `git_head`, `exp06_source_closure_sha256`, `run_type`; `gl_seed` is the
frozen `sha256('gl:<eval_seed>:<room>:<idx>')[:8]` little-endian formula and `invert` calls
`torch.manual_seed` with it **only** when the flag is on (asserted: no `manual_seed` call at
all when off); `side_labels` are room-frame signs, identical in both frames, and a
microphone on the speaker axis is refused; `room_summary` is the pinned summary dict plus
`meta`, and `write_room_outputs` keeps the pinned file names. `evaluate_room` is
`eval_haa.py`'s loop with the seeded inversion and the side-label array; `main` writes the
records before the loop and `metrics_all<tag>.json` after it.

**Cycle 5 — `tools/exp06_haa_pipeline.sh` and the end-to-end contract test.** The queue
takes `<gpu> <job>… [--dry-run]`, refuses the **whole** queue before anything runs if any
job is malformed, sources `tools/exp06_launch.sh` as a library for `say`/`run`/`own_launch`/
`run_child`/`close_child`/`abort`/`finalize`, writes the finalizer's mandatory `--job-spec`
before the first child, runs stage 1 / stage 2 / evaluation children in exclusive
directories with the recipes of §6.2, finalizes each child immediately, and finalizes the
job from its own closed log and `child_exit.json`. The job root is the only place a live
`launch.pid` exists; `child()` shadows `OWNER` so child finalizations never claim the owner
exception. `--dry-run` prints every command; goldens cover a full finetune seed and the
three zero-shot jobs line for line. The end-to-end test builds nine real children from
`prepare`/`write_records`/`room_summary`/`write_room_outputs` with a mocked GPU loop over a
synthetic four-room cache, certifies each with the real finalizer, writes the job spec by
sourcing the pipeline's own `job_spec`, and requires `admissible_arm: true`; eight further
cases flip each frozen job-spec field (`backbone`, `frame`, `seed`, `expect`, `init`,
`init_sha256`, `rooms`, `heading`) and one plants a live pid inside a child — all refused.

**Cycle 6 (scope update) — the finalizer's heading consumer contract.** `_heading_binding`
now requires the child's recorded `haa_root`, resolves `<haa_root>/<room>`, re-reads each
record with `read_heading_json(path, room_dir=…)` so the four cache inputs are rehashed, and
refuses any record whose `admissibility` is not `confirmatory`. New tests
(`tests/test_exp06_finalize_haa_consumer.py`): a cache edited after estimation refuses the
child (and the same child is accepted once the cache is restored), a `diagnostic` record is
refused, an absent/empty/mistyped/missing `haa_root` is refused, the room frame needs none,
and all three rooms of a stage-1 child are re-verified — plus the two moved T60 regressions.

## Validation commands and outcomes

| command | outcome |
|---|---|
| `python -m pytest tests/test_exp06_*.py -q -p no:cacheprovider` (baseline, before any change) | 508 passed, 3 skipped |
| `pytest tests/test_exp06_finalize.py -k 'omitted_room_measures or infinite'` (cycle 0 red) | 4 failed (`DID NOT RAISE`) |
| `pytest tests/test_exp06_finalize.py tests/test_provenance.py` (cycle 0 green) | 308 passed |
| `pytest tests/test_exp06_eval.py` | 19 passed |
| `pytest tests/test_exp06_eval_launch.py` | 16 passed |
| `pytest tests/test_exp06_haa.py` | 23 passed |
| `pytest tests/test_exp06_haa_pipeline.py` | 21 passed |
| `pytest tests/test_exp06_finalize_haa_consumer.py` | 12 passed |
| `pytest tests/test_exp06_finalize.py` (after the cycle-6 contract change) | 272 passed |
| `python -m py_compile` on all five new modules and all six touched test modules | OK |
| `bash -n tools/exp06_haa_pipeline.sh` | OK |
| `git diff --check` | clean; `git status --porcelain` empty |
| `git diff main...HEAD --diff-filter=M --name-only` | empty (no file on `main` modified) |
| `python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider` (final) | **631 passed, 3 skipped** in 661 s (the 3 skips are the CUDA-only tests) |

Two further read-only confirmations that the pinned code is untouched (outside the
required command list):

| command | outcome |
|---|---|
| `pytest tests/test_exp03_record_tools.py` | 31 passed, 1 skipped |
| `pytest tests/test_exp04_eval.py tests/test_exp04_eval_launch.py tests/test_exp05_eval.py tests/test_exp05_eval_launch.py` | 142 passed, 1 skipped |

`ckpt/exp06/` does not exist: nothing was written through the worktree's `ckpt` symlink.

`python -m pytest tests/test_provenance.py -q` was green at the start (32 passed) and is
included in the final run above.

## Discrepancies and deviations (nothing was deviated from silently)

1. **`--checkpoint-epoch` is `required=True`.** The prompt gives a default only for
   `--checkpoint-role`. Rather than invent a default epoch that would be silently recorded
   in the manifest and both outputs, the flag is mandatory (the fail-closed option).
2. **`model_factory` is a named function**, not the prompt's inline
   `lambda b, n: build_xrir_exp06(b, n)` — identical semantics, but routing is then
   testable by identity for every registered name.
3. **`registry_sha256()` is re-implemented** in `tools/exp06_eval.py` and
   `tools/exp06_haa_finetune.py` (three lines, the definition of `tools.exp06_train`)
   instead of imported, so the evaluator's recorded source closure does not acquire the
   trainer and the dataset module. A test asserts it equals `exp06_finalize.registry_sha256()`.
4. **exp_04's `execute_run` cannot re-check the exp_06 metadata.** It compares only
   `eval_manifest_sha256`, `conditions`, `n_samples` (plus exp_05's hard-coded tier fields)
   against the child's output `meta`. The exp_06 fields are bound in the manifest
   (`check_fields` before it is written) and revalidated by the child's own
   `validate_manifest`, but a post-exit tamper of `per_sample_yaw.json`'s exp_06 meta would
   not be caught at finalisation. Closing this would require editing the pinned launcher.
5. **`tools/exp06_haa_eval.py` imports the record helpers from `tools.exp06_haa_finetune`**
   (`resolve_root`, `select_frame`, `load_headings`, `build_dataset`, `data_identity`,
   `provenance_fields`, `record_args`, `write_records`) rather than duplicating ~80 lines.
   Consequence: the evaluation child's recorded closure is a superset that includes
   `train_xRIR_backbone.py`. That is the fail-closed direction (a trainer edit invalidates
   evaluation completions, never the reverse), but it over-binds the evaluator.
6. **`provenance._inventory` (a private helper of the pinned module) is used** to build the
   HAA cache `data_identity`, because no public HAA-shaped equivalent exists and the
   finalizer recomputes `inventory_sha256` with exactly that definition; duplicating the
   hashing would risk drift.
7. **`--root` added to both HAA wrappers** (the pinned scripts have none, taking
   `HAA_XRIR_ROOT`/`~/data_cache/HAA_xrir` implicitly). Its default is the pinned
   `DEFAULT_ROOT`; the resolved value is recorded as `haa_root`, which the scope update's
   consumer contract requires, and it lets the tests point at a synthetic cache.
8. **`--seed` added to the HAA evaluation parser** (the pinned `eval_haa.py` has none),
   because the round-2a contract requires an integer `seed` in every HAA child's `args.json`,
   distinct from `eval_seed`. It is also recorded in `provenance.effective_args`.
9. **Heading JSON naming.** The plan does not fix it; `<heading-dir>/<room>.json` is assumed
   by `load_headings` and by the pipeline's job-spec writer, matching round 2a's test
   fixtures.
10. **`load_headings` (producer side) also requires `confirmatory` and re-verifies the cache
    inputs**, symmetric with the new consumer check — beyond the prompt's wording, but the
    alternative is a child that runs happily on a record its own finalizer will reject.
11. **Two commits exceed the ≤200-changed-line guidance**: `aa61fc4` (294 lines — the
    fine-tuning wrapper, whose epoch loop is copied faithfully from the pinned script) and
    `140c3c9` (212 lines — the pipeline shell script). Splitting either would have committed
    a module or script that does not run. Cycle 4 was split into 4a/4b precisely to stay
    under the limit, and cycle 5 into 5a/5b.
12. **`tests/test_exp06_finalize.py` was edited**, which the scope update asked me to avoid.
    The cycle-6 contract makes every existing HAA fixture inadmissible (its heading records
    are `diagnostic`, there is one shared cache directory rather than one per room, and no
    `args.json` records `haa_root`), so the fixture had to move with the contract. The diff
    against the branch point is confined to four hunks in the fixture region: a
    `confirmatory()` helper, the `heading_jsons` fixture (per-room cache directories, a
    clean-tree closure, a `root` key), and one added `haa_root=…` line in each of
    `haa_train_args` and `haa_eval_args`. The cycle-0 regressions I had appended to that
    file were **moved out** into `tests/test_exp06_finalize_haa_consumer.py`, so the file
    contains no other change of mine. The concurrent Coder's regions (provenance/approvals,
    receipt pid, diagnostic receipts, dtype) are untouched.
13. **The pipeline defines `finalize_job` and `open_job` locally.** `tools/exp06_launch.sh`'s
    `finalize` takes no `--children/--expect/--job-spec`, so it is not reusable for the job
    level; everything else (`say`, `run`, `own_launch`, `run_child`, `close_child`, `abort`,
    `finalize` for children) is reused by sourcing it with `EXP06_LAUNCH_LIB=1`. The new
    helpers live in the new pipeline file, which also honours `EXP06_PIPELINE_LIB=1` so the
    tests can call `job_spec` directly.
14. **The HAA job's `child_exit.json` records the live pipeline's pid** (`--child-pid $$`),
    because a job has no separate child process. The finalizer requires only a positive
    integer there, and the job root's `launch.pid` is admitted through `--owner-pid`.
15. **The pipeline's job-spec writer does not itself require a decided, confirmatory
    record.** A refused record would write `heading: null` for that room and `load_job_spec`
    refuses it at finalisation (and every child refuses it earlier). I left the writer thin
    rather than add an untested second gate; flagged here as a residual ordering wrinkle.
16. **Plan §6.3 lists `train_completion` as an exp_06-specific binding**, but exp_04's
    `MUTABLE_INPUTS` already contains it, so — following the round-2b prompt — only
    `heading` is validated and hashed by the exp_06 launcher; `train_completion` is passed
    through to exp_04 unchanged.
17. **`prepare(..., repo=…)` was added to both HAA wrappers** so a child can record the
    repository it was launched from. In production the default (`REPO` from `__file__`) is
    correct; the end-to-end test needs it because it must record a repository whose HEAD
    equals the working tree.
18. **The end-to-end test commits the working-tree entry points into a throwaway clone.**
    The finalizer's three-way closure check compares working tree, spawn bytes and reviewed
    blobs, so a child's repository must have those bytes at HEAD — true in production, made
    true in the test by cloning and committing. It also means that test exercises the real
    `source_closure`/`closure_record` path, not a stub.
19. **`metrics_<room><tag>.json` gained a `meta` key** (as the prompt asks) and the child
    directory also contains the pinned `metrics_all<tag>.json`; the finalizer hashes only
    the four required artefacts and does not forbid extra files.
20. **Stage-1 heading records cover the union of `--rooms` and `--val-rooms`**, so a
    validation room outside the training rooms is still rotated consistently; the finalizer
    only requires `set(rooms) ⊆ set(heading)`.
21. **Not covered by any test:** the two `main()` functions' GPU sections (the epoch loop and
    the per-room forward pass) — CPU-only this round, as the prompt requires; the plan's
    ladder rung 4 smokes are where they will first execute.
