**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# Round 3a fix cycle — Codex findings 1–3 and nit 4

Worktree `/home/yixunhu/codespace/xRIR_code_wt2b`, branch `exp06-round2b`, from the reviewed
tip `895f512` to `7f29877`. CPU only (`CUDA_VISIBLE_DEVICES=''`), no GPU work, no process
signalled, nothing written under `/home/yixunhu/codespace/xRIR_code` except this report, and
nothing read or touched in `/home/yixunhu/codespace/xRIR_code_wt` or in
`worklog/worklog_yixun/archive_codex_round2a_superseded_2026-09-15/`.

`git diff main...HEAD --diff-filter=MD --name-only` is **empty**: no file that exists on `main`
was modified or deleted. Only the four files this cycle is allowed to touch changed:
`tools/exp06_mirror_probe.py`, `tools/exp06_haa_pipeline.sh` and their two test modules
(348 insertions, 44 deletions in total).

## Commits

| SHA | Changed lines | Red | What |
|---|---:|---:|---|
| `85c5488` | 130 | **8 failing** | red: G1 must bind the bytes it evaluated (findings 1–2, nit 4) |
| `f108f12` | 151 | — | capture → load from the capture → re-check before publication |
| `8934990` | 51 | **5 failing** | red: a refused preparation must not touch an unowned root (finding 3) |
| `f62c750` | 41 | — | `OWNED_ROOT`: receipt and `_ABORTED_` rename only for a root this attempt created |
| `7f29877` | 19 | (guard) | cover every gate arm loading from the captured bytes |

Changed lines are additions + deletions, tests included; every commit is ≤ 200 (maximum 151)
and carries the two trailer lines. Each implementation commit is preceded by its own genuinely
failing test commit, and each red failure was inspected to be the intended cause (never an
import or fixture error):

- `85c5488` — the 4 parametrised `test_an_input_that_moves_while_the_probe_runs_is_refused`
  cases, `test_a_head_that_moves_while_the_probe_runs_is_refused`,
  `test_the_record_binds_the_panorama_and_a_confirmatory_heading`,
  `test_a_diagnostic_heading_never_binds_the_gate`,
  `test_the_probe_runs_from_the_bytes_it_hashed`.
- `8934990` — the 2 parametrised
  `test_a_failed_preparation_never_renames_a_root_it_did_not_create` cases,
  `test_a_root_whose_pid_file_cannot_be_written_is_left_alone`, and the 2 surviving cases of
  `test_a_failed_preparation_leaves_a_structured_receipt`, which now also assert
  `owned_root is True`.
- `7f29877` is a test-only commit added after its implementation, so it was green on arrival.
  It is a genuine guard, not a tautology: reverting the `states` argument in `full_gate`'s
  `_load` call makes it fail (verified by mutation, then reverted — the working tree was clean
  again before the commit).

## Per finding

### Finding 1 (blocker) — the record now binds the bytes that were evaluated

`build_record` is bracketed by one capture and one re-check.

- `capture_inputs(args)` runs **before any computation**: it reads each of the three
  checkpoints and the heading JSON into memory and hashes **those bytes**
  (`hashlib.sha256`), hashes the five cache files (finding 2), and takes the source closure
  with `git_state` (`_source_record()`).
- Loading is bound to the capture: `state_from_blob(blob) = load_model_state(io.BytesIO(blob))`
  and `_load(..., states)` uses the captured state dict for that path (it raises rather than
  silently re-reading if a path is missing from the capture). Both probes receive `states`,
  so no on-disk rewrite can be evaluated. The heading is parsed from the captured bytes with
  `_validate_heading_record(json.loads(...))`.
- `revalidate_inputs(captured)` runs **immediately before publication** and refuses
  (`SystemExit`, so `write_manifest` is never reached and no record is written) if any
  checkpoint, the heading JSON, any of the five cache files, any file of the captured source
  closure or the git identity differs.
- The published record carries the pre-computation identities: `checkpoints[*].sha256` and
  `heading.sha256` come from the capture, not from a second hash.

Regressions (`tests/test_exp06_mirror_probe.py`): a checkpoint / the heading JSON / a cache
file that changes while the probe runs is refused with no record written; the source closure's
commit moving mid-run is refused; `test_the_probe_runs_from_the_bytes_it_hashed` corrupts every
checkpoint file after the capture and shows the probe still runs from the captured bytes and
that `revalidate_inputs` then refuses; `test_every_gate_arm_loads_the_captured_bytes` does the
same for all four arms of the full gate; the unchanged-input path still writes the record with
the captured hashes.

Because the heading JSON is parsed from the captured bytes and the full gate runs after the
legacy reproduction, a heading replaced between the two parts is caught by the final re-check
(the record's `k` is the one that was parsed and hashed).

### Finding 2 — the panorama is bound

`PROBE_CACHE_FILES = ('depth.npy', 'meta.json', 'rirs.npy', 'speaker_xyz.npy', 'xyzs.npy')` is
hashed at capture and re-checked before publication, and the record gained
`cache = {'room_dir': ..., 'sha256': {...}}`. The heading record's four input hashes are now
compared against these captured cache digests instead of a second disk read, so heading
verification and the probe binding rest on **one** identity of the cache. The heading record
schema is unchanged. Regression: changing only `depth.npy` — with the heading JSON and its four
input hashes still matching — is refused.

### Finding 3 — failure cleanup never touches an unowned root

`open_job` now creates the job root with an **exclusive** `mkdir` and sets `OWNED_ROOT=1` only
then; a root that already exists is still adopted (a resumed queue skips completed children),
but with `OWNED_ROOT=0`. `prepare_job` resets `OWNED_ROOT=0` before calling `open_job`, so any
path that does not explicitly grant ownership is treated as unowned. `prepare_failed` therefore:

- **owned** → receipt `<root>/preparation_failure.json` and the `_ABORTED_prepare_<reason>`
  rename, exactly as before (`owned_root: true`, `aborted_dir` set);
- **unowned** (pre-existing root, or a failed acquisition such as an unwritable `launch.pid`)
  → writes `<root>_PREPARE_FAILED_<UTC>.json` **beside** the root, never inside it, never
  renames anything, and says `UNOWNED <root>` (`owned_root: false`, `aborted_dir: null`).

Both branches still `return 1`, so non-zero propagation and the "no child is launched" gate are
unchanged. Regressions: a populated pre-existing root (refused at `open_job`, and refused at
`job_spec` after an adopted acquisition) keeps its contents and its name, with the failure file
beside it, exit non-zero and zero launches; a root whose `launch.pid` cannot be written (it is a
directory) is left byte-for-byte alone with the same outcome; the owned-root receipt and rename
are unchanged, including the second-refusal collision suffix.

The `preparation_failure.json` schema went to `schema_version: 2` (`owned_root` added,
`aborted_dir` nullable).

### Nit 4 — confirmatory heading admission

`_verified_heading` now refuses any record whose `admissibility` is not `confirmatory` (in
addition to the existing decided-roll and cache checks), and the emitted heading binding carries
`admissibility`. Regression: a record written from a dirty-tree closure (a genuinely
`diagnostic` record, accepted by `write_heading_json`) is refused and no record is written.

### Nit 5 — commit size

Every commit of this cycle is ≤ 200 changed lines including tests (130 / 151 / 51 / 41 / 19).

## Validation

All commands in the prescribed environment (`/home/yixunhu/miniconda3/envs/xRIR/bin/python`,
`PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt2b`,
`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=4`, `PYTHONDONTWRITEBYTECODE=1`, `NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`,
`TMPDIR` in the session scratchpad, pytest with `-p no:cacheprovider`).

| Command | Outcome |
|---|---|
| `python -m py_compile tools/exp06_mirror_probe.py tests/test_exp06_mirror_probe.py tests/test_exp06_haa_pipeline.py` | passed |
| `bash -n tools/exp06_haa_pipeline.sh` | passed |
| `git diff --check 895f512..HEAD` | clean |
| `git diff main...HEAD --diff-filter=MD --name-only` | empty |
| `python -m pytest tests/test_exp06_mirror_probe.py -q` | 54 passed, 2 skipped (baseline 46 + 2) |
| `python -m pytest tests/test_exp06_haa_pipeline.py -q` | 39 passed (baseline 37) |
| `python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q` (at `f62c750`) | **833 passed, 6 skipped** in 880 s |
| `python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q` (at `7f29877`) | **834 passed, 6 skipped** in 893 s |

The 2 skips are the two CUDA parity tests (alignment and composed forward); they remain
unexecuted under the CPU-only mandate, as disclosed in round 3a. `tests/test_provenance.py`
stayed green throughout.

## Choices, deviations and discrepancies

1. **Source-closure re-check re-hashes instead of re-importing.** `revalidate_inputs` re-hashes
   every file the captured closure names and re-reads `git_state`, rather than running a second
   hermetic `source_closure` import (seconds per call, on every CLI run and every record test).
   It binds exactly what `_source_record` records; a new import edge still changes the importing
   file's bytes, and any tracked change or HEAD move flips `dirty`/`diff_sha256`/`HEAD`. Noted
   as the cheaper — not weaker — option; the stricter alternative is a one-line change.
2. **Strictness this buys.** A repo edit (outside gitignored paths) by a concurrent agent while
   G1 runs now refuses publication. That is the intended reading of finding 1 — the closure is
   an input — but it means a confirmatory G1 run needs a quiescent tree. `__pycache__/` and
   `ckpt/` are gitignored, so bytecode and the run's own output cannot cause a spurious refusal.
3. **An existing job root is still adopted, not refused.** Only *ownership* is recorded. Making
   `open_job` refuse a pre-existing root would break the resume path that `child()`'s
   `completion.json` SKIP exists for; the fail-closed part is that an unowned root is never
   written into or renamed.
4. **`_validate_heading_record` is imported privately** from `tools/exp06_heading.py` (as
   `_closure_digest` already was) so the heading can be parsed from the captured bytes. No
   heading-tool file was modified.
5. **Refusals raise `SystemExit('refusing: ...')`** (exit 1), consistent with the module's
   existing heading refusal, rather than the unused `EXIT_INPUT = 2`. No record is written in
   any refusal path.
6. **Record schema bumped 1 → 2** (`cache` block, `heading.admissibility`). No consumer of
   `gate_g1.json` exists yet; round 3b's producer-admission wiring should read the new fields.
7. **Two existing tests were edited** (authorised regression edits): the `open_job` case moved
   out of `test_a_failed_preparation_leaves_a_structured_receipt` into the new unowned-root
   test, and the surviving cases now assert `owned_root is True`.
8. **Coordinator's pid guidance (mid-cycle).** The finding-3 tests launch no child and use no
   `--child-pid` / `child_exit.json` / `child.pid` value; the three `os.getpid()` uses in
   `tests/test_exp06_haa_pipeline.py` (lines 407, 492, 553) are pre-existing and untouched here,
   and belong to the separate `exp06-window` integration fix.
9. **Still open from round 3a:** both CUDA parity tests need a GPU run before production G1, and
   the real-checkpoint legacy reproduction (`EXP06_REAL_PROBE=1`) was not re-run this cycle — no
   numerical code changed, only how the checkpoint bytes reach `load_state_dict`.
