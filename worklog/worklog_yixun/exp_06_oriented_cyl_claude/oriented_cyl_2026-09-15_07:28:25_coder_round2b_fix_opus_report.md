**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# Round 2b fix cycle — Codex findings 1–4 and nits 5–6

Worktree `/home/yixunhu/codespace/xRIR_code_wt2b`, branch `exp06-round2b`, from the reviewed
tip `d6a4f42` to `3df20b8`. CPU only (`CUDA_VISIBLE_DEVICES=''`), no GPU work, no process
signalled, nothing written under `/home/yixunhu/codespace/xRIR_code` except this report and
nothing touched in `/home/yixunhu/codespace/xRIR_code_wt`.

`git diff main...HEAD --diff-filter=M --name-only` is **empty**: no file that exists on `main`
was modified. The round's own files (`tools/exp06_{eval,eval_launch,haa_finetune,haa_eval}.py`,
`tools/exp06_haa_pipeline.sh`, their tests) and the round-2a `tools/exp06_finalize.py` +
`tests/test_exp06_finalize.py` (the should-fixes below) are the only changes.

## Commits

| SHA | Changed lines | Red | What |
|---|---:|---:|---|
| `a49e3a3` | 58 | **5 failing** | red: the validation-only heading binding (finding 4) |
| `c1f9829` | 24 | — | `_heading_binding` revalidates the training ∪ validation rooms (finding 4) |
| `93b397f` | 106 | **13 failing** | red: the job-spec identity a child is launched under (finding 3) |
| `052bf34` | 105 | — | `--job-spec` recorded by both wrappers; job admission requires it (finding 3) |
| `a1a851d` | 138 | **12 failing** | red: a failed job preparation launches nothing (finding 1, nit 5) |
| `497824b` | 185 | — | `prepare_job` gate, `run_queue` failure accounting, quoted init (finding 1, nit 5) |
| `5189b0d` | 92 | **31 failing** | red: the post-run check of the finished outputs (finding 2) |
| `3df20b8` | 106 | — | exp_06-owned output validation and quarantine (finding 2) |

Red counts are the failures actually observed on each red commit: 5 in
`tests/test_exp06_finalize_haa_consumer.py`; 11 in `tests/test_exp06_haa.py` plus 2 in
`tests/test_exp06_haa_pipeline.py` (a third pipeline mutation failed only because my expected
cause did not match the finalizer's existing wording, so the expectation was re-specified to
that wording rather than counted); 12 in `tests/test_exp06_haa_pipeline.py`; 31 in
`tests/test_exp06_eval_launch.py`.

Every commit is ≤ 200 changed lines (additions + deletions, tests included; maximum 185) and
carries the two trailer lines. Each implementation commit is preceded by its own genuinely
failing test commit; the red counts above are the failures observed on the red commit, each
inspected to be the intended cause (never an import or fixture error).

## Per finding

**1 (blocker) — job preparation is now a gate.** `tools/exp06_haa_pipeline.sh`:
`run_finetune` / `run_zeroshot` call a new `prepare_job`, which checks the status of
`open_job`, `job_spec` and a new `check_spec` explicitly (`|| { say "REFUSED …"; return 1; }`),
because the queue calls these functions through `||`, where bash suppresses `errexit`. The
declaration writer now derives every roll from `tools.exp06_haa_finetune.load_headings` over
**all four rooms** against the selected cache (`HAA_XRIR_ROOT`), so a heading that is refused,
not `confirmatory`, or no longer hashing its cache inputs aborts the job before any child —
and a `k: null` can no longer be written at all. `check_spec` then validates the written file
with the finalizer's own `load_job_spec` (the "equivalent import" the prompt allows; see
*Deviations*). `run_queue` counts each job's failure and prints `QUEUE_FAILED <n>` with a
non-zero exit; `QUEUE_DONE` now means every job succeeded. The job functions moved above the
library guard so faults can be exercised without a queue.
*Regressions* (`tests/test_exp06_haa_pipeline.py`): stubbed failures of `open_job`, `job_spec`
and `check_spec` each reach **zero** child launches and a non-zero status; a private cache
whose `dampened_room` is flat (heading `refused`) stops the job that never trains on that room,
writing no `job_spec.json`, with the intact cache as the positive control launching all nine
children and declaring `{room: 128}`; a failed child stops its own job after exactly one
launch; a finalizer failure on each of two jobs yields `QUEUE_FAILED 2` and no `QUEUE_DONE`.

**2 — the finished outputs are checked against their manifest.** `tools/exp06_eval_launch.py`
gains `validate_outputs` / `quarantine` / `certify_outputs`, run after `execute_run` returns:
the manifest must still hash to `completion.eval_manifest_sha256`; both outputs' `meta` must
carry `model_class`, `registry_sha256`, `checkpoint_role`, `checkpoint_epoch`, `heading`,
`frame` **and** exp_04's `conditions`, `n_samples`, `eval_manifest_sha256`, each present, of
the manifest's type and equal to it; and both files must still hash to what the completion
bound. A failure renames the run `<name>_QUARANTINED_<reason>` (never deletes), writes
`quarantine.json` with the reason and detail, and `main` exits 3.
*Regressions* (`tests/test_exp06_eval_launch.py`): the **24** cases (six fields × corrupt/remove
× both outputs), each with the completion's hash refreshed so the corruption precedes
certification, all quarantine; plus changed output bytes, a changed manifest, a missing output,
a missing `meta`, an intact run passing untouched, and `main` failing on quarantine.

**3 — first admission and launch-time identity.** Every frozen-field mutation now runs against
a job with **no** completion (any prior one is removed first), asserts the specific refusal
cause and requires that no completion appears. `tools/exp06_haa_finetune.py` and
`tools/exp06_haa_eval.py` take `--job-spec <path>` and record `job_spec`, `job_spec_sha256`,
`job_init` and `job_init_sha256` in `args.json` / `provenance.effective_args` before anything
runs; an unusable declaration (missing, unparsable, not a record, no init name, no init sha256)
refuses the child. The finalizer's job branch (`verify_child` → `check_job_spec` →
new `check_job_identity`) requires every child to have recorded this declaration's digest and
initialisation; the identity is compared **after** the per-field checks, so a contradicted
field still names itself. The pipeline passes `--job-spec` to all nine children.
*Regressions*: `init: somebody_elses_pretrain` against otherwise valid children is refused for
identity before any completion exists; a byte-different copy of the same declaration is refused;
a child that recorded no declaration is refused.

**4 — validation-only headings.** `_heading_binding` re-verifies the union of `args["rooms"]`
and the effective validation rooms (`args["val_rooms"]`, else the training rooms) — validation
selects `best.pth`, so those records belong to the input-revalidation contract. A declared
`val_rooms` that is not a nonempty list of rooms is refused; training-room membership stays
checked by `_rooms_and_frame`.
*Regressions* (`tests/test_exp06_finalize_haa_consumer.py`): training `hallway`, validating
`class_room`, with the `class_room` record replaced by invalid text (and its recorded sha256
updated, so the refusal is the revalidation and not the hash) → refused, no completion; the
union is bound when both records are good while `rooms` stays `['hallway']`; an absent, empty
or mistyped `val_rooms` binding is refused.

**5 (nit) — shell parsing.** `init_of` now sets `INIT_BACKBONE` / `INIT_CKPT` instead of
echoing one whitespace-joined string, so a checkpoint path containing spaces survives (test:
`--init /tmp/…/path with spaces/epoch_012.pth` appears whole in the dry run). A job is exactly
`<allowed-init>:<integer seed>`: the seed is everything after the **first** colon, so
`cyl_or:anything:0`, `cyl_or:0:1` and `cyl_or:-1` are refused (new parameters on the existing
refusal test).

**6 (nit) — the record.** The round-2b report counted two oversized commits; there were
**three**: `aa61fc4` (294), `140c3c9` (212) and `4b30735` (227, a test commit — tests count as
code). This report is the correction; the earlier report was left untouched because the
Planner's instruction confines writes in the main tree to this file. All eight commits of this
fix cycle are ≤ 200 changed lines.

## Validation

All commands in the xRIR env (`/home/yixunhu/miniconda3/envs/xRIR/bin/python`) with
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, `PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt2b`,
`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest `-p no:cacheprovider`.

| Command | Outcome |
|---|---|
| `python -m py_compile` on the 9 exp_06 tools + 5 changed test files | passed |
| `bash -n tools/exp06_haa_pipeline.sh`, `bash -n tools/exp06_launch.sh` | passed |
| `git diff --check 12e06fc..HEAD`, `git diff --check` (working tree) | clean |
| `git diff main...HEAD --diff-filter=M --name-only` | empty (no file from `main` modified) |
| `pytest tests/test_exp06_*.py tests/test_provenance.py -q` (13 + 1 files) | **687 passed, 3 skipped** in 687 s |
| `pytest tests/test_exp03_record_tools.py -q` | 31 passed, 1 skipped (pinned files unchanged) |

The 687 is the round-2b 631 plus exactly the 56 tests added here (5 for finding 4, 10 for
finding 3, 11 for finding 1 and nit 5, 30 for finding 2).
Intermediate greens: `tests/test_exp06_finalize_haa_consumer.py` 17 passed after finding 4;
`tests/test_exp06_haa.py` 32 passed and the finalize + pipeline + consumer trio 311 passed after
finding 3; `tests/test_exp06_haa_pipeline.py` 33 passed after finding 1;
`tests/test_exp06_eval_launch.py` 46 passed after finding 2.

## Deviations and discrepancies (nothing silent)

1. **`check-job-spec` subcommand not added.** Finding 1 allows "a `check-job-spec` subcommand
   *or equivalent import*"; the Planner's instruction confines `tools/exp06_finalize.py` edits
   to the job-admission and `_heading_binding` regions, and a subcommand would also touch
   `main()`/`build_parser`. The pipeline therefore validates through
   `python -c "from tools import exp06_finalize; exp06_finalize.load_job_spec(path, expect)"`
   (`CHECK_SPEC_PY`), printing `CHECKSPEC <path> expect=<expect>` in the dry run. The validating
   code is exactly the finalizer's.
2. **The heading verifier called is `tools.exp06_haa_finetune.load_headings`**, which is the
   exp_06 heading consumer: it calls `exp06_heading.read_heading_json(path, room_dir=<cache>/<room>)`
   (hence `verify_heading_inputs`) and adds the decided / `confirmatory` requirements. Using it
   keeps one definition of "a heading a child may bind" for the pipeline and the children.
3. **`check_job_spec` / `verify_child` signatures changed** (both gain the child's `args.json`);
   `load_job_spec` now also returns the declaration's own `job_spec_sha256`. Both are private to
   `tools/exp06_finalize.py` — no other module or test referenced them.
4. **The child's job identity is read from `args.json` at admission**, not added to each child's
   completion evidence, so the edit stays inside the job-admission region and no child
   completion changes shape. `args.json` is already re-validated against
   `provenance.effective_args` by the re-run that precedes the check.
5. **The frozen-field test removes any existing job completion before mutating** *and* asserts
   the specific cause and the absence of a completion afterwards. The module fixture is still
   shared with the positive test (rebuilding nine children per parameter is minutes of CPU), but
   no mutation can now pass on a publication conflict.
6. **`tests/test_exp06_finalize.py` fixtures were reordered** so a job's declaration is written
   before its children (as the pipeline does) and every child records it; the end-to-end fixture
   in `tests/test_exp06_haa_pipeline.py` likewise. These are fixture-region edits to a
   round-2a test file, of the kind the review's disposition item 12 accepted.
7. **A failed preparation leaves the job root and its `launch.pid` behind** (the launcher has
   exited, so the pid is dead and `refuse_live_launch` is unaffected). No `_ABORTED_` rename is
   applied at job level; the SOP's rename is defined for children, and a re-run re-prepares the
   same root. Flagged rather than invented.
8. **Dry runs verify nothing** (no python is executed), so the heading gate is exercised only in
   the real path — where the fault tests cover it.
9. **`validate_outputs` compares against the completion mapping `execute_run` returned**, which
   is the record it wrote; it does not re-read `completion.json` from disk.
10. The GPU loops of the HAA wrappers are still not executed (round-2b's item 21); nothing in
    this cycle changes that.
