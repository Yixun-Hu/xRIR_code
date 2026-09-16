**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 round 3b, fix cycle 3 — the five closure-blocking findings of close review 2

Worktree `/home/yixunhu/codespace/xRIR_code_wt2b`, branch `exp06-round2b`, from `07001a0`
to `6ac832c`. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`,
`PYTHONDONTWRITEBYTECODE=1`, pytest with `-p no:cacheprovider`), no GPU work, no process
signalled, no `pkill`/`pgrep`. Nothing was read or written under
`/home/yixunhu/codespace/xRIR_code_wt`, nothing was written under `ckpt/`, and the only
file written in the main tree is this report. `git diff 07001a0..HEAD --name-only` lists
exactly the four permitted files: `tools/exp06_summarize_haa.py`, `tools/exp06_compare.py`,
`tests/test_exp06_summarize_haa.py`, `tests/test_exp06_compare.py`. No amend, rebase,
reset or push.

## Commits (11: 5 red, 5 implementation, 1 test-only regression; both trailer lines on all)

| commit | subject | changed lines (incl. tests) |
|---|---|---|
| `c0d4753` | red -- the owner pid and its digest come from one read (finding 6) | 43 |
| `efe66ea` | parse and hash one launch.pid snapshot (finding 6) | 13 |
| `597589a` | red -- the exp_05 tier is read from the bytes its digest identifies (finding 5) | 24 |
| `0718c62` | parse and hash one exp_05 args snapshot (finding 5) | 18 |
| `9014b8a` | red -- a record that changed after the job certified it is refused (finding 2b) | 30 |
| `ee8d1c9` | bind the parent's digest before delegating, and enforce it after (finding 2b) | 33 |
| `ed98fe4` | red -- every child-provenance dependency is retained and revalidated (finding 2a) | 57 |
| `f668747` | retain every validated child-provenance dependency (finding 2a) | 41 |
| `cf9c38b` | red -- primary admission registers the effective validation rooms (finding 3) | 50 |
| `93c1a75` | register the effective validation rooms of each stage (finding 3) | 13 |
| `6ac832c` | regression -- a dependency contradicting an earlier binding is refused (finding 2a) | 16 |

Maximum commit size 57 changed lines.

## Red counts (failing tests at each red commit, before its implementation)

| cycle | red commit | failing | how it failed |
|---|---|---|---|
| finding 6 | `c0d4753` | 1 | the published owner digest was the substituted marker's, not the parsed pid's bytes |
| finding 5 | `597589a` | 1 | `DID NOT RAISE`: the substituted M-tier parse was admitted against legacy bytes |
| finding 2b | `9014b8a` | 2 | `DID NOT RAISE` for both the child completion and the job spec |
| finding 2a | `ed98fe4` | 4 | the dependency paths were absent from the input map (`None`), and publication accepted the mutations |
| finding 3 | `cf9c38b` | 2 | no `val_rooms` deviation was recorded, and the primary refusal named none |

Total 10 genuinely failing tests. `6ac832c` is test-only; it was verified red by checking
`ee8d1c9`'s `tools/exp06_summarize_haa.py` into the working tree (`DID NOT RAISE`), then
the module was restored with `git checkout HEAD --` before the commit.

## Per-finding fix and regression

**2a — every validated child-provenance dependency is retained and revalidated.**
New `child_dependencies()` re-reads each child's `provenance.json` pinned to the digest
`finalizer.verify_child` just re-hashed, and binds, with the digests that validation
confirmed: every `data_identity`/`train_data_identity` inventory entry under the record's
own `data_root` (the five HAA cache files per room — `meta.json`, `rirs.npy`, `xyzs.npy`,
`speaker_xyz.npy`, `depth.npy`), every file of every recorded source closure resolved
against the repo (`tools/exp06_haa_finetune.py`, `tools/exp06_haa_eval.py`,
`sim_to_real/finetune_haa.py` and the rest of both closures), and every mutable input.
`bind_child` calls it for every child, so the shared map `write_outputs` revalidates
immediately before publication now covers them; a digest that is not 64 hex characters is
refused rather than bound, and a second, different digest for one path is the existing
`contradictory bindings` refusal. Regressions: the review's exact probe, parametrized over
`hallway/rirs.npy`, `hallway/depth.npy` and the child's entry-point source — mutate after
admission, and `write_outputs` refuses with `input changed during analysis`, leaving
neither output behind; a retention test that walks both a training and an evaluation
child's provenance and requires every inventory entry, every closure file and every
mutable input to be in the map with its recorded digest; and the dependency-level
contradiction regression (`6ac832c`).

**2b — the parent's certified digest is bound before delegation and enforced after.**
`verify_job` binds `record['children'][name]` for a child completion, and
`record['job_spec']['sha256']` for the job spec, *before* `verify_child` /
`load_job_spec` runs. `_read_json` gained an `expected` parameter: the bytes it parses
must hash to the certified digest, so the completion re-read after validation can never
describe different bytes, and the spec's own `job_spec_sha256` (the loader's hash of what
it read) must equal the certified one. No later read replaces a validated identity.
Regression: a parametrized race that rewrites the file the instant its content is first
read — the child completion with the review's own mutation (its `log` binding) and the
spec re-serialised — both refused.

**3 — primary admission registers the effective validation rooms.**
`child_recipe` now compares `args.val_rooms or args.rooms` (exactly the wrapper's default)
with the stage's registered training rooms: stage 1 = `class_room`, `complex_room`,
`hallway`; stage 2 = its own room. A deviation is refused in primary mode and reported by
`--sensitivity`, like every other recipe deviation. Regressions: the unit table (default
and explicitly-equal `val_rooms` admitted; `['hallway']`, all four rooms, or another room
deviating, at 1000/10 and 200/2) and a finalized-job regression that rewrites stage 1's
`args.json` and `provenance.effective_args` to `val_rooms: ['hallway']`, re-certifies the
child and the job through the real finalizer, and requires the primary refusal to name the
validation rooms while sensitivity lists the deviation.

**5 — one exp_05 args snapshot.** `check_exp05_tier` reads the checkpoint-adjacent
`args.json` once (`raw = path.read_bytes()`), derives both the record and the digest from
that buffer, reconciles that digest with the binding the run's `train_args` mutable input
already made (`bind(path, parsed)`), with `args_json_sha256` and with the `train_args`
declaration, and publishes it as the tier block's `args_json` identity. Regression: the
review's transient substitution — a patched reader serving M-tier arguments while the file
on disk is the legacy record — is refused for `legacy_M` instead of admitted.

**6 — one `launch.pid` snapshot.** `job_owner` reads the marker once, parses the pid from
that buffer and hashes that same buffer, then compares the pid with `owner_pid` and retains
that digest. Regression: the race probe rewrites the marker at its first read; the admitted
owner pid and the retained digest are both the original bytes' (and the changed file is
then caught by the publication recheck).

## Validation

| check | result |
|---|---|
| `python -m py_compile` of the four changed files | OK |
| `git diff --check 07001a0..HEAD` | clean |
| `git diff --name-only 07001a0..HEAD` | exactly the four permitted files |
| `git status --porcelain` (worktree) | clean |
| `pytest tests/test_exp06_compare.py` | 78 passed |
| `pytest tests/test_exp06_summarize_haa.py` (after each summariser cycle) | 70 → 74 passed |
| full suite before `6ac832c` | **1217 passed, 8 skipped, 0 failed, 0 xfail** (31 min 31 s) |
| **final** `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 pytest tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider --tb=short` | **1218 passed, 8 skipped, 0 failed, 0 xfail** (28 min 21 s) |

(1207 before this cycle + 11 new tests = 1218.)

## Discrepancies and deviations

1. **The finding-3 finalized-job regression uses the nine-child fixture, not a real
   1000/10 + 200/2 job** (a real one is a ~31-hour training). Stage 1's `val_rooms` is set
   to `['hallway']` and both the child and the job are re-certified through the real
   `tools.exp06_finalize`, so the refusal is measured on a genuinely finalized job whose
   recipe otherwise deviates only in the fixture's shortened budget; the registered
   1000/10 + 200/2 arguments are covered exactly at the `child_recipe` level. Every byte
   the test rewrites (stage 1's `args.json`, `provenance.json`, `completion.json` and the
   job `completion.json`) is restored in a `finally`.
2. **`6ac832c` is a test-only commit** with no implementation after it; its red state was
   established by reverting the module in the working tree rather than by a red commit,
   and the commit message records that.
3. **A window remains inside the delegated finalizer readers, and this round may not close
   it.** `finalizer.load_job_spec` parses the spec and then hashes it itself; a
   substitution restored between those two operations inside that function would still
   parse content the digest does not describe. What this round guarantees is that the
   digest the loader produced equals the one the parent job certified and the one bound
   for publication. Closing the inner window needs an edit in `tools/exp06_finalize.py`,
   which is a pinned/training-path module this round must not touch — the Planner may want
   to carry it as an open item for the merged branch.
4. No change to the sensitivity semantics the review accepted: primary production remains
   "run without `--sensitivity`", and a sensitivity run still labels its banner, its
   deviations and both decision verdicts.
