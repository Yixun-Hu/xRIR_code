**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-16

Pre-launch round FIX for exp_06 (the single should-fix of
`oriented_cyl_codex_code_prelaunch_review.md`). Worktree
`/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-prelaunch`, from the reviewed tip
`8bc7cf8`. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), Python
`/home/yixunhu/miniconda3/envs/xRIR/bin/python` (3.8), `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest with `-p no:cacheprovider`. No GPU work, no
process signalled that this round did not start (no `pkill`/`pgrep` at all), no write under
`ckpt/` outside pytest temporary directories, and the main tree untouched except this report
(the confirmatory pretraining on GPU 1 was never approached).

## Commits (2; `8bc7cf8..c867e24`)

| # | SHA | Subject | Changed lines (files) |
|---|---|---|---|
| 1 | `9381b6a` | red tests for allow-listed artefacts linked out of the smoke tree | 40 + / 0 − = **40** (1: `tests/test_exp06_finalize.py`) |
| 2 | `c867e24` | a registered HAA artefact is a regular file of the smoke tree | 36 + / 5 − = **41** (1: `tools/exp06_finalize.py`) |

Both well under 200 changed lines; both carry the two trailers
(`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`,
`Claude-Session: https://claude.ai/code/session_01NT7CPAUaJ71xgzoS8JGDVw`), verified 2/2. The
red commit precedes the implementation commit and was genuinely failing at its own tree state
(see the red counts below). No amend, rebase, reset or push. `git diff --name-status
8bc7cf8..HEAD` lists exactly the two permitted files; `git diff --name-only main...HEAD` is
unchanged from the reviewed round (the same five files).

## The finding and the fix

Codex finding 1: `haa_smoke_artifacts()` checked artefact **names** only, and `artifacts()`
used `is_file()` and `provenance.sha256_file()`, both of which follow symlinks. An
allow-listed name pointing outside `ckpt/exp06/_smoke` was therefore hashed and finalised with
`passed: true`, so a passing training diagnostic could have handed the eval rung an external
checkpoint.

**Red (commit `9381b6a`), 5 failures.** Four parametrised cases
(`test_an_allow_listed_artefact_may_not_link_outside_the_smoke_tree`:
`haa_smoke_train`/`best.pth`, `haa_smoke_train`/`args.json`,
`haa_smoke_eval`/`per_sample_hallway.json`, `haa_smoke_eval`/`provenance.json`) reproduced the
reviewer's result exactly — **`Failed: DID NOT RAISE <class 'ValueError'>`**, i.e. finalization
accepted the external symlink — and one case
(`test_a_symlinked_artefact_directory_is_refused`, the `run/` directory itself symlinked out)
failed on the wrong cause (`args.json records the save_dir …`, no mention of a symlink). Each
case also asserts that no `completion.json` was written, and the file-symlink cases then
restore the same bytes as a regular file and assert the diagnostic finalises with
`passed: true`, so the accept path is pinned too.

**Implementation (commit `c867e24`).** New helper
`confined(path, root, label)` in `tools/exp06_finalize.py`: `root` must be an ancestor, then
every component below the already resolved `root` is checked with `Path.is_symlink()`
(`lstat`, so a broken link refuses as well) and the result must still satisfy
`path.resolve() == path`. `artifacts(run_dir, names, root=None)` gained the optional `root`;
when given, every registered artefact is confined before `is_file()` and hashing.
`haa_smoke_artifacts()` now confines the artefact directory (`<attempt>/run`), confines
`args.json` **before** reading it (so a crafted external `args.json` cannot even supply the
`save_dir` comparison), and passes `root=<resolved attempt dir>` to `artifacts()`, which covers
every allow-listed name including `args.json`, `provenance.json` and `metrics_all<tag>.json`.
Refusals are named (`the artefact … is a symlink, not a real entry of …`) and, as before, no
completion is published — `finalize()` writes `completion.json` only after the evidence
functions return.

Scope note (no silent deviation): `root` defaults to `None`, so `full`, `haa_train` and
`haa_eval` keep their previous behaviour. The prompt scoped the fix to the two HAA diagnostic
run types, whose contract is the location rule; the other run types are not required to live in
`_smoke` and bind their artefacts through the argument/source gates instead. The helper is
available should the Planner want the same confinement there.

## Validation

| Check | Result |
|---|---|
| `python -m py_compile tools/exp06_finalize.py tests/test_exp06_finalize.py` | passed |
| `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh` | passed (both unchanged this round) |
| `git diff --check 8bc7cf8..HEAD` | clean |
| Red run (`-k "link_outside or symlinked_artefact"`, at `9381b6a`'s tree) | **5 failed, 302 deselected** |
| `pytest tests/test_exp06_finalize.py -q` after the fix | **307 passed** (was 302 before this round) |
| `pytest tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider -rf` | **1 failed, 1252 passed, 7 skipped** in 30:05 — the failure is the known approvals-drift guard, nothing else; no xfail |

The single failure is
`tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`,
the same one the reviewed round reported (1247 passed then, 1252 now = +5 new tests). It cannot
be made green from inside this round: `oriented_cyl_results_assets/approved_digests.json` is not
a permitted file and re-filling it here would be the self-approval the fail-closed design
forbids. The seven skips are the pre-existing "skip if absent" cases.

## Digests at the new tip `c867e24`

Recomputed with `tools.exp06_profiles.compute_code_digests(repo, HEAD)` (content-determined, so
they survive the merge as long as the bytes do). Four keys still differ from the committed
approvals — the same four the reviewer named — but two of the four values **moved** with this
fix:

| Key | Class | Approved now | At `8bc7cf8` (reviewed) | At `c867e24` |
|---|---|---|---|---|
| `finalize` | TRAINING_KEYS | `3da0479ab2629ebca4ff2cfafe193481bd13b6c2f1b7e1addb8ba9a3926a060c` | `9e3a000ba089aa0f4cb40372d07f5e4b7a4e3b851b4884c95dbf5a04aa470b80` | **`5ef318e8da734e6076f89237014d1e2366078692b1da83ef93b304d1dc609399`** |
| `summarize_haa` | non-training | `774c6868691d2521e665caa8bd9ee88ffb389ba81117247a66e144c894a3a5c1` | `c2b56e66f6aa32c7732080219437aa6fb0f71c78298be304ff7fdc9d365f228a` | **`2ad63b6dea015f996f1568a4e6e8d1513d356c83246b96dd19cc98c3fa4cefb4`** (it imports the finalizer) |
| `smoke` | TRAINING_KEYS | `eebc949a2ebfa5bf63ca4cf7603c5bf60256fc3ca7d10acbe7743daee813abaa` | `00e3b91b8289cc0f1fda8c2ab2ce3b151ca897f4a4c746f93863e3d338091d00` | `00e3b91b8289cc0f1fda8c2ab2ce3b151ca897f4a4c746f93863e3d338091d00` (unchanged by this fix) |
| `launch_sh` | TRAINING_KEYS | `157e2435ba717eb97e9046cf0120ddbaa0ea03860daaf55e303c2396164eb3a8` | `c64b60df4632b83617c1343f0d0a56c6ac7cbab24db95719a2aa8734f3977773` | `c64b60df4632b83617c1343f0d0a56c6ac7cbab24db95719a2aa8734f3977773` (unchanged by this fix) |

Unchanged and matching the approvals: `trainer`, `recipe`, `encoder`, `factory`, `profiles`
(TRAINING_KEYS) and `bootstrap`, `compare`, `eval`, `eval_launch`, `evaluator_exp03`,
`haa_eval`, `haa_finetune`, `haa_pipeline_sh`, `heading`, `mirror_probe`, `probe_align`.

A6 still applies: merge only after the running pretraining has completed and finalised with its
original code; the approvals refill (the four keys above, `finalize` and `summarize_haa` at
their **new** values) and the integrative review remain the Planner's steps before any HAA
execution.
