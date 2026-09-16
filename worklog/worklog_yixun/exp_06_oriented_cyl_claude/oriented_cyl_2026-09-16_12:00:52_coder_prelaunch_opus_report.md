**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-16

Pre-launch round for exp_06 (plan §10a amendments A4 and A5). Worktree
`/home/yixunhu/codespace/xRIR_code_wt`, new branch `exp06-prelaunch` from main `515136f`.
CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), Python
`/home/yixunhu/miniconda3/envs/xRIR/bin/python` (3.8.20), `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest with `-p no:cacheprovider`. No GPU work, no
process signalled that this round did not start, no write under `ckpt/` outside pytest
temporary directories, `/home/yixunhu/codespace/xRIR_code_wt2b` never read, and the main
tree untouched except this report.

## Commits (11; `515136f..8bc7cf8`)

| # | SHA | Subject | Changed lines (files) |
|---|---|---|---|
| 1 | `c61e86d` | red tests for the A5 rung-4 smoke budgets | 70 + / 4 − = **74** (1) |
| 2 | `040f2b1` | red tests for the finalised HAA diagnostics (A4) | 149 + / 5 − = **154** (1) |
| 3 | `0e4b6b2` | A5 rung-4 smoke budgets as environment parameters | 58 + / 15 − = **73** (3) |
| 4 | `cd1fc17` | finalised HAA diagnostics, run types `haa_smoke_train`/`haa_smoke_eval` (A4) | 108 + / 7 − = **115** (2) |
| 5 | `c956a3a` | red tests for the smoke mode's finalised HAA rungs | 72 + / 3 − = **75** (1) |
| 6 | `343d7d6` | the smoke mode runs and finalises the two HAA rungs (A4) | 34 + / 1 − = **35** (1) |
| 7 | `a6cf010` | red regression for the job-spec parse/hash window | 49 + / 0 − = **49** (1) |
| 8 | `1027101` | `load_job_spec` reads the job spec once (round-3b finding 6) | 20 + / 4 − = **24** (1) |
| 9 | `7954508` | regression for the passed gate between the two HAA rungs (test-only) | 22 + / 0 − = **22** (1) |
| 10 | `2664512` | red test for a job spec that changes during its validation | 41 + / 19 − = **60** (1) |
| 11 | `8bc7cf8` | a job spec that changed during its validation is stale, not certifiable | 6 + / 1 − = **7** (1) |

Maximum 154 changed lines per commit (< 200). Every commit carries both trailers
(`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`,
`Claude-Session: https://claude.ai/code/session_01NT7CPAUaJ71xgzoS8JGDVw`), verified 11/11.
No amend, rebase, reset or push. Author identity `Claude Opus 5 <noreply@anthropic.com>`.

**Files changed (exactly the five the prompt permits, nothing else):**
`tools/exp06_launch.sh`, `tools/exp06_finalize.py`, `tools/exp06_smoke.py`,
`tests/test_exp06_launch.py`, `tests/test_exp06_finalize.py`.
`tools/exp06_haa_pipeline.sh` is byte-identical to main; `git diff 515136f --name-only`
lists only those five paths.

**Commit ordering disclosure.** Commit 3 (cycle 1's implementation) landed *after* commit 2
(cycle 2a's red tests): the implementation was written and verified green before commit 2
but was not staged until after it. Every implementation commit is still preceded by its own
red commit, and each red commit was genuinely failing at the tree state where it was made.

## Cycle 1 — A5 budgets (`c61e86d` red → `0e4b6b2`)

`tools/exp06_launch.sh` gained `SMOKE_ALARM_S="${EXP06_SMOKE_ALARM_S:-300}"` and
`SMOKE_MAX_GB="${EXP06_SMOKE_MAX_GB:-6}"`; both rung-4 smoke argv now interpolate them, so
the values appear in the dry-run output and in every receipt (`alarm_seconds`, `max_gb` were
already recorded). The probe's 2400 s / 46 GB budget is unchanged, as required.
`preflight()` passes `--min-free-gb "$SMOKE_MAX_GB"` for mode `smoke` only.

`tools/exp06_finalize.py` gained `gpu_free_gib(gpu)` (`nvidia-smi --query-gpu=memory.free
--format=csv,noheader,nounits -i <g>`; an unreadable, non-numeric, empty or multi-line reply
is a named refusal) and a `min_free_gb` parameter on `preflight()`/`preflight_main`, which
refuses when the card has less free than the budget and records `min_free_gb` and
`gpu_free_gib` in the preflight record. The free-memory census is separate from the
compute-app census `probe`/`full` use.

- **Red:** 6 failed / 32 deselected in `tests/test_exp06_launch.py`.
- **Green:** 38 passed (`tests/test_exp06_launch.py`), 286 passed
  (`tests/test_exp06_finalize.py`).
- **Deviation disclosed in the commit message:** the red CLI case asserted that a
  *subprocess* preflight would be admitted, which no subprocess can reach (the approvals
  digests are monkeypatched in-process only). It now drives `preflight_main` directly, which
  exercises the same argparse wiring and both outcomes.

## Cycle 2 — A4 finalised HAA diagnostics (`040f2b1` red → `cd1fc17`; `c956a3a` red → `343d7d6`)

Split into two red/implementation pairs to stay under the line budget.

**Finalizer (`cd1fc17`).** New diagnostic run types `haa_smoke_train` / `haa_smoke_eval`,
added to `RUN_TYPES`, `DIAGNOSTIC` and `ENTRY_MODULES` (entry module
`tools.exp06_smoke`, since the runner is what records the provenance). They keep the whole
smoke contract — runner receipt with identity, budgets, window and outcome; `provenance.json`
bound three ways; approvals and orchestration closures; closed log with the
`EXP06_CHILD_EXIT` marker and `child_exit.json`; never an arm (`diagnostic: true`,
`admissible_arm: false`) — and replace the `--no-save` argv guard (now gated by
`NO_SAVE_REQUIRED = ('smoke', 'probe')`) with:

- `smoke_run_dir()` — the run directory must resolve strictly inside `ckpt/exp06/_smoke`;
- `haa_smoke_artifacts()` — the wrapper's own `args.json` must name the artefact directory
  (`<run dir>/run`) as its `save_dir`, every registered file must exist and is hashed into
  the completion, and anything else in that directory is refused as an unregistered write;
- a receipt whose `entry` is the wrapper the run type registers
  (`exp06_haa_finetune` / `exp06_haa_eval`).

`tools/exp06_smoke.py` gained the two run types in `RUN_TYPES` so the runner can write the
matching receipt and `provenance.json` (the only change it needed).

`passed_main` (`python tools/exp06_finalize.py passed --run-dir <dir> --run-type <type>`)
exits 0 only when that directory's own `completion.json` records the run type and
`passed: true`.

**Launcher (`343d7d6`).** `smoke` mode now runs §9 (c) and (d) through the same `diagnostic`
helper as the other rungs — exclusive run directory, `launch.pid`, drained sink, end marker,
`child_exit.json`, finalisation, `_ABORTED_<reason>` on failure — with the exact §9 argv, the
CPU fixture built first, and `require_passed` between them.

- **Red:** 13 failed / 11 passed (finalizer selection); 5 failed / 36 passed (launcher).
- **Green:** 24 passed (finalizer selection); 81 passed
  (`test_exp06_smoke.py` + `test_exp06_finalize_haa_consumer.py` + `test_exp06_launch.py`);
  342 passed (`test_exp06_finalize.py` + `test_exp06_haa_pipeline.py`); 41 passed
  (`test_exp06_launch.py` after the launcher change).

### Two spec decisions taken where the prompt left a choice (both fail-closed)

1. **`metrics_all<tag>.json` is in the evaluation allow-list.** The prompt's literal list is
   `metrics_<room>.json`, `per_sample_<room>.json`, `args.json`, `provenance.json`, but
   `tools/exp06_haa_eval.py:242` writes `metrics_all<tag>.json` unconditionally beside the
   per-room pair. Without it in the list the contract could never be satisfied by a real
   evaluation smoke. The per-room names are derived from the wrapper's own `rooms` and `tag`
   in `args.json`, never guessed. The training list is exactly the six files
   `tools/exp06_haa_finetune.py` writes (`provenance.json`, `args.json`, `history.jsonl`,
   `summary.json`, `best.pth`, `last.pth`) — verified by reading every write in both wrappers.
2. **A failed HAA diagnostic is refused, not recorded as `passed: false`.** The prompt lists
   "receipt outcome not ok" among the refusals, and the next rung consumes this one's
   `best.pth`, so `haa_smoke_*` finalisation requires the diagnostic to have succeeded
   (receipt `outcome == 'ok'`, receipt `exit_status == 0`, child exit 0) and always records
   `passed: true`. This differs from `smoke`/`probe`, which still record a failed run with
   `passed: false`. Operationally the launcher then aborts the rung as
   `_ABORTED_finalize_refused` (exit 2) instead of `_ABORTED_child_failed_<n>`; the receipt,
   provenance and log are preserved under the renamed directory, and no completion exists —
   which is exactly what `require_passed` needs to see. If the Planner prefers the sibling
   behaviour (a `passed: false` completion for a failed HAA smoke), the change is one
   `_require` in `diagnostic_evidence`.

Two further notes for the reviewer:

- The `smoke`/`probe` run types are deliberately **not** subjected to the `_smoke` location
  rule: the fit/timing probe legitimately lives at `<attempt-root>/probe_<UTC>`.
- The HAA rungs' live child lifecycle is covered by the existing `diagnostic` harness tests
  (shared code, exercised with a stub `PYTHON`), not by running the real wrappers, which
  need a GPU and the HAA cache. Commit 9 adds the one piece of new launcher logic that the
  dry-run goldens cannot reach.

## Cycle 3 — `load_job_spec` single snapshot (`a6cf010` red → `1027101`)

`load_job_spec` reads the file once with `Path.read_bytes()`, decodes and parses that buffer,
and binds `hashlib.sha256(data).hexdigest()` as `job_spec_sha256`; `haa_job_evidence` records
that digest instead of re-hashing the file. Unreadable bytes and undecodable or non-object
JSON are named refusals (`unreadable job spec: …`, `job spec is not a JSON object`), so the
existing `not_json` refusal case is unchanged. No other behaviour change.

The red case reproduces the Codex finding exactly: with a spec whose bytes are substituted
after the first read, the pre-fix loader bound `c4c73e22…` (the substituted file) while
having validated `171d4b1a…` (the parsed bytes). The second case asserted that no later
consumer re-hashes the spec, guarding `provenance.sha256_file` against the spec path across a
whole job finalisation; cycle 4 replaces it (see below).

`tools/exp06_summarize_haa.py:615-619` already compares the loader's returned digest with the
one the job certified; that comment ("the loader hashes what it read") only becomes true with
this fix, so the summariser inherits the guarantee. It is not in this round's file list and
was not edited.

- **Red:** 2 failed / 299 deselected.
- **Green:** 13 passed (both regressions plus the 11 `test_the_job_spec_is_required_and_validated`
  cases).

## Cycle 4 — the spec must still hold the bytes it was validated from (`2664512` red → `8bc7cf8`)

Reading the spec once removed a refusal a **reviewed** test depends on. With the parse and
the digest taken from one buffer, a spec substituted immediately after that read was
certified from bytes the file no longer held, and
`tests/test_exp06_summarize_haa.py::test_a_record_that_changed_after_the_job_certified_it_is_refused[job_spec]`
(close review 2, finding 2b — a module this round may not edit) went red. That test races a
substitution into the first read of the file and requires `verify_job` to refuse.

`load_job_spec` therefore re-hashes the file once validation of the snapshot is complete and
refuses when the two differ — exactly the "stale log" rule `finalize` already applies to the
log it validated (`tools/exp06_finalize.py`). The single-snapshot guarantee is untouched: the
declaration and its bound digest still come from one buffer, and the second read can only
refuse, never supply evidence.

`test_no_consumer_rehashes_the_job_spec` (added in cycle 3) was **replaced**, not kept: its
premise — nothing ever hashes the spec again — is contradicted by the staleness check. Its
requirement is now pinned by `test_the_job_records_the_digest_the_loader_bound`, which changes
the file the instant the loader returns (after its own staleness check has passed), so
anything that re-hashed the file on the way to the completion would publish bytes the
validation never saw.

- **Red:** 1 failed / 2 passed (the substituted parametrisation).
- **Green:** 14 passed (the two regressions plus the 11 job-spec validation cases and the
  no-substitution parametrisation); 4 passed in the summariser's snapshot-identity cases,
  including the reviewed refusal that cycle 3 had broken.
- **Disclosure:** this is a behavioural change on top of cycle 3 that the prompt did not ask
  for. It was taken rather than leaving a reviewed refusal broken, and rather than editing
  `tools/exp06_summarize_haa.py` or its tests, which this round may not touch.

## Validation

| Check | Result |
|---|---|
| `python -m py_compile tools/exp06_finalize.py tools/exp06_smoke.py` | passed |
| `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh` | passed |
| `git diff --check`, `git diff main --check` | clean |
| `git diff 515136f --name-only` | the five permitted files only |
| `tools/exp06_haa_pipeline.sh` vs `main` | byte-identical |
| Full smoke dry-run inspected end to end | 6 `exp06_smoke.py` rungs, budgets 300 s / 6 GB, `--min-free-gb 6`, fixture before the HAA rungs, `passed` gate between them |
| `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider -rf` | **1 failed, 1247 passed, 7 skipped** in 27:05; the single failure is the approvals-drift guard `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`, explained below |

Per-module confirmations at the branch tip (separate runs): `test_exp06_launch.py` +
`test_exp06_smoke.py` + `test_exp06_recipe.py` 183 passed; `test_exp03_record_tools.py` +
`test_provenance.py` 63 passed, 1 skipped (the pinned exp_03 closure is untouched);
`test_exp06_heading.py` + `test_exp06_eval.py` 75 passed;
`test_exp06_finalize_haa_consumer.py` + `test_exp06_probe_align.py` 43 passed, 1 skipped;
`test_exp06_compare.py` + `test_exp06_bootstrap.py` + `test_exp06_mirror_probe.py`
189 passed, 2 skipped; `test_exp06_eval_launch.py` + `test_exp06_haa.py` 85 passed;
`test_exp06_train.py` + `test_exp06_approvals_api.py` + `test_exp06_profiles.py` 80 passed,
1 failed (see below). No xfail. The skips are the pre-existing "skip if absent" cases.

### The one expected failure: `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`

This is the approvals-drift guard doing its job, not a regression. The committed
`oriented_cyl_results_assets/approved_digests.json` still pins the digests from the
`b9f6ebe` re-fill, and this round changes four of them (`finalize`, `smoke`, `launch_sh`,
`summarize_haa`, listed below) — the test names exactly those four. It cannot be made green
from inside this round: `approved_digests.json` is not in the file list this prompt permits,
and re-filling it here would be precisely the self-approval the fail-closed design forbids.
A5 already schedules the re-fill after the Codex review and the merge, and the same pattern
was recorded on 2026-09-16T08:29 for the previous code-changing branch. Every other module
is green.

## TRAINING_KEYS digests that change

Computed with `tools.exp06_profiles.compute_code_digests(repo, HEAD, keys=…)` at the branch
tip `8bc7cf8` (content-determined, so they survive the merge to `main` unchanged as long as
the file bytes are):

| Key | Approved now | This branch |
|---|---|---|
| `finalize` | `3da0479ab2629ebca4ff2cfafe193481bd13b6c2f1b7e1addb8ba9a3926a060c` | `9e3a000ba089aa0f4cb40372d07f5e4b7a4e3b851b4884c95dbf5a04aa470b80` |
| `smoke` | `eebc949a2ebfa5bf63ca4cf7603c5bf60256fc3ca7d10acbe7743daee813abaa` | `00e3b91b8289cc0f1fda8c2ab2ce3b151ca897f4a4c746f93863e3d338091d00` |
| `launch_sh` | `157e2435ba717eb97e9046cf0120ddbaa0ea03860daaf55e303c2396164eb3a8` | `c64b60df4632b83617c1343f0d0a56c6ac7cbab24db95719a2aa8734f3977773` |

Unchanged TRAINING_KEYS: `trainer`, `recipe`, `encoder`, `factory`, `profiles`.

**Outside `TRAINING_KEYS`, one further `code` key changes** and needs the same re-fill before
the round-3 producers run, because `tools/exp06_summarize_haa.py` imports the finalizer:

| Key | Approved now | This branch |
|---|---|---|
| `summarize_haa` | `774c6868691d2521e665caa8bd9ee88ffb389ba81117247a66e144c894a3a5c1` | `c2b56e66f6aa32c7732080219437aa6fb0f71c78298be304ff7fdc9d365f228a` |

`haa_finetune`, `haa_eval`, `eval`, `eval_launch`, `mirror_probe`, `bootstrap`, `compare`,
`heading`, `probe_align`, `haa_pipeline_sh`, `evaluator_exp03` are unchanged.

The final commit of this branch is the intended launch tip once merged, reviewed and the
approvals re-filled; note that `main` advanced to `7796d6f` (worklog-only) during the round,
which touches no closure file.
