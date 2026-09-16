**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

Training-gate fix cycle 2 for exp_06 "oriented_cyl": the four findings of the Codex
`full_train` review 2 (`oriented_cyl_codex_code_full_train2_review.md`, reviewing
`f18383a`). Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`,
CPU only (`CUDA_VISIBLE_DEVICES=''`). `/home/yixunhu/codespace/xRIR_code_wt2b` was not
read; no process I did not start was signalled; nothing under `ckpt/` was written.

## Commits (f18383a..a313253)

| SHA | subject | changed lines |
|---|---|---|
| `0f3b79e` | red test for the registered smokes' child dispatch (finding 1) | +75/-0 = 75 |
| `b979f0c` | the launcher tells the exp_06 child what diagnostic it is (finding 1) | +67/-13 = 80 |
| `a74d985` | red tests for approvals bound to a reviewed commit (finding 2) | +90/-15 = 105 |
| `5201430` | preflight binds the approvals to their reviewed commit (finding 2) | +43/-3 = 46 |
| `a83f045` | red tests for committed approvals at finalisation (finding 2) | +93/-10 = 103 |
| `79b53f0` | finalisation reads approvals from their reviewed blob (finding 2) | +8/-3 = 11 |
| `f82c33d` | red tests for complete consumed-data coverage (finding 3) | +98/-0 = 98 |
| `7868207` | bind the held-out waveforms and both geometry splits (finding 3) | +83/-20 = 103 |
| `df6a762` | red tests for diagnostic receipt consistency (finding 4) | +69/-0 = 69 |
| `58ccc27` | derive a diagnostic's pass from consistent evidence (finding 4) | +71/-6 = 77 |
| `a313253` | regression for the exploratory flag reaching the children (finding 1) | +14/-0 = 14 |

Maximum 105 changed lines; both trailer lines present on all eleven commits
(`git log -1 --format=%B | grep -c` = 2 each). `git diff main...HEAD --diff-filter=M
--name-only` is empty: no tracked file of the repository was modified, so the exp_03
pinned-closure rule and the exp_04/05 digests are untouched. Files touched this cycle:
`tools/exp06_{train,finalize,smoke,profiles}.py`, `tools/exp06_launch.sh`, and
`tests/test_exp06_{train,launch,finalize,finalize_gate,profiles,smoke}.py`.

## Finding 1 — the registered smokes failed at startup (blocker)

**Fix.** `tools/exp06_launch.sh` now forwards `--run-type smoke|probe` (and
`--exploratory` when the launch is exploratory) into the **child** argv for the
`exp06_train` entries; the pinned `trainer` entry's argv is byte-for-byte what plan §9
registers, since its parser knows neither flag. `tools/exp06_train.py` grew
`check_admission(args)`, called from `main()` in place of the inline guard: without
`--approved` a run is admitted only when `--run-type` is `smoke` or `probe` **and**
`--no-save` is set; `full` without `--approved` is refused as before.

**Safest-option choice (reported, not silent).** The prompt left open what a diagnostic
that would write to its save-dir should do. I took the fail-closed option: a `smoke` or
`probe` without `--approved` and without `--no-save` is refused
(`a smoke run that writes to its save-dir must name its approvals (--approved), or run
with --no-save`), so no diagnostic can leave behind an artefact that could later be
mistaken for an arm. `check_admission` lives in `main()`, not `parse_args()`, so the
parser stays pure for the trainer-parity introspection test.

**Regressions.** `tests/test_exp06_launch.py::test_the_printed_diagnostic_commands_reach_
the_real_entry[smoke|probe]` parses the launcher's own dry-run output, and dispatches
every printed `(entry, child argv)` pair through the real
`tools.exp06_smoke._invoke(...)` with the dataset stubbed and the model builders raising
at the first line that needs a GPU — asserting the entry is reached, i.e. no ValueError
on the way. `test_a_full_child_without_its_approvals_is_still_refused` strips `--approved`
from the printed *full* child argv and requires the refusal (and reaches the GPU line
with it). `test_an_exploratory_launch_tells_its_children_so` pins the flag forwarding,
`tests/test_exp06_train.py::test_a_diagnostic_is_admitted_without_approvals_only_while_it_
saves_nothing` pins the `--no-save` condition, and the §9 golden strings were updated.

Before the fix the smoke case failed exactly as the review reported
(`ValueError: a full run must name the approvals it is admitted under (--approved)`,
after `train samples: 8 test samples: 8`); the probe case already carried `--run-type
probe` and passed.

## Finding 2 — approvals bound to a reviewed commit

**Fix.** `tools.exp06_profiles.committed_bytes(path, repo, commit)` reads the blob with
`git cat-file -p <commit>:<repo-relative path>`, and
`load_approved_digests(path, repo=None, commit=None)` requires, when both are given, that
the file lie inside the repository, be tracked at that commit, and be byte-identical to
the blob; the identity then carries `repo_relative` and `committed_at`. Supplying only
one of the two is itself a refusal. `exp06_finalize.approval_gate` (preflight) binds every
mode it admits, and `verify_approvals` (finalisation) binds with the record's
`reviewed_commit`; the only exception is a run recorded `exploratory`, which by contract
is a `smoke`/`probe` and is never `admissible_arm`. The recorded `approvals.sha256` is
still compared to the file on disk first, so it equals the committed blob's sha256
whenever the binding holds. The completion records `approvals.committed_at`.

Per the prompt's self-correction, the two approvals copies are unchanged: the worktree
template `tools/exp06_approved_digests_template.json` stays the null template, and the
filled approvals stay at the record path
`worklog/.../oriented_cyl_results_assets/approved_digests.json`, which becomes a tracked
path of this repository when the second reviewed commit adds it.

**Deliberate scope choice (reported).** The review's minimal fix names *preflight and
finalisation*; I did not add the committed check to `exp06_train.approvals_binding` at
spawn. The spawn-time hash still binds the file, preflight has already refused an
uncommitted or foreign file a moment earlier, and finalisation refuses it again, so the
confirmatory path is gated on both sides. This also keeps the trainer usable from a dirty
checkout for exploratory diagnostics.

**Regressions.** `tests/test_exp06_profiles.py`: a temp repository with the template
committed — binding succeeds and names the path and commit; an external copy, an
untracked sibling, a working-tree edit and an unknown commit each refuse; a lone `repo=`
or `commit=` refuses. `tests/test_exp06_launch.py::test_confirmatory_approvals_must_be_
committed_at_the_reviewed_commit` walks the same four cases through `preflight('full')`
and shows an exploratory smoke still reading an external file.
`tests/test_exp06_finalize_gate.py`: `test_full_approvals_must_be_the_blob_of_the_
reviewed_commit` (external copy, untracked sibling),
`test_approvals_edited_after_the_reviewed_commit_are_refused` (the review's hole —
preflight's cleanliness check excludes `worklog/`), and
`test_an_exploratory_diagnostic_may_read_approvals_from_anywhere`.

**Fixture change.** `tests/test_exp06_finalize.py::approvals` now writes the filled
approvals *and* a null sibling into the throwaway clone at the registered record path and
commits them (two commits: placeholder, then filled), returning the tracked path; a new
`null_approvals` fixture serves the "nothing approved yet" case. The clone's working tree
is still clean, and the extra commits change no source blob, so every closure digest is
unchanged.

## Finding 3 — complete consumed-data coverage

**(a) test-split waveforms.** `tools/exp06_train.py` gained `heldout_wav_paths(data_root)`
and `heldout_wav_identity(data_root)`: the membership comes from the pinned dataset's own
split logic (`xRIR_Dataset(split='test').file_list`), and the record is the same inventory
shape `tools.provenance` uses. `provenance_fields(..., heldout=...)` records it as
`test_wav_identity`, and `main()` computes it whenever the run has a provenance
destination. The finalizer's `verify_heldout_identity` derives the membership again here,
never from the record, rehashes every file, and reports
`test_wav_files` / `test_wav_bytes` / `test_wav_sha256`; `full_evidence` requires it.

**Naming (reported).** The provenance key is `test_wav_identity` as specified, but the two
functions are `heldout_wav_*` rather than `test_wav_*`, so that no module-level callable
in `tools/` is named `test_…` and collectable by pytest if it were ever star-imported.

**(b) geometry splits.** `verify_geometry_identity` no longer reads the required splits
from the record: it requires `splits == list(exp06_train.GEOMETRY_SPLITS)` exactly and
derives the expected membership from that constant. A new shared `_membership(key,
entries, expected)` refuses duplicates (a set comparison could not see a path recorded
twice), omissions and additions, and is used by both inventories; `_identity_root`
factors out the shared `data_root` agreement check.

**Regressions.** `tests/test_exp06_finalize_gate.py`:
`test_heldout_membership_is_derived_from_the_datasets_own_split`,
`test_a_full_completion_rehashes_every_held_out_waveform`,
`test_a_changed_test_waveform_is_refused` (the review's reproduction, on
`Bathrooms_idx_18/S000_R000_hybrid_IR.wav`),
`test_an_incomplete_test_wav_inventory_is_refused[absent|extra|wrong_root]`,
`test_a_geometry_inventory_of_one_split_is_refused` (a correctly hashed
`splits=['train']` record, also with the held-out depth map then changed), and
`test_a_duplicated_inventory_entry_is_refused[geometry_identity|test_wav_identity]`.

## Finding 4 — diagnostic receipt consistency

**Fix.** `exp06_finalize.check_receipt_consistency(fields, record, window)` runs after the
field/type checks and derives `passed`. It requires: `entry` in
`tools.exp06_smoke.ENTRIES` (imported, so the two lists cannot drift); `argv` a list of
`str`; `entry` equal to the startup provenance's `entry`; `[entry] + argv` equal to the
recorded child `command`; `alarm_seconds` and `max_gb` finite and strictly positive
(`_positive`); `outcome == 'ok'` only with `exit_status == 0`, a peak within
`max_gb·2^30` and a wall time within `alarm_seconds`; any other outcome only with a
non-zero status; and the receipt's `started_at`/`ended_at` inside the child execution
window that `child_exit.json` records. `passed` is then
`outcome == 'ok' and exit_status == 0 and child_exit == 0`, so a valid failure receipt
still produces a completion with `passed: false` (never `admissible_arm`).

**Two judgement calls (reported).**
1. *Over-budget with `outcome: ok` refuses rather than degrading to `passed: false`* —
   the fail-closed reading. To make that impossible to trigger spuriously, the runner was
   made self-consistent: `tools/exp06_smoke.py::_publish` now also converts a
   retrospective over-run (`wall_s > alarm_seconds` with status 0) into `aborted_alarm`,
   exactly as it already did for the memory ceiling. The one-second watchdog tick could
   otherwise let an entry return just past its deadline and publish `ok`.
2. *One second of slack at the end of the window* — `child-exit` writes the child's
   `ended_at` truncated to the second (`replace(microsecond=0)`), while the receipt's
   `ended_at` carries microseconds, so the recorded window can end up to a second before
   the receipt legitimately does. The start needs no slack (the launcher's `date -u`
   truncates downwards, before the spawn). This is documented in the function and pinned
   by `test_the_marker_second_is_the_only_slack_at_the_end_of_the_window`.

**Regressions.** `test_contradictory_diagnostic_receipts_are_refused` covers all twelve
mutations the review accepted (`outcome='failed'` and `'aborted_memory'` at status 0; a
4 GiB peak against 3 GiB; `wall_s=301` against 300; a zero alarm and a zero memory budget;
`entry='invented'`; `argv=['--no-save', 17, {}]`; an entry and an argv disagreeing with the
startup command; and both timestamps outside the child window).
`test_a_consistent_failure_is_recorded_as_not_passed` keeps the valid failure path, and
`tests/test_exp06_smoke.py::test_a_run_that_outlasted_its_alarm_is_published_as_an_abort`
pins the runner-side flip.

## Validation

| check | result |
|---|---|
| `python -m py_compile` on the six `tools/exp06_*.py` and the six touched test modules | OK |
| `bash -n tools/exp06_launch.sh` | OK |
| `git diff --check` | OK (no output) |
| `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider` | **679 passed, 4 skipped** in 734.7 s |
| `pytest tests/test_exp03_record_tools.py -q` (pinned-file guard) | **31 passed, 1 skipped** |
| `git diff main...HEAD --diff-filter=M --name-only` | empty |

Red counts before each implementation commit: finding 1 — 1 failed (the review's
`ValueError`, smoke case; probe already green); finding 2 preflight — 8 failed;
finding 2 finalisation — 3 failed; finding 3 — 1 failed + 30 fixture errors;
finding 4 — 13 failed. Baseline on `f18383a` for the same three files I re-ran first was
91 passed; the suite grew from the previously reported 639 passed / 4 skipped to
679 passed / 4 skipped (+40 cases).

## Discrepancies and open items

- **Spawn-time approvals binding** is deliberately not added to `exp06_train`
  (finding 2 above): preflight and finalisation carry it, as the review's minimal fix
  asks. If the Planner wants it at spawn as well, `approvals_binding` would need a
  `repo`/`commit`/`exploratory` triple and the trainer's provenance test would move onto a
  temp clone.
- **`passed` is advisory to the launcher.** `tools/exp06_launch.sh::diagnostic` aborts a
  rung on a non-zero *child status*, not on `passed: false`. Every case that can now
  produce `passed: false` with a zero status is a forged receipt (the runner always exits
  non-zero when it publishes a non-`ok` outcome), so the ladder still stops at a genuinely
  failed rung — but the completion, not the exit code, is the record of truth.
- **The confirmatory geometry + held-out rehash cost.** Finalisation now rehashes the
  test-split waveforms in addition to the ~10.2 GB of geometry; on the registered mirror
  that is 6 337 extra files, small next to the 60 s geometry pass, but it is a second
  full-inventory read at the end of a 31 h run.
- **`probe_align` is still absent from the approvals template** (carried over from
  review 1's disposition; round 3 work, outside this gate).
- The heading consumer readback (`_heading_binding` calling `read_heading_json` without
  `room_dir`) remains the deferred round-2b item; untouched here.
