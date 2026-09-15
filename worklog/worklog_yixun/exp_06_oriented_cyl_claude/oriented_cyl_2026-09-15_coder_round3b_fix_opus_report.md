**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

Round 3b FIX cycle. Worktree `/home/yixunhu/codespace/xRIR_code_wt2b`, branch `exp06-round2b`,
from tip `0ca806c`. CPU only (`CUDA_VISIBLE_DEVICES=''`), `OMP_NUM_THREADS=4`,
`PYTHONDONTWRITEBYTECODE=1`, pytest with `-p no:cacheprovider`. No process was signalled;
no `pkill`/`pgrep` was used at all. Nothing under `/home/yixunhu/codespace/xRIR_code_wt`
or the main tree was touched except this report and the record copy of the approvals
template.

## Cycle −1 — merge of the training branch

`git merge --no-ff exp06-window` (tip `5091f6c`) → merge commit **`90ce3ba`**
(`exp06: merge the training branch (5091f6c) into the round-3 branch`).

Three conflicts, all resolved preserving both sides; no test was dropped.

| File | Conflict | Resolution |
|---|---|---|
| `tools/exp06_haa_pipeline.sh` | both sides re-documented `open_job` (round 3a's `OWNED_ROOT` comment vs the training branch's `own_launch` comment) | the merged body has **both** behaviours (exclusive `mkdir` setting `OWNED_ROOT`, then `own_launch` recording `$$`), so the comment now states both |
| `tests/test_exp06_haa_pipeline.py` | round 3b added a local `dead_pid()`; the training branch imports the shared `DEAD_PID` from `test_exp06_finalize` | kept the shared constant (same pid rule, plus a non-Linux fallback) and removed the local duplicate; every fixture still writes a guaranteed-dead pid |
| `tests/test_exp06_profiles.py` | add/add — the training branch's tests of the real `tools/exp06_profiles.py` vs round 3b's tests of the lazy adapter | the training branch's file keeps the name; round 3b's 212 lines moved verbatim to the new `tests/test_exp06_approvals_api.py`. Both sides' tests survive |

`tests/test_exp06_finalize.py` was not touched on this branch, so the training branch's
version merged cleanly, as instructed.

### Integration failures after the merge

Full suite at `90ce3ba`: **5 failed, 1132 passed, 8 skipped** (22 min). One cause, three
symptoms, fixed in the single separate commit **`76e8f6d`**
(`exp06: the merged approvals module is the one round 3b reads`, 47 changed lines):

1. `test_exp06_approvals_api.py::test_the_absent_approvals_module_is_a_named_refusal` —
   asserted `tools.exp06_profiles` is not importable. The merge brought it. The test now
   names an absent module explicitly, and a new test pins that the real committed template
   loads through the adapter's schema.
2. `test_exp06_summarize_haa.py::test_production_refuses_without_the_approvals_module`,
   `::test_the_cli_refuses_a_production_run_on_this_branch`,
   `::test_the_cli_writes_a_draft_and_the_legacy_receipt` and
   `test_exp06_compare.py::test_the_cli_refuses_a_production_run_on_this_branch` — the
   adapter now reached the real loader and refused for two wrong reasons: the loader
   returns a frozen `MappingProxyType` (`approved digests: section code is not a record`)
   and the committed template carried no `code.probe_align` (`code is missing
   probe_align`). Fixed by detaching the frozen mapping in
   `exp06_approvals_api.load_approved_digests` and by registering `probe_align`.

### `code.probe_align`

Added to `tools/exp06_profiles.py`'s `CODE_SPECS` (→ `CODE_KEYS`, `present_keys`) and to
**both** copies of the template: worktree `tools/exp06_approved_digests_template.json` and
the record copy
`worklog/.../oriented_cyl_results_assets/approved_digests.json` (the only main-tree file
edited besides this report; left uncommitted in the main tree for the Planner).

## Reconciled A3 job-completion field names

The merged finalizer is authoritative. Round 3b had assumed a different shape; the
summariser now reads the real one (`tools/exp06_summarize_haa.py::JOB_FIELDS`).

| Round 3b assumed | Merged finalizer (authoritative) |
|---|---|
| `job_spec_sha256` (string) | **`job_spec`** = `{path, sha256}` — re-read and rehashed, then parsed by `finalizer.load_job_spec` |
| `owner` = `{pid, path, sha256}` | **`owner_pid`** (int) |
| `children` = `{relative name: sha256}` | same |
| `run_type='haa_job'`, `expect`, `schema_version`, `run_dir`, `diagnostic`, `admissible_arm`, `backbone`, `frame`, `heading`, `seed`, `init_sha256` | same |
| (not assumed) | additionally `repo`, `child_exit`, `log` (`{path,sha256}` or null), `artifacts` (`{}`), `init` |
| assumed: no `child_exit_receipt` | correct — and no `child_exit_time` either; **both** are now refused if present (`FORBIDDEN_JOB_FIELDS`) |

Job-level `heading` is `{room: k}` (the job spec's rolls), not the per-room binding
records; the per-child binding records (with `path`/`sha256`) are what `arm_headings`
compares across seeds and against `artifacts.heading`.

## Fix commits

All ≤ 200 changed lines including tests; every implementation commit is preceded by a
genuine red commit; every commit carries the two trailer lines.

| SHA | Lines | Red | Summary |
|---|---|---|---|
| `76e8f6d` | 47 | — | integration fix (above) + `code.probe_align` |
| `e5e2620` | 15 | — | **F11** six refusal regexes now assert a diagnostic phrase, not a word a tmp path also contains |
| `447be92` | 40 | 1 | red — a legacy receipt may not omit a registered arm |
| `6e35927` | 22 | — | **F6** enumeration derived from `LEGACY_ARMS`; `arms` parameter removed from the writers; a receipt whose `arms` is anything but the registered two is refused |
| `727f0a0` | 41 | 2 | red — void/empty HAA cohort is a reportable cell |
| `578f884` | 96 | — | **F8** `cell_rows` no longer raises; `void_reasons` names an empty cohort; `decision_cell` applies the policy *before* bootstrapping; `intervals` returns `None`; H2/D/renderer report `not available`; the bootstrap's zero-width refusal is untouched |
| `9da6d8b` | 72 | 5 | red — per-role closures, heading identity, frozen protocol |
| `096956f` | 71 | — | **F1/F3** `arm_closures` (per entry point), `arm_headings` (one record per room), `check_arm_identities` (vs `code.haa_finetune`/`code.haa_eval` and `artifacts.heading`), `child_protocol`, `check_test_indices` |
| `928592a` | 49 | — | statistics tests moved onto `synthetic_arm` dicts (test-only, green) |
| `09ccbfa` | 151 | — | **F2** the forged new-arm completions retired (no provenance, no log, no receipt, one constant closure) |
| `d21d884` | 63 | 4 | red — admission over children the finalizer really certified (reuses the pipeline tests' `finetune_seed`) |
| `5b8bfdc` | 171 | — | **F1/F2** `job_completion` reconciled to the A3 schema; `verify_job` rehashes the job spec, calls `verify_child` per child, refuses `admissible_arm: false`, re-derives `job_lineage` and compares the job's own fields, applies the frozen protocol / DiffRIR indices / per-sample `meta.heading` equality; `load_new_arm` composes the four jobs and compares closures and headings with the approvals |
| `45eb6bc` | 46 | 2 | red — summariser compares actual identities |
| `468263c` | 87 | — | **F3** producer closure vs `code.summarize_haa`; exp_02 `stats.json`/`summary.txt` vs `reused.exp02_*`; G1 artifact vs `artifacts.gate_g1` through a new `--gate-g1`; `--write-legacy-receipt` passes the same gate; verified producer identity published |
| `8e7da03` | 48 | 3 | red — bound inputs, final revalidation, staged publication |
| `cfa7844` | 104 | — | **F10/F12** digests captured with the bytes consumed; side-split cache geometry bound; `recheck_inputs` immediately before publication; staged output pair with cleanup |
| `9fb0538` | 81 | — | H3 fixture rewritten to write real evidence (repo, closure files, data inventory, reference manifest, log, flags, reconciling aggregate) |
| `d05c063` | 105 | 14 | red — H3 completion/source/data/reference evidence |
| `949689f` | 111 | — | **F4/F5** `_UNBOUND` sentinel; completion flags + child log; `provenance.revalidate`; closure and data-inventory bindings with stat identity; mutable-input names and the exp_06 arm's training bindings; metrics reconciliation; `check_reference` parses the manifest (K = 8, seed, semantic hash, exact query/index order) |
| `b1047c9` | 73 | 19 | red — explicit admission routes |
| `454c843` | 88 | — | **F7** `route_of` + `check_route`; exp_05 route reads the hash-approved exp_05 record and validates M-tier metadata and its `train_args` binding; exp_04 route consumes `reused.exp04_approved_digests_sha256` |
| `b148e7b` | 66 | 5 | red — H3 eligibility, producer identity, safe publication |
| `6c6a156` | 105 | — | **F3/F9/F10/F12** complete-arm + no-deviation gate for decisions (`H3: available/unavailable/exploratory`, named suppressions); `check_producer_identity` vs `code.compare`; `recheck_inputs`; staged output pair |
| `92c0dec` | 14 | 1 | red — the checkpoint identity of every arm is published |
| `df7ea24` | 19 | — | **F5** `arm_checkpoints` requires one checkpoint per arm and publishes path/digest/epoch/route |

Red counts above are the number of failing tests the red commit introduced.

## Per-finding summary

1. **Role-specific closures** — `arm_closures` groups by the entry point the child's role
   names and requires consistency *within* each role; `check_arm_identities` compares each
   with `code.haa_finetune` / `code.haa_eval`. Positive test on a real finalised job shows
   the two actual, distinct digests.
2. **Full child verification** — `verify_job` re-reads and rehashes the bound job spec,
   calls `finalizer.verify_child` (provenance, closure, artefacts, log + exit receipt,
   heading JSONs re-read through `exp06_heading.read_heading_json`, agreement with the job
   spec) for every expected child, refuses `admissible_arm: false`, re-derives
   `job_lineage` (init → stage1 → stage2 → eval; zero-shot → job init) and requires the
   job's own lineage fields to equal it, then applies the frozen protocol, the DiffRIR
   indices and per-sample `meta.heading` equality. Fixtures are real finalised children.
3. **Producer approvals** — actual-versus-approved comparisons for the summariser's own
   closure, the children's per-role closures, the heading artefacts, the G1 artifact, the
   exp_02 canonical record, the comparer's own closure, and the exp_04/exp_05 approvals
   records. `--write-legacy-receipt` is gated the same way; the verified identity is in
   the output.
4. **H3 bindings** — `_UNBOUND` sentinel (a declared `null` is a missing binding);
   completion `confirmatory`/`allow_dirty_used`/`log`; `provenance.revalidate` over the
   declarations; every closure file and every data-inventory file bound with stat
   identity; mutable-input names constrained and the exp_06 arm's training bindings
   required; `_check_metrics_reconciliation` composed.
5. **H3 protocol** — `num_shot == 8` in the manifest and in the reference; the reference is
   parsed with `reference_manifest.load_manifest`/`manifest_hash` (seed, K, semantic hash)
   and the per-sample `query`/`index` must be exactly its entries; k = 0, TF32 off, batch
   16, condition P, split/count unchanged; role-specific checkpoint identities (C =
   approved `artifacts.epoch_012` incl. epoch, A/B = exp_01's weights) are now recorded in
   the published result. Mutation tests keep every other hash internally consistent.
6. **Legacy receipt** — expected enumeration from `LEGACY_ARMS`; partial/empty declarations
   refused; tampering in either arm refused; a real-root test (skipped without
   `ckpt/sim2real`) checks both arms are enumerated.
7. **exp_05 route** — explicit `exp04`/`exp05`/`exp06` routes; the exp_05 route reads the
   hash-approved exp_05 record for its evaluator identity and validates the M-tier
   metadata and `train_args` binding; exp_05-shaped positive test.
8. **Void outcomes** — cohort/exclusion diagnostics before any resampling; a void or empty
   cohort yields a reportable `void` H1/H1b cell and `not available` downstream; the
   bootstrap helper's zero-width refusal is unchanged.
9. **H3 eligibility** — production reporting requires the complete A/B/C set (missing B →
   `H3: unavailable`, verdict `suppressed (no H3 comparator)`); an exploratory or
   deviation-carrying admission → `H3: exploratory` and `suppressed (draft)` everywhere.
10. **Final revalidation** — hashes captured with the bytes consumed (receipt verification,
    arm admission, side-split cache); all bindings rechecked immediately before
    publication; mutation-during-analysis tests for both producers.
11. **Nit** — the six flagged regexes now assert diagnostic phrases.
12. **Nit** — both producers stage their output pair through temporaries with cleanup and
    preflight distinct, absent paths.

## Validation

| Command | Outcome |
|---|---|
| `git merge --no-ff exp06-window` | 3 conflicts, resolved; merge `90ce3ba` |
| full suite at `90ce3ba` | 5 failed, 1132 passed, 8 skipped (integration failures above) |
| `python -m py_compile` on every changed module | clean |
| `git diff --check 0ca806c..HEAD` | clean |
| `git diff --name-only 90ce3ba HEAD` | only `tools/exp06_{approvals_api,approved_digests_template.json,compare,profiles,summarize_haa}` and `tests/test_exp06_{approvals_api,compare,eval_launch,finalize,haa,summarize_haa}.py` |
| final full suite (`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider`) | **1169 passed, 8 skipped, 0 failed, 0 xfail** (22 min) |
| trailer audit of every commit `90ce3ba..HEAD` | both lines present on all 26 |

## Deviations and discrepancies (nothing silent)

1. **New CLI flag `--gate-g1`** on the summariser. §6.4 requires the G1 artifact of
   `artifacts.gate_g1` for this producer, and comparing it needs the file. Fail-closed
   choice: when `artifacts.gate_g1` is approved and `--gate-g1` is absent, production
   refuses.
2. **Signature changes.** `load_new_arm(root, arm, init_sha256=None, repo=REPO,
   approved=None)`; `legacy_receipt` / `write_legacy_receipt` lost the `arms` parameter
   (finding 6) and gained `identity=`; `analyse` gained `producer` and `extra_inputs`;
   `write_outputs` (comparer) gained `stats`.
3. **`arms[arm]['closure']` is now `{role: digest}`**, not a single digest — the direct
   consequence of finding 1. It is published under the same key.
4. **Test architecture.** The statistics/side-split/rendering tests read `synthetic_arm`
   dicts rather than going through admission, and the summariser's two CLI tests use an
   explicit `stub_new_arms` seam; admission itself is tested against a real finalised job
   built with the pipeline tests' own `finetune_seed`/`clone`/`cache` fixtures. Building
   three complete arms of real children (31 children each) for every statistical assertion
   was not affordable; the seam is explicit and named.
5. **`exp05_approvals()` is a module-level seam.** exp_05's approvals record is all-null on
   this branch, so a production exp_05-route run refuses (correct). The positive test
   monkeypatches this one function rather than the loader, so the production path is
   unchanged.
6. **`verify_job` does not call `refuse_live_launch` on the job root.** `verify_child`
   still enforces it for every child (so no job is admitted while a child is alive). A
   summary run weeks later would otherwise refuse a valid job whose recorded launcher pid
   had been reused by an unrelated process. The job's `owner_pid` is still required to be
   a positive integer.
7. **`test_a_changed_source_or_data_file_is_refused` matches
   `'(digest |revalidate source)'`** because `provenance.revalidate` itself names closure
   drift (`source.<name>.<path>`) before the explicit `bind` loop reaches it.
8. **The `t60` phrase list needed `nothing measures T60`** as well as `never measures` —
   the finalizer has two distinct refusals for that room.
9. **Not done, and out of the listed findings:** the exp_06 approvals are *not* required to
   be the blob committed at HEAD, although `tools.exp06_profiles.load_approved_digests`
   supports `repo=`/`commit=`. Adding it would make every production run depend on the
   approvals file being committed; it is a separate, plan-level decision. Recorded here as
   an open item.
10. **A repo-relative `manifest_path` fails closed, not open.** `paired_compare.admit_run`
    resolves it against `fields['repo']`; `exp06_compare.check_reference` resolves it as
    given. Every retained exp_04 run and every exp_06 launch records an absolute path (the
    launcher resolves them in `parse_args`), and a relative one would raise `OSError` out
    of `load_manifest` and become an admission deviation, i.e. a refusal. Left as is
    rather than adding an untested resolution path; noted as a known limitation.
11. **`arm_inputs` no longer rehashes.** A caller that constructs arm dicts by hand (the
    test seam above) contributes no inputs; production always goes through
    `verify_legacy_receipt` / `load_new_arm`, which do.
