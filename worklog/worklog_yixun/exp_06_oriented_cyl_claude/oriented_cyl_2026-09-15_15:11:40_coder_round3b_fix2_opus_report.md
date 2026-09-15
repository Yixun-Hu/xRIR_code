**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 round 3b, fix cycle 2 — the seven blocking findings of the close review

Worktree `/home/yixunhu/codespace/xRIR_code_wt2b`, branch `exp06-round2b`, from `df7ea24`
to `07001a0`. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), no process
signalled, no `pkill`/`pgrep`. Only four tracked files were touched —
`tools/exp06_compare.py`, `tools/exp06_summarize_haa.py` and their two test files. No
pinned, digest-pinned or training-path module was edited (`git diff df7ea24..HEAD
--name-only` lists exactly those four). This report is the only file written under the
main tree.

## Commits (18; 9 red, 9 implementation; every one has both trailer lines)

| commit | subject | changed lines (incl. tests) |
|---|---|---|
| `2e670da` | red -- H3 must enforce the registered experiment (finding 1) | 109 |
| `7b19f94` | H3 admits the registered split, inventory and reference (finding 1) | 36 |
| `76a3606` | red -- role, registry, class and epoch identities (finding 4) | 93 |
| `9acdc0c` | H3 binds the registered model, registry and epoch of each role (finding 4) | 51 |
| `55e975e` | red -- the exp_05 route establishes its args/tier binding (finding 5) | 100 |
| `2fae051` | the exp_05 route re-derives its tier from the checkpoint args.json (finding 5) | 94 |
| `cacfa86` | red -- production approvals are bound to the reviewed commit (finding 7) | 77 |
| `e391513` | production approvals are the bytes committed at the reviewed HEAD (finding 7) | 49 |
| `48f14a0` | red -- the A3 owner is read from the job root launch.pid (finding 6) | 27 |
| `b2361fd` | bind the job owner to the root launch.pid (finding 6) | 23 |
| `74fa776` | red -- the registered HAA recipe of plan 6.2 and the sensitivity mode (finding 3) | 73 |
| `f0d7963` | admit only plan 6.2's recipe, or label the run a sensitivity analysis (finding 3) | 104 |
| `747f3a6` | red -- parsed bytes, contradictions and the complete revalidated set (finding 2) | 52 |
| `e97bcab` | bind bytes where they are read and revalidate the complete set (finding 2) | 62 |
| `f30ac22` | red -- every HAA child artefact is bound and revalidated (finding 2) | 42 |
| `9b62055` | carry every verified child artefact into the revalidated set (finding 2) | 67 |
| `4617219` | red -- the comparer's own closure is bound and revalidated (finding 2) | 15 |
| `07001a0` | bind the comparer's producer closure for revalidation (finding 2) | 7 |

Maximum commit size 109 lines. No amend, rebase, reset or push.

## Red counts (failing tests at each red commit, before its implementation)

| cycle | red commit | failing |
|---|---|---|
| finding 1 | `2e670da` | 6 |
| finding 4 | `76a3606` | 7 |
| finding 5 | `55e975e` | 7 |
| finding 7 | `cacfa86` | 6 (4 comparer + 2 summariser) |
| finding 6 | `48f14a0` | 1 |
| finding 3 | `74fa776` | 4 |
| finding 2 (bindings) | `747f3a6` | 4 |
| finding 2 (child artefacts) | `f30ac22` | 2 |
| finding 2 (comparer producer) | `4617219` | 1 |

Total 38 genuinely failing tests, each red at collection-time-valid module level (no
cycle was red only through a collection error).

## Per-finding summary

1. **(blocker) H3 now enforces the registered experiment.** `SPLIT` is built from
   `exp04_profiles.COMMON['dataset']` plus `REFERENCES[8]`, so admission compares each
   run against the registered inventory digest, the registered seed-specific reference
   semantic hash, the canonical query digest and the 17-room population, and validates
   every reference entry as an eight-reference (K = 8) draw. The review's three
   counterexamples — empty inventory with a recomputed digest, `refs: []` everywhere,
   and consistently invented query names — are each refused. The real exp_04 seed-42..46
   control runs still admit as arm A (positive regression retained).
2. **(blocker) The summariser carries the complete verified evidence forward.** A new
   `bind()` records the bytes each reader actually read and refuses a contradictory
   second digest for one path; `_read_json` binds the exact bytes it parsed, so job and
   child completions are hashed as they are consumed rather than afterwards. `verify_job`
   threads one `inputs` map and `bind_child` adds every child artefact, its log, its exit
   receipt, the heading records it read and the weights it started from or evaluated;
   the legacy receipt is verified *before* the historical runs are loaded; side-cache
   reads can no longer overwrite an earlier binding; the producer closure files and the
   approvals receipt are bound for both producers. A child per-sample/metrics mutation
   after admission is now refused at publication (new test, through admission *and*
   `write_outputs`).
3. **(blocker) An exp_06 HAA admission profile for plan §6.2.** `S1_ROOMS`, `STAGES`
   (1000/10 and 200/2), `TRAIN_RECIPE` (lr 1e-4, weight decay 1e-4, decay_epochs 50,
   gamma 0.1, batch_size 0, accum 1, TF32 on, max_len 9600, depth_variant default,
   num_shot 8, eval_seed 0) and `EVAL_RECIPE` (split test, num_shot 8, eval_seed 0,
   max_len 9600, depth_variant default, max_samples 0, `gl_seed_per_query` **false**,
   tag '') are checked by `child_recipe` for every child, including the per-stage room
   sets. A confirmatory admission refuses any deviation; `--sensitivity` admits them and
   labels the outputs (`mode: "sensitivity"`, a `SENSITIVITY` banner, and every H1/H1b
   verdict prefixed `sensitivity: `). The diagnostic nine-child fixture (four epochs) is
   now refused by confirmatory admission and only the explicitly sensitivity-flagged
   tests use it.
4. **H3 identities.** Each role registers its backbone and epoch (`C`:
   cylindrical_oriented/12, `A`/`B`: exp_01's `CONTROL`/`CYL`). `exp06_registry()` takes
   the reviewed registry digest, the model-class map and `METADATA_FIELDS` from
   `tools.exp06_eval`, so an exp_06 run must carry that registry digest, the model class
   its backbone maps to and matching exp_06 output metadata; the approved `epoch_012`
   must itself be epoch 12; a baseline that declares any other epoch is refused, and a
   historical manifest without the field publishes its registered epoch 12.
5. **exp_05 route.** `check_exp05_tier` requires the checkpoint-adjacent `args.json`,
   binds it, compares its digest with both `args_json_sha256` and the `train_args`
   mutable input (and that the binding really points at that file), re-derives `legacy_M`
   and the tier configuration with exp_05's own rules (`exp05_params.TIERS`, `tier_of`),
   compares `param_counts` with the registered `M_cyl` counts, and reconciles all eight
   `TIER_FIELDS` across the manifest, both outputs and the completion.
6. **A3 owner binding.** `job_owner` reads and parses the job root's `launch.pid`,
   requires equality with the completion's `owner_pid`, and retains the binding
   (`{path, pid, sha256}`) for publication. Liveness is deliberately not checked (pid
   reuse in retrospective analysis); no signal of any kind is sent. A missing, unreadable
   or disagreeing `launch.pid` is refused.
7. **Approvals bound to the committed bytes.** Both producers now pass `repo`/`commit`
   into `exp06_profiles.load_approved_digests` for non-exploratory runs, defaulting to
   the repository HEAD and overridable with the new `--approved-commit`. A well-formed
   record outside the repository, or one not tracked at the selected commit, is refused;
   the receipt (path, sha256, repo-relative path, committed_at) is published and
   revalidated with every other input. Exploratory runs keep the unbound read and record
   the labelled deviations.

## Validation

| command | outcome |
|---|---|
| `git diff --check 0ca806c..HEAD` | clean |
| `git diff df7ea24..HEAD --name-only` | the four files above only |
| `python -m py_compile tools/exp06_{compare,summarize_haa}.py tests/test_exp06_{compare,summarize_haa}.py` | ok |
| `pytest tests/test_exp06_compare.py` | 77 passed |
| `pytest tests/test_exp06_{compare,summarize_haa,profiles,approvals_api}.py tests/test_provenance.py tests/test_exp03_record_tools.py` | 265 passed, 2 skipped |
| `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 pytest tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py -q -p no:cacheprovider` | **1207 passed, 8 skipped, 0 failed, 0 xfail** (26 min 14 s) |
| commit trailers | 18/18 commits carry both lines |

The skips are pre-existing environment-gated cases; the real exp_04 arm-A admission
regression and the real exp_02 canonical-cell regression both ran and passed.

## Discrepancies and deliberate choices (never silent)

* **`verify_job(..., sensitivity=...)` defaults to `False`** (confirmatory). The prompt
  left the default open; the fail-closed default means the existing diagnostic fixtures
  had to say `sensitivity=True` explicitly, which is exactly the "diagnostic fixtures
  stay separate from confirmatory admission" the review asked for.
* **`--sensitivity` labels rather than suppresses.** H1/H1b verdicts are prefixed
  `sensitivity: `, `mode` is `sensitivity` in the JSON and the summary carries a
  `SENSITIVITY` banner plus every recipe deviation. H2/D cells are not individually
  relabelled; the banner and `mode` cover the whole document.
* **Arm B's exp_05 tier is registered in `ROLES`, not looked up by checkpoint hash.**
  `EXP05_M` pins `M_cyl`'s role, tier, backbone, sha256 and parameter counts; admission
  requires the evaluated checkpoint to hash to that registered sha256. This keeps the
  synthetic tests able to substitute their own roles table (as they already do for
  checkpoints) without weakening production, where `EXP05_M['sha256']` is exp_01's CYL.
* **`check_exp05_tier` hashes the args.json and then re-reads it to parse.** The two
  reads are adjacent statements with no intervening I/O and the file is bound, so a
  change is caught at publication; it is nonetheless a (tiny) re-read that the stricter
  `_read_json`-style "parse the bytes you hashed" pattern used in the summariser avoids.
  Left as is rather than adding untested code after the green commit.
* **`exp06_registry()` and `tier_fields()` import lazily**, so `tools.exp06_compare`
  still imports without torch until an admission runs; the registry digest is then the
  writer's own (`tools.exp06_eval.registry_sha256`), not a separately maintained pin.
* **The registered dataset now travels in `SPLIT`**, which `analyse` copies into the
  published record (`result['split']`), so the H3 JSON grows the registered identities.
* The template/record approvals file was **not** edited; `code.probe_align` was already
  registered by the previous fix cycle.

## Post-merge wiring the Planner still owes (the review's closure section, plus this cycle)

1. Restore the pending approvals schema in the record copy and keep it byte-identical to
   `tools/exp06_approved_digests_template.json`.
2. Recompute the `code` digests at the **final merged revision** with
   `exp06_profiles.compute_code_digests` and commit the reviewed approvals record — the
   producers now require that record to be a tracked path whose blob at the reviewed
   commit equals the bytes read.
3. Produce the legacy receipt (`--write-legacy-receipt`), approve it, and fill
   `reused.legacy_receipt` (path + sha256).
4. Fill the artefact identities: `artifacts.epoch_012` (path, epoch **12**, sha256),
   `artifacts.heading.<room>` for the four rooms, `artifacts.gate_g1`.
5. Fill the `reused` identities: `exp02_stats_sha256`, `exp02_summary_sha256`,
   `exp04_*` and `exp05_approved_digests_sha256`.
6. Pass explicit `--approved` and `--gate-g1` paths, and absolute `manifest_path` values
   in any command that writes a run (the repo-relative limitation fails closed).
7. New: pass `--approved-commit <reviewed HEAD>` when the analysis runs at a later commit
   than the one the approvals were reviewed at; the default is the repository HEAD.
8. New: never pass `--sensitivity` for the primary §6.2 comparison — it exists only for a
   labelled sensitivity analysis of runs outside the registered recipe.
9. New: the confirmatory HAA jobs must actually run §6.2's recipe (1000/10 and 200/2
   epochs, TF32 on, unseeded Griffin-Lim); `tools/exp06_haa_pipeline.sh` already passes
   `$S1`/`$S2`, so no change is needed there, but any shortened rehearsal will be refused
   from the primary tables.
10. New: every job root must retain its `launch.pid` — the summariser now binds it and
    refuses a job that has lost it.
11. New: arm B of H3 is registered as exp_05's `M_cyl` (exp_01's cylindrical weights). If
    the Planner instead evaluates that checkpoint through `tools/exp06_eval.py`, the run
    takes the exp_06 route and the exp_05 tier checks do not apply; the role's
    `backbone`/`epoch` registration still does.
