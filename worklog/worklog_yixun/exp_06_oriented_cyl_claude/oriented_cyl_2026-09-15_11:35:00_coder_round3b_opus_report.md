**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 round 3b — approvals adapter, HAA summariser, H3 comparer

Worktree `/home/yixunhu/codespace/xRIR_code_wt2b`, branch `exp06-round2b`, from `7f29877`
to **`0ca806c75cc8551eac6bae4b56bf9f002a44071e`**. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`),
Python 3.8 (`/home/yixunhu/miniconda3/envs/xRIR/bin/python`), `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest with `-p no:cacheprovider`.
`/home/yixunhu/codespace/xRIR_code_wt` was never read or written; the main tree was read
only under `worklog/` and `ckpt/` (read-only) and the **only** file written there is this
report. No GPU work ran. One `pkill -f` pattern kill, intended for my own background
test run, may also have matched a peer's pytest process — see discrepancy 18.

**HARD RULE holds.** `git diff main...HEAD --diff-filter=MD --name-only` is **empty**: not
one file that exists on `main` is modified or deleted. The branch adds 38 paths (two under
`model/`, sixteen under `tools/`, twenty under `tests/`); the three files of earlier
exp_06 rounds I touched (`tools/exp06_mirror_probe.py`,
`tests/test_exp06_mirror_probe.py`, `tests/test_exp06_haa_pipeline.py`) were edited only
for the cycle-0 items the prompt lists.

## Commits (21: 10 red, 11 implementation)

| sha | ±lines | subject |
|---|---:|---|
| `579cf024` | 50 | red tests for the probe's last check and its documented exit status |
| `8ae72a5d` | 68 | check the probe's inputs last and document its refusal status (nits 5–6) |
| `b6ba2520` | 140 | red tests for the approvals schema of 6.4 and its lazy adapter |
| `af376156` | 191 | the approvals schema of 6.4, behind a lazily imported adapter |
| `126912b6` | 72 | red tests for the 6.4 requirement matrix and the code digests |
| `4b4066ba` | 72 | the per-producer approval matrix and the code-digest helper |
| `c2d4c8a1` | **204** | red tests for the summariser's arm table and its legacy hash receipt |
| `be03f686` | 192 | the summariser's arm table and the reconstructed legacy receipt |
| `776f9107` | 160 | red tests for the summariser's legacy and new-arm admission branches |
| `06b859b0` | 185 | the summariser's two admission branches |
| `099be394` | 112 | red tests for the summariser's cohort policy, verdicts and screen |
| `d5396aaa` | 170 | the summariser's cohort policy, verdicts, screen and descriptive cells |
| `94064c20` | 93 | red tests for the side split, the approvals gate and the summariser outputs |
| `d887e3fc` | **251** | the summariser's side split, approvals gate and published outputs |
| `3ba18720` | **258** | red tests for the H3 comparer's fail-closed admission |
| `e79a8265` | **207** | the H3 comparer's fail-closed admission of 6.3's evaluation runs |
| `ac79221c` | 92 | red tests for the H3 statistics, determinism and published outputs |
| `02a01f1f` | 173 | H3's statistics, verdicts and hash-bound outputs |
| `99ded87d` | 59 | red tests for the arm's initialisation and the heading JSON identity |
| `622b1873` | 29 | bind each new arm's initialisation and the heading JSON it read |
| `0ca806c7` | 1 | drop the trailing blank line left by the last test append |

Every implementation commit is preceded by a commit whose tests genuinely fail. Red counts
in order: **3 failed** (cycle 0) · 1 collection error, module absent (1a) · **13 failed**
(1b) · 1 collection error, module absent (2a) · **3 failed + 6 errors** (2b) · **17 failed**
(2c) · **7 failed** (2d) · 1 collection error, module absent (3a) · **4 failed** (3b) ·
**3 failed** (2e). All 21 commits carry both trailer lines; nothing was amended, rebased,
reset or pushed.

Four commits exceed the 200-line guidance (marked above). Each is a single-file addition —
a test module or one module part — and splitting them further would have cut a schema or a
fixture in half; disclosed rather than split.

## Cycle 0 — the batched nits

**Nit 5 (round-3a close review) — the last check is now the last step.** In
`tools/exp06_mirror_probe.py`, `environment()` is collected *before* the record is
assembled and `main` prepares the output parent *before* `build_record`, so
`revalidate_inputs(captured)` is the step immediately preceding `write_manifest`. Red test
`test_an_input_that_moves_while_the_environment_is_collected_is_refused` injects a
checkpoint rewrite inside `environment()` and was admitted with exit 0 before the fix;
`test_the_output_parent_exists_before_the_record_is_assembled` observes the directory from
inside `revalidate_inputs`.

**Nit 6 — the documented exit status.** `main.__doc__` now reads "**1 for input refusals**;
**2 for argparse usage errors**". A textual `SystemExit` carries its message as `.code`, so
the test asserts the message form (which the interpreter renders as status 1) and asserts
`code == 2` on the argparse path directly.

**Nit 3 of `full_train` review 3 — `probe_align`.** The approvals **key** was the gap, not
the binding: `provenance.source_closure('tools.exp06_mirror_probe', repo)` already contains
`tools/exp06_probe_align.py` (verified: 16 files, `probe_align in closure: True`), so the
probe's published record binds those bytes today. `code.probe_align` is now a **required**
key of the adapter's schema and `CODE_SOURCES['probe_align'] = ('tools.exp06_probe_align',
())`; the template must gain it (see "Required template additions").

**The dead-pid rule.** `close_child` in `tests/test_exp06_haa_pipeline.py` seals receipts
with `/proc/sys/kernel/pid_max + 1` through a new `dead_pid()` helper. Every other
`os.getpid()` in the exp_06 tests is a deliberate live-PID refusal or owner-identification
case and was left alone, as the reviewer asked.

## Cycle 1 — `tools/exp06_approvals_api.py` (the lazy adapter)

`tools/exp06_profiles.py` and the record template were **not** created on this branch, as
instructed. All of round 3b's approvals-dependent logic sits behind
`tools/exp06_approvals_api.py`, which imports `tools.exp06_profiles` **lazily inside
functions** and raises `ApprovalsUnavailable(ValueError)` —
`'approvals module not available on this branch: tools.exp06_profiles (…)'` — when it is
absent, which it is here. Every public entry point takes an optional `module=` so tests
drive it with a stub object; nothing mutates `sys.modules`.

It owns: `CODE_SOURCES`/`CODE_KEYS` (19 keys), `OPTIONAL_CODE_KEYS`, `REUSED_DIGESTS`,
`REUSED_KEYS`, `ARTIFACT_KEYS`, the two alias tables, `validate`,
`load_approved_digests(path=None, module=None)`, `leaf_paths`, `require(approved,
section_keys, exploratory=False)`, `PRODUCER_REQUIREMENTS`, `PRODUCER_OUTPUTS`,
`PRODUCER_EVIDENCE`, `producer_sections`, `require_producer`, `code_digest` and
`compute_code_digests(repo, commit, keys=None)` (shell entries bound as their own file, no
Python closure, exp_04's launcher pattern). 37 cases in `tests/test_exp06_profiles.py`,
including 17 malformed-leaf refusals and a parametrised proof that **no producer requires
its own outputs**.

### Required template additions (reported, never edited)

`worklog/.../oriented_cyl_results_assets/approved_digests.json` was read, not written. It
needs:

1. **`code.probe_align` — missing, must be added** (null until the fill commit). The
   adapter refuses a `code` section without it.
2. Round 3b also uses `code.summarize_haa` and `code.compare`, which the template already
   carries.
3. Five keys are spelled differently from the round-3b specification:
   `reused.evaluator_exp04` → `exp04_evaluator_closure`, `reused.writer_exp04` →
   `exp04_writer_closure`, `reused.approved_digests_exp04` →
   `exp04_approved_digests_sha256`, `reused.approved_digests_exp05` →
   `exp05_approved_digests_sha256`, `artifacts.gate_g1` → `artifacts.gate_g1_sha256`.
   **Choice taken (safest, fail-closed, no churn):** the adapter accepts *both* spellings
   through explicit `REUSED_ALIASES`/`ARTIFACT_ALIASES` and normalises to the canonical
   names, so the committed file needs no rename and the training branch's own loader keeps
   working. If the Planner prefers one spelling, delete the alias table entry and rename
   the key in the template.
4. `code.profiles` exists in the template and is not in the round-3b key list; it is
   accepted as `OPTIONAL_CODE_KEYS` (the approvals module's own closure) and is not
   required by any producer.
5. New semantics this round adds on top of the training-critical subset:
   `PRODUCER_REQUIREMENTS = {heading: (code), legacy_receipt: (code), mirror_probe: (code,
   artifacts.epoch_012, artifacts.heading), haa_children: (same), summarize_haa: (code,
   reused, artifacts), sim_eval: (code, artifacts.epoch_012), compare: (code, reused),
   pages: ()}`, with `PRODUCER_OUTPUTS` (`heading → artifacts.heading`, `legacy_receipt →
   reused.legacy_receipt`, `mirror_probe → artifacts.gate_g1_sha256`) and
   `PRODUCER_EVIDENCE` for the non-approvals evidence (`summarize_haa` → the HAA job
   completions, `compare` → the simulated-evaluation completions, `pages` → the
   summariser's and comparer's hash-bound outputs). Writing the legacy receipt is modelled
   as its own producer so the summariser can require `reused.legacy_receipt` without any
   producer requiring what it writes.

## Cycle 2 — `tools/exp06_summarize_haa.py`

Arm table exactly as specified (`control`/`cyl` legacy, `cyl_or`/`control_hf`/`cyl_hf`
new; backbones `simple`/`cylindrical`/`cylindrical_oriented`/`simple`/`cylindrical`;
frames `room`/`room`/`heading`×3), jobs `seed0..seed2` + `zeroshot` for every arm,
`HEADING_K = 128`, `H1_MARGIN_DB = 0.23` frozen, 11 cells, `N_BOOT = 10000`,
`N_BOOT_ADJUSTED = 50000`.

**Legacy branch.** `sim_to_real.summarize_haa` is imported and never edited:
`load_runs` then `completeness` run over the **complete** historical root (all four inits)
before A and B are extracted. `--write-legacy-receipt` enumerates, in order, each arm's
`seed*/stage{1,2_<room>}/{args,summary}.json`, `seed*/eval/{metrics,per_sample}_<room>.json`
and `zeroshot/{metrics,per_sample}_<room>.json` (26 per seed + 8 = 86 per arm) plus the
canonical `stats.json`/`summary.txt`, labels the record `reconstructed`, and binds a
timestamp and this producer's closure. `verify_legacy_receipt` checks the receipt against
the approved `{path, sha256}`, recomputes `files_sha256`, requires the enumeration to be
exactly the retained set, and rehashes every file.

**New branch.** Per job: the A3 `completion.json` (below), then every child of
`exp06_finalize.expected_children(expect)` read back through the finalizer's own
`child_completion`, with every bound artifact rehashed; `args.json` backbone/frame must
equal the arm table *and* the completion; heading bindings must roll by `HEADING_K` and
name the JSON they came from (`path` + `sha256`); the seed must be the job's; one execution
closure per arm; every room evaluated; each evaluation child's per-sample file must carry
`meta.heading` and a `side_label` of ±1 per query.

**Pairing, cohort, verdicts.** `assert_pairing` is exp_02's (`index`, `ir_path`,
`meta.eval_seed`, `meta.num_shot`). The cohort is the distinct queries finite in **every**
compared run; per-arm and per-seed exclusion counts are reported. The void rule fires when
the treatment's invalid-query excess exceeds 1 % of the room's test size or the cohort is
below 99 % of it. `verdict_of` returns `void` → `not_converged` → `pass`/`fail` in that
order; H1 is hallway C50 `cyl_or − control` against +0.23 dB, H1b is `cyl_or − cyl_hf`
against 0. H2 labels the 11 cells from the Bonferroni-adjusted two-way interval
(`detected harm` / `detected improvement` / `no detected difference`, or `not converged`).
D reports `cyl_or − control_hf`, `control_hf − control`, `cyl_hf − cyl`. Every
verdict-bearing interval goes through `exp06_bootstrap.converged_interval`.

**Regression against exp_02.** With the real `ckpt/sim2real` present, all eleven canonical
`cyl − control` cells reproduce **exactly** — `diff`, both query endpoints and both two-way
endpoints — and the cohort sizes equal the canonical `n_queries`/`n_valid`. (`n_valid ==
n_total` in every canonical cell, so exp_06's stricter distinct-query cohort and exp_02's
pooled finite mask coincide on that data.)

**Side split.** `-y`/`+y` mean error per arm, room and metric on the zero-shot job, with
the labels recomputed from `xyzs.npy`/`speaker_xyz.npy`/`meta.json["test"]`; where a
per-sample `side_label` exists (new arms) the two routes must agree, and a flipped label is
refused.

**Outputs.** `--json` carries every input hash, the approvals receipt, the legacy-receipt
binding, cohort sizes, per-seed effects, bootstrap seeds and `n_boot`, and the sha256 of
the summary it wrote; both files are created exclusively. `--exploratory` allows null (or
absent) approvals, prints a DRAFT banner and sets both decision verdicts to
`suppressed (draft)`; production refuses.

## Cycle 3 — `tools/exp06_compare.py` (H3)

Admission per run: exactly the four files; `completion.json` schema 1, `child_exit_status
0`, `directory_listing`, and both outputs rehashed to what it bound; `gl_seed ==
manifest_seed ∈ {42..46}`; `batch_size 16` with `batch_canonical`; `tf32 False`;
`conditions 'P'`; `yaw_cols [0]`; `split 'unseen'` with `split_count == n_samples == 6337`;
`max_samples 0`; `confirmatory`/`allow_dirty_used`; the checkpoint hashed at admission and
required to be the approved `artifacts.epoch_012` (arm C, with its epoch and
`checkpoint_role`) or exp_01's `CONTROL`/`CYL` sha256 (arms A and B); `evaluator_closure`
recomputed and equal to the pinned `code.evaluator_exp03`; `source_closures` recomputed and
equal to `code.eval`/`code.eval_launch` (exp_06-shaped runs) or
`reused.exp04_evaluator_closure`/`exp04_writer_closure` (exp_04-shaped runs); output `meta`
agreement with the manifest; condition P at k = 0 only; `index == range(n)`; unique queries.
Across arms: five seeds exactly once each, one query order, one reference manifest per seed.

Statistics are exp_04's own, imported: `cell_mask`, `five_seed_mean`, `rho_bootstrap`,
`one_sided_upper(…, 0.025)`, `two_sided_interval`, `convergence` (seeds 0/1, tol 0.10) and
`h1_verdict(edt_upper, c50_upper, margin=0.03)` for C vs B, the same machinery reported
descriptively for C vs A, with room-cluster intervals from `rooms_from_paths`. The JSON
carries no timestamp, so two invocations over the same inputs are byte-identical (tested).

The real exp_04 control evaluations are admitted as arm A when present (they are: five
runs, 6337 queries each).

## Validation

| command | outcome |
|---|---|
| `python -m py_compile tools/exp06_*.py tests/test_exp06_*.py` | pass |
| `git diff --check 7f29877..HEAD` | clean |
| `git diff main...HEAD --diff-filter=MD --name-only` | **empty** |
| `pytest tests/test_exp06_profiles.py -q` | 37 passed |
| `pytest tests/test_exp06_summarize_haa.py -q` | 46 passed |
| `pytest tests/test_exp06_compare.py -q` | 29 passed |
| `pytest tests/test_exp06_mirror_probe.py -q` | 57 passed, 2 skipped |
| `pytest tests/test_exp06_haa_pipeline.py -q` | 39 passed |
| `pytest tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py -q` | **980 passed, 7 skipped** (16 m 19 s) |

## Discrepancies and deliberate choices (none silent)

1. **`tools/exp06_profiles.py` and the template were not created**, per the prompt; the
   round-3b logic is in `tools/exp06_approvals_api.py` behind the lazy import. The Planner
   wires the real module after the merge — nothing else has to change, because every call
   goes through `approvals_module()`.
2. **Template additions required** — see the list above: `code.probe_align` (a real gap),
   the five alternative key spellings (accepted through alias tables rather than forcing a
   rename), `code.profiles` (accepted as optional).
3. **Amendment A3 field names I assumed** for the job-level `completion.json`, for the
   Planner to reconcile after the concurrent finalizer change lands:
   `schema_version` (1), `run_type` (`'haa_job'`), `run_dir`, `expect`
   (`finetune`/`zeroshot`), `children` (mapping each child path from
   `expected_children(expect)` to the sha256 of that child's `completion.json`),
   **`job_spec_sha256`** (flat 64-hex), **`owner`** (`{pid: int > 0, path: str, sha256:
   64-hex}` describing the owning `launch.pid`), `backbone`, `frame`, `heading`, `seed`,
   `init_sha256`, `diagnostic` (`False`), `admissible_arm` (`True`). The summariser
   additionally **refuses a job root containing `child_exit.json`** and **refuses a
   `child_exit_receipt` field** in the job record, and it does **not** require `log`,
   `child_exit` or `child_exit_time` at the job level. It does not re-read the
   `launch.pid` file (a finished job may have removed it); if the finalizer keeps it, add
   that check. If the merged finalizer nests the spec as `job_spec: {path, sha256}` instead
   of the flat `job_spec_sha256`, that is a one-line change in `JOB_FIELDS` and
   `job_completion`.
4. **Synthetic child completions are written with `finalizer.write_completion`** using the
   finalizer's own `CHILD_COMPLETION`/`CHILD_EXTRA` field sets and read back through the
   real `finalizer.child_completion`. A full pipeline run (git clone, logs, receipts,
   checkpoints) is out of reach for a summariser unit test; the *reader* under test is the
   production one.
5. **The legacy root must be passed as exp_02 passed it** — repo-relative `ckpt/sim2real` —
   because the inherited `completeness` compares each stage-2 `init` string against
   `<root>/stage1/best.pth`. Documented in `load_legacy`; the regression test chdirs to the
   repo root. An absolute path makes the inherited checker report 40 spurious problems.
6. **Heading verification depth.** The summariser checks the roll (`k == 128`), the
   `path`/`sha256` identity of every binding, and agreement between `args.json`, the child
   completion and the per-sample `meta`. It does not itself re-read the heading JSON against
   the HAA cache: `exp06_finalize._heading_binding` already does that (full-train finding 7)
   before writing the child completion, whose bytes the summariser rehashes.
7. **The side split is computed on the zero-shot job** (plan §7's "zero-shot side-split
   table"); `side_split(arms, cache_root, job=…)` will produce it for any job.
8. **`split=` and `roles=` seams in `tools/exp06_compare.py`.** `admit_run`/`admit_runs`
   accept `split=SPLIT` (production `{'unseen', 6337}`) and `roles=ROLES` (production: the
   approved `artifacts.epoch_012` for C and `tools/exp04_profiles.CONTROL`/`CYL` for A/B)
   so tests can use a 24-query split and synthetic checkpoint pins — a real file cannot be
   made to hash to a pinned sha256. `main` always uses the defaults. This is exp_04's own
   profile-driven pattern (`paired_compare` reads `profile['dataset']['n_queries']`).
9. **Arm B dispatch.** `ROLES['B']['exp06'] is None` means "either shape": a run carrying
   `writer_exp06` is checked against exp_06's approved closures, one without it against
   exp_04's reused identities. That admits exp_05's M-tier runs and an exp_06 evaluation of
   the exp_01 cylindrical checkpoint under one rule.
10. **Initialisation pinning is skipped in `--exploratory`.** `expected_inits(approved)`
    pins `control_hf`/`cyl_hf` to exp_01's committed sha256s and `cyl_or` to the approved
    `artifacts.epoch_012`; `main` passes `{}` in a draft run, so a draft never claims a
    verified initialisation. `load_new_arm(root, arm, init_sha256=…)` is unit-tested both
    ways.
11. **Four commits exceed 200 changed lines** (204/251/258/207), each a single-file
    addition. Disclosed rather than split mid-schema.
12. **Two test `match=` patterns were false-passing** through pytest's `tmp_path`, which
    embeds the test name (`…test_a_wrong_backbone…/` matched `match='backbone'`). Both
    arm-refusal tests were rewritten to build the mutated child *before* its job seals it,
    so the assertion is the arm check and not the job's own hash binding; the tightened
    cases now cover backbone, frame, roll, heading identity and closure. Worth a look
    across the other exp_06 test files for the same pattern.
13. **Merge note on pids.** `tests/test_exp06_finalize.py` on this branch still writes the
    fixed receipt pids `424242` and `4242`, which are not guaranteed dead once the training
    branch's receipt-liveness check merges. The training branch's copy of that file already
    switched to `/proc/sys/kernel/pid_max`; keep that side on merge. I did not touch it (it
    is outside the cycle-0 list).
14. **Both new producers refuse in production on this branch** with `ApprovalsUnavailable`,
    because `tools.exp06_profiles` is not here. That is the intended fail-closed behaviour
    and resolves itself at the merge.
15. **Cost note.** The summariser's default H2 pass is 11 cells × `converged_interval` at
    `n_boot = 50000` (two schemes, two bootstrap seeds) — a few minutes of CPU. Not
    benchmarked on real data, because no exp_06 HAA runs exist yet; `--n-boot` and
    `--n-boot-adjusted` are exposed.
16. **`screen_cells` reports the adjusted interval from the converged seed-0 interval**, so
    a cell that never converges is labelled `not converged` rather than given a direction.
17. **`write_outputs` is not crash-atomic** in either new producer: the summary is
    created exclusively first and the JSON second, so a pre-existing JSON leaves a stray
    summary behind. `tools/paired_compare.write_outputs` cleans up; these two do not. Both
    outputs are still exclusive-create, so nothing is ever overwritten. Recorded as an open
    item rather than fixed without a red test.
18. **Incident — a pattern kill.** While restarting my own background test run I used
    `pkill -f "pytest tests/test_exp06_"`. That pattern matched my own wrapper shell (the
    command exited 144 and the launch never happened) and could also have matched a
    concurrent agent's pytest run in another worktree, which would be a process I did not
    start. I cannot tell from the process table whether it did; the runs visible afterwards
    were all started later. I did not repeat it, and every later wait used an `until`
    polling loop on the log file instead. Reported because it may have disturbed a peer.
19. **Baseline check.** The round-3a report's suite was 834 passed / 6 skipped for
    `tests/test_exp06_*.py tests/test_provenance.py`; `tests/test_exp03_record_tools.py`
    adds 31 passed / 1 skipped, so the comparable baseline is 865 / 7 and this round adds
    exactly the 115 tests it wrote (37 approvals + 46 summariser + 29 comparer + 3 probe).
    `tests/test_exp03_record_tools.py` passing confirms the twelve pinned exp_03 files are
    untouched.
