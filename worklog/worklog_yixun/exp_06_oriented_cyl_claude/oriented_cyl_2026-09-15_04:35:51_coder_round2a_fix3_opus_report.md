**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 oriented_cyl — round 2a, fix cycle 3 (Codex close review 2, findings 1–5)

Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-window`, from `bf51484` to
`a977a68`. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`,
`PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt`,
`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONDONTWRITEBYTECODE=1`,
`NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`), no GPU work, no process signalled, nothing written
under `ckpt/` or the main tree except this report. Only `tools/exp06_finalize.py` and
`tests/test_exp06_finalize.py` were touched this cycle;
`git diff main...HEAD --diff-filter=M --name-only` is empty (no file that exists on `main` was
modified). The superseded archive folder and the `archive/codex-round2a-2026-09-15` branch were
not read.

## 1. Commits (changed lines = insertions + deletions; every one ≤ 200)

| # | Commit | Changed | Subject |
|---|---|---:|---|
| 1 | `7c5b74f` | 30 | red tests for a live child inside a job (finding 1) |
| 2 | `77946c8` | 6 | refuse a live child during job admission (finding 1) |
| 3 | `1be0997` | 52 | red tests for unbound child receipts and evidence (finding 2) |
| 4 | `2741f37` | 51 | bind every child's receipt and evidence to the re-run (finding 2) |
| 5 | `d8005a9` | 63 | red tests for unbound per-sample results (finding 3) |
| 6 | `961d7e0` | 27 | bind per-sample results to their room and protocol (finding 3) |
| 7 | `32db42a` | 29 | red tests for malformed metric values (finding 4) |
| 8 | `d94bf1b` | 76 | type and recompute every metric value (finding 4) |
| 9 | `4a9f40c` | 30 | red CLI tests for nested job-schema failures (finding 5) |
| 10 | `e7fb365` | 20 | type the whole job schema before it is used (finding 5) |
| 11 | `a977a68` | 4 | document where a pipeline's launch.pid belongs (finding 1) |

Maximum 76 changed lines; every commit carries the two required trailer lines. Each
implementation commit is preceded by its own red-test commit (TDD pairs 1/2, 3/4, 5/6, 7/8,
9/10); commit 11 is documentation only and changes no behaviour.

## 2. Red counts (measured at the red commit, before the paired implementation)

| Red commit | Failing cases | Which |
|---|---:|---|
| `7c5b74f` | 2 | `test_a_live_child_is_never_certified_by_a_job[child.pid, launch.pid]` |
| `1be0997` | 6 | `…must_bind_what_the_job_validated[receipt_redirected, receipt_pid, exit_time, closure_digest, closure_absent]`, `…registry_and_head_they_ran_under` |
| `d8005a9` | 11 | `…bound_to_the_room_and_protocol[8 cases]`, `…another_protocol[3 cases]` |
| `32db42a` | 9 | `…parsed_and_cross_checked[count_string, count_negative, count_object, count_boolean, count_above_n, median_object, median_wrong, summary_extra_key]`, `…never_count_as_missing[values0]` |
| `4a9f40c` | 4 | `…named_cli_refusals[rooms_mixed, run_dir_null, init_null, heading_out_of_range]` |

Two added cases were green before their fix and are kept as guards, stated here so the count is
not overclaimed: `…parsed_and_cross_checked[n_boolean]` (`n_samples=True` already failed the
equality with 3 entries) and `…never_count_as_missing[values1]` (three NaN with a non-null mean
already failed the old mean check).

## 3. Per finding: fix and regression

**Finding 1 — child liveness in job admission.** `verify_child` now calls
`refuse_live_launch(path)` before it reads any of the child's evidence, with no owner exception:
a live `child.pid` or `launch.pid` under any child refuses the job. The job's own directory keeps
the `--owner-pid` exception in `finalize`, which runs first, so the declared owner can certify
only once every child is dead. Regressions: a valid nine-child job plus `stage1/child.pid` (and,
separately, `stage1/launch.pid`) containing the test's own pid is refused with `alive`, both with
and without `--owner-pid` naming that same live process; the companion test shows the job's own
live `launch.pid` plus a dead `stage1/child.pid` still finalizing.

**Finding 2 — receipt/evidence binding.** `rehash_bound_evidence` now (a) re-validates the closed
log and requires the freshly hashed digest to equal the one the completion bound, (b) requires
`completion.child_exit_receipt.path` to *resolve* to the canonical `<child>/child_exit.json`,
(c) compares the binding's `sha256`, `child_pid`, `started_at` and `ended_at` field by field with
the canonical receipt that validation returned, and (d) requires `child_exit_time` to equal both
the validated marker timestamp and the receipt's `ended_at`. The receipt's `status` is bound to
the completion's `child_exit` by `child_exit_receipt(path, record['child_exit'], …)` (the
completion records the status as `child_exit`, not inside the receipt binding), and
`child_completion` still requires `child_exit == 0`. `verify_child` now compares the **whole**
generated evidence mapping (`sorted(set(evidence) - {'artifacts'})`, each field required to be
present) instead of the old `CHILD_EVIDENCE + CHILD_EXTRA` subset, so `source_closure_sha256`,
`epochs`, `samples` and any future field are bound; the constant `CHILD_EVIDENCE` is gone.
Both HAA evidence builders now return a shared `child_identity(record)` =
{`source_closure_sha256`, `registry_sha256`, `git_head`}, and `haa_child_arguments` requires the
recorded `registry_sha256` to equal `registry_sha256()` recomputed from `BACKBONES_EXP06` at
finalisation (previously only the `full` run type checked it). Regressions: the three review
reproductions — receipt binding redirected to `stage1/args.json` with that file's correct hash
while the canonical receipt's `child_pid` changes, `child_exit_time: "never"`, a false
`source_closure_sha256` — plus a removed `source_closure_sha256`, a completion whose bound
receipt `child_pid` is off by one, and an HAA child whose provenance registry digest is false.

**Finding 3 — per-sample room/protocol binding.** `haa_eval_evidence` now requires
`per_sample['index']` entries to be the writer's non-negative integers, `ir_path` to be a list of
exactly `'{room}/{index[i]}'` (what `sim_to_real/eval_haa.py` line 96 writes), and the per-sample
`meta` to record `split`, `num_shot` and `eval_seed` equal (type-strictly) to `args.json` —
`room` too when the writer records one. `metrics_<room>.json` already had to agree with `args` on
`backbone`, `checkpoint`, `room`, `split`, `num_shot`, `eval_seed`, so all three protocol fields
are bound on both files. The success fixtures now emit realistic writer output: `write_haa_eval`
derives `ir_path` per room (the old `x/0, x/1, x/2` is gone) and `eval_meta` carries the real meta
keys. Regressions: the review's altered hallway file (`class_room/<i>` paths, `split="val"`,
`num_shot=1`, `eval_seed=777`) refused on the standalone evaluation path (8 cases) and on the job
path (3 cases), where the child completion carries the correct hash of the altered file.

**Finding 4 — malformed metric values.** `haa_metrics` now requires `n_samples` to be a strict
`int` equal to the per-sample count; each of `c50_outliers`, `t60_invalid`, `edt_invalid` to be a
strict `int` in `[0, n_samples]`; every per-sample metric element to be `int`/`float` with `bool`
excluded (NaN stays the writer's invalid measurement) *before* the finite values are filtered,
via `_numbers`; and each summary block to be exactly the `{mean, median, n}` that
`eval_xRIR_backbone.summarize` writes, with `n` an `int` equal to the recomputed count, `mean`
and `median` finite and equal to the recomputed values (`statistics.median`, numpy's convention)
when `n > 0`, and both `null` when `n == 0`, via `_summary_block`. The per-sample element check
now also runs for `t60` in `NO_T60_ROOMS`, where only the `null` summary is required. The test
helper's median was `sorted(finite)[len//2]`, which is not what `np.median` writes for an even
count; it now uses `statistics.median` (fixture correction, not a contract change). Regressions:
`c50_outliers: "nonsense"`, `edt_invalid: -5`, `t60_invalid: {}`, `edt_invalid: true`,
`c50_outliers: 4` with three samples, `edt_error_s.median: {"nonsense": true}`, a wrong median, a
summary with an extra `std` key, and per-sample EDT `["oops", {}, true]` paired with
`n=0, mean=null, median=null`.

**Finding 5 — nested job schema.** `load_job_spec` now requires `init` to be a non-empty string,
`rooms` to be a list of strings *before* it is sorted (so `["hallway", 1, null, {}]` is a named
refusal, not a `TypeError`) and equal to the pipeline rooms, and every heading roll to be a
strict `int` in `[0, 512)`; `child_completion` requires a non-empty string `run_dir` before
`Path()` sees it. Regressions run through `main()` (exit status 2, named cause on stderr, no
`completion.json`): `rooms=["hallway", 1, null, {}]`, child `run_dir: null`, `init: null`, and a
heading roll of 512.

## 4. Validation

All commands in the `xRIR` env (`/home/yixunhu/miniconda3/envs/xRIR/bin/python`, 3.8) from the
worktree, `-p no:cacheprovider`.

- Baseline before any change: `pytest tests/test_exp06_finalize.py -q` → **220 passed** (207 s).
- Per-cycle red runs: the counts in §2 (each red commit's failures are listed there).
- Per-cycle green runs after each implementation commit: fix 1 `-k "live_child or job_owner or
  job_refusals or job_completion or zeroshot"` → 27 passed; fix 2 `-k "job or haa"` → 103 passed;
  fix 3 `-k "per_sample_results or another_protocol or haa_eval or dampened or job"` → 77 passed;
  fix 4 `-k "metrics or corrupt_per_sample or haa_eval or dampened or per_sample_results"` → 51
  passed; fix 5 `-k "nested_job_schema or job_spec"` → 15 passed.
- `python -m py_compile tools/exp06_finalize.py tests/test_exp06_finalize.py` → OK.
- `bash -n tools/exp06_launch.sh` → OK (file untouched this cycle).
- `git diff --check bf51484..HEAD` → clean.
- `git diff main...HEAD --diff-filter=M --name-only` → empty.
- Final, at `a977a68`: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_exp06_*.py
  tests/test_provenance.py -q -p no:cacheprovider` → **523 passed, 3 skipped** (7 m 40 s; the 3
  skips are the CUDA-only cases). The same command at `e7fb365` (before the documentation commit)
  also gave 523 passed, 3 skipped. Previous round's total was 488 passed, 3 skipped; 35 cases
  were added.
- Working tree clean at `a977a68`.

## 5. Discrepancies, choices and consequences

1. **A live `launch.pid` inside a child now refuses the job (operational consequence for round
   2b).** The fix prompt is explicit that the owner exception applies only to the job's own
   `launch.pid`, so `tools/exp06_haa_pipeline.sh` must write its pid at the **job root**
   (`seed<s>/launch.pid`) and give children only `child.pid`; if it copies the launcher pid into
   each child directory (the pretraining launcher's per-attempt pattern), job-level finalization
   will refuse while the pipeline is alive — which is by construction, since the pipeline is the
   process that invokes the finalizer. This is documented in the module header (commit
   `a977a68`) and is the fail-closed reading of the instruction.
2. **`registry_sha256` and `git_head` added to HAA child evidence.** The prompt lists both among
   the "generated child-evidence fields" that must equal the freshly derived evidence, but the
   HAA evidence builders generated neither. Rather than compare fields that do not exist, I made
   them real: `haa_child_arguments` recomputes `registry_sha256()` and refuses a child whose
   provenance disagrees, and `child_identity` puts both into every HAA child's completion. This
   widens the HAA child completion schema by two fields (plan §6.4 lists the required evidence,
   not an exclusive set) and required one fixture line (`haa_provenance` now records the real
   registry digest instead of `'a' * 64`).
3. **Full-set evidence comparison.** `verify_child` compares every key the role validator
   generated except `artifacts` (still compared item-wise in both directions), and requires each
   to be present in the completion. Any evidence field added in round 2b is therefore bound
   automatically, and `CHILD_EVIDENCE` was deleted as dead.
4. **Summary keys are exact, not a subset.** `eval_xRIR_backbone.summarize` writes exactly
   `{mean, median, n}` (no `std`), so `_summary_block` requires that exact key set; a block with
   an extra key is refused. The prompt's "`mean`/`median`/`std`…" is answered by the writer,
   which emits no `std`.
5. **Mean and median are recomputed for every metric, including the unfiltered ones.**
   `sim_to_real/eval_haa.py` filters non-finite values only for `edt`/`c50`/`t60` and passes
   `env`/`stft_mse`/`loss` through unfiltered, so a NaN in one of the latter would make the
   writer's `n` the full length while this checker recomputes over the finite values and refuses.
   That is deliberate (fail-closed: a NaN aggregate is not admissible evidence) and cannot arise
   from the current writer, whose `env`/`stft_mse`/`loss` are always finite.
6. **Counts are bounded, not cross-checked against the NaN counts.** `edt_invalid` and
   `c50_outliers` happen to equal the number of non-finite per-sample values in the current
   writer, but `measure_edt` could in principle return NaN without raising, and `t60_invalid` is
   0 for `NO_T60_ROOMS` although every `t60` is NaN there. Requiring `0 ≤ count ≤ n_samples`
   (the review's "sensible bounds") avoids a false refusal on a genuine run.
7. **Unchanged from earlier cycles, still open:** the round-2b entries `tools/exp06_haa_finetune`
   and `tools/exp06_haa_eval` do not exist yet, so the HAA paths are still exercised against a
   stub closure repo; `--job-spec` is consumed here but nothing writes it yet (round 2b must
   write it before execution and bind it); and the finalizer's `haa_job` branch validates whatever
   `--children` it is given, so the launcher must pass exactly the directories it created.
