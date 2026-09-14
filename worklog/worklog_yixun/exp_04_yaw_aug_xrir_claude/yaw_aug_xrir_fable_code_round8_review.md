**Reviewer:** Claude Fable 5.1 (Agent subagent, model fable) · **Date:** 2026-09-14

# exp_04 yaw_aug_xrir — code review, round 8 (result generators, binder, checker)

**Scope.** Commits `f41473c`, `929ab47`, `4827cc1`, `4f0abb6` (`git diff bb9fa91..4f0abb6`, 575 insertions): `worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/yaw_aug_xrir_results_assets/{make_results_md.py, make_results_html.py, bind_provenance.py, check_record.py, static_checks.sh, README.md}` and `tests/test_exp04_record_tools.py`. Specification: `coder_prompts/round8_prompt.md`, plan §3 (verdict wording) and §8, the SOP results-page rule, exp_03's generators/binder as the model, producers `tools/paired_compare.py` and `tools/results_table.py`.

**Method and environment.** Isolated clone at `4f0abb6` under the session scratchpad (`…/scratchpad/e4r8/e4r8_clone`), xRIR interpreter, `PYTHONPATH=<clone>`, **CPU only (`CUDA_VISIBLE_DEVICES=''`), no GPU work, no process signalled**. The real repository was not modified except for this file; the real `ckpt/` was only *read* (`json.load` of one completed eval run's manifest/completion/metrics and of the final attempt's manifest/completion, `ls`, `readlink`) to check the binder's shape assumptions against live artefacts. Every changed line was read; the coder's tests were replayed; real producer JSON was built from the producers' own fixtures (default and a non-degenerate perturbed variant) and both generators, the binder and the checker were attacked through their CLIs and module entry points.

---

## Findings

### 1. BLOCKER — module-name collision with exp_03's pinned record tools breaks the full test suite
- **Where.** `tests/test_exp04_record_tools.py:14-15` (`sys.path.insert(0, str(ASSETS))`) and every `importlib.import_module('make_results_md' | 'make_results_html' | 'bind_provenance' | 'check_record')` (lines 42, 56, 85-86, 150, 163, 189, 206); `make_results_html.py:4` (`from make_results_md import …`); `check_record.py:5` (`from bind_provenance import collect, require`).
- **What.** exp_03's pinned `tests/test_exp03_record_tools.py:13-14` inserts *its* assets directory and imports `check_sweep_acceptance`, which imports exp_03's `bind_provenance` at collection time; `sys.modules['bind_provenance']` is then exp_03's for the whole pytest session. Verified: `pytest tests/test_exp03_record_tools.py tests/test_exp04_record_tools.py` → **13 failed** (all 13 binder params: `check_record.py:5: ImportError: cannot import name 'collect' from 'bind_provenance' (…exp_03…/bind_provenance.py)`); reverse order → **29 failed** (additionally `module 'make_results_md' has no attribute 'load'/'INPUTS'` because exp_03's assets directory ends up first on `sys.path`). exp_03's tests pass in both orders (28 passed, 4 skipped), so the fix must be on exp_04's side (exp_03's files are pinned).
- **Why it matters.** The SOP's rung-2 full suite (`python -m pytest tests`) and every later integrative review run the whole `tests/` directory; collection of exp_03's module makes every exp_04 record test fail. The coder's "131 passed" is the three-file subset only, and `static_checks.sh` runs only `tests/test_exp04_record_tools.py -k refusals`, so the collision was never exercised.
- **Minimal fix.** Never import the record modules by bare name. In the assets add a by-path loader that caches under a unique key in `sys.modules`:
  ```python
  import importlib.util, sys
  from pathlib import Path
  def _sibling(name):
      key = 'exp04_record_' + name
      if key not in sys.modules:
          spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(name + '.py'))
          module = importlib.util.module_from_spec(spec); sys.modules[key] = module; spec.loader.exec_module(module)
      return sys.modules[key]
  ```
  `make_results_html.py`: `_md = _sibling('make_results_md'); arguments, display, tables = _md.arguments, _md.display, _md.tables`. `check_record.py`: `_binder = _sibling('bind_provenance'); collect, require = _binder.collect, _binder.require`. Test file: drop the `sys.path.insert`, load the four modules with the same loader keyed on `ASSETS / (name + '.py')` (the checker's `collect` must be the *same* function object as the test's `binder.collect`, i.e. the same cached module, or `monkeypatch.setattr(binder, 'load_approved_digests', …)` in `test_binding_roundtrip_and_refusals` no longer reaches the checker). Add a regression that runs with exp_03's assets directory already on `sys.path` and exp_03's `bind_provenance` in `sys.modules`, and add the two-file run (both orders) to `static_checks.sh`. Verify with `python -m pytest tests -q` on CPU.

### 2. SHOULD-FIX — the binder does not bind the canonical producer outputs, so a re-signed tamper is undetectable
- **Where.** `bind_provenance.py:59-89` (`collect`) records runs, attempt, probe receipt, audit, the approval blob and HEAD; nothing records the five canonical JSONs, their `.provenance.json` sidecars or their summaries (`H*_K*.txt`, `model_comparison.md`).
- **What.** The generators trust the sidecar as the sole root (`make_results_md.py:30-36`). Verified: editing an estimate in `H2_K8.json` and re-signing `outputs[<json>]` in its sidecar is accepted by both generators (`[tamper_resigned] rc=0`), and `check_record.py` has nothing to compare it against; the only trace is the identity line in the generated pages, which are regenerable. exp_03's binder recorded the canonical gate JSON/summary digests for exactly this reason, and the SOP asks for "binding … as in exp_03".
- **Minimal fix.** `--canonical PATH …` (the five JSONs): for each, `stamp(json)`, `stamp(json + '.provenance.json')` and every path in the sidecar's `outputs`; store under `canonical` in the report (the checker then recomputes it for free). Also require that each sidecar's `inputs` digests for the bound run files (eval_manifest/completion/per_sample/metrics) equal the report's run digests — that closes the chain runs → sidecar → JSON → page. Recommended in the same fix round, before `check_record` is used as the record's acceptance check.

### 3. SHOULD-FIX — presentation is a raw JSON dump (fitness, not correctness)
- **Where.** `make_results_md.py:18-19` (`display` = `json.dumps`), `:82-84` (whole nested dicts as single cells for exclusions / seed_means / convergence / protocol), `:114` (`html.escape(..., quote=True)` inside Markdown cells); same content in the HTML.
- **What.** Every number is faithful (verified exhaustively, see below) but rendered as 17-significant-digit floats (`0.01581138830084184`), `null`/`true`/`false` cells, and JSON blobs with `&quot;` entities in the Markdown; the TABLE block prints the same JSON values with more digits than the producer's own `model_comparison.md` (6 significant digits); there is no verdict block or legend — the reader finds "aggregate verdict: …" only in a section title. This is far from the SOP's "maximally human-readable" page and from exp_03's page.
- **Minimal fix (display only, no arithmetic).** A `fmt(v)` = `'{:.6g}'.format(v)` for floats (optionally `'{:+.2f} %'.format(100*v)` for ratio columns, if the Planner wants the plan's units, with a round-trip test `abs(float(shown)/100 − v) ≤ 0.5e-4`); `quote=False`; a verdict block per family above the tables listing the JSON verdict strings; `exclusions` flattened to columns (`n`, `baseline_invalid`, `newly_invalid`, `excluded`, `valid` per arm and joint), `convergence` to (`gate`, `passed`, `movement`, `width`, `ratio`, `tolerance`), protocol as one `Field | Value` row per scalar. Keep the byte-stability and traceability tests. Planner's call; `_results.md` is a committed deliverable, so I would do it while the chain runs.

### 4. SHOULD-FIX — the binder never looks at the run directory's actual contents
- **Where.** `bind_provenance.py:38-39` compares `completion['directory_listing']` with the completion's own `outputs`, never with the directory.
- **What.** Verified: an extra `extra.json` in a run directory is accepted. `tools/paired_compare.py:293-295` refuses `run directory contents` ≠ the four files, so the binder can bind a directory the producers would refuse (or one that gained a file after analysis).
- **Minimal fix.** `require({p.name for p in directory.iterdir()} == set(completion['directory_listing']) | {'completion.json'}, 'run directory contents')` (train: against `directory_listing` keys plus `completion.json`).

### 5. NIT — `check_record` accepts any existing commit as `git_HEAD`
`check_record.py:12` takes HEAD from the report; `bind_provenance.py:86` only checks the commit exists (verified: swapping in `HEAD~1` passes; a bogus HEAD surfaces as a `subprocess.CalledProcessError` traceback rather than a refusal). Cheap strengthening: also require `git rev-parse <HEAD>:<approved relpath>` == `approved_digests.git_blob`, and wrap the git error into `ValueError`.

### 6. NIT — only the JSON's own digest in the sidecar is verified
`make_results_md.py:30` checks `side['outputs'][<json>]`; the sibling summary (`H*_K*.txt`) is not verified although exp_03 bound its summary hash. For paired_compare the summary is immutable (exclusive create), so `all(sha(p) == d for p, d in side['outputs'].items())` is safe; skip the `.md` for TABLE_V1 (living table, `--force-md`).

### 7. NIT — `--diag` has no duplicate guard
`make_results_md.py:96`: passing an already-listed JSON duplicates its tables and figures and repeats its digest in the identity line (verified: `--diag TOST_K8.json H2_K8.json` → 7 figures, two "TOST_K8 — protocol" sections). Refuse duplicate resolved paths.

### 8. NIT — `static_checks.sh` scope
`static_checks.sh:9` runs `git diff --check` over the whole working tree (it failed for the coder on the live session log; the real tree happened to be clean when I ran it read-only) and hardcodes the repo path as exp_03's did. Scope it to the record assets and the test file (or `bb9fa91..HEAD`), and add the cross-module isolation run from finding 1.

### 9. NIT — binder runtime
`snapshot` revalidates each eval run's full data inventory (≈ 16 s per run per the round-7 note → ≈ 16 min for ~60 runs) and the training run's 296k-file inventory; `check_record` repeats all of it. Acceptable for a one-off; `paired_compare` avoids the repeat with a per-path cache (`data_stats`).

### 10. NIT — verdict vocabulary not asserted
`make_results_md.py:49-50` accepts any non-empty string. Verbatim rendering is correct per spec; asserting the plan vocabulary ({non-inferior, non-inferior on EDT only, non-inferior on C50 only, not shown}, {supported, partially supported, not supported}, {equivalent, equivalence not established}) is cheap defence in depth.

### 11. NIT — README operational notes
State that inputs must stay at the absolute paths recorded in their sidecars (a copy moved elsewhere is refused: `[moved_copy] provenance sidecar mismatch`), so pages are generated from the `ckpt/yaw_aug/…` paths and the JSON/sidecar copies in the assets folder are for the record only.

### 12. NIT — the generators cannot admit an all-descriptive profile
`make_results_md.py:49-50` makes a profile with zero decision-driving cells a draft (`draft = not verdicts`), and `:29` hard-wires the producer key to `producer_paired_compare` for every non-TABLE_V1 profile. Not a defect today (no such producer exists) but it shapes the descriptive-grid decision — see Fitness.

---

## Verified (commands and outputs)

Hygiene
- `git clone … && git checkout 4f0abb6`; `git diff --stat bb9fa91..4f0abb6`: 7 files, 575 insertions; only the record assets and `tests/test_exp04_record_tools.py`; `git diff --name-only … | grep -E '^tools/|exp_03'` → none.
- Per-commit changed lines (`--numstat`): `f41473c` 199, `929ab47` 0 (rename `test_exp04_record_tools.py` → `tests/`), `4827cc1` 184 (182+2), `4f0abb6` 196 — all < 200; each carries `Co-Authored-By: OpenAI Codex gpt-6-astra <noreply@openai.com>`.
- `py_compile` of the 4 Python assets + the test file: OK; `git diff --check bb9fa91..4f0abb6`: clean; `bash -n static_checks.sh`: ok.
- CPU subset `pytest tests/test_exp04_record_tools.py tests/test_paired_compare.py tests/test_results_table.py -q -p no:cacheprovider` (`CUDA_VISIBLE_DEVICES=''`): **131 passed** in 26 s. New tests: 1+8 (Markdown) + 2+4+1 (HTML/presentation) + 13 (binder) = **29**. Whole suite collects 1227 tests on CPU.
- `static_checks.sh` (a copy with `cd`/`NUMBA_CACHE_DIR` pointed at the clone; the original hardcodes the real repo): exit 0; `-k refusals` selects 25 tests (8 + 4 + 13).

Generators (default fixture materialised with `pytest --basetemp`, then both CLIs run twice)
- Byte-stable: `results_1.md` = `results_2.md` (`cefb29e9…`), `results_1.html` = `results_2.html` (`85b895c1…`). Identity lines: `generated by make_results_md.py from <5 sha256> at 4f0abb64…` (clone HEAD); HTML likewise.
- Traceability: 23 Markdown tables / 304 rows; 28 HTML `<h2>` sections / 397 rows. Every data cell equals `display()` of a value in the five JSONs (or a sidecar value, or the fixed labels primary/supportive/descriptive/—); the 18 TABLE composites `mean ± sd unit` decompose into JSON values (multi-word unit "log-STFT MSE" handled). HTML rows 0–282 are identical to the Markdown rows; the 107 extra HTML rows are the `--diag` duplicate placed before the footer; provenance footer: 5 rows identical, 6th = the diag input. No arithmetic anywhere (`tables()` copies fields; the only composition is string concatenation).
- Non-degenerate fixture (aug arm perturbed per angle/seed, n_boot 20 000): H1_K8/K1 "non-inferior", H2_K8 "not supported", TOST_K8 14 decision cells with mixed "equivalent" / "equivalence not established", all rendered verbatim (aggregate verdicts in the section titles, TOST per cell; individual H1/H2 cells show "—"). SVG: H2 strip 8/8 cells (point = estimate, segment estimate → one-sided upper), TOST strip 14/14 (segment = companion interval, dashed lines invert to ±0.02 = profile margin), diagnostic r_k curves 7/7 per metric; **max |inverted coordinate − JSON value| / scale ≤ 3.3e-16**.
- HTML: no `http(s)://` or `//` references, no `<script>`, `prefers-color-scheme` dark block, explicit body background, `<title>`, `role="img"` + `<title>` on every SVG.

Refusals (both CLIs; on refusal the output file is not created)
- moved copy (sidecar keys point elsewhere) → `provenance sidecar mismatch`; TABLE JSON in the `--h1-k8` slot → `wrong profile`; producer-made exploratory JSON (`analyze(exploratory=True)`, no verdict) → `exploratory JSON refused`; tampered estimate without re-signing → `provenance sidecar mismatch`; one TOST cell verdict removed (re-signed) → `verdict absent`; H2 verdict removed → `verdict absent`, with `--draft-ok` → rc 0 and `DRAFT — missing verdict; tests only` in both outputs; sidecar deleted → `FileNotFoundError`; `--out` = an input JSON or its sidecar → `output overlaps canonical input` (input unchanged, digest `bf14d18…` still bound). Coder's parametrised refusals (profile digest, closure digest, coverage, convergence gate) replayed green. Tamper + re-sign → accepted (finding 2).

Binder / checker (synthetic fixture from the coder's test, plus my attacks)
- Report keys: `schema_version, git_HEAD, runs[{path, manifest, completion, outputs, log}], training{…}, probe_receipt, audit, approved_digests{path, sha256, git_blob, blob}, inputs`. Second `bind` → `FileExistsError`, bytes unchanged; `check_record` passes without calling `git rev-parse` (HEAD taken from the report).
- Refused: edited output digest in the report, extra report key, run dropped from `runs`+`inputs`, `inputs.audit` swapped, `child_exit_status` 1 (`eval status`), duplicate run (`empty or duplicate run set`), attempt `mode` ≠ full, all 12 coder mutations (manifest/completion/output/train_manifest/train_output/probe/audit/approved/log/echo/incomplete/missing_digest). Accepted: another existing commit as `git_HEAD` (finding 5), `_ABORTED_`-named directory with a valid completion (name not checked; harmless — aborted runs have no completion), extra file in the run directory (finding 4).

Real artefacts (read-only)
- `ckpt/xRIR_simple_yawaug_8_shot/final` → `attempt_20260913T102051`; `train_manifest.json`: `mode` full, `allow_dirty` False, `mutable_inputs` {control_args, effective_args, probe_receipt, train_inventory}, `train_data_identity` via `inventory_file`, `effective_args` present, closures {launcher, training}; `completion.json`: `directory_listing == outputs` (both dicts), `train_manifest_sha256` matches the file, outputs include the 12 epochs + history/args/effective_args/train_manifest — every binder assumption holds. Training closure: no drift; launcher closure drifted (`tools/exp04_launcher.py`, `tools/exp05_gates.py`) — tolerated by the binder's `source_drift=[]` path (reviewed-blob consistency, not current bytes).
- `ckpt/yaw_aug/eval/aug_k8_seed42_k0`: reviewed `f19b9b6`, confirmatory, `allow_dirty_used` False; closures entrypoint 14 / writer 15 / frozen evaluator 12 files, **no drift at the real HEAD `ecbb96e`** and no `commits_after_reviewed`; completion listing/outputs/status exactly as the binder and `admit_run` expect; `metrics_yaw.json` P['0'] carries c50/consistency/decay/edt/log_mse/loss/stft/t60, so TABLE_V1's five columns (`make_results_md.py:68`) exist for real runs. `meta.checkpoint` is the resolved attempt path; `admit_run` and the binder compare resolved paths, so the `final` symlink is fine.
- `approved_digests.json` is still the all-null template (`d750597`); the binder refuses it (`approval pins are not final`), as intended before the A9 fill.

---

## Fitness for the final record (what the Planner still needs)

1. **Descriptive grid and epoch-9 diagnostic have no producer** (coder's README is accurate). Neither generator can carry them today: `paired_compare` needs five seeds, `results_table` only the approved epoch-12 checkpoint, the approval schema has a fixed key set (`tools/exp04_profiles.py:27-34`), and `load()` treats a profile with no decision-driving cells as a draft (finding 12). Two clean options: **(a)** keep them out of `_results.md`/the page and report them in `_analysis.md` as single-seed diagnostics citing the bound run digests from `binding_report.json` (they are bound evidence even without a producer; §3 says they never enter a table or a verdict); **(b)** a short round 9: `tools/descriptive_grid.py` (new file) emitting `{profile, profile_digest, inputs, run_flags, producer_closure_sha256, cells[…decision_driving=False…]}` + sidecar through the shared `admit_run`, a `producer_descriptive_grid` pin in `load_approved_digests`'s shape check (this edits `exp04_profiles.py`, inside both producers' closures, so it must land **before** the A9 fill), and three generator changes (per-profile producer key, non-draft when no driving cells, `--diag` curves for that profile). (a) is safe if the chain finishes first; (b) only if the pins are not yet filled.
2. **A9 fill (verified against the live artefacts).** `checkpoints.aug` = `{path: 'ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth'` (the `final/…` string — `admit_run` compares it to `AUG.checkpoint`, not the resolved attempt path), `epoch: 12`, `sha256` of that file}`; `closures.evaluator` = `source_closures.entrypoint.sha256` and `closures.writer` = `source_closures.writer.sha256` from any completed eval manifest (identical across runs at `f19b9b6`); `closures.training_launcher` = the train manifest's `source_closures.launcher.sha256` (recorded, not compared); `closures.producer_paired_compare` / `producer_results_table` = `producer_identity('tools.paired_compare'|'tools.results_table')['sha256']` at the fill commit's HEAD with a clean tree (invariant to the JSON edit); `schema_version: 1`. Commit the JSON first — `load_approved_digests` requires the bytes to equal the HEAD blob, so no producer, generator or binder run can precede that commit. **From now until binding, no commit may touch any file of the evaluator closures** (they include `tools/provenance.py`, `tools/exp04_eval*.py`, `eval_yaw_rotation.py`, `tools/yaw_rotation.py`, `tools/reference_manifest.py`, `model/`, `utils/`, the dataset module): the producers and the binder compare working-tree bytes with the spawn-time digests and refuse on drift. Round 8 touched none of them.
3. **Order of operations once the chain finishes:** (i) fix finding 1 (and ideally 2, 4) and re-run the full CPU suite; (ii) fill `approved_digests.json` → commit; (iii) producers, each to an exclusive path under `ckpt/yaw_aug/analysis/`: `paired_compare --profile H1_K8 --runs-a <5 aug K8 k0> --runs-b <5 control K8 k0>`, `H1_K1` with the K = 1 standalone runs, `H2_K8 --runs-a <5 aug P blocks> --runs-b <5 control P blocks>`, `TOST_K8 --runs-a <5 aug P blocks>`, `results_table --profile TABLE_V1 --runs <30 k0 runs> --json … --md model_comparison.md` — pass the completed retries, not the `*_ABORTED_setup_failed` directories; (iv) `make_results_md.py`/`make_results_html.py` from those absolute paths with `--out` in the record folder (`yaw_aug_xrir_results.md`, `yaw_aug_xrir_01_results.html`); copy the JSONs + sidecars into the assets folder for the record (copies are not valid inputs, finding 11); (v) `bind_provenance.py --runs <all completed confirmatory eval dirs, diagnostics included> --attempt ckpt/xRIR_simple_yawaug_8_shot/final --probe-receipt ckpt/xRIR_simple_yawaug_8_shot/_probe_20260913T101537_probe.json --audit ckpt/yaw_aug/alignment_audit.json --out ckpt/yaw_aug/binding_report.json` (+ `--canonical …` once finding 2 lands), copy the report into the record, `check_record.py ckpt/yaw_aug/binding_report.json` (budget ≈ 20 min each); (vi) `static_checks.sh` on a clean tree; (vii) one commit of pages, report copy, `commits_yaw_aug_xrir.md`. The identity lines and the report carry the *pre-commit* HEAD (as in exp_03); do not regenerate after committing or the identity lines churn.

---

## Verdict

**Request changes.**

Blocking findings the Coder must fix before the round closes:
1. Finding 1 — module-name collision with exp_03's record tools (`tests/test_exp04_record_tools.py:14-15` + bare-name imports; `make_results_html.py:4`; `check_record.py:5`): the full suite fails 13–29 exp_04 tests depending on collection order. Fix by by-path loading under unique `sys.modules` keys (same cached module for binder and checker), add the isolation regression, and demonstrate `python -m pytest tests -q` (CPU) green in both orders.

Strongly recommended in the same fix round (non-blocking): findings 2 (bind the canonical JSONs/sidecars/summaries), 4 (directory-contents check), 3 (presentation) at the Planner's discretion.
