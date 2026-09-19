# Coder report — exp_06 `oriented_cyl`, fix cycle for Codex review R1–R5 (branch `exp06-fullfix`)

**Coder:** Claude Opus 5 (Claude Code, Agent tool, max effort) · **Date:** 2026-09-18 02:10 local
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-fullfix`, base `ac05dd9` (main `ddfcb66`), **tip `31d2933`**
**Review answered:** `oriented_cyl_codex_code_fullfix_review.md` (OpenAI Codex `gpt-6-astra`, reasoning xhigh) — R1, R2, R3 (blockers), R4, R5.
**Environment:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout. No GPU touched, no process signalled, nothing merged, `main` untouched. The only file written outside the worktree is this report.

## 1. Commits (nine — red test, then implementation; every one ≤ 200 changed lines)

| commit | subject | changed lines |
|---|---|---|
| `19eed6a` | exp06 R5 (red): omitting the headings is the claim that there are none | 21 |
| `09a2ee7` | exp06 R5: a producer of every room offers every heading, or offers nothing admissible | 7 |
| `6cdf1ff` | exp06 R1 (red): the approvals must admit the bytes that would run | 124 |
| `5921bb5` | exp06 R1: the producer gate compares the approvals with the bytes on disk | 57 |
| `adc1a67` | exp06 R2 (red): a receipt of another exp_01 arm may not launch the baseline | 52 |
| `4240086` | exp06 R2: one role -> registration rule, applied by the launcher and the comparer | 59 |
| `a129caa` | exp06 R3 (red): a receipt is evidence of what the training directory still holds | 86 |
| `243bb59` | exp06 R3: the receipt is verified against the directory, not against itself | 66 |
| `31d2933` | exp06 R4: the integration test admits the manifest the launcher itself built | 98 |

Maximum 124 changed lines, tests included. Every commit carries the two required trailers
(`Co-Authored-By: Claude Opus 5`, `Claude-Session: …session_01NT7CPAUaJ71xgzoS8JGDVw`).

**Recorded, not rewritten** (as the round prompt asks): the previous round's F2/F3 commits exceeded the
size rule — `35c035a` 516, `1168bb2` 361, `fc9748d` 343 and `fd00c2b` 221 changed lines — exactly as the
review's process note says. History is left as it is; this round's split is the corrected practice.

## 2. What each finding became

### R1 (blocker) — the approvals now authenticate the bytes that would run
`tools/exp06_approvals_api.py`. `enforce_producer` no longer asks the approvals module for digests
alone. New readers in the same module, built from its own `CODE_SOURCES` map (so
`tools/exp06_profiles.py`, a `TRAINING_KEYS` member, keeps its identity):

* `code_files(key, repo)` — the files a key's identity is taken over (import closure + shell files);
* `present_code_key(key, repo)` — whether this checkout has them (a later round's key is still named,
  never guessed);
* `code_closure(key, repo, commit)` → `provenance.closure_record(...)` = `(file records, digest)`;
* `code_closures(keys, repo, commit)` → `{key: {'files', 'sha256'}}`;
* `code_digest` is now `code_closure(...)[1]` — same value, one reader.

For every required key `enforce_producer` adds a deviation when the key is absent from the checkout,
when its digest is not the approved one (as before), when a file of its closure has **no reviewed blob
at `commit`**, or when that file's `working_tree_sha256` differs from its `reviewed_blob_sha256` — each
naming the key and the path. Exploratory runs keep the deviations and stay `diagnostic`.

Regressions (`tests/test_exp06_approvals_api.py`): a temporary git repository holding one file-only
`code` key and its own committed approvals reproduces the review's two cases without touching this
repository's tree — (a) the source committed forward of the approvals' commit, (b) the source edited
and never committed — both refused naming `code.launch_sh` and `tools/exp06_launch.sh`; (c) the clean
checkout at the approvals' own commit is admitted with `deviations == []`. A stub-level test covers a
missing reviewed blob, the named refusal and the exploratory path.

### R2 (blocker) — one role → registration rule, applied by both callers
`tools/exp06_train_evidence.py`: `REGISTERED = {'baseline': exp04_profiles.CYL}` and
`registered_checkpoint(role, approved=None)` — exp_01's registered cylindrical sha for `baseline`, the
approved `artifacts.epoch_012` for `arm`, `None` for `diagnostic`. `check()` applies it whenever a
caller supplies no registration, so launcher and comparer can no longer disagree;
`tools/exp06_eval_launch.py::training_evidence` also passes it explicitly. The comparer keeps passing
its own role registration — the same `exp04_profiles.CYL` object in production.

Regression: a receipt built from a *control* training directory, offered to a `--checkpoint-role
baseline` launch, is refused (`not the registered exp_01 …`) **before `execute_run`**, with no output
directory created. Confirmed against the real artefacts as well (probe, read-only): the receipt of the
real `ckpt/xRIR_cyl_8_shot` is admitted for `8ba344ad…` and refused for the control's `651e3a37…`.

### R3 (blocker) — the receipt is verified against the directory, not against itself
`tools/exp06_legacy_train_receipt.py`: the writer's membership rule is extracted as `receipt_names`
(required files, the epoch checkpoints present, then optional `best.pth`/`last.pth`), and
`receipt_files` is that list hashed. `verify()` now, before hashing anything:

* requires a `source_closure` record with `strict is True` and no `drift` — `--allow-dirty`'s "never
  confirmatory" contract, enforced on the reading side (the writer records `strict` so a reader can
  tell);
* requires the receipt's enumeration to be **exactly** `receipt_names(train_dir)` as the directory
  stands now, naming what is missing and what is unexpected;

and after the per-artefact hashing (still routed through the caller's binder):

* re-reads the retained `args.json` and `history.jsonl` and requires the epoch counts to agree with the
  receipt, the certified checkpoint to be the last epoch the history records, `args.backbone` to be the
  receipt's backbone, and the receipt's `registered` block to agree with its own checkpoint hash, arm,
  backbone and epoch count.

Authority and consistency are deliberately split: *which* registered arm the weights must be stays
R2's `registered_sha256` (both callers now supply it); this is the internal-consistency and
completeness half, which keeps the reader usable for the control receipt the writer also registers.

Reader-side regressions: the two-file receipt with a recomputed `files_sha256` is refused naming
`history.jsonl`; an enumerated file the directory does not hold, and a file retained after the receipt,
are both refused; `drift`, `strict: False` and a missing closure are refused; four contradiction cases
(epochs, backbone, arm, registered weights) are refused. The clean real receipt of
`ckpt/xRIR_cyl_8_shot` — **18 artefacts**, read-only — is still accepted.

### R4 — the integration test admits the manifest the launcher built
`tests/test_exp06_compare.py::write_run` accepts `launched=<fields>` and writes the outputs, the
per-sample file and the completion around exactly those fields. `tests/test_exp06_sim_evidence.py` now
builds the fields with the launcher and admits **them**: the stubbed closures are the real files of
this repository (self-consistent digests the comparer re-binds), the stubbed data identity is the
fixture split's, and the approvals carry the closure digests the launcher really records. The
runbook's explicit `--approved` / `--approved-commit` are part of the generated argv, and a new test
shows both reaching 6.4's gate (`enforce_producer('sim_eval', repo, commit, approved_path=…,
checkpoint=…)`, arm B passing no checkpoint). Probe: corrupting one recorded closure digest inside the
built fields makes `admit_run` refuse (`closure digest writer_exp06`) — the composition the review
asked for now has something to catch.

### R5 — omitted headings are the empty mapping
`enforce_producer` treats `headings=None` as `{}` for the producers of `HEADING_COMPLETE`, so the HAA
queue must offer all four rooms however the claim is spelled. Both spellings are tested; a producer of
one room (the hallway mirror probe) is unaffected.

## 3. Design decisions worth recording

* **No approvals-schema change.** No key added or renamed; `tools/exp06_approved_digests_template.json`
  and the record copy `oriented_cyl_results_assets/approved_digests.json` are untouched, as are
  `tools/exp06_profiles.py` and every other protected path.
* **R1's readers live in `exp06_approvals_api`**, built from its own `CODE_SOURCES`, precisely so the
  training-critical `profiles` closure does not change under an already-running experiment.
* **R2's registration is one object.** `REGISTERED['baseline']` and the comparer's
  `ROLES['B']['checkpoint']` are both `tools.exp04_profiles.CYL`; `check()` resolves the rule for any
  caller that names none, which is what closed the launcher/comparer gap.
* **Membership is a rule, not a list.** `receipt_names` ignores files outside it (the real directory's
  `eval_*.json` / `.log` and `best_epoch01.pth` are not artefacts of the training run), so the rule is
  stable and the real 18-artefact enumeration is unchanged.

## 4. Verification (CPU only, `CUDA_VISIBLE_DEVICES=''`)

* **Full suite** — `python -m pytest tests -q -p no:cacheprovider`, 39 min 29 s:
  **1 failed, 3080 passed, 50 skipped** (3066 passed before this round; +14 tests).
  `grep -E '^(FAILED|ERROR)'` over the whole log returns one line and nothing else — the expected
  approvals-drift guard.
* **exp_06 suite** — `pytest tests/test_exp06_*.py`: **1 failed, 1290 passed, 6 skipped**, 27 min 32 s
  (same single expected failure).
* **Expected failure** — `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`,
  and only that test (the other 23 in the file pass), naming exactly the six keys of §5:
  `compare`, `eval_launch`, `finalize`, `haa_pipeline_sh`, `mirror_probe`, `summarize_haa`.
* Per file, at the tip: `test_exp06_approvals_api.py` **50 passed** (45 before),
  `test_exp06_legacy_train_receipt.py` **29** (22), `test_exp06_eval_launch.py` **72** (71),
  `test_exp06_sim_evidence.py` **12** (11), `test_exp06_compare.py` **95** (95).
* `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh` — passes.
* `python -m py_compile` of every changed `.py` — passes. `git diff --check` — clean.
* `git diff main --stat`: **16 files, +2573 / −60** — the same 16 files as `ac05dd9`, no new file;
  the intersection of `git diff main --name-only` with every protected path (exp_03's twelve pins,
  `model/`, `treble_multi_room_dataset/`, `utils/`, `train_xRIR_backbone.py`, `tools/exp04_*`,
  `tools/exp05_*`, `tools/paired_compare.py`, `tools/provenance.py`, `tools/exp06_profiles.py`,
  `tools/exp06_approved_digests_template.json`, `sim_to_real/`, `worklog/`) is **empty**.
* **Real-artefact probe** (read-only on `ckpt/`, receipt written to the scratchpad): the writer
  produced the receipt of the real `ckpt/xRIR_cyl_8_shot` with `strict=True` — 18 artefacts,
  `drift: []` — and `exp06_train_evidence.check('baseline', …)` admitted it (`route=reconstructed`,
  `arm=cyl`, `epochs=12`, 19 bound inputs) while refusing the control's `651e3a37…`.

## 5. Recomputed `code` digests at `31d2933` (`exp06_profiles.compute_code_digests(wt, 'HEAD')`)

Six keys differ from the record's approvals; the other fourteen match.

| Key | SHA-256 | vs `ac05dd9` |
|---|---|---|
| `compare` | `ea97c6d72a74e4e7df49a0c042d753cb86c89f8989555841b6b8a8cdc41419e6` | new |
| `eval_launch` | `7314166eeb58a66c5096c0e67fdc1c9a1bfec7fc40d6c57cb6e1ad8fd756cd4f` | new |
| `finalize` | `1edbaf2caaffc6ba4e65d6a7fc6bda119f7e4597523f0072bb323c0af7af6b18` | unchanged this round |
| `haa_pipeline_sh` | `2ad3e562c1e322d3a035e2aed3bb28d55c940d20e0e9516d88c242a7df0055a7` | unchanged this round |
| `mirror_probe` | `ac503c0f265576c478c45dff4c3376cfa029d47c434fcd11db3e5c7182309597` | new |
| `summarize_haa` | `3f497fc4eb71efdff624b07a71822e8a6bf000df96580f9ecfb77e0b440267e6` | new |

`compare`, `eval_launch`, `mirror_probe` and `summarize_haa` moved because their import closures carry
`tools/exp06_approvals_api.py`, `tools/exp06_train_evidence.py` and/or
`tools/exp06_legacy_train_receipt.py`. **They must be recomputed at the actual merge tip**, not taken
from this table.

## 6. Open items for the Planner

1. **Re-fill the approvals at the merge tip.** Until the record's `approved_digests.json` is re-filled
   there, the drift guard fails for exactly those six keys and every producer refuses — the intended
   fail-closed state, not a regression.
2. **`artifacts.gate_g1` stays null** until the mirror probe writes it; unchanged by this round.
3. **No arm-B receipt exists yet.** The writer now records `source_closure.strict` and `verify`
   requires it, so the receipt must be produced by the runbook's `breceipt` step at or after this tip
   (a receipt written by an older writer would be refused for recording no strictness). Nothing to
   migrate — the step has not run.
4. **Operational consequence of R1:** a producer now refuses when any file of its own `code` closures
   differs on disk from the blob committed at the approvals' commit. The runbook already passes
   `--approved-commit $(git rev-parse HEAD)` and the launchers already refuse a tree dirty outside
   `worklog/`; this makes "no writer in the main tree while confirmatory runs launch" an enforced rule
   rather than a convention.
5. **Not touched here:** the launcher-v3 certification, renderer edge cases and binder atomicity that
   exp_03/exp_04 record as open remain open.
