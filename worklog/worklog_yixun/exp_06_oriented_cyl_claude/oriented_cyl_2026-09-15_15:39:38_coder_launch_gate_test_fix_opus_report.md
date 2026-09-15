**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 launch-gate test fix (tests only)

## Scope and rules

Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-launch-gate`. CPU only
(`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), `PYTHONPATH` = the worktree,
`PYTHONDONTWRITEBYTECODE=1`, `NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest with
`-p no:cacheprovider`. One file changed, `tests/test_exp06_profiles.py`; no tool, template,
record or other tracked file was touched; nothing was written under the main tree except this
report; no process was signalled; `/home/yixunhu/codespace/xRIR_code_wt2b` was never read.

## Commit

`4dd37ec7a574e044825a566f42bcc46e7de68ef6` — *exp06: validate the populated approvals record
instead of the null template* — **71 changed lines** (63 insertions, 8 deletions) in
`tests/test_exp06_profiles.py`, the only file in the commit. Trailers: `Co-Authored-By: Claude
Opus 5 <noreply@anthropic.com>` and `Claude-Session:
https://claude.ai/code/session_01NT7CPAUaJ71xgzoS8JGDVw`.

## The finding and the fix

Codex finding 1 (`oriented_cyl_codex_approvals_fill_review.md`): the authorized approvals fill
(plan §6.4 — the `code` section filled in a *second reviewed commit* before any confirmatory
execution) necessarily breaks
`tests/test_exp06_profiles.py::test_the_record_copy_is_byte_identical_when_it_is_present`,
which demanded the record copy equal the null template byte-for-byte. The test could only fail
once the lifecycle it is supposed to guard ran, so it was removed and replaced with validation
of the record *as populated*. No tool, no template and no record byte changed.

The template's null assertions were already separate — `test_the_template_loads_and_round_trips`
and `test_only_the_pinned_exp03_evaluator_is_filled` take the `template` fixture, which reads
`TEMPLATE_PATH` — and were left untouched. Three new tests, sharing a module-scoped `record`
fixture that skips when the asset is absent from the checkout:

1. `test_the_record_copy_is_schema_valid_and_keyed_like_the_template` — the record loads through
   `load_approved_digests(path)` (so every schema rule applies: sections, nested shapes, sha256
   or null per leaf), the returned value round-trips to the file's JSON, the identity names the
   path and the sha256 of its bytes, and the record and the template agree on key **sets** in
   every section including the three nested ones (`NESTED`) — an approval the template does not
   name is one no producer reads.
2. `test_the_record_copy_binds_to_the_bytes_committed_at_head` — `load_approved_digests(path,
   repo=REPO, commit=HEAD)` binds the file to the blob committed at HEAD, and the identity's
   `repo_relative` / `committed_at` / `sha256` are checked. The sub-check is skipped when the
   record is not tracked at HEAD; the helper `tracked_at` returns False (rather than raising) if
   git cannot answer at all.
3. `test_every_filled_record_digest_is_the_one_this_checkout_computes` — every non-null `code`
   entry must be present in `compute_code_digests(REPO, HEAD)` **and** equal to it; `TRAINING_KEYS`
   must all be filled; `require(...)` must return `[]`. Absence is derived from the checkout, never
   listed: a key this checkout cannot compute (a later round's module) has no computed digest, so
   the single dict comparison also refuses any digest claimed for an absent module, while leaving
   such keys free to be null. The null `reused` / `artifacts` sections are allowed, as they are
   filled only after the runs.

## Validation

Every command below ran in the worktree with the environment named above.

**Red (before the change).**
`python -m pytest tests/test_exp06_profiles.py -q -p no:cacheprovider` →
**1 failed, 20 passed in 39.77s**; the failure is
`test_the_record_copy_is_byte_identical_when_it_is_present`, `assert ... At index 332 diff:
b'"' != b'n'` — byte 332, exactly the failure Codex reported at 53307b6.

**The facts the new tests assert, checked directly first** (scratchpad probe, read-only):
record sha256 `eba8ad3ac7713bb5deb1365fdb07d43a0ddc321bf8483dedafeff32c53bcbd9e` (the value in the
Codex review); it binds at HEAD; all 15 non-null `code` digests equal
`compute_code_digests(REPO, HEAD)`; the null keys are exactly `bootstrap`, `compare`,
`mirror_probe`, `summarize_haa`, whose modules are absent from the checkout;
`require(record, TRAINING_KEYS, ...)` → `[]`; record and template key sets agree everywhere.

**The new tests bite (negative controls, no repository file touched — a pytest plugin in the
scratchpad points `profiles.APPROVED_DIGESTS_PATH` at a tampered copy):**

| Tampered record | Outcome |
|---|---|
| `code.trainer` → `'a'*64` | digest test FAILS |
| `code.trainer` → `null` (a TRAINING key) | digest test FAILS |
| `code.bootstrap` filled although the module is absent | digest test FAILS |
| an extra `code.invented` key | refused at load; all three record tests ERROR |
| the record file missing | all three record tests SKIP cleanly |

Binding is meaningful on the real asset as well: `load_approved_digests(record, repo, '4d7a6cb')`
and `... 'fc1ee2a'` are both refused with *differ from the bytes committed at …* (the pre-fill
and the `fill_note` commits), and the byte-edit refusal in a temp repository is already pinned by
the pre-existing `test_approvals_that_no_reviewed_commit_carries_are_refused[edited-committed]`.

**Green (after the change).**
- `python -m py_compile tests/test_exp06_profiles.py` → ok.
- `git diff --check` → clean.
- `python -m pytest tests/test_exp06_profiles.py -q -p no:cacheprovider` → **23 passed in 38.18s**
  (21 before → one test removed, three added).

## Discrepancies and deviations (nothing silent)

1. **Branch base.** The prompt says `exp06-launch-gate` was created from main `53307b6`; the
   branch actually sits on `3975afe` ("exp_07: integrative review … round-6 relocation prompt"),
   a worklog-only commit on top of `53307b6`. `git diff 53307b6..3975afe --name-only` touches only
   `worklog/worklog_yixun/exp_07_seen_protocol_claude/*`, so no source file, no approvals byte and
   no closure digest differs between the two: the record still binds at HEAD and
   `compute_code_digests(REPO, HEAD)` reproduces all 15 filled digests. `git diff 3975afe..HEAD
   --name-only` is exactly `tests/test_exp06_profiles.py`.
2. **Null-key allowance is a one-directional rule, on purpose.** The prompt asks that the planned
   null keys be allowed and that "absent" be derived from the checkout. I assert *filled ⊆
   computed and equal there*, plus `TRAINING_KEYS ⊆ filled`. I deliberately did **not** assert the
   converse (that every present key is filled, or that the null set equals the absent set): both
   would be true today and would go stale the moment a round-3 module lands before its approval is
   filled — the same staleness that produced this finding. The rule as written refuses a digest
   claimed for a module that is not here, refuses drift on anything filled, and refuses a null
   training key, which is what admission depends on.
3. **`require` is exercised with `current=digests`** (the module-scoped `computed` fixture) so the
   test does not recompute 15 source closures a second time; the digests come from
   `compute_code_digests(REPO, HEAD)` either way.
4. **Test placement.** The new fixture, helper and three tests are appended at the end of the file,
   after the committed-blob binding tests they build on, rather than in the deleted test's slot.
5. No other deviation from the prompt; tools, template, the record asset and every other tracked
   file are byte-identical to `3975afe`.

**Required suite (after the change).**

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests/test_exp06_profiles.py \
    tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py \
    -q -p no:cacheprovider
```

→ **901 passed, 4 skipped, 91 warnings in 1103.00s (0:18:22)**, exit code 0. **0 failures** —
the launch gate's green-suite condition now closes. (The warnings are the pre-existing
torchvision `weights` deprecation and pytest's own notices; the 4 skips are the pre-existing
environment-dependent ones — this branch adds none: `tests/test_exp06_profiles.py` alone is
23 passed, 0 skipped.)

## Status

Codex finding 1 is fixed in tests only, and the change itself is what now needs review before the
launch gate is declared closed. The approvals record, the template and every tool are untouched.
