**Coder:** Claude Opus 5 (Agent subagent, model opus) · **Date:** 2026-09-15

# exp_06 launch-gate test fix 2 — the non-git collection path (tests only)

## Scope and rules

Worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-launch-gate`, from its tip
`4dd37ec`. CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), `PYTHONPATH` = the
worktree, `PYTHONDONTWRITEBYTECODE=1`, `NUMBA_CACHE_DIR=/tmp/exp06_opus_numba`, pytest with
`-p no:cacheprovider`. One file changed, `tests/test_exp06_profiles.py`; no tool, template,
approval byte, record or any other tracked file was touched (`git diff main...HEAD
--name-only` is exactly that one path, and `--diff-filter=M` the same). Nothing was written
under the main tree except this report. `/home/yixunhu/codespace/xRIR_code_wt2b` was never
read. No `pkill`/`pgrep -f`; the only processes signalled were two of my own detached pytest
runs, by their exact pids.

## Commit

`4ea1b2b97067e53d708202a364775999e3fe52b7` — *exp06: resolve HEAD lazily so this module
collects outside a git checkout* — **134 changed lines** (111 insertions, 23 deletions) in
`tests/test_exp06_profiles.py`, the only file in the commit. Trailers: `Co-Authored-By:
Claude Opus 5 <noreply@anthropic.com>` and `Claude-Session:
https://claude.ai/code/session_01NT7CPAUaJ71xgzoS8JGDVw`.

## The finding and the fix

Codex should-fix 1 (`oriented_cyl_codex_launch_gate_test_review.md`): the module-level
`HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()`
ran at **import**, so outside a checkout the file failed to *collect* (git exits 128 →
collection error → pytest exits 2) before the `tracked_at()` skip added in `4dd37ec` could
run. Four changes, all in the test module:

1. **`git_head()`** replaces the module-level constant: `subprocess.run` (not
   `check_output`), returning the stripped stdout when the exit status is 0 and `None`
   otherwise, `None` as well on `OSError` (no git binary at all), memoised in a module-level
   list so the one `git rev-parse` per session is kept. Nothing git-dependent now runs at
   import time.
2. **A module-scoped `head` fixture** yields that commit and calls `pytest.skip('git identity
   is unavailable here: not a checkout, or no git')` when it is `None`. The `computed`
   fixture takes `head` (so every digest test skips through it), and the five tests that used
   the constant directly now take `head`: `test_the_shell_launcher_is_bound_as_a_file`,
   the two `require` tests, `test_the_record_copy_binds_to_the_bytes_committed_at_head`
   (including its `tracked_at(head, ...)` call) and
   `test_every_filled_record_digest_is_the_one_this_checkout_computes`.
3. **The `committed` fixture** — the throw-away repository the committed-blob binding tests
   build — wraps its `git init`/`add`/`commit`/`rev-parse` in `try/except OSError` →
   `pytest.skip('git is unavailable here')`. It does not need *this* checkout's identity, so
   it still runs (and passes) outside one; it only skips where git itself is missing.
4. **The regression**, `test_this_module_still_collects_outside_a_git_checkout`: copies this
   module's own bytes to `<tmp>/not_a_checkout/tests/test_exp06_profiles.py` and runs a real
   `python -m pytest tests/test_exp06_profiles.py -q -p no:cacheprovider --junitxml …`
   subprocess with `cwd` there, `PYTHONPATH=profiles.REPO` (so `tools` still imports from
   this worktree), `PYTHONDONTWRITEBYTECODE=1` and `GIT_CEILING_DIRECTORIES=<tmp>` so git
   cannot walk up into a repository by accident. It asserts the child's **exit code is 0**
   (with the child's own output as the failure message), then parses the JUnit XML and
   asserts: no `failure`/`error` case at all; every `@needs_git` test skipped; the
   template/schema tests passed (`test_the_template_loads_and_round_trips`,
   `test_the_training_keys_are_the_launch_critical_ones`, a malformed-file parametrisation,
   and the record-schema test when the record asset is present); and more passes than skips.
   The set of must-skip tests is read off a `needs_git` decorator that records the function
   name in `GIT_DEPENDENT`, not a hand-kept list, so a git-dependent test added later is
   covered the moment it is decorated. The regression skips itself in the child through the
   `EXP06_PROFILES_OUTSIDE_GIT_CHILD` environment variable, so it cannot recurse.

Production code, the template and the approval bytes are untouched; `tools/exp06_profiles.py`
is byte-identical to `3975afe`.

## Validation

All commands ran in the worktree with the environment named above; the interpreter is
`/home/yixunhu/miniconda3/envs/xRIR/bin/python` (3.8.20, pytest 8.3.5).

**Red (regression added, fix not yet applied).**
`python -m pytest tests/test_exp06_profiles.py -q -p no:cacheprovider -k collects_outside`
→ **1 failed, 23 deselected in 1.75s**. The assertion message is the child's own output:

```
ERROR collecting tests/test_exp06_profiles.py
tests/test_exp06_profiles.py:16: in <module>
    HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
E   subprocess.CalledProcessError: Command '['git', 'rev-parse', 'HEAD']' returned non-zero exit status 128.
------------------------------- Captured stderr --------------------------------
fatal: not a git repository (or any of the parent directories): .git
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.18s
...
assert 2 == 0
```

i.e. exactly Codex's reproduction — git exit 128, no tests collected, one collection error,
child pytest exit 2 — reproduced by a real pytest process rather than in memory.

**Green — the outside-git child, replayed by hand** (module copied to a non-git
`<scratch>/nogit/tests/`, `cwd` there, `PYTHONPATH` = the worktree):

```
.........sssss.......sss                                                 [100%]
SKIPPED [7] git identity is unavailable here: not a checkout, or no git
SKIPPED [1] this is the child run that the outside-git regression spawns
16 passed, 8 skipped in 0.20s        EXIT=0
```

The seven skips are exactly the seven `@needs_git` tests; the 16 passes are the template,
schema, malformed-file, committed-blob-binding and record-schema tests.

**Negative control — the regression bites.** A copy in which
`test_the_pinned_exp03_evaluator_digest_reproduces` loses `@needs_git` and calls
`git rev-parse HEAD` itself gives the child `1 failed, 16 passed, 7 skipped`, **exit 1**, so
the parent's `returncode == 0` assertion fails. A git-dependent test added without the skip
therefore cannot pass this regression.

**Hygiene.**
- `python -m py_compile tests/test_exp06_profiles.py` → ok.
- `git diff --check` → clean.
- `git diff main...HEAD --name-only` → `tests/test_exp06_profiles.py` only (and the same
  under `--diff-filter=M`); `git diff 4dd37ec..HEAD --name-only` → the same single path.
- `python -m pytest tests/test_exp06_profiles.py -q -p no:cacheprovider` → **24 passed in
  45.54s** (23 before + the regression), 0 skipped in this checkout.

## Discrepancies and deviations (nothing silent)

1. **`committed`-fixture tests are not made to skip outside a checkout, only where git is
   missing.** The prompt says "make every git-dependent test skip cleanly (binding, digests,
   and any other test that uses HEAD)". The five `committed`-fixture cases
   (`test_approvals_can_be_bound_to_their_committed_blob`,
   `test_approvals_that_no_reviewed_commit_carries_are_refused[4]`,
   `test_binding_needs_both_a_repository_and_a_commit`) do build a git repository, but their
   own one in `tmp_path`; they never read this checkout's identity, so outside a checkout
   they still *pass*, which is strictly better than skipping. They are guarded against the
   git binary being absent (`OSError` → skip) so the file is clean in a no-git environment
   too. The binding test that does depend on this checkout —
   `test_the_record_copy_binds_to_the_bytes_committed_at_head` — is `@needs_git` and skips.
2. **The record-schema test is asserted to pass in the child only when the record asset is
   present.** `test_the_record_copy_is_schema_valid_and_keyed_like_the_template` takes the
   `record` fixture, which skips when
   `oriented_cyl_results_assets/approved_digests.json` is absent from the checkout (the
   pre-existing behaviour from `4dd37ec`). Asserting it passes unconditionally would make the
   regression fail on a checkout that has no record, so the assertion is guarded by
   `profiles.APPROVED_DIGESTS_PATH.is_file()`. On this branch the asset is tracked, so it
   passes.
3. **`GIT_CEILING_DIRECTORIES` is set for the child.** Not asked for; it stops git from
   discovering a repository above the temporary directory, so the regression cannot silently
   turn green because the temp root happened to sit inside a checkout.
4. **The must-skip set is derived, not listed.** `needs_git` is a plain decorator that adds
   the function name to `GIT_DEPENDENT`, deliberately *not* a `pytest.mark` — there is no
   `pytest.ini`/`pyproject.toml` in this repo to register a custom marker and `conftest.py`
   is out of scope, so a mark would emit `PytestUnknownMarkWarning` on every run.
5. **`_HEAD` is a module-level list used as a one-slot memo** rather than
   `functools.lru_cache`, to avoid adding an import for a single cached call; `None` is a
   legitimate cached answer, which a truthiness-based cache would recompute.
6. **Branch base.** As in the previous report, the branch sits on `3975afe` (a worklog-only
   commit on top of main `53307b6`), not directly on `53307b6`; no source file or
   approval byte differs between the two.
7. No other deviation. The prompt's remaining constraints hold: tests only, one file, red
   before green, ≤ 200 changed lines, both trailers, no GPU work, no question asked.

## Status

Codex should-fix 1 is closed in tests only. The change itself is what now needs review before
the launch gate is declared closed; the approvals record, the template and every tool remain
byte-identical to `3975afe`.

**Required suite (after the change, at `4ea1b2b`).**

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests/test_exp06_profiles.py \
    tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py \
    -q -p no:cacheprovider
```

→ **903 passed, 4 skipped, 91 warnings in 1283.44s (0:21:23)**, exit code 0. **0 failures.**
(901 → 903 because the profiles module is collected twice — explicitly and through the glob —
so the one new test counts twice. The 4 skips and the 91 warnings are the pre-existing
environment-dependent ones; this branch adds neither: `tests/test_exp06_profiles.py` alone is
24 passed, 0 skipped.)
