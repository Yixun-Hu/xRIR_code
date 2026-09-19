# Coder report — exp_06 `oriented_cyl`, micro-round fix3 (close-review 2, single should-fix), branch `exp06-fullfix`

**Coder:** Claude Opus 5 (Claude Code, Agent tool, max effort) · **Date:** 2026-09-18, local
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-fullfix`, base of this round
`b6de5d2`, **tip `ec7b7bf`**. `main` untouched (it was `2642e4e` at the start); nothing merged, no
GPU used, no foreign process signalled, nothing written in the main tree except this report.
**Review answered:** `oriented_cyl_codex_code_fullfix_close2_review.md` (verdict *approve with
changes*), its one should-fix: `tools/exp06_legacy_train_receipt.py::check_producer` validated the
recorded `source_closure.commit` only as 40 hex characters, and Codex replaced it with
`git rev-parse HEAD^{tree}` — a tree object — after which `git show <tree>:<path>` served exactly
HEAD's blobs and the receipt passed `verify()`, `launcher.training_evidence()/build_fields()` and
`exp06_compare.admit_run()` as arm B.
**Environment:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`,
`PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=''` throughout.

## 1. Commits (two: red tests, then the implementation)

| commit | subject | changed lines |
|---|---|---|
| `64d934f` | exp06 R3 (red): forty hex digits name a tree as readily as a commit | 25 |
| `ec7b7bf` | exp06 R3: the commit a receipt records is a commit object, not any object | 14 |

Both carry the required trailers (`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`,
`Claude-Session: https://claude.ai/code/session_01NT7CPAUaJ71xgzoS8JGDVw`), verified with
`git log --format='%(trailers:key=…)'`. Files touched this round:
`tools/exp06_legacy_train_receipt.py`, `tests/test_exp06_legacy_train_receipt.py`,
`tests/test_exp06_sim_evidence.py`. No pinned or forbidden path is among them.

## 2. The fix

`tools/exp06_legacy_train_receipt.py`, immediately after the `COMMIT` format check in
`check_producer` (the only place the recorded identifier is admitted):

```python
kind = _object_type(repo, commit)
_require(kind == 'commit', 'the producer closure records {}, which is not a commit in '
         '{} but {}'.format(commit[:12], repo, kind or 'no object at all'))
```

with a new five-line helper next to `_reviewed_blob`:

```python
def _object_type(repo, identifier):
    """What git calls this object in ``repo``: ``commit``, ``tree``, ``blob`` or nothing."""
    found = subprocess.run(['git', 'cat-file', '-t', identifier],
                           cwd=str(repo), capture_output=True, text=True)
    return found.stdout.strip() if found.returncode == 0 else None
```

The refusal names what was found — `tree`, `blob`, or `no object at all` for a well-formed 40-hex id
the repository does not carry (previously that case was refused later and for the wrong reason, as
an unreviewed blob). The identifier reaching `git cat-file` has already passed the 40-hex regex, so
it can name no ref or path. `check_producer`'s docstring now states the rule ("a git **commit**
object, since forty hex digits name a tree or a blob just as well and `git show <tree>:<path>` reads
the very same reviewed blobs"). Nothing else changed: the writer, the artefact hashing, the digest
rule and the reviewed-blob check at the recorded commit are untouched, and a *historical* commit is
still as acceptable as HEAD (one `git cat-file -t` per `verify`).

## 3. Tests (+4 parametrised cases, red first)

`tests/test_exp06_legacy_train_receipt.py` — new module-level helper `git_object(spec, repo=REPO)`
(`git rev-parse`) and two cases in the `closure_cases` fixture / `CLOSURE_CASES`, both keeping every
other field of the writer's real closure and only swapping the recorded commit:

* `tree_object` — `git rev-parse <commit>^{tree}`, exactly Codex's probe. **Red before the fix:**
  `DID NOT RAISE` (the receipt verified).
* `blob_object` — `git rev-parse <commit>:tools/exp06_legacy_train_receipt.py`. **Red before the
  fix:** raised, but `Regex pattern did not match` — it was refused as an unreviewed blob, not as a
  non-commit.

`tests/test_exp06_sim_evidence.py` — the same two closures added to `CLOSURE_DAMAGE`, so the
existing parametrised end-to-end test drives them through the runbook's own argv:
`launcher.training_evidence(args)` must refuse with `not a commit` before anything is created, and
the launcher's `build_fields()` output with `train_completion` repointed at the damaged receipt must
be refused by `exp06_compare.admit_run(..., 'B', …)`. Both were red before the fix.

Acceptance kept (all ran, none skipped, verified with `-rs -v`):
`test_a_receipt_written_at_an_older_commit_is_still_evidence` (the commit that added the writer),
`test_the_real_exp01_cylindrical_training_gets_a_receipt` and
`test_the_real_exp01_receipt_still_launches_and_admits_arm_b` (the writer's default strict path on
the retained, read-only `ckpt/xRIR_cyl_8_shot`, through `training_evidence` → `build_fields` →
`admit_run` as arm B).

## 4. Verification at the tip `ec7b7bf`

* `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_exp06_legacy_train_receipt.py
  tests/test_exp06_sim_evidence.py tests/test_exp06_eval_launch.py tests/test_exp06_compare.py -q
  -p no:cacheprovider` → **231 passed**, 1:27. (Per the round prompt, no full-suite run: no other
  test file mentions the receipt — `grep -l 'legacy_train_receipt\|train_receipt' tests/*.py`
  returns exactly these four.)
* `python -m py_compile` of the three changed files: clean. `git diff --check`: clean. Worktree
  clean at the tip.

## 5. Recomputed code digests (`exp06_profiles.compute_code_digests(worktree, 'HEAD')`)

Two keys differ from `main`'s approvals (their closures import the receipt module); all eighteen
other keys are byte-identical to the approved values — i.e. exactly the set the previous cycle
already left unapproved, with new values:

| key | at `ec7b7bf` | at `b6de5d2` (previous tip) | main's approved |
|---|---|---|---|
| `eval_launch` | `0a73f84a5a1c17533cfc3e2ef62c24928fb157103e1f4d69bf8d2474ad7ff205` | `db58a3a072484040995230e7b8718f883c51cbbbc13dcf243529fd5a519afe2d` | `7314166eeb58a66c…` |
| `compare` | `43b3461f334aaeef0ce16d1a9f166c91917c73d4b57ce250fe2c6ab8fe921dae` | `53785b8bf7a47f44f6a73e01619cb9552e4ecb4a3b829828d5dc180081d680ff` | `ea97c6d72a74e4e7…` |

`tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`
therefore stays red in this checkout for the same keys as before, until the Planner's approvals
re-fill at the merge tip.
