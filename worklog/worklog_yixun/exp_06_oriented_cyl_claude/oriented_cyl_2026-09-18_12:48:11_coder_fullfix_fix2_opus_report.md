# Coder report — exp_06 `oriented_cyl`, second fix cycle (close-review R3), branch `exp06-fullfix`

**Coder:** Claude Opus 5 (Claude Code, Agent tool, max effort) · **Date:** 2026-09-18, local
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-fullfix`, base `31d2933`, **tip `b6de5d2`**.
`main` was `8ed7684` when this round started (it has since moved to `2642e4e`, whose own message
records the exp_07 runtime-approval test as known-red until the pin fill — see §4). It already carries the merge of `31d2933` (`e1cb5b4`) and the six-key approvals
re-fill (`f150294`); `git merge-base main HEAD` is therefore `31d2933` and this branch is exactly the
five commits below.
**Review answered:** `oriented_cyl_codex_code_fullfix_close_review.md` (OpenAI Codex `gpt-6-astra`,
reasoning xhigh) — the single remaining blocker **R3: the legacy training receipt's producer-closure
validation**. R1, R2, R4 and R5 were closed by the previous cycle and are untouched here.
**Environment:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`,
`PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=''` throughout. No GPU touched,
no process signalled, nothing merged, `main` untouched. This report is the only file written outside the
worktree.

## 1. Commits (five — red test, then implementation; every one ≤ 200 changed lines)

| commit | subject | changed lines |
|---|---|---|
| `fa2ed7a` | exp06 R3 (red): a producer closure is its records, not two summary flags | 102 |
| `ffa5b91` | exp06 R3 (red): the launcher and the comparer read the same producer evidence | 104 |
| `dbf9f07` | exp06 R3: the producer closure is validated, not read for its summary flags | 88 |
| `50e0a15` | exp06 R3 (red): a closure that omits the writer is not the writer's closure | 17 |
| `b6de5d2` | exp06 R3: the closure a receipt records names the writer that wrote it | 7 |

Maximum 104 changed lines (tests included). Every commit carries both required trailers
(`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_01NT7CPAUaJ71xgzoS8JGDVw`), verified with
`git log --format='%(trailers:key=…)'`.

Files changed on the branch (`git diff main...HEAD --stat`): five, 300 insertions / 18 deletions —
`tools/exp06_legacy_train_receipt.py`, `tests/test_exp06_legacy_train_receipt.py`,
`tests/test_exp06_sim_evidence.py`, `tests/test_exp06_compare.py`, `tests/test_exp06_eval_launch.py`.
No protected/pinned path is in that list (exp_03's twelve files, the trainer closure, `sim_to_real`,
exp_04/exp_05 tooling, `tools/paired_compare.py`, `tools/provenance.py`, `tools/exp06_profiles.py`,
the approvals template and its record copy, anything under `worklog/`).

## 2. The fix (R3)

`tools/exp06_legacy_train_receipt.py`. `verify` used to read the producer closure for two summary
flags — `strict is True` and a falsey `drift` — which is why Codex's receipt carrying
`{"strict": true}` and nothing else was admitted by the registered-baseline evidence check, by
`launcher.build_fields()` and by `exp06_compare.admit_run()` as arm B. The closure is now validated as
evidence, in a new `check_producer(closure, repo=REPO)` that `verify` calls in place of the two flag
tests (`verify` gains a `repo=None` keyword, defaulting to the module's own repository root; no caller
signature changes):

1. `strict is True` — kept, and still refuses an `--allow-dirty` receipt first, with the old wording.
2. **Producer identity**: `entry_module == 'tools.exp06_legacy_train_receipt'`.
3. **Commit**: a 40-hex string (`COMMIT` regex).
4. **Membership**: a non-empty `files` list; every record must carry a `path` and both
   `reviewed_blob_sha256` and `working_tree_sha256` as 40- or 64-hex (`FILE_HASH`, the two widths
   `closure_record` can write); the paths must be sorted and unique; and the writer's own source
   (`tools/exp06_legacy_train_receipt.py`) must be among them — a closure of `tools/provenance.py`
   alone is internally consistent and says nothing about the bytes that wrote the receipt.
5. **Drift is derived**, not trusted: any record whose `working_tree_sha256 != reviewed_blob_sha256`
   is drift, and the receipt's own `drift` list is still honoured (union of the two), so both the
   contradictory-hash probe and the self-declared-drift probe are refused, each naming the paths.
6. **Digest**: `_closure_digest(files)` recomputes `provenance.closure_record`'s own digest —
   `sha256(json.dumps([[path, reviewed_blob_sha256], …], sort_keys=True))`, the rule
   `tools/exp06_finalize.closure_digest` and `tools/exp06_heading._closure_digest` already write out —
   and must equal the recorded `sha256`.
7. **Reviewed blobs at the recorded commit**: for every record, `git show <commit>:<path>` in `repo`,
   hashed exactly as `closure_record` hashes it, must equal the recorded `reviewed_blob_sha256`. A
   commit that is not HEAD is as valid as HEAD (a receipt is written once and read later), and a
   commit the repository does not carry fails here, naming the paths.

**Dirty worklog files.** 6.4's allowance needs no exception: the closure is this tool's *import*
closure (`tools/exp04_profiles.py`, `tools/exp06_legacy_train_receipt.py`, `tools/provenance.py`), so a
`worklog/` path can never appear in it and can never be drift. A test asserts the closure's paths are
all under `tools/`. Artefact hashing is unchanged and still routed through the caller's `hasher`
binder; the writer is unchanged.

**Two deliberate limits, for the reviewer.** (a) The reader does not require the recorded closure to
equal the *current* checkout's import closure: that would refuse a receipt written at an earlier commit
whose producer has since gained an import, and the executing bytes are R1's gate
(`exp06_approvals_api.enforce_producer`), not this one. The floor is the writer's own source plus the
reviewed-blob check at the recorded commit. (b) Verification now needs a git checkout that carries the
recorded commit — the runbook writes (`breceipt`) and reads (`sim`, `compare`) in the same tree, and
after the merge the commit is an ancestor of `main`, so this holds; a receipt carried to an unrelated
repository would be refused.

## 3. Tests (+18 cases; the stubbed producer fixture is gone)

`tests/test_exp06_legacy_train_receipt.py` — new `producer_identity(repo, commit)` helper builds the
writer's **real** closure with `provenance.source_closure` + `closure_record` (working-tree hashes set
to the reviewed blobs: a clean checkout of that commit), computed once at import as `PRODUCER`, before
any test can monkeypatch `tools.provenance`. Reader refusals, one per probe the review reported plus
the membership floor: `strict_only` ({"strict": true}), `wrong_producer`, `no_commit`,
`empty_membership`, `corrupt_digest`, `contradictory` (working ≠ reviewed), `missing_reviewed`,
`unreviewed_blob` (well-formed hashes, digest re-derived, blob not the one at that commit), and the
truncated closure that omits the writer. Plus: the digest rule equals `closure_record`'s on the real
closure, the closure carries no `worklog/` path, and a receipt whose closure is bound to the **commit
that added the writer** (an older commit, skipped only if that is HEAD) is still accepted.

`tests/test_exp06_sim_evidence.py` — the same seven damaged closures pushed through the runbook's own
argv: `launcher.training_evidence(args)` refuses each before anything is created, and the launcher's
own `build_fields()` output, written as the eval manifest with `train_completion` repointed at the
damaged receipt, is refused by `exp06_compare.admit_run(..., 'B', …)` with `training evidence: …`. And
the acceptance case the review asked to keep: a module-scoped fixture writes the receipt with the
**writer's default strict path on the real `ckpt/xRIR_cyl_8_shot`** (read-only, 18 artefacts, ~2.6 GB
re-hashed at write, at launch and at admission), and that receipt passes `training_evidence`,
`build_fields()` and `admit_run()` as arm B against `exp04_profiles.CYL` — verified live in this
worktree, not a stub.

`tests/test_exp06_compare.py` and `tests/test_exp06_eval_launch.py` — their receipt fixtures now call
`producer_identity()` instead of `{'entry_module': 'x', 'files': [], 'sha256': 'b'*64, …}`.

## 4. Verification

* **exp_06 suite** (`CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_exp06_*.py -q -p no:cacheprovider`,
  at `dbf9f07`, i.e. before the two-line membership floor and its test): **1308 passed, 6 skipped,
  1 failed**, 35:51. The one failure is the expected approvals-drift guard (below). At the tip the
  four affected files were re-run together: `tests/test_exp06_{legacy_train_receipt,sim_evidence,
  eval_launch,compare}.py` → **227 passed** (40 + 20 + …), 1:22 + 0:14.
* **Full suite once** (`CUDA_VISIBLE_DEVICES='' python -m pytest tests -q -p no:cacheprovider`, at the
  tip `b6de5d2`): **3098 passed, 50 skipped, 2 failed**, 51:47. The two failures:
  1. `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`
     — the **expected** drift guard. In this checkout it names six keys, because the branch predates
     the Planner's re-fill on `main`: `compare`, `eval_launch`, `mirror_probe`, `summarize_haa`,
     `finalize`, `haa_pipeline_sh`. Against `main`'s re-filled approvals (`f150294`) only **`compare`
     and `eval_launch`** still differ — the two this round moves (§5).
  2. `tests/test_exp07_profiles.py::test_the_new_arm_checkpoint_digests_come_from_the_runtime_approval[seen_simple-simple-0]`
     — **not mine and not this branch's**: exp_07's `ckpt/exp07/seen_simple/final/epoch_012.pth` now
     exists on disk while `checkpoints.seen_simple.sha256` is still `null` in exp_07's approvals. That
     file is byte-identical on `main` (checked with `git show main:…`), so the failure reproduces on
     `main` and is for the Planner's exp_07 fill, not for this round. This branch changes no exp_07 file.
* `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh`: **passed** (neither changed).
* `python -m py_compile tools/exp06_legacy_train_receipt.py`: **passed**.
* `git diff --check` and `git diff main...HEAD --stat`: clean; working tree clean at the tip.

## 5. Recomputed code digests (`exp06_profiles.compute_code_digests(wt, 'HEAD')` at `b6de5d2`)

Exactly two keys move against `main`'s current approvals — the receipt module sits in both closures:

| key | recomputed sha256 |
|---|---|
| `compare` | `53785b8bf7a47f44f6a73e01619cb9552e4ecb4a3b829828d5dc180081d680ff` |
| `eval_launch` | `db58a3a072484040995230e7b8718f883c51cbbbc13dcf243529fd5a519afe2d` |

The other seventeen keys are unchanged, including `evaluator_exp03`
(`5ba818d8…`, exp_03's pin) and the four keys the previous round moved (`mirror_probe` `ac503c0f…`,
`summarize_haa` `3f497fc4…`, `finalize` `1edbaf2c…`, `haa_pipeline_sh` `2ad3e562…`), which already
match `main`. After the merge, the approvals re-fill for this round is those **two** keys, and the
runbook's expected-digest list needs the same two updated.

## 6. Open items

* Nothing from R3 is left open. R1, R2, R4 and R5 were not touched.
* The exp_07 `seen_simple` pin (failure 2 above) is the Planner's, independent of exp_06.
* The receipt can only be verified inside a git checkout that carries its recorded commit (§2b).
