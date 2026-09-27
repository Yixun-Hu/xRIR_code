# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `c8acda2`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU, no process signals
**Started:** 2026-09-27 02:1x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_review.md` (request changes, five blockers) and `coder_prompts/round2_fix_opus_prompt.md`

**Red-first is evidenced in history this time**: every blocker has a RED commit whose
tests fail against the code as it then stood, followed by a GREEN commit.

## Commits

| SHA | Subject |
|---|---|
| `2b02035` | exp11 bookkeeping: round-2 Codex review and the round-2 prompts (record only) |
| `c7a4602` | **fix 1 (RED)** the adapter arms' zero-shot evaluations must be finalizable |
| `3e64f6f` | **fix 1 (GREEN)** the contextual checkpoint contract at evaluation admission |
| `441db03` | **fix 2 (RED)** exp_11's approval values must be enforced at their gates |
| `c8e6c4e` | **fix 2 (GREEN)** enforce every exp_11 approval value at its consuming gate |
| `ddfdfad` | **fix 3 (RED)** the diagnostic receipt must be bound to its own provenance |
| `eea0cca` | **fix 3 (GREEN)** restore the diagnostic receipt bindings |
| `e98b490` | **fixes 4 and 5 (RED)** the ceiling may only tighten; recovery must match its arm |
| `72839af` | **fixes 4 and 5 (GREEN)** validate the ceiling; recovery must be its arm's own |
| `ef0dd3a` | **fix 6 (RED)** the bundled smoke must cover the adapter route |
| `fca7a4f` | **fix 6 (GREEN)** adapter training and evaluation smokes in the bundled shell |

## Blocker 1 [P1] — J/K zero-shot evaluations could not finalize

`tools/exp11_finalize.py` demanded the complete `simple_adapter` state (268 tensors) of
every evaluation checkpoint, but the J/K queue's zero-shot job legitimately evaluates the
historical A/E **base** checkpoint (266), which the loader reads through
`load_base_checkpoint`. Phase 1b and the final publication were therefore unreachable.

**The contract admission now applies** (`checkpoint_contract` + `verify_evaluation_checkpoint`,
tested in `tests/test_exp11_zeroshot_adapter.py`):

* base mode **only** for an adapter arm whose child evaluates the very initialisation its
  job spec declared at launch — `checkpoint_sha256 == job_init_sha256`. That is the same
  equality `job_lineage` requires of every child of a `zeroshot` job and
  `check_job_identity` ties to the spec, so the signal is the child's own launch-time
  declaration, not a guess from the file's contents;
* in base mode the checkpoint must carry **exactly** the pinned `xRIR` parameter set and
  **no** adapter state at all (a base checkpoint that smuggled adapter keys is refused);
* everything else — every later stage or evaluation checkpoint, and a fine-tuned J/K
  evaluation whose checkpoint lost its adapter — stays on the strict adapter path;
* the contract that was applied is recorded as the completion's **`checkpoint_mode`**
  (added to `CHILD_EXTRA['exp11_haa_eval']`), so a reader never has to infer it.

Tests: both J and K zero-shot admitted; a fine-tuned J/K evaluation without its adapter
refused; an adapter checkpoint offered in base mode refused; a child that declares no job
initialisation stays strict; the non-adapter arms are untouched; and `job_lineage` still
ties a zero-shot checkpoint to the job's own initialisation.

## Blocker 2 [P1] — exp_11 approval values were never enforced

Committed bytes plus non-null fields say nothing about whether a value describes what
ran. Two new gates in `tools/exp06_summarize_haa.py`, both called from `main`:

* **`check_exp11_producer`** compares exp_11's `code.summarize_haa` with the closure this
  run is executing. The historical `check_producer_identity` (exp_06's record) is
  untouched and still runs, so neither replaces the other.
* **`check_exp11_reused`** compares all three `reused` identities with the records
  actually consumed — exp_04's approvals (which `exp04_aug_checkpoint` reads for arms E,
  G and K), exp_06's approvals (heading artefacts and arm C's epoch), and the legacy
  receipt by **path and digest** — and binds those bytes into the published inputs.

In `tools/exp11_haa_pipeline.sh` the queue gate now checks exp_11's own
`reused.approved_digests_exp04` **before** reading that record to resolve arm K's
initialisation, alongside the exp_06 pin it already checked.

Tests (`tests/test_exp11_approval_enforcement.py`) use **incorrect-but-populated** values
throughout: a wrong `code.summarize_haa`, a wrong `approved_digests_exp04`, a wrong
`approved_digests_exp06`, a wrong `legacy_receipt.sha256`, a **nonexistent**
`legacy_receipt.path`, and a receipt path that is not the one this run was admitted
through. A source-level test requires the pipeline to check the exp_04 pin before
`exp04_aug_checkpoint` is called.

## Blocker 3 [P2] — diagnostic receipt bindings restored

`check_receipt_identity` makes the three comparisons the frozen exp_06 finalizer makes and
exp_11 had dropped: the receipt's `runner_closure_sha256` against the digest its
provenance recorded, `git_head` (now **required** in `DIAGNOSTIC_RECEIPT`) against the
provenance's HEAD, and the two `exploratory` flags. `diagnostic_evidence` calls it as soon
as the provenance is loaded, before any consistency or artifact work. One mutation test
per binding, plus the missing-`git_head` refusal.

## Blocker 4 [P2] — the 36 h ceiling may only be tightened

`check_ceiling` refuses `EXP11_FULL_CEILING_S` unless it is a whole number of seconds in
`(0, 129600]`, before anything starts. An **explicitly empty** value is refused rather
than silently replaced, because only an *unset* variable now takes the default
(`${EXP11_FULL_CEILING_S-…}`, not `:-`). So the value GNU `timeout` receives can never be
the `0` that disables it, nor anything above the registered ceiling. Dry-run tests for
`0`, `-1`, `36h`, empty, `129601`, `1000000` (all refused, and no child command printed)
and for `1`, `3600`, `129600` (accepted, with the ceiling reaching `timeout` verbatim).

## Blocker 5 [P2] — recovery may not promote another arm's attempt

`check_attempt` refuses recovery unless the resolved `--attempt` is a **direct child of
the selected arm's root** *and* the profile its recorded arguments select
(`exp11_recipe.select_profile`) is that arm's — both checked after `preflight` and before
the finalizer call and the promotion link. `EXP11_PRETRAIN_ROOT` makes the roots
overridable so the dry-run tests need no fixture inside `ckpt/`.

Tests: a valid **H** attempt offered to `--arm I` is refused; an attempt in the right
directory whose args select the *other* profile is refused; the arm's own attempt is
accepted. The `launch_finalize_I` golden was removed in the same commit — with the check
running in dry-run, a golden of a nonexistent attempt would be a golden of a refusal — and
the argv it pinned (`preflight --mode finalize`, the `exp11_finalize.py --run-type
exp11_train` call and the `PROMOTE` target) is asserted in the acceptance test instead.

## Process item 6 — adapter smokes in the bundled shell

`tools/exp11_launch.sh` smoke mode gains rungs (e) and (f): `simple_adapter` fine-tuning
from the committed fixture with `--adapter-heading-json-dir`, then the evaluation of its
`best.pth`, gated on the fine-tuning rung's own completion certifying it passed. The
oriented rungs prove nothing about the adapter route — a different backbone, a different
cue flag, and the contextual loading mode blocker 1 added — so J and K now reach their
queue with a reviewed smoke of their own. The `launch_smoke_H` golden was regenerated in
the same commit and a test pins both adapter commands.

## Process item 6 — the ten production commits over 200 lines

Recorded as exceptions, in full, as the review asked. Round 2's `.md` accounted for four;
these are all ten (test-only and generated-record commits excluded):

| SHA | Subject | insertions |
|---|---|---|
| `769a6ef` | `tools/exp11_recipe.py` — H_RECIPE and I_RECIPE | 216 |
| `b59e234` | `tools/exp11_train.py` — the exp_06 loop on BACKBONES_EXP11 | 275 |
| `3458146` | finalizer part 1 — run types, closures, orchestration, approvals | 260 |
| `b39b7d0` | finalizer part 3 — HAA children and the adapter-heading binding | 203 |
| `216aef3` | finalizer part 4 — job admission, spec, lineage and children | 208 |
| `c7b1d7b` | `tools/exp11_smoke.py` — enumerated diagnostic kinds with budgets | 236 |
| `c0b10a9` | HAA entry points with the faithful orchestration loop and cue routing | 442 |
| `8f5ff9a` | `tools/exp11_launch.sh` — smoke/probe/full/finalize, `--arm H\|I` | 283 |
| `e978c2e` | `tools/exp11_haa_pipeline.sh` and golden dry-run tests | 970 |
| `e7cb7f4` | summariser arms H/I/J/K, admission adapter, phases and A′ | 351 |

Reasons, briefly: `e978c2e` is dominated by ten **generated** golden files; `c0b10a9`,
`b59e234` and `8f5ff9a` are single entry points/launchers mirroring exp_06 modules of the
same size whose `main` is one unit; `3458146`, `b39b7d0`, `216aef3` and `c7b1d7b` are the
finalizer and smoke runner split as far as they could be while each state still imports;
`769a6ef` and `e7cb7f4` are single schema/registry units whose parts reference one
another. No history was rewritten; the exceptions stand recorded rather than hidden.

The fix cycle itself kept every production commit under the limit.

## Frozen files

Unchanged. The only exp_06 module this round touches remains `tools/exp06_summarize_haa.py`
(its `summarize_haa` key moves, as the round-2 scope allows); `tools/exp06_finalize.py`,
`tools/exp06_launch.sh`, `tools/exp06_smoke.py`, `tools/exp06_profiles.py`,
`tools/exp06_approvals_api.py` and every other pinned file are imported or sourced, never
edited.

## Digests after the fix cycle (last code commit `58bd991`)

**exp_06** — exactly **one** key moved against the record re-filled at `9f98bbb`, as the
round-2 scope allows; the other nineteen are unchanged:

| key | approved at 9f98bbb | now |
|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `2868e66c1167…` |

**exp_11** — all eight keys present in this checkout:

| key | digest |
|---|---|
| `train` | `d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571` |
| `finalize` | `3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e` |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` |
| `launch_sh` | `0bdbb5d3d5c291bb2d0e10a9cb09d3ebd66c7078b28f56118e697c4eef4b44cd` |
| `smoke` | `62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e` |
| `summarize_haa` | `2868e66c116752f205af1299408998b0c3fcebbfc61e26878a0f4a7ac1ff614f` |

Four exp_11 keys moved against the reviewed tip `c8acda2` — `finalize`,
`haa_pipeline_sh`, `launch_sh` and `summarize_haa` (which imports `tools/exp11_finalize.py`,
so it moves with it). These are values for the **eventual reviewed merge**, not an
authorisation to fill the approvals now: the record stays all-null and every exp_11
producer refuses today.

## Tests

* **Targeted set** (`tests/test_exp11_*.py` + `tests/test_exp09_sim_eval_closures.py`):
  **152 passed** in 153 s. That includes the two new files
  (`test_exp11_zeroshot_adapter.py`, `test_exp11_approval_enforcement.py`) and the new
  cases in `test_exp11_finalize.py` and `test_exp11_shell.py`.
* **Summariser suites** (`test_exp06_summarize_haa.py`, `test_exp11_summarize_haa.py`):
  **169 passed** in 507 s — the pinned exp_06/exp_09 oracle and the exp_11 phase-1 oracle
  still reproduce their payloads and rendered summaries byte for byte, so blocker 2's two
  new gates changed no published number.
* `bash -n` on both shells, `py_compile` on every exp_11 module and test, and
  `git diff --check`: clean.
