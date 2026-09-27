# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 2**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `2c64a2a`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU, no process signals
**Started:** 2026-09-27 04:0x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close_review.md` (request changes, three residual blockers) and `coder_prompts/round2_fix2_opus_prompt.md`

## Commits

| SHA | Subject |
|---|---|
| `8506456` | **fix2 1 (RED)** the exp_06 pin must name the record `main()` actually loaded |
| `6b02824` | **fix2 1 (GREEN)** the exp_06 pin names the record this run actually loaded |
| `a6b3200` | **fix2 2 and 3 (RED)** overflow ceilings pass; promotion uses the unresolved path |
| `0b10d7c` | **fix2 2 and 3 (GREEN)** bounded ceiling; canonical attempt; scoped root override |
| `1c4498e` | **fix2 4** exempt the copied Codex reviews from the whitespace check |

Every commit in this cycle is well under 200 changed lines.

## Blocker 1 [P1] — the exp_06 pin named the default record, not the consumed one

`check_exp11_reused` hashed `approvals_api.approved_path_default()` while `main()` loaded
whatever `--approved` selected, so exp_11's `reused.approved_digests_exp06` could match the
default file while arm C's epoch and the heading artefacts came from a record nobody
pinned (the reviewer's fixture: consumed `a99f44f9…`, pinned `eb6e58e8…`, both producer
checks and the reuse check accepting).

**Fix.** The gate now takes `exp06_receipt` — the identity `main()` already holds from
`approvals(...)` — compares **its** digest with `reused.approved_digests_exp06`, and binds
that same consumed identity into the published inputs. The receipt is **required**: a
caller that supplies none is refused rather than quietly served the default, so the gate
can never fall back to re-resolving a path. `main()` passes `approvals_receipt` through.

**Tests** (`tests/test_exp11_approval_enforcement.py`): an alternate **committed** record
whose bytes differ only in one artefact digest is refused; a byte-identical copy at
another path is admitted (identity, not location, is what is approved); the consumed
record's identity is the one bound into the inputs; a call with no receipt is refused; and
`main()`'s call site is asserted to pass `approvals_receipt`.

**Red-first.** All four new tests failed on the intended assertion —
*"check_exp11_reused must be given the exp_06 approvals record main() loaded from
--approved; it hashes approved_path_default() instead"* — because the helper asserts the
contract before calling, so none of them failed on a `TypeError` or an import.

## Blocker 2 [P2] — the ceiling check failed open on overflow

`EXP11_FULL_CEILING_S=9223372036854775808` (and a 39-digit value) made bash's `-gt`/`-le`
error out; the `if` was then false, validation "succeeded", and that value reached GNU
`timeout`.

**Fix.** `check_ceiling` validates a **bounded decimal string first** — no leading zeros,
at most six digits (129600 has six) — so the numeric comparison that follows can never be
outside bash's range. Anything else refuses by name.

Verified directly: `2**63` → exit 2, `0` → exit 2, `129601` → exit 2, `129600` → exit 0
with `timeout --kill-after=60 129600`, `3600` → exit 0 with `… 3600`. Dry-run tests also
cover a 39-digit value, `0129600`, `129600.0`, `1e5` and `" 3600"`, and assert that no
`timeout` line is printed and that `integer expression expected` never appears.

## Blocker 3 [P2] — promotion used the unresolved path

`check_attempt` validated the *resolved* attempt while promotion linked the *typed*
basename, so `--attempt <root>/final` (with `final -> attempt_review`) promoted
`final -> final`, and an external alias left a dangling target under the arm root.

**Fix — the canonical attempt is carried through.** `check_attempt` prints
`ATTEMPT canonical <realpath>` and sets `CANONICAL_ATTEMPT`, which `abort`, the finalizer
call and `promote` all use; the typed path is never used again after validation. The
resolved directory must additionally be named `attempt_*`, so a `final` that points at
something else is refused rather than promoted.

**Chosen behaviour (stated, as the prompt asks): aliases are *resolved*, not rejected.**
An existing-`final` alias and an external alias both promote **the canonical
`attempt_<UTC>`** — `final -> attempt_20260927T000000` — and the finalizer runs with
`--run-dir <canonical>`. Both are tested, as are the two refusals (an attempt that does
not resolve into the arm root; a resolved directory not named `attempt_*`).

**`EXP11_PRETRAIN_ROOT` is now a test-context override.** It moves the attempt roots *and*
the roots `preflight` scans for live pid files, so `apply_root_override` honours it only
with `EXP11_TEST_ROOTS=1` **and** `--dry-run`, and refuses it by name otherwise. Both
sides are tested: the override alone refuses, and the override without `--dry-run`
refuses, while the full test context is honoured.

## Process item 4 — corrections to the fix-cycle-1 record

**Red-first evidence, qualified.** In `e98b490` (fixes 4 and 5 RED) the six *ceiling*
parametrisations failed for the intended reason (the launcher exited 0 and printed the
child command where a refusal was required). The three *recovery* tests in that same
commit failed on `NameError: name 'json' is not defined` — a missing import in the new
test helper — so **they did not exercise recovery at that commit**; the import was added
in the GREEN commit `72839af` and the recovery behaviour was only demonstrated there. The
red-first evidence for blocker 5 of cycle 1 is therefore **weaker than the other cycles'**
and is recorded as such rather than claimed. In this cycle every RED run failed for the
behaviour under test, which I verified before committing each one.

**`e978c2e` line count.** Round 2's report described it as "almost entirely generated
golden files". That understates it: the commit adds **401 production insertions**
(`tools/exp11_haa_pipeline.sh`), 117 test insertions and 452 generated golden lines. It is
therefore a genuine >200-line production exception, and the corrected figure belongs in the
list of ten:

| SHA | production insertions | note |
|---|---|---|
| `769a6ef` | 216 | `tools/exp11_recipe.py` |
| `b59e234` | 275 | `tools/exp11_train.py` |
| `3458146` | 260 | finalizer part 1 |
| `b39b7d0` | 203 | finalizer part 3 |
| `216aef3` | 208 | finalizer part 4 |
| `c7b1d7b` | 236 | `tools/exp11_smoke.py` |
| `c0b10a9` | 442 | HAA entry points |
| `8f5ff9a` | 283 | `tools/exp11_launch.sh` |
| **`e978c2e`** | **401** | `tools/exp11_haa_pipeline.sh` — *not* mostly goldens |
| `e7cb7f4` | 351 | summariser arms/admission/phases/A′ |

No history was rewritten; the exceptions stand recorded.

**`git diff --check`.** The only complaints in the range were the two-space hard line
breaks of the **copied Codex review** I added to the record in `2b02035`. A review is not
edited after it is written and those breaks are meaningful Markdown, so `1c4498e` adds a
`.gitattributes` scoped to this record directory marking `*codex*review*.md` as
`-whitespace`. `git diff --check c8acda2..HEAD` and the working-tree check are now both
clean, and no reviewer's bytes were touched.

## Tests

* **Targeted set** — all `tests/test_exp11_*.py` plus `tests/test_exp06_summarize_haa.py`
  and `tests/test_exp09_sim_eval_closures.py`: **339 passed** in 500 s. The pinned
  exp_06/exp_09 and exp_11 phase-1 oracles still reproduce their payloads and rendered
  summaries byte for byte, so blocker 1's tightened gate changed no published number.
* `bash -n` on both shells, `py_compile` on every exp_11 module and test, and
  `git diff --check` over `c8acda2..HEAD` **and** the working tree: clean.

## Digests (last code commit `3119e22`)

**exp_06** — exactly **one** key moved against the record re-filled at `9f98bbb`; the
other nineteen are unchanged:

| key | approved at 9f98bbb | now |
|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `deff6d49d8e9…` |

**exp_11** — all eight keys present in this checkout:

| key | digest |
|---|---|
| `train` | `d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571` |
| `finalize` | `3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e` |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` |
| `launch_sh` | `737414cc75bd89e11c045c203f26be02a9c9f4fed7b923b565adce8ade4bcd17` |
| `smoke` | `62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e` |
| `summarize_haa` | `deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a` |

Three keys moved against `2c64a2a`: `launch_sh` (blockers 2 and 3), `summarize_haa`
(blocker 1) and nothing else; `finalize`, `train`, `haa_finetune`, `haa_eval`,
`haa_pipeline_sh` and `smoke` are byte-for-byte what the close review recomputed. These
are values for the **eventual reviewed merge**, not an authorisation to fill the approvals
now: exp_11's record stays all-null and every exp_11 producer refuses today.
