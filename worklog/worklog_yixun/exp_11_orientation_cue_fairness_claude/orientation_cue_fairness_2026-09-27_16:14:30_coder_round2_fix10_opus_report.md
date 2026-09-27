# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 10**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `55dc3dc`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 16:1x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close9_review.md` via `coder_prompts/round2_fix10_opus_prompt.md`

**Signals:** nothing outside this cycle's own helpers was signalled; lock holders are ended
through their own `Popen` handles. `kill -0` remains a probe.

## Commits

| SHA | Subject |
|---|---|
| `ea92e3b` | exp11 fix10 (RED): a pid file says one thing, and both readers must read it so |
| `64b0484` | exp11 fix10 (GREEN): one pid-record grammar, written once and read twice |
| `56e67b0` | exp11 fix10 (RED): registration and resolution must not interleave |
| `d0f9b95` | exp11 fix10 (RED): the resolver must yield to a registration in flight |
| `d7c9e62` | exp11 fix10 (GREEN): one registration lock, held by whoever is changing the arm |
| `5046ca4` | exp11 bookkeeping: fix-9 counts corrected; close review 9 (record only) |

## Blocker 1 [P1] — one pid-record grammar

Three readers, three answers. `pid_alive` rejected a numeric record with no trailing
newline; `registration_complete` accepted it and, by deleting *every* newline, also
accepted `99999991\n99999992\n` — a "complete registration" no liveness check would ever
probe, so the arm read as quiet with a trainer in it and the marker was withdrawn. The
trainer's own reader stripped surrounding whitespace, a third answer again.

The grammar is now written once and is the **whole file**:

```
^[0-9]{1,10}\n?$
```

digits, at most one trailing newline, nothing else — no leading blank, no second record,
no CR, no other bytes. `pid_record <file>` in the launcher prints the pid or returns 1,
and **both** `pid_alive` and `registration_complete` go through it. `pid_record(path)` in
the trainer is the same grammar as a **bytes** `fullmatch` (text mode would translate
`1457170\r\n` into a valid record) with no `strip`. An invalid record is not a
registration and never withdraws the marker.

Both implementations are parametrised over the reviewer's table plus the cases that
caught the old readers out:

| record | verdict |
|---|---|
| `1457170` | valid |
| `1457170\n` | valid |
| `1234567890\n` | valid (ten digits is the cap) |
| `\n1457170\n` | invalid |
| `99999991\n99999992\n` | invalid |
| ` 1457170\n` | invalid |
| `1457170\r\n` | invalid |
| `1457170\n\n` | invalid |
| `12345678901\n` | invalid |
| (empty) | invalid |

**A consequence worth naming:** `train.pid` held `"<pid> <start time>"`, which under this
grammar is no record at all — a live trainer no reader could parse, exactly the failure
this closes. It now holds only the pid; the `launching` marker already carries the
timestamp.

## Blocker 2 [P2] — registration and resolution may not interleave

The trainer opened `train.pid` before writing it, so the resolver's
scan → tombstone → rename could run between the trainer's checks and its write — and the
open descriptor survives a rename, so the retired directory received the registration
after the arm had been declared quiet again.

**A registration lock, distinct from the publication lock.** `<arm-root>/.registration.lock`.
It has to be a different lock: a full-mode launcher holds the *publication* lock for its
child's whole run, so a trainer could never take that one. This one is held only across a
registration and across a resolution — exactly the pair that must not interleave.

* **`register_trainer`** takes it **blocking with a 60 s bound** (a resolution is short; a
  trainer that waits forever is a trainer nothing can end) and only then checks — the
  directory exists, its canonical parent is this arm root, no tombstone, a launch record
  present — and writes `train.pid` **atomically**: a temp file in the same directory and
  one `os.replace`. The file is opened only under the lock. A timeout writes nothing and
  exits 3. `--no-save` diagnostics take no lock and only a tombstone refuses them.
* **`resolve_unregistered`** takes the same lock **non-blocking** around its whole
  critical section — scan, tombstone, rename, marker removal — in addition to the
  publication lock it already holds, and refuses *"a registration in progress … a trainer
  is between its checks and its write"*.
* **The scans do not take it**, and the launcher says why: an in-flight registration is
  already covered there by the `launching` marker or by a complete live `child.pid`, and
  a recovery that had to queue behind every trainer would be a new way to stall an arm.

**Regressions** (real locks, the real sourced lifecycle, a stub importing the real
`register_trainer`): the reviewer's schedule — a registration holds the lock, the resolver
refuses, nothing is tombstoned or renamed, and the resolution succeeds once released; the
mirror schedule — the resolver went first, so the waking trainer refuses on the tombstone
and leaves no `train.pid` in the retired directory; the registration completes in the
directory it checked; the timeout path waits its bound and writes nothing; the write is a
rename (no temp file survives, no partial pid is observable); a recovery is **not** blocked
by a held registration lock.

## Report correction

Fix 9's per-commit table is corrected to close review 9's counts (`ff610ff` +48,
`3c32cb2` +19/−5, `c530e32` +79, `e4bd6a3` +37/−4 tests, `71ac3c9` +7, `b49d90b` +56), and
`71ac3c9` is named as that cycle's last production commit.

## Line counts

| commit | production | tests |
|---|---:|---:|
| `ea92e3b` (RED) | — | +98 |
| `64b0484` (GREEN) | +46/−18 | +5/−6 |
| `56e67b0` (RED) | — | +55 |
| `d0f9b95` (RED) | — | +78 |
| `d7c9e62` (GREEN) | +99/−15 | +5/−16 |
| range `55dc3dc..HEAD` | **+144 / −32 = 176 changed** | +224/−5 |

Corrected on 2026-09-27 to close review 10's counts (the four per-commit figures above;
the aggregate was right). Every **production/test** commit is under 200 changed lines;
no exception this cycle. The record-only commits `5046ca4` and `51e34b1` are larger, as
bookkeeping commits are.

## Digests

Taken at `5046ca4`, the cycle's tip (its last production commit is `d7c9e62`).

**exp_06** — exactly **one** key differs from the record re-filled at `9f98bbb`:

| key | approved at 9f98bbb | fix 9 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `a1c2b1c0dfc1…` | `a1c9b623dcfb…` |

**exp_11** — all eight keys, and which moved against fix 9's tip `55dc3dc`:

| key | digest | vs `55dc3dc` |
|---|---|---|
| `train` | `633d13605c584dd9e0f0a3d3fe854dbc88f4c8d7aada2c2b6254a2431a0e2519` | **moved** |
| `finalize` | `8a416d9754f3fcef65fd566e01db5fab87ba7a2148d814546a8ddf056190a76e` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `9952138e58ee1ceee3086b0fd5a70f4c92a0edaa8f0fc478d7162100c9d4599f` | **moved** |
| `smoke` | `bbdd9a5f907d2822ee8e72c90d1448fffc58ec3a598d49ea2ea779d21e68c1c1` | **moved** |
| `summarize_haa` | `a1c9b623dcfbebba4bfc39ba3ef8a81753871935dc7cffbe92f16a70c95e8708` | **moved** |

The same five as the last two cycles, for the same two files: `launch_sh` through its own
shell bytes, and `train`, `finalize`, `smoke`, `summarize_haa` because
`tools/exp11_train.py` is in all four source closures. exp_11's approvals record remains
all-null and every producer refuses today.

## Test results

* **Targeted set** — all fourteen `tests/test_exp11_*.py` plus
  `tests/test_exp06_summarize_haa.py` and `tests/test_exp09_sim_eval_closures.py`:
  **452 passed** in 614 s
  (`tests/test_exp11_shell.py` alone: **106**, `tests/test_exp11_train.py`: **49**,
  `tests/test_exp11_lock_holder.py`: **7**). Fix 9 was 411; the 41 new passes are this
  cycle's grammar table (twice over) and the registration-lock regressions.
* `bash -n` on both shells, `py_compile` on every changed module and test, and
  `git diff --check 55dc3dc..HEAD`: clean. `tools/exp06_launch.sh` and
  `tools/exp06_finalize.py` are byte-identical to `55dc3dc`.
* **Full CPU suite** — run detached at `9f22647` (the same tree as the last code
  commit `d7c9e62`) to
  `orientation_cue_fairness_2026-09-27_16:30_suite_full_cpu_9f22647.log` in this record.
  **1 failed, 3782 passed, 55 skipped** in 3249 s (54:08). The one
  failure is the standing, expected
  `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`:
  `code.summarize_haa` has moved and the record is re-filled at the reviewed merge, as
  round 1 did with `9f98bbb`, not on this branch. Fix 9 was 3741 passed; the 41 new
  passes are this cycle's regressions. Nothing else regressed.
