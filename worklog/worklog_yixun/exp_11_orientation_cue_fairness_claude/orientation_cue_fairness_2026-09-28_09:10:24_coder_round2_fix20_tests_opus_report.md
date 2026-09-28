# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 20 (tests only)**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, fast-forwarded to main (round 2 merged at `6a299b1`, exp_06 re-filled at `6baa337`, exp_11's approvals record filled at `6b70843`)
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` — no GPU
**Started:** 2026-09-28 08:5x EDT
**Input:** the Planner's post-merge suite (`orientation_cue_fairness_2026-09-28_07:55:15_suite_full_cpu.log`) via `coder_prompts/round2_fix20_tests_only_opus_prompt.md`

**No production file is touched by this cycle**: the diff is two test files plus one
helper in a third. All 28 approvals keys are unchanged (below), and
`git diff 6b70843..HEAD -- tools/ model/` is empty, so the record filled at `6b70843`
still describes the code at HEAD.

**Signals:** none were sent to any process. The one signal-related change *removes* a
dependency on how the suite was launched.

## Commits

| SHA | Subject |
|---|---|
| `66a9efd` | exp11 fix20 (tests): the INT replay needs a shell that can trap INT |
| `464a943` | exp11 fix20 (tests): the approvals tests no longer assert the calendar |

## 1 & 2 — two tests asserted that the live record was empty

`test_the_approval_gate_refuses_the_all_null_record` and `test_committed_blob_binding`
both read the **live** record and asserted `blob['code']['train'] is None`. That was true
until a reviewer filled it at `6b70843` and is a deterministic failure ever after: they
were asserting the calendar, not a property.

* The gate's refusal is now shown against the **committed all-null template**
  (`tools/exp11_approved_digests_template.json`, committed at `781cf83`) — a real,
  committed, all-null record, so `load_approved_digests`' repo-and-commit binding is
  exercised exactly as before, and the refusal is the gate's, not the fixture's. The
  second refusal (an exploratory `full`) is unchanged.
* `test_committed_blob_binding` keeps everything about the live record that is timeless
  — its shape (`sorted(blob['code']) == sorted(CODE_KEYS)`), `committed_at`, the
  repo-relative path, the sha256 identity, and the two refusals (a file outside the
  repository; a repo without a commit) — and says nothing about its *state*.
* The binding property itself — *the bytes of that commit, not the file on disk* — moves
  into a repository of its own: `test_the_bytes_bound_are_the_bytes_of_that_commit`
  builds a tmp repo with `git init`, commits the all-null template at the record's
  repo-relative path, then commits a filled record over it, and asserts that binding at
  the **first** commit still yields the all-null bytes with its own sha256, that the
  second yields the filled one, and that the file on disk matches the later identity.
* A complementary test, `test_the_approval_gate_admits_the_filled_record_at_head`,
  asserts the other side: with the record filled, a confirmatory `full` launch at HEAD is
  **admitted**, with `committed_at == HEAD`. It skips while the record is all-null, so
  the pair is valid in both states — and when the record is filled it doubles as an
  exp_11 drift guard: it fails if the filled digests stop describing the code at HEAD.

## 3 — the INT replay depended on how pytest was launched

`test_the_int_handler_terminates_too` printed `NO_INT_TRAP` and exited 9 in the
nohup-launched suite while passing interactively. The cause, reproduced exactly:

```
parent ignores SIGINT    -> "trap -- '' SIGINT\n"
parent default           -> "trap -- 'echo handled' SIGINT\n"
```

A shell that inherits SIGINT **ignored** cannot trap it — bash accepts the `trap`
command and keeps the ignore — so `trap -p INT` reports an empty handler, the replay
finds nothing to run, and the test refuses. `lib` now starts its child with SIGINT and
SIGTERM reset to their defaults (`preexec_fn`), so the replay tests the launcher rather
than the launch method of the suite.

**Production consequence, for the record:** a real exp_11 launcher is started with
`nohup setsid … &`, so it inherits SIGINT ignored and **never sees SIGINT at all** —
TERM is the operative signal, and its trap registers regardless of the entry
disposition. The launcher's header does not say this today; it documents `on_signal`
and the 143/130 exits. Since this cycle may touch no production file, it is recorded
here rather than in the header, as a one-line comment worth adding next time the
launcher is opened for another reason.

## The diff

```
tests/test_exp11_finalize.py   +31 −4
tests/test_exp11_profiles.py   +39 −1
tests/test_exp11_shell.py      +24 −2     (the `lib` helper and the INT test's comment)
```

No `tools/`, no `model/`, no shell, no record producer. Both commits are well under 200
changed lines.

## The digest check — all 28 keys unchanged

Recomputed with `tools.exp06_profiles.compute_code_digests` and
`tools.exp11_profiles.compute_code_digests` and compared with the records committed at
HEAD:

```
exp_06 keys 20 · differing from the record at HEAD: []
exp_11 keys  8 · differing from the record at HEAD: []
```

and `git diff 6b70843..HEAD -- tools/ model/` is empty. Nothing this cycle can move an
approvals key, because nothing in any closure changed.

## Targeted results

| run | result |
|---|---|
| `tests/test_exp11_finalize.py`, `tests/test_exp11_profiles.py`, `tests/test_exp11_shell.py` | **253 passed, 1 skipped** (218 s) |
| `test_the_int_handler_terminates_too` under `setsid nohup … &` — the way the Planner's suite was launched | **passed** |

The skip is the read-only bind-mount check from fix 19, which runs only where the system
already has a mount with two spellings of one inode.

## The colon-free basetemp

Every run in this cycle used `--basetemp=/tmp/pytest-exp11-<stamp>` and removed it
afterwards. Per the coordinator's addendum, the full-suite basetemp is
**`/tmp/pytest-exp11-20260928-091024`** — colon-free by construction. The peer's finding
explains the 21 failures in the Planner's post-merge suite that are **not** exp_11's:
that run used a basetemp path containing `07:55:15`, and tests that prepend a shim
directory under the pytest scratch root to `PATH` had their entry split at the colons,
so the `nvidia-smi` shim (9 `tests/test_exp06_launch.py` preflight tests) and the
`mv`/`cp` shims (12 `tests/test_exp10_record_tools.py` tests) were never found and the
real commands ran. No code change is needed for those, here or in exp_10.

## The full suite

| run | result |
|---|---|
| full CPU suite at **`464a943688123ef58e1b1f0a5af04e0c005027ad`**, `--basetemp=/tmp/pytest-exp11-20260928-091024` | **4197 passed, 56 skipped, 0 failed** (4054 s) |

The log is `orientation_cue_fairness_2026-09-28_09:10_suite_full_cpu_464a943.log`, whose
first two lines are the HEAD and the basetemp. **Nothing fails**, in the whole repository:

* the three exp_11 tests this cycle fixed now pass — including
  `test_the_int_handler_terminates_too`, in a suite launched exactly as the Planner's
  was (`setsid nohup … &`), which is the point of the fix;
* the 9 `tests/test_exp06_launch.py` preflight tests and the 12
  `tests/test_exp10_record_tools.py` tests pass, because this basetemp has no colons and
  their PATH shims were found;
* `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`
  passes for the first time since round 2 began: the re-fill at `6baa337` is exactly the
  merge-time action every fix report said it was waiting for.

It was launched with a backgrounded `setsid nohup`, i.e. with SIGINT ignored on entry —
deliberately, so that the INT test is shown to be independent of that. The basetemp held
5.4 GB and was removed as soon as the run finished; the root filesystem is at 34 GB free.

## Next tests-only item (cycle 21), per the coordinator's second addendum

Not done here — the full suite was already running when the addendum arrived, and this
cycle may touch no production file and should not grow a second subject mid-run:

* Harden the PATH-prepended fakes so they survive a **relative `TMPDIR`** as well as a
  colon in `--basetemp`: create the shim directory with `tempfile.mkdtemp(dir='/tmp')`
  (absolute, colon-free) rather than under `tmp_path`, and clean it up.
* The one site in this repository is `tests/test_exp06_launch.py:71` (the `nvidia-smi`
  fake: `monkeypatch.setenv('PATH', str(binary) + os.pathsep + os.environ['PATH'])`),
  plus the peer's `mv`/`cp` shims in `tests/test_exp10_record_tools.py`. **No exp_11 test
  prepends anything to PATH** — checked by grep — so exp_11 has nothing of its own to
  harden; the fix belongs to the exp_06 file (frozen for exp_11's *production* closure
  but a test file, so a tests-only cycle may touch it if the Planner says so) and to the
  peer's file.
* Add the regression the addendum describes: build the fake under a `tmp_path`
  subdirectory whose name contains a colon and assert the shim is still the one that
  runs.
