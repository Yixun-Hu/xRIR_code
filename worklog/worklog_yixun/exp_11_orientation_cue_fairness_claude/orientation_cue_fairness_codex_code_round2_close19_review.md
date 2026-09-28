**Close review 19 — exp_11 `orientation_cue_fairness`, round 2**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized temporary test execution under `/tmp` |
| Worktree | `/home/yixunhu/codespace/xRIR_code_wt` |
| Branch | `exp11-cue` |
| HEAD | `46f0a76ce3b1fb3ae0fe5445116cf43c06832e87` |
| Last production commit | `659747480963eb103f43650a4d15dfd24c6d440e` |
| Completed | 2026-09-28 11:51:11 UTC / 07:51:11 EDT |
| Execution | CUDA disabled; numeric threads limited to one per worker; at most four workers; pytest cache disabled |

**Verdict: approve. Round 2 can be CLOSED and `46f0a76` merged, with the approval records filled as specified below.** Both close-18 P2 findings are resolved. No new production blocker was found. The documentation and coverage limitations below are nonblocking.

**P2-1 is closed.** In [tools/exp11_launch.sh:155](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:155), the embedded Python canonicalizes only the attempt. Grep and inspection confirm that no `.resolve()` or path-name equality remains in the launcher. The profile and `attempt_<UTC>` checks remain; containment is decided by `same_parent` at line 194.

Independent replay used the real sourced lifecycle, publication and registration locks, and a live stub calling the real `register_trainer`:

- The stub registered and was detected alive.
- Concurrent recovery through the arm alias refused while the publication lock was held.
- Normal exit produced the trainer exit record, child exit receipt and end marker, and removed `launching`.
- Subsequent locked recovery accepted the alias and actually published `final -> attempt_20260927T000000`.
- An attempt in another directory refused without changing `final`.

This verifies lifecycle and publication behavior; it does not claim model finalization.

The existing bind spellings remain:

```text
/usr/share/hunspell
/var/snap/firefox/common/host-hunspell

Both: st_dev=66306, st_ino=8257996
```

Through the sourced launcher, `same_path(first, second)` returned success, as did `same_parent(second/imaginary_attempt, first)`. Thus, if a valid attempt existed beneath the second spelling, its containment would pass with the first configured as root. No mount or protected attempt was created.

The committed bind test **skips**, but its explanation is inaccurate: [tests/test_exp11_shell.py:2333](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:2333) compares only read-only mountpoints. It misses the source alias because `/usr/share/hunspell` is not itself a separate mountpoint. The skip does not establish that this machine lacks an identity-equivalent pair.

**P2-2 is closed.** [tools/exp11_train.py:312](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:312) derives the canonical attempt and its arm, checks actual directory existence, and compares the normalized lexical parent against that arm by followed `stat` identity. An explicit `arm_root` must additionally identify the same arm. Unknown identity refuses before registration.

Fourteen independent subprocess cases used real registration locks and live registered children:

| Case | Result |
|---|---|
| Default `attempt_X/sub/..` | Registers |
| Default `arm//attempt_X` | Registers |
| Default `arm/./attempt_X` | Registers |
| Default trailing slash | Registers |
| Default `arm_alias/attempt_X`, alias → arm | Registers |
| Different arm’s attempt symlink → canonical attempt | Exit 3; no sidecar or registration lock in either arm |
| Missing component before `..` | Refuses actual lookup |
| Actual path exists; normalized lexical parent absent | Refuses unknown identity; no lock or sidecar |
| Explicit identity-equivalent arm alias | Registers |
| Explicit different arm root | Refuses before lock/sidecar |
| Arm alias → arm, whose attempt points into a third arm | Refuses |
| Arm alias → distinct directory, whose attempt points into the canonical arm | Refuses |
| Alternate attempt symlink within the same arm | Registers correctly |
| Exact same-name self-referential construction | Refuses ELOOP |

The requested symlink construction cannot establish wrong-arm acceptance in a stable namespace:

1. If `arm_other` identifies a distinct directory, its attempt symlink into `arm` fails the parent identity comparison.
2. If `arm_other -> arm`, its lexical parent identifies `arm` itself.
3. If that arm’s attempt then points into a third arm, canonicalization changes the comparison’s other operand, and identity fails.
4. Combining `arm_other -> arm` with `arm_other/attempt_X -> arm/attempt_X` makes `arm/attempt_X` point to itself.
5. A different attempt name within the same arm can succeed legitimately; its directory, tombstone and lock remain in that arm.

`normpath` only proposes a parent to compare. It cannot make a different directory acquire the canonical parent’s device/inode, and it does not choose the sidecar destination or lock.

There is a coverage nuance at [tests/test_exp11_train.py:525](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_train.py:525): the named “missing lexical parent” test actually fails raw lookup because `gone` is absent. The independent replay also covered the stronger case: `prefix/link -> actual/deep`, with `prefix/link/../arm/attempt_X` existing while normalized `prefix/arm` does not. That reached the identity gate and refused.

**Validation completed: 384 selected tests passed, one skipped.**

| Selected checks | Result |
|---|---:|
| `tests/test_exp11_shell.py` | 212 passed, 1 skipped — **213 collected** |
| `tests/test_exp11_train.py` | **65 passed** |
| Path-probe and PID-record files | 98 passed |
| Sim-eval closures, registry identity and pinned-output comparisons | 9 passed |

Shell syntax, in-memory compilation of all ten exp_11 Python modules, and `git diff --check f2f6943 46f0a76` also pass.

For RED evidence, the trainer source from **`44656e1`** was extracted under `/tmp` and replayed against the four spelling regressions. Result: **1 failed, 3 passed**. The failure was exactly default `attempt_X/sub/..`, exiting 3 because the old code treated `attempt_X/sub` as its arm. HEAD passes all four. These repeated cases are not added to the distinct-test total above.

The six commits and production totals match the requested range:

```text
launcher: +5 / −5
trainer:  +33 / −16
total:    +38 / −21
tests:    +127 / −2
```

The retained full-suite log begins exactly:

```text
HEAD 659747480963eb103f43650a4d15dfd24c6d440e
```

It records **1 failed, 4003 passed, 56 skipped in 3595.29 seconds**. The sole failure is the stale exp_06 `summarize_haa` approval digest. This is the tested production HEAD; `46f0a76` adds bookkeeping. I inspected that retained evidence rather than rerunning the full suite.

The timeout report correctly withdraws the approximately-180-second overall bound. Each reader/probe invocation receives `READER_TIMEOUT_S`—default 30, validated within 1–300—plus the five-second kill grace. This is a per-question timeout policy, not a bound on a complete scan.

The remaining nonblocking record corrections are:

| Location | Correction |
|---|---|
| [tools/exp11_launch.sh:49](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:49) | The old approximately-60-second example remains and omits kill grace. |
| [tools/exp11_launch.sh:552](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:552) | The health check asks four path/identity questions, plus two PID-reader questions. |
| [Coder report:102](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_06:43:58_coder_round2_fix19_opus_report.md:102) | “Adds four” should distinguish four path probes from six total timed health-check children. |
| [Coder report:177](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_06:43:58_coder_round2_fix19_opus_report.md:177) | The bind test’s discovery limitation does not establish absence of aliases. |
| [Coder report:86](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_06:43:58_coder_round2_fix19_opus_report.md:86) | All four production/test commits are below 200 changed lines; the two record-only commits total **634** and **249**, respectively. |

**Closure hygiene passes.**

- All **28 code digests**—20 exp_06 and eight exp_11—agree between `6597474` and `46f0a76`.
- Against the exp_06 record at `9f98bbb`, only `code.summarize_haa` moves; the other 19 match.
- Five exp_11 keys move versus `f2f6943`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged. All eight exact values appear below.
- Sim-eval closures remain unchanged; all three `tests/test_exp09_sim_eval_closures.py` tests pass.
- The 28 checked frozen source/shell files, plus the frozen exp_06 JSON template, match `9f98bbb`. All 12 exp_03 pinned source records match their reviewed hashes.
- Pinned exp_06/exp_09 serialized outputs and rendered summaries, and the exp_11 phase-1 comparison, are byte-identical.
- `BACKBONES_EXP06` is unchanged.
- `ckpt/exp11/` is absent.

Frozen identities:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479

BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

**The exact merge fill follows.** It applies when the merge retains the reviewed production bytes. Any production edit, including launcher comments, requires recomputing affected closure hashes.

In exp_06’s existing approval record, replace **only** `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
3956692a6ae37d0c7bb59dcc11941f7e186b110e85937d144bf7cdd128d1de0a
```

Preserving every other byte yields a **3,190-byte** record with SHA256:

```text
38fe553af63644b6b9ea79df3da7ffc09596304698cbdafed7ef0b83798c2bcb
```

The complete exp_11 fill is:

```json
{
  "artifacts": {
    "simpor_epoch_012": {
      "epoch": null,
      "path": null,
      "sha256": null
    },
    "simpor_yaw_epoch_012": {
      "epoch": null,
      "path": null,
      "sha256": null
    }
  },
  "code": {
    "finalize": "6a747806cdd5a84809c4f2b2529245e4fb0ca9f777ef4b174287dae653bca574",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "a932e5bfd087394ee467700e803aa78973037be62cfa0ef9a7fd2d92efd5caac",
    "smoke": "22f989b7541c90dfca53420bb775a6c82b6927999d0c430943dff475ced03092",
    "summarize_haa": "3956692a6ae37d0c7bb59dcc11941f7e186b110e85937d144bf7cdd128d1de0a",
    "train": "56004f44a2f88625882904ac622c6179c126cfef89333714eca5ff7ccc6fc0a8"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "38fe553af63644b6b9ea79df3da7ffc09596304698cbdafed7ef0b83798c2bcb",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one final newline, this exp_11 record is **1,316 bytes**, SHA256:

```text
4efa0eae70e6f80da1ec1050c3b252d949c8b72e33bc436697f3e4deb2accdea
```

No review or approval record was saved. All review temporary directories were removed; disk remains approximately 35 GB free. HEAD and worktree status are unchanged, with only the pre-existing untracked close19 prompt.
tokens used
76,630
**Close review 19 — exp_11 `orientation_cue_fairness`, round 2**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized temporary test execution under `/tmp` |
| Worktree | `/home/yixunhu/codespace/xRIR_code_wt` |
| Branch | `exp11-cue` |
| HEAD | `46f0a76ce3b1fb3ae0fe5445116cf43c06832e87` |
| Last production commit | `659747480963eb103f43650a4d15dfd24c6d440e` |
| Completed | 2026-09-28 11:51:11 UTC / 07:51:11 EDT |
| Execution | CUDA disabled; numeric threads limited to one per worker; at most four workers; pytest cache disabled |

**Verdict: approve. Round 2 can be CLOSED and `46f0a76` merged, with the approval records filled as specified below.** Both close-18 P2 findings are resolved. No new production blocker was found. The documentation and coverage limitations below are nonblocking.

**P2-1 is closed.** In [tools/exp11_launch.sh:155](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:155), the embedded Python canonicalizes only the attempt. Grep and inspection confirm that no `.resolve()` or path-name equality remains in the launcher. The profile and `attempt_<UTC>` checks remain; containment is decided by `same_parent` at line 194.

Independent replay used the real sourced lifecycle, publication and registration locks, and a live stub calling the real `register_trainer`:

- The stub registered and was detected alive.
- Concurrent recovery through the arm alias refused while the publication lock was held.
- Normal exit produced the trainer exit record, child exit receipt and end marker, and removed `launching`.
- Subsequent locked recovery accepted the alias and actually published `final -> attempt_20260927T000000`.
- An attempt in another directory refused without changing `final`.

This verifies lifecycle and publication behavior; it does not claim model finalization.

The existing bind spellings remain:

```text
/usr/share/hunspell
/var/snap/firefox/common/host-hunspell

Both: st_dev=66306, st_ino=8257996
```

Through the sourced launcher, `same_path(first, second)` returned success, as did `same_parent(second/imaginary_attempt, first)`. Thus, if a valid attempt existed beneath the second spelling, its containment would pass with the first configured as root. No mount or protected attempt was created.

The committed bind test **skips**, but its explanation is inaccurate: [tests/test_exp11_shell.py:2333](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:2333) compares only read-only mountpoints. It misses the source alias because `/usr/share/hunspell` is not itself a separate mountpoint. The skip does not establish that this machine lacks an identity-equivalent pair.

**P2-2 is closed.** [tools/exp11_train.py:312](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:312) derives the canonical attempt and its arm, checks actual directory existence, and compares the normalized lexical parent against that arm by followed `stat` identity. An explicit `arm_root` must additionally identify the same arm. Unknown identity refuses before registration.

Fourteen independent subprocess cases used real registration locks and live registered children:

| Case | Result |
|---|---|
| Default `attempt_X/sub/..` | Registers |
| Default `arm//attempt_X` | Registers |
| Default `arm/./attempt_X` | Registers |
| Default trailing slash | Registers |
| Default `arm_alias/attempt_X`, alias → arm | Registers |
| Different arm’s attempt symlink → canonical attempt | Exit 3; no sidecar or registration lock in either arm |
| Missing component before `..` | Refuses actual lookup |
| Actual path exists; normalized lexical parent absent | Refuses unknown identity; no lock or sidecar |
| Explicit identity-equivalent arm alias | Registers |
| Explicit different arm root | Refuses before lock/sidecar |
| Arm alias → arm, whose attempt points into a third arm | Refuses |
| Arm alias → distinct directory, whose attempt points into the canonical arm | Refuses |
| Alternate attempt symlink within the same arm | Registers correctly |
| Exact same-name self-referential construction | Refuses ELOOP |

The requested symlink construction cannot establish wrong-arm acceptance in a stable namespace:

1. If `arm_other` identifies a distinct directory, its attempt symlink into `arm` fails the parent identity comparison.
2. If `arm_other -> arm`, its lexical parent identifies `arm` itself.
3. If that arm’s attempt then points into a third arm, canonicalization changes the comparison’s other operand, and identity fails.
4. Combining `arm_other -> arm` with `arm_other/attempt_X -> arm/attempt_X` makes `arm/attempt_X` point to itself.
5. A different attempt name within the same arm can succeed legitimately; its directory, tombstone and lock remain in that arm.

`normpath` only proposes a parent to compare. It cannot make a different directory acquire the canonical parent’s device/inode, and it does not choose the sidecar destination or lock.

There is a coverage nuance at [tests/test_exp11_train.py:525](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_train.py:525): the named “missing lexical parent” test actually fails raw lookup because `gone` is absent. The independent replay also covered the stronger case: `prefix/link -> actual/deep`, with `prefix/link/../arm/attempt_X` existing while normalized `prefix/arm` does not. That reached the identity gate and refused.

**Validation completed: 384 selected tests passed, one skipped.**

| Selected checks | Result |
|---|---:|
| `tests/test_exp11_shell.py` | 212 passed, 1 skipped — **213 collected** |
| `tests/test_exp11_train.py` | **65 passed** |
| Path-probe and PID-record files | 98 passed |
| Sim-eval closures, registry identity and pinned-output comparisons | 9 passed |

Shell syntax, in-memory compilation of all ten exp_11 Python modules, and `git diff --check f2f6943 46f0a76` also pass.

For RED evidence, the trainer source from **`44656e1`** was extracted under `/tmp` and replayed against the four spelling regressions. Result: **1 failed, 3 passed**. The failure was exactly default `attempt_X/sub/..`, exiting 3 because the old code treated `attempt_X/sub` as its arm. HEAD passes all four. These repeated cases are not added to the distinct-test total above.

The six commits and production totals match the requested range:

```text
launcher: +5 / −5
trainer:  +33 / −16
total:    +38 / −21
tests:    +127 / −2
```

The retained full-suite log begins exactly:

```text
HEAD 659747480963eb103f43650a4d15dfd24c6d440e
```

It records **1 failed, 4003 passed, 56 skipped in 3595.29 seconds**. The sole failure is the stale exp_06 `summarize_haa` approval digest. This is the tested production HEAD; `46f0a76` adds bookkeeping. I inspected that retained evidence rather than rerunning the full suite.

The timeout report correctly withdraws the approximately-180-second overall bound. Each reader/probe invocation receives `READER_TIMEOUT_S`—default 30, validated within 1–300—plus the five-second kill grace. This is a per-question timeout policy, not a bound on a complete scan.

The remaining nonblocking record corrections are:

| Location | Correction |
|---|---|
| [tools/exp11_launch.sh:49](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:49) | The old approximately-60-second example remains and omits kill grace. |
| [tools/exp11_launch.sh:552](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:552) | The health check asks four path/identity questions, plus two PID-reader questions. |
| [Coder report:102](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_06:43:58_coder_round2_fix19_opus_report.md:102) | “Adds four” should distinguish four path probes from six total timed health-check children. |
| [Coder report:177](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_06:43:58_coder_round2_fix19_opus_report.md:177) | The bind test’s discovery limitation does not establish absence of aliases. |
| [Coder report:86](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_06:43:58_coder_round2_fix19_opus_report.md:86) | All four production/test commits are below 200 changed lines; the two record-only commits total **634** and **249**, respectively. |

**Closure hygiene passes.**

- All **28 code digests**—20 exp_06 and eight exp_11—agree between `6597474` and `46f0a76`.
- Against the exp_06 record at `9f98bbb`, only `code.summarize_haa` moves; the other 19 match.
- Five exp_11 keys move versus `f2f6943`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged. All eight exact values appear below.
- Sim-eval closures remain unchanged; all three `tests/test_exp09_sim_eval_closures.py` tests pass.
- The 28 checked frozen source/shell files, plus the frozen exp_06 JSON template, match `9f98bbb`. All 12 exp_03 pinned source records match their reviewed hashes.
- Pinned exp_06/exp_09 serialized outputs and rendered summaries, and the exp_11 phase-1 comparison, are byte-identical.
- `BACKBONES_EXP06` is unchanged.
- `ckpt/exp11/` is absent.

Frozen identities:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479

BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

**The exact merge fill follows.** It applies when the merge retains the reviewed production bytes. Any production edit, including launcher comments, requires recomputing affected closure hashes.

In exp_06’s existing approval record, replace **only** `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
3956692a6ae37d0c7bb59dcc11941f7e186b110e85937d144bf7cdd128d1de0a
```

Preserving every other byte yields a **3,190-byte** record with SHA256:

```text
38fe553af63644b6b9ea79df3da7ffc09596304698cbdafed7ef0b83798c2bcb
```

The complete exp_11 fill is:

```json
{
  "artifacts": {
    "simpor_epoch_012": {
      "epoch": null,
      "path": null,
      "sha256": null
    },
    "simpor_yaw_epoch_012": {
      "epoch": null,
      "path": null,
      "sha256": null
    }
  },
  "code": {
    "finalize": "6a747806cdd5a84809c4f2b2529245e4fb0ca9f777ef4b174287dae653bca574",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "a932e5bfd087394ee467700e803aa78973037be62cfa0ef9a7fd2d92efd5caac",
    "smoke": "22f989b7541c90dfca53420bb775a6c82b6927999d0c430943dff475ced03092",
    "summarize_haa": "3956692a6ae37d0c7bb59dcc11941f7e186b110e85937d144bf7cdd128d1de0a",
    "train": "56004f44a2f88625882904ac622c6179c126cfef89333714eca5ff7ccc6fc0a8"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "38fe553af63644b6b9ea79df3da7ffc09596304698cbdafed7ef0b83798c2bcb",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one final newline, this exp_11 record is **1,316 bytes**, SHA256:

```text
4efa0eae70e6f80da1ec1050c3b252d949c8b72e33bc436697f3e4deb2accdea
```

No review or approval record was saved. All review temporary directories were removed; disk remains approximately 35 GB free. HEAD and worktree status are unchanged, with only the pre-existing untracked close19 prompt.

