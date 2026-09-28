**Close review 18 — exp_11 `orientation_cue_fairness`, round 2**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized `/tmp` fixtures and owned helper processes |
| Worktree | `/home/yixunhu/codespace/xRIR_code_wt` |
| Branch | `exp11-cue` |
| HEAD | `f2f69433884c33642c6fd9ed98af67908e0d2c5f` |
| Last production commit | `d420ddb4443cf3212899b01534608b3f39491165` |
| Completed | 2026-09-28 10:35:03 UTC / 06:35:03 EDT |
| Execution | UID 1013; CUDA disabled; numeric worker threads limited to one; at most four concurrent workers; pytest cache disabled; temporary fixtures under `/tmp` |

**Verdict: request changes. Round 2 cannot be CLOSED, and `f2f6943` should not be merged yet.**

The close-17 path-payload vulnerabilities are eliminated from the probe protocol. Two P2 gaps remain.

1. **[P2] Recovery still makes an earlier string-based containment decision.**

   [tools/exp11_launch.sh:152](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:152) retains:

   ```python
   directory, base = Path(attempt).resolve(), Path(root).resolve()
   if directory.parent != base:
   ```

   This runs before the new `same_parent` gate. Therefore the audit’s claim that every pathname comparison uses identity is false, and an identity-equivalent bind alias can be rejected.

   Read-only verification against an existing bind mount showed:

   ```text
   /usr/share/hunspell
   /var/snap/firefox/common/host-hunspell

   Both: st_dev=66306, st_ino=8257996
   exp11_pathprobe.same: same
   Path.resolve() equality: False
   ```

   `sameparent` for an attempt pathname beneath the second spelling and the first arm root returns `same`; the inline comparison necessarily rejects that layout before reaching the identity gate. I did not create a mount or an attempt in these protected directories.

   **Required:** replace this comparison with an in-process `samestat`, or remove the redundant comparison and retain the later identity gate. Add a regression covering identity-equivalent parents with different resolved spellings. This is a conservative rejection, not a demonstrated unsafe publication.

2. **[P2] The normal trainer entry point rejects the promised terminal `sub/..` spelling.**

   [tools/exp11_train.py:323](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:323) derives the default arm root from `given.parent`. For:

   ```text
   arm/attempt_20260928T010203/sub/..
   ```

   that parent is `arm/attempt_20260928T010203/sub`, although the canonical attempt’s parent is `arm`.

   Independent replay established:

   ```text
   same(input, canonical attempt): same
   sameparent(canonical attempt, actual arm): same
   register_trainer(input): exit 3
   register_trainer(input, arm_root=actual_arm): succeeds
   ```

   The default call creates neither `train.pid` nor a lock. Its diagnostic says the attempt belongs to `arm`, rather than the supplied `…/attempt_…/sub`.

   [main:389](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:389) supplies no `arm_root`. The new [regression:491](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_train.py:491) supplies it explicitly, masking the runtime refusal.

   **Required:** support this spelling through the default entry point and test without that override, while preserving rejection of an attempt symlink into another arm. Simply trusting the canonical parent would remove that existing protection.

The remaining requested checks produced the following results.

| Check | Finding |
|---|---|
| Probe verdict grammar | No pathname or link target appears in a successful verdict. `valid_path` and probe canonical-path output are gone. |
| Identity spellings | Symlinked arm roots, `//`, `/./`, trailing slash and existing `sub/..` resolve to the same identity. Trainer default-entry qualification is blocker 2. |
| Hard links | Regular-file hard links return `same`. Correct: this asks inode identity. Ordinary attempt directories cannot be hard-linked this way. |
| Bind mounts | A bind alias normally retains device/inode identity; the existing mount above returns `same`. Different resolved strings do not imply different identity. |
| Copies | An independent identical-content copy returns `different`. Appropriate: content equality does not establish attempt identity. |
| Dangling `final` | Unknown; publication, conflict, abort and resolution guards refuse. |
| Scan exclusion | Uses `same_path` after all three PID checks. A live registered trainer still refuses even when its directory is the resolution target under another spelling. |
| Scan cost | One additional probe per reached directory when a resolution target is supplied. Ordinary scans without an exclusion target incur no such identity probes. |
| Attempt naming | Real `%Y%m%dT%H%M%S` timestamps always match the digits/`T` pattern. The pattern is permissive rather than a calendar validator. |
| `final` as input | The scan never enumerates literal `final`. Recovery canonicalizes it to its attempt target; resolution recognizes that target as published and refuses before mutation. |
| Trainer tombstone | `arm/attempt/sub/..` finds the tombstone and exits 3 in both save and `--no-save` modes. |
| Cross-arm alias | Refuses before creating the wrong arm’s registration lock. |
| `--no-save` | Remains sidecar-free and lock-free; an absent diagnostic directory is allowed when its tombstone check is answerable. |
| Trainer/launcher lock identity | No case was found where an accepted registration uses a different arm lock from the launcher’s identity decision. A whole-arm bind alias exposes the same lock inode. |
| Health check | Four path questions: present, absent, same file, different files; plus the two existing PID-reader questions. |
| Threat model | Both implementations explicitly defend against malformed answers and trust the pinned module’s truthful answers. |

The probe’s `sameparent` means the parent **as written**. Lifecycle callers canonicalize the attempt before asking that question. The trainer’s default-root discrepancy is why blocker 2 matters.

I independently replayed the real sourced lifecycle with real publication locks and a live helper registered through `register_trainer`. The malformed-answer matrix included:

```text
present
present 81a4 /forged parent/x
present 81a4␠
absent yes
link /x
same different
yes
```

All seven returned unknown through each of `probe_path`, `file_present`, `probe_link`, `same_path` and `same_parent`. Injecting them separately into `link` and `same` made `publishes`, `already_published`, both replacement settings of `check_final_conflict`, `abort_recovery` and `resolve_unregistered` refuse with status 2. Malformed containment and scan-exclusion answers also refused.

The attempt, publication, log and registered live helper remained intact; no tombstone was written. Only my own helper was terminated through its subprocess handle. Separate dangling-`final` replays returned unknown across all five publication/recovery checks. A quiet resolution target reached through a symlinked root and `//./…/sub/..` was correctly excluded.

Thus close-17’s space-bearing and overlong path payloads cannot authorize decisions through this protocol: there is no path field to accept. Raw-byte checking still rejects extra lines and NUL padding. This does not imply protection against a pinned module deliberately returning a false but grammatically valid `same`.

`publishes` performs two probes, but all production publication mutations occur under the arm’s publication lock, which the caller holds. The trainer never retargets `final`. Cooperative lifecycle participants therefore cannot change `final` between those probes. An unrelated manual filesystem mutation or mount change can bypass advisory locking; that is outside the stated coordination model.

The audit table is otherwise consistent with the implementation: sidecar decisions use the probe; scan exclusion and publication checks use identity; resolution containment uses `same_parent`; `readlink` is informational. Checked enumeration, classification and rename-path operations remain justified. **The inline comparison in blocker 1 prevents accepting the audit as complete.**

**Validation completed: 304 distinct selected tests passed.**

| Selected checks | Result |
|---|---:|
| `tests/test_exp11_shell.py` | 209 passed |
| `tests/test_exp11_pathprobe.py` | 23 passed |
| `tests/test_exp11_train.py` | 62 passed |
| `tests/test_exp09_sim_eval_closures.py` | 3 passed |
| Registry isolation and pinned-output comparisons | 6 passed |
| Exact exp_04 `[True-promote]` case | 1 passed |

Shell syntax and `git diff --check 2e109d0 f2f6943` also pass. Passing tests do not cover the two findings above.

For RED evidence, I replayed `test_no_verdict_carries_a_path` using the actual module and test from **`2e76a95`**, extracted under `/tmp`. It failed as expected because stdout still contained:

```text
present 81b4 /tmp/.../launching
```

The eight-commit history matches the requested range. Production changes total **+211/−151**; test changes total **+169/−43**. The reported file counts—23 pathprobe, 209 shell and 62 trainer tests—are correct.

The retained full-suite log begins exactly:

```text
HEAD d420ddb4443cf3212899b01534608b3f39491165
```

It records **1 failed, 3997 passed, 55 skipped**, in 3454.56 seconds. The sole failure is the stale exp_06 `summarize_haa` approval digest. No superseded `87f4e4d` partial suite log remains in the worktree. I inspected the retained full-suite evidence; I did not rerun the entire suite.

Several record corrections are still needed:

- Fix-17’s actual suite HEAD is correctly amended to `fef89ba`; both retained logs expose that SHA.
- The revised timeout accounting correctly identifies six ordinary per-attempt calls and approximately 105 seconds for three TERM-ignoring sidecar probes. Approximately 180 seconds is six base 30-second budgets, **not a universal bound**: kill grace, health checks, identity probes, additional entries and early refusal affect actual duration. [Launcher:45](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:45) still contains the obsolete approximately-60-second explanation.
- The [report:122](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_05:30:00_coder_round2_fix18_opus_report.md:122) still puts the exp_04 completion transaction on a background thread. Only the SIGTERM sender runs there; `execute_attempt` runs on the main thread. “Timing-sensitive” is supported; co-tenancy is unproven. The isolated case passes, and exp_04 files are unchanged since `9f98bbb`.
- The [report:143](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_05:30:00_coder_round2_fix18_opus_report.md:143) incorrectly says every production/test commit is below 200 changed lines. `87f4e4d` totals **252**: 197 production plus 55 test lines.
- The earlier oversized record-only commits are named, but their quoted counts are wrong in this checkout. `git show --numstat` totals **672** for `a700a24` and **707** for `cad6018`, rather than 624 and 538. This cycle’s record-only commits are **654** lines for `98c82b3` and **248** for `f2f6943`.
- [Launcher:552](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:552) says “three questions” although the path health check asks four.

Closure hygiene passes for the reviewed candidate:

- All **28 exp_06/exp_11 code digests** agree between `d420ddb` and `f2f6943`.
- Against the exp_06 record at `9f98bbb`, only `code.summarize_haa` moves; the other 19 keys match.
- Five exp_11 keys move versus `2e109d0`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged. All eight exact values appear below.
- Sim-eval closures remain unchanged; all three closure tests pass.
- The 28 checked existing frozen files match `9f98bbb`; all 12 exp_03 pinned source records match their reviewed hashes.
- `BACKBONES_EXP06` is unchanged. Pinned exp_06/exp_09 and exp_11 phase-1 output comparisons are byte-identical.
- `ckpt/exp11` is entirely absent.
- Git status is unchanged: only the pre-existing untracked close18 prompt. No review or approval record was saved.

The frozen identities are:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479

BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

The exact fill manifest below applies to a merge retaining **the reviewed `f2f6943` production bytes**. It is not authorization to fill or merge this rejected candidate; recompute it after correcting the blockers.

In exp_06’s existing approvals record, replace only `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
85caf08ce7cbf2d479349fbf67825c8cb0feff9be8822d41ef76f08d2f4716d7
```

Preserving every other byte produces a **3,190-byte** exp_06 record with SHA256:

```text
c33e2475da5dbf3ad6d2144d9faac28d140a42f20ba8bfae2d8c514b12c870ac
```

The corresponding complete exp_11 fill is:

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
    "finalize": "197f5bdeb6753e48cad1075299f2cd4bd86fd4ba1a7cfe7639ee95a39142c213",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "d4dab1e4a51fcac65b5ea6f307c66f43e601939f9670361219000d24b1202d19",
    "smoke": "c8710213e7eadf16193e8f664e7dd7c60ecee13fd34ad6ffd01f307eac2a83ff",
    "summarize_haa": "85caf08ce7cbf2d479349fbf67825c8cb0feff9be8822d41ef76f08d2f4716d7",
    "train": "d712fa89fc7b35f7785f970cb1562d432c5120cfdab55848e32318ddc5c0dc30"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "c33e2475da5dbf3ad6d2144d9faac28d140a42f20ba8bfae2d8c514b12c870ac",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one final newline, that exp_11 record is **1,316 bytes**, SHA256:

```text
7459520df0d736bb18882da223f7e57df90a7fbc1c58458e4d1100de3804467e
```
tokens used
104,526
**Close review 18 — exp_11 `orientation_cue_fairness`, round 2**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized `/tmp` fixtures and owned helper processes |
| Worktree | `/home/yixunhu/codespace/xRIR_code_wt` |
| Branch | `exp11-cue` |
| HEAD | `f2f69433884c33642c6fd9ed98af67908e0d2c5f` |
| Last production commit | `d420ddb4443cf3212899b01534608b3f39491165` |
| Completed | 2026-09-28 10:35:03 UTC / 06:35:03 EDT |
| Execution | UID 1013; CUDA disabled; numeric worker threads limited to one; at most four concurrent workers; pytest cache disabled; temporary fixtures under `/tmp` |

**Verdict: request changes. Round 2 cannot be CLOSED, and `f2f6943` should not be merged yet.**

The close-17 path-payload vulnerabilities are eliminated from the probe protocol. Two P2 gaps remain.

1. **[P2] Recovery still makes an earlier string-based containment decision.**

   [tools/exp11_launch.sh:152](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:152) retains:

   ```python
   directory, base = Path(attempt).resolve(), Path(root).resolve()
   if directory.parent != base:
   ```

   This runs before the new `same_parent` gate. Therefore the audit’s claim that every pathname comparison uses identity is false, and an identity-equivalent bind alias can be rejected.

   Read-only verification against an existing bind mount showed:

   ```text
   /usr/share/hunspell
   /var/snap/firefox/common/host-hunspell

   Both: st_dev=66306, st_ino=8257996
   exp11_pathprobe.same: same
   Path.resolve() equality: False
   ```

   `sameparent` for an attempt pathname beneath the second spelling and the first arm root returns `same`; the inline comparison necessarily rejects that layout before reaching the identity gate. I did not create a mount or an attempt in these protected directories.

   **Required:** replace this comparison with an in-process `samestat`, or remove the redundant comparison and retain the later identity gate. Add a regression covering identity-equivalent parents with different resolved spellings. This is a conservative rejection, not a demonstrated unsafe publication.

2. **[P2] The normal trainer entry point rejects the promised terminal `sub/..` spelling.**

   [tools/exp11_train.py:323](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:323) derives the default arm root from `given.parent`. For:

   ```text
   arm/attempt_20260928T010203/sub/..
   ```

   that parent is `arm/attempt_20260928T010203/sub`, although the canonical attempt’s parent is `arm`.

   Independent replay established:

   ```text
   same(input, canonical attempt): same
   sameparent(canonical attempt, actual arm): same
   register_trainer(input): exit 3
   register_trainer(input, arm_root=actual_arm): succeeds
   ```

   The default call creates neither `train.pid` nor a lock. Its diagnostic says the attempt belongs to `arm`, rather than the supplied `…/attempt_…/sub`.

   [main:389](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:389) supplies no `arm_root`. The new [regression:491](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_train.py:491) supplies it explicitly, masking the runtime refusal.

   **Required:** support this spelling through the default entry point and test without that override, while preserving rejection of an attempt symlink into another arm. Simply trusting the canonical parent would remove that existing protection.

The remaining requested checks produced the following results.

| Check | Finding |
|---|---|
| Probe verdict grammar | No pathname or link target appears in a successful verdict. `valid_path` and probe canonical-path output are gone. |
| Identity spellings | Symlinked arm roots, `//`, `/./`, trailing slash and existing `sub/..` resolve to the same identity. Trainer default-entry qualification is blocker 2. |
| Hard links | Regular-file hard links return `same`. Correct: this asks inode identity. Ordinary attempt directories cannot be hard-linked this way. |
| Bind mounts | A bind alias normally retains device/inode identity; the existing mount above returns `same`. Different resolved strings do not imply different identity. |
| Copies | An independent identical-content copy returns `different`. Appropriate: content equality does not establish attempt identity. |
| Dangling `final` | Unknown; publication, conflict, abort and resolution guards refuse. |
| Scan exclusion | Uses `same_path` after all three PID checks. A live registered trainer still refuses even when its directory is the resolution target under another spelling. |
| Scan cost | One additional probe per reached directory when a resolution target is supplied. Ordinary scans without an exclusion target incur no such identity probes. |
| Attempt naming | Real `%Y%m%dT%H%M%S` timestamps always match the digits/`T` pattern. The pattern is permissive rather than a calendar validator. |
| `final` as input | The scan never enumerates literal `final`. Recovery canonicalizes it to its attempt target; resolution recognizes that target as published and refuses before mutation. |
| Trainer tombstone | `arm/attempt/sub/..` finds the tombstone and exits 3 in both save and `--no-save` modes. |
| Cross-arm alias | Refuses before creating the wrong arm’s registration lock. |
| `--no-save` | Remains sidecar-free and lock-free; an absent diagnostic directory is allowed when its tombstone check is answerable. |
| Trainer/launcher lock identity | No case was found where an accepted registration uses a different arm lock from the launcher’s identity decision. A whole-arm bind alias exposes the same lock inode. |
| Health check | Four path questions: present, absent, same file, different files; plus the two existing PID-reader questions. |
| Threat model | Both implementations explicitly defend against malformed answers and trust the pinned module’s truthful answers. |

The probe’s `sameparent` means the parent **as written**. Lifecycle callers canonicalize the attempt before asking that question. The trainer’s default-root discrepancy is why blocker 2 matters.

I independently replayed the real sourced lifecycle with real publication locks and a live helper registered through `register_trainer`. The malformed-answer matrix included:

```text
present
present 81a4 /forged parent/x
present 81a4␠
absent yes
link /x
same different
yes
```

All seven returned unknown through each of `probe_path`, `file_present`, `probe_link`, `same_path` and `same_parent`. Injecting them separately into `link` and `same` made `publishes`, `already_published`, both replacement settings of `check_final_conflict`, `abort_recovery` and `resolve_unregistered` refuse with status 2. Malformed containment and scan-exclusion answers also refused.

The attempt, publication, log and registered live helper remained intact; no tombstone was written. Only my own helper was terminated through its subprocess handle. Separate dangling-`final` replays returned unknown across all five publication/recovery checks. A quiet resolution target reached through a symlinked root and `//./…/sub/..` was correctly excluded.

Thus close-17’s space-bearing and overlong path payloads cannot authorize decisions through this protocol: there is no path field to accept. Raw-byte checking still rejects extra lines and NUL padding. This does not imply protection against a pinned module deliberately returning a false but grammatically valid `same`.

`publishes` performs two probes, but all production publication mutations occur under the arm’s publication lock, which the caller holds. The trainer never retargets `final`. Cooperative lifecycle participants therefore cannot change `final` between those probes. An unrelated manual filesystem mutation or mount change can bypass advisory locking; that is outside the stated coordination model.

The audit table is otherwise consistent with the implementation: sidecar decisions use the probe; scan exclusion and publication checks use identity; resolution containment uses `same_parent`; `readlink` is informational. Checked enumeration, classification and rename-path operations remain justified. **The inline comparison in blocker 1 prevents accepting the audit as complete.**

**Validation completed: 304 distinct selected tests passed.**

| Selected checks | Result |
|---|---:|
| `tests/test_exp11_shell.py` | 209 passed |
| `tests/test_exp11_pathprobe.py` | 23 passed |
| `tests/test_exp11_train.py` | 62 passed |
| `tests/test_exp09_sim_eval_closures.py` | 3 passed |
| Registry isolation and pinned-output comparisons | 6 passed |
| Exact exp_04 `[True-promote]` case | 1 passed |

Shell syntax and `git diff --check 2e109d0 f2f6943` also pass. Passing tests do not cover the two findings above.

For RED evidence, I replayed `test_no_verdict_carries_a_path` using the actual module and test from **`2e76a95`**, extracted under `/tmp`. It failed as expected because stdout still contained:

```text
present 81b4 /tmp/.../launching
```

The eight-commit history matches the requested range. Production changes total **+211/−151**; test changes total **+169/−43**. The reported file counts—23 pathprobe, 209 shell and 62 trainer tests—are correct.

The retained full-suite log begins exactly:

```text
HEAD d420ddb4443cf3212899b01534608b3f39491165
```

It records **1 failed, 3997 passed, 55 skipped**, in 3454.56 seconds. The sole failure is the stale exp_06 `summarize_haa` approval digest. No superseded `87f4e4d` partial suite log remains in the worktree. I inspected the retained full-suite evidence; I did not rerun the entire suite.

Several record corrections are still needed:

- Fix-17’s actual suite HEAD is correctly amended to `fef89ba`; both retained logs expose that SHA.
- The revised timeout accounting correctly identifies six ordinary per-attempt calls and approximately 105 seconds for three TERM-ignoring sidecar probes. Approximately 180 seconds is six base 30-second budgets, **not a universal bound**: kill grace, health checks, identity probes, additional entries and early refusal affect actual duration. [Launcher:45](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:45) still contains the obsolete approximately-60-second explanation.
- The [report:122](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_05:30:00_coder_round2_fix18_opus_report.md:122) still puts the exp_04 completion transaction on a background thread. Only the SIGTERM sender runs there; `execute_attempt` runs on the main thread. “Timing-sensitive” is supported; co-tenancy is unproven. The isolated case passes, and exp_04 files are unchanged since `9f98bbb`.
- The [report:143](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_05:30:00_coder_round2_fix18_opus_report.md:143) incorrectly says every production/test commit is below 200 changed lines. `87f4e4d` totals **252**: 197 production plus 55 test lines.
- The earlier oversized record-only commits are named, but their quoted counts are wrong in this checkout. `git show --numstat` totals **672** for `a700a24` and **707** for `cad6018`, rather than 624 and 538. This cycle’s record-only commits are **654** lines for `98c82b3` and **248** for `f2f6943`.
- [Launcher:552](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:552) says “three questions” although the path health check asks four.

Closure hygiene passes for the reviewed candidate:

- All **28 exp_06/exp_11 code digests** agree between `d420ddb` and `f2f6943`.
- Against the exp_06 record at `9f98bbb`, only `code.summarize_haa` moves; the other 19 keys match.
- Five exp_11 keys move versus `2e109d0`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged. All eight exact values appear below.
- Sim-eval closures remain unchanged; all three closure tests pass.
- The 28 checked existing frozen files match `9f98bbb`; all 12 exp_03 pinned source records match their reviewed hashes.
- `BACKBONES_EXP06` is unchanged. Pinned exp_06/exp_09 and exp_11 phase-1 output comparisons are byte-identical.
- `ckpt/exp11` is entirely absent.
- Git status is unchanged: only the pre-existing untracked close18 prompt. No review or approval record was saved.

The frozen identities are:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479

BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

The exact fill manifest below applies to a merge retaining **the reviewed `f2f6943` production bytes**. It is not authorization to fill or merge this rejected candidate; recompute it after correcting the blockers.

In exp_06’s existing approvals record, replace only `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
85caf08ce7cbf2d479349fbf67825c8cb0feff9be8822d41ef76f08d2f4716d7
```

Preserving every other byte produces a **3,190-byte** exp_06 record with SHA256:

```text
c33e2475da5dbf3ad6d2144d9faac28d140a42f20ba8bfae2d8c514b12c870ac
```

The corresponding complete exp_11 fill is:

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
    "finalize": "197f5bdeb6753e48cad1075299f2cd4bd86fd4ba1a7cfe7639ee95a39142c213",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "d4dab1e4a51fcac65b5ea6f307c66f43e601939f9670361219000d24b1202d19",
    "smoke": "c8710213e7eadf16193e8f664e7dd7c60ecee13fd34ad6ffd01f307eac2a83ff",
    "summarize_haa": "85caf08ce7cbf2d479349fbf67825c8cb0feff9be8822d41ef76f08d2f4716d7",
    "train": "d712fa89fc7b35f7785f970cb1562d432c5120cfdab55848e32318ddc5c0dc30"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "c33e2475da5dbf3ad6d2144d9faac28d140a42f20ba8bfae2d8c514b12c870ac",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one final newline, that exp_11 record is **1,316 bytes**, SHA256:

```text
7459520df0d736bb18882da223f7e57df90a7fbc1c58458e4d1100de3804467e
```

