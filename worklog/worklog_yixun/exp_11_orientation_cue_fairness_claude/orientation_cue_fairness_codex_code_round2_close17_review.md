**Close review 17 — exp_11 `orientation_cue_fairness`, round 2**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized execution for `/tmp` fixtures and owned helpers |
| Worktree | `/home/yixunhu/codespace/xRIR_code_wt` |
| Branch | `exp11-cue` |
| HEAD | `2e109d037ebe93aff98ed3afdc92fb88a2d77c41` |
| Last production commit | `e918f4b7b0d57a601806c0940db83ae322b831ba` |
| Review completed | 2026-09-28 09:00:28 UTC / 05:00:28 EDT |
| Execution | UID 1013; CUDA disabled; one CPU thread per worker, at most four workers; pytest cache disabled; temporary writes under `/tmp` |

**Verdict: request changes. Round 2 cannot be CLOSED, and `2e109d0` should not be merged.**

The three original injections now refuse correctly. Independent replays found a remaining P1 in path-field validation and three P2 regressions.

1. **[P1] Space-bearing and overlong encoded path fields still authorize lifecycle decisions.**

   [tools/exp11_launch.sh:387](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:387) and [tools/exp11_pathprobe.py:61](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pathprobe.py:61) accept embedded spaces. Both also count characters instead of filesystem-encoded bytes.

   Using the real sourced launcher, real locks, and wrappers changing only the selected probe:

   | Injection | Operation | Observed result |
   |---|---|---|
   | `link /forged parent/x` for `final`, which actually publishes the target | `resolve_unregistered` | Exit **0**; tombstone written; attempt retired; `final` left dangling |
   | Same | Isolated `abort_recovery`, with a live stub registered through `register_trainer` | Exit **0**; attempt and log renamed; `final` dangling; stub still alive |
   | `present 41ed /forged parent/launching` for the actual marker | `require_quiet_arm` | Exit **0**, despite the unresolved marker |
   | `present 41ed <path>/launching`, with a 2,434-character / 4,834-byte UTF-8 path | `require_quiet_arm` | Exit **0**, despite exceeding the claimed 4,096-byte limit |

   The isolated abort replay does **not** establish a bypass of the complete recovery entry point’s preceding live-process scan. The resolution replay reaches the actual scan, registration lock, tombstone and rename.

   **Required:** enforce the specified path-field grammar, including rejection of embedded spaces, and measure the length in bytes. Add regressions that preserve the actual publication and attempt under these injections.

2. **[P2] Terminal `..` canonicalizes to the wrong directory and bypasses a diagnostic tombstone check.**

   [tools/exp11_pathprobe.py:136](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pathprobe.py:136) treats terminal `..` like an empty or `.` component:

   ```text
   canonical("/base/sub/..") → "/base/sub"
   actual resolved path     → "/base"
   ```

   Consequently, the CLI can print the mode of one directory alongside another directory’s canonical path.

   The trainer consequence is reproducible without injecting a verdict: with `arm/attempt_X.resolved` present, `register_trainer("arm/attempt_X", no_save=True)` exits **3**, but the equivalent `arm/attempt_X/sub/..` succeeds. It checks `arm/attempt_X/sub.resolved` instead.

   **Required:** resolve terminal `..` with its filesystem meaning. Add the equivalent-path tombstone regression.

3. **[P2] The trainer normalizes malformed canonical fields before validating them.**

   At [tools/exp11_train.py:309](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:309), `Path(...)` precedes `valid_path(...)`.

   Injecting any of these canonical results for an otherwise valid attempt makes registration succeed and write `train.pid`:

   ```text
   /valid-arm//attempt_X
   /valid-arm/./attempt_X
   /valid-arm/attempt_X/
   ```

   `valid_path(raw_result)` returns false for each, but `Path` removes the offending spelling before validation.

   **Required:** validate the raw canonical string before constructing `Path`, as the arm-root branch already does.

4. **[P2] The basename check rejects an existing attempt supplied with a trailing slash.**

   [tools/exp11_launch.sh:422](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:422) compares the returned basename with either the entire slash-stripped input or the basename of the unstripped input. For `/arm/attempt_X/`, neither comparison matches.

   Through the real locked lifecycle, an ordinary unresolved attempt resolves successfully with its normal spelling, while the equivalent trailing-slash spelling returns **2**, claiming the directory cannot be inspected.

   **Required:** derive the comparison basename from the appropriately slash-stripped input, retaining the explicit handling of root and special components. This is separate from correctly rejecting a dangling link with a trailing slash.

The requested replay results were:

| Check | Result |
|---|---|
| `link garbage` on the actual published `final`: resolution and isolated abort | **2**, attempt/publication/log preserved; registered stub remains alive |
| `present 41ed relative` on `launching` | **2**, scan refuses |
| Relative targets, `//abs`, `/abs/../x`, 5,000-character paths, CR-containing paths | Unknown; tested attempts and publications preserved |
| Modes `f000`, `ffffffff`, and valid type bits with excessive length (`0000041ed`) | Unknown |
| Embedded spaces | **Fails**, as finding 1 records |
| `missing/../leaf`, `dangling/../leaf`, dangling link plus trailing slash | Unknown |
| Resolvable `sub/../leaf` and `sub/../gone` | Correctly present and absent |
| Terminal `..` | **Fails**, as finding 2 records |
| Timeout values `1`, `30`, `300` | Accepted |
| Timeout values `0`, `301`, `3s`, and 36 digits | Refused with exit **2** |

The basename rule accepts an ordinary attempt reached through its arm’s symlink: `arm_alias/attempt_X` produces the canonical arm directory plus `attempt_X`, and the shell accepts it. The probe also correctly follows a final-component symlink to either a directory or a regular file and reports the target’s mode.

The production publication guards inspect `final` through `link`. Production creates ordinary `launching`, `train.pid`, `train.exit` and `completion.json` files; I found no generated lifecycle layout requiring those sidecars to be symlinks. Resolution additionally probes its caller-supplied attempt pathname.

However, “a probed name that is itself a symlink becomes unknown” is too broad. The shell checks **basename equality**, not symlink identity. Different-basename targets return unknown; a sidecar symlink pointing to another `launching` file passes. I replayed both cases.

**Validation completed: 297 passed.**

| Selected checks | Result |
|---|---:|
| `tests/test_exp11_shell.py` | 207 passed |
| `tests/test_exp11_pathprobe.py` | 18 passed |
| `tests/test_exp11_train.py` | 60 passed |
| `tests/test_exp09_sim_eval_closures.py` | 3 passed |
| Exact exp_04 `[True-promote]` case from the first suite log | 1 passed |
| Pinned summarizer/admission output comparisons and registry checks | 8 passed |

The existing tests therefore pass while missing the independently reproduced defects above. Shell syntax and `git diff --check f966461 2e109d0` also pass.

For RED evidence, the exact `009836d` pathprobe and test versions were replayed under `/tmp`. `test_a_missing_component_before_a_dotdot_is_not_an_absence` failed as expected: it returned `absent` instead of `unknown`.

The report’s canonicalization-order and identity-fallback corrections are accurate: canonicalization precedes the directory probes, literal identity fallbacks can compare equal, and all three PID checks precede the exclusion. The marker timestamp fallback at [tools/exp11_launch.sh:816](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:816) substitutes `now`, producing age zero and refusal at a positive grace period. With grace zero, that fallback does not itself refuse.

Several report corrections remain necessary:

- The claim that field validation handles embedded spaces is contradicted by finding 1. The “4,096 bytes” claim also exceeds the implementation.
- The [timeout accounting](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_02:52:12_coder_round2_fix17_opus_report.md:120) is incorrect. A fully scanned attempt has **six outer timeout calls**: three PID readers and three sidecar probes. The reader’s internal Python path probe shares its existing deadline. Three TERM-ignoring sidecar probes took **18.21 seconds** with timeout 1 plus kill grace 5, returning unknown and preserving the fixture. At the default, that sidecar group can take approximately **105 seconds**, not 70. Resolution adds self-tests and other probes; 140 seconds is not a universal bound.
- Both retained full-suite logs expose HEAD **`fef89bad716fac8a039c2a4774fcac36a562d850`**, rather than the report’s claimed exact HEAD `e918f4b`. Their production bytes are identical.
- The exp_04 transaction runs on the main thread; only its SIGTERM sender runs on a background thread, at [tests/test_exp04_launcher.py:381](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp04_launcher.py:381). Unchanged source, the isolated pass and the successful rerun support an unrelated timing-sensitive flake. The logs do not establish co-tenancy as its cause.
- “Every commit is under 200 changed lines” applies to the six implementation/test commits. The two bookkeeping commits add **624** and **538** lines.

The retained full-suite results are accurately reported as:

| Log | Result |
|---|---|
| First run | **2 failed, 3987 passed, 55 skipped** |
| Rerun | **1 failed, 3988 passed, 55 skipped** |

The additional first-run failure is the exp_04 capture assertion. No exp_04 source or test file changed since `9f98bbb`. The rerun’s sole failure is the expected stale exp_06 `summarize_haa` approval digest. I reviewed those logs and replayed the isolated failure; I did not repeat the full CPU suite.

Production changes total **+153/−21**; tests total **+179/−0**. The eight-commit history and record-only changes after `e918f4b` match the stated scope.

Closure hygiene passes:

- All **28 exp_06/exp_11 code digests** agree between `e918f4b` and `2e109d0`.
- Against exp_06’s approval record at `9f98bbb`, only `code.summarize_haa` moves; the other **19 keys** match.
- Five exp_11 keys move versus `f966461`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged.
- Sim-eval closures remain unchanged; all three closure tests pass.
- Existing frozen sources and all 12 exp_03 pinned source records match their reviewed bytes. Pinned oracle comparisons pass.
- `ckpt/exp11` is entirely absent after testing.
- Git status remains unchanged, containing only the pre-existing untracked close17 prompt.

The frozen identities are:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479

BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

The following is the **exact fill manifest for a content-identical merge of `2e109d0`**. It does not authorize merging this candidate; affected digests must be recomputed after fixes.

Replace only exp_06 `code.summarize_haa` with:

```text
0f5fdeab9c57de5702ab3e773209bbaaf95536b131eb57140dece834f21c05df
```

Preserving every other byte of the 3,190-byte exp_06 approval record produces this record SHA256:

```text
c79538299609ebb78b0d79cb82fbb2fdfe9c2959dcedb89b1a8bd432faf24e2d
```

The exp_11 manifest is:

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
    "finalize": "d30f729992ff3690ced21fd268f1af6cb147d095b245f8a2f07bcd18bc3ff8cf",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "2149526de3524f5602a1f3e5775f676e8b10fda309010be744fa8089085cb8a8",
    "smoke": "e81f8389019770cf88fc4193bda223a95e1e5c22d497fedb72aa773d3f5ed0da",
    "summarize_haa": "0f5fdeab9c57de5702ab3e773209bbaaf95536b131eb57140dece834f21c05df",
    "train": "be309c629c72a541e636075c4dbeec9c80687e22bbcf075597870b39f73d747a"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "c79538299609ebb78b0d79cb82fbb2fdfe9c2959dcedb89b1a8bd432faf24e2d",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one trailing newline, this exp_11 record is **1,316 bytes**, SHA256:

```text
64063b13b14d99467e18fde994956d81b58d1fc951f5d9969559e2465a7380f8
```

No review file or approval record was saved.
tokens used
84,679
**Close review 17 — exp_11 `orientation_cue_fairness`, round 2**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized execution for `/tmp` fixtures and owned helpers |
| Worktree | `/home/yixunhu/codespace/xRIR_code_wt` |
| Branch | `exp11-cue` |
| HEAD | `2e109d037ebe93aff98ed3afdc92fb88a2d77c41` |
| Last production commit | `e918f4b7b0d57a601806c0940db83ae322b831ba` |
| Review completed | 2026-09-28 09:00:28 UTC / 05:00:28 EDT |
| Execution | UID 1013; CUDA disabled; one CPU thread per worker, at most four workers; pytest cache disabled; temporary writes under `/tmp` |

**Verdict: request changes. Round 2 cannot be CLOSED, and `2e109d0` should not be merged.**

The three original injections now refuse correctly. Independent replays found a remaining P1 in path-field validation and three P2 regressions.

1. **[P1] Space-bearing and overlong encoded path fields still authorize lifecycle decisions.**

   [tools/exp11_launch.sh:387](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:387) and [tools/exp11_pathprobe.py:61](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pathprobe.py:61) accept embedded spaces. Both also count characters instead of filesystem-encoded bytes.

   Using the real sourced launcher, real locks, and wrappers changing only the selected probe:

   | Injection | Operation | Observed result |
   |---|---|---|
   | `link /forged parent/x` for `final`, which actually publishes the target | `resolve_unregistered` | Exit **0**; tombstone written; attempt retired; `final` left dangling |
   | Same | Isolated `abort_recovery`, with a live stub registered through `register_trainer` | Exit **0**; attempt and log renamed; `final` dangling; stub still alive |
   | `present 41ed /forged parent/launching` for the actual marker | `require_quiet_arm` | Exit **0**, despite the unresolved marker |
   | `present 41ed <path>/launching`, with a 2,434-character / 4,834-byte UTF-8 path | `require_quiet_arm` | Exit **0**, despite exceeding the claimed 4,096-byte limit |

   The isolated abort replay does **not** establish a bypass of the complete recovery entry point’s preceding live-process scan. The resolution replay reaches the actual scan, registration lock, tombstone and rename.

   **Required:** enforce the specified path-field grammar, including rejection of embedded spaces, and measure the length in bytes. Add regressions that preserve the actual publication and attempt under these injections.

2. **[P2] Terminal `..` canonicalizes to the wrong directory and bypasses a diagnostic tombstone check.**

   [tools/exp11_pathprobe.py:136](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pathprobe.py:136) treats terminal `..` like an empty or `.` component:

   ```text
   canonical("/base/sub/..") → "/base/sub"
   actual resolved path     → "/base"
   ```

   Consequently, the CLI can print the mode of one directory alongside another directory’s canonical path.

   The trainer consequence is reproducible without injecting a verdict: with `arm/attempt_X.resolved` present, `register_trainer("arm/attempt_X", no_save=True)` exits **3**, but the equivalent `arm/attempt_X/sub/..` succeeds. It checks `arm/attempt_X/sub.resolved` instead.

   **Required:** resolve terminal `..` with its filesystem meaning. Add the equivalent-path tombstone regression.

3. **[P2] The trainer normalizes malformed canonical fields before validating them.**

   At [tools/exp11_train.py:309](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:309), `Path(...)` precedes `valid_path(...)`.

   Injecting any of these canonical results for an otherwise valid attempt makes registration succeed and write `train.pid`:

   ```text
   /valid-arm//attempt_X
   /valid-arm/./attempt_X
   /valid-arm/attempt_X/
   ```

   `valid_path(raw_result)` returns false for each, but `Path` removes the offending spelling before validation.

   **Required:** validate the raw canonical string before constructing `Path`, as the arm-root branch already does.

4. **[P2] The basename check rejects an existing attempt supplied with a trailing slash.**

   [tools/exp11_launch.sh:422](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:422) compares the returned basename with either the entire slash-stripped input or the basename of the unstripped input. For `/arm/attempt_X/`, neither comparison matches.

   Through the real locked lifecycle, an ordinary unresolved attempt resolves successfully with its normal spelling, while the equivalent trailing-slash spelling returns **2**, claiming the directory cannot be inspected.

   **Required:** derive the comparison basename from the appropriately slash-stripped input, retaining the explicit handling of root and special components. This is separate from correctly rejecting a dangling link with a trailing slash.

The requested replay results were:

| Check | Result |
|---|---|
| `link garbage` on the actual published `final`: resolution and isolated abort | **2**, attempt/publication/log preserved; registered stub remains alive |
| `present 41ed relative` on `launching` | **2**, scan refuses |
| Relative targets, `//abs`, `/abs/../x`, 5,000-character paths, CR-containing paths | Unknown; tested attempts and publications preserved |
| Modes `f000`, `ffffffff`, and valid type bits with excessive length (`0000041ed`) | Unknown |
| Embedded spaces | **Fails**, as finding 1 records |
| `missing/../leaf`, `dangling/../leaf`, dangling link plus trailing slash | Unknown |
| Resolvable `sub/../leaf` and `sub/../gone` | Correctly present and absent |
| Terminal `..` | **Fails**, as finding 2 records |
| Timeout values `1`, `30`, `300` | Accepted |
| Timeout values `0`, `301`, `3s`, and 36 digits | Refused with exit **2** |

The basename rule accepts an ordinary attempt reached through its arm’s symlink: `arm_alias/attempt_X` produces the canonical arm directory plus `attempt_X`, and the shell accepts it. The probe also correctly follows a final-component symlink to either a directory or a regular file and reports the target’s mode.

The production publication guards inspect `final` through `link`. Production creates ordinary `launching`, `train.pid`, `train.exit` and `completion.json` files; I found no generated lifecycle layout requiring those sidecars to be symlinks. Resolution additionally probes its caller-supplied attempt pathname.

However, “a probed name that is itself a symlink becomes unknown” is too broad. The shell checks **basename equality**, not symlink identity. Different-basename targets return unknown; a sidecar symlink pointing to another `launching` file passes. I replayed both cases.

**Validation completed: 297 passed.**

| Selected checks | Result |
|---|---:|
| `tests/test_exp11_shell.py` | 207 passed |
| `tests/test_exp11_pathprobe.py` | 18 passed |
| `tests/test_exp11_train.py` | 60 passed |
| `tests/test_exp09_sim_eval_closures.py` | 3 passed |
| Exact exp_04 `[True-promote]` case from the first suite log | 1 passed |
| Pinned summarizer/admission output comparisons and registry checks | 8 passed |

The existing tests therefore pass while missing the independently reproduced defects above. Shell syntax and `git diff --check f966461 2e109d0` also pass.

For RED evidence, the exact `009836d` pathprobe and test versions were replayed under `/tmp`. `test_a_missing_component_before_a_dotdot_is_not_an_absence` failed as expected: it returned `absent` instead of `unknown`.

The report’s canonicalization-order and identity-fallback corrections are accurate: canonicalization precedes the directory probes, literal identity fallbacks can compare equal, and all three PID checks precede the exclusion. The marker timestamp fallback at [tools/exp11_launch.sh:816](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:816) substitutes `now`, producing age zero and refusal at a positive grace period. With grace zero, that fallback does not itself refuse.

Several report corrections remain necessary:

- The claim that field validation handles embedded spaces is contradicted by finding 1. The “4,096 bytes” claim also exceeds the implementation.
- The [timeout accounting](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_02:52:12_coder_round2_fix17_opus_report.md:120) is incorrect. A fully scanned attempt has **six outer timeout calls**: three PID readers and three sidecar probes. The reader’s internal Python path probe shares its existing deadline. Three TERM-ignoring sidecar probes took **18.21 seconds** with timeout 1 plus kill grace 5, returning unknown and preserving the fixture. At the default, that sidecar group can take approximately **105 seconds**, not 70. Resolution adds self-tests and other probes; 140 seconds is not a universal bound.
- Both retained full-suite logs expose HEAD **`fef89bad716fac8a039c2a4774fcac36a562d850`**, rather than the report’s claimed exact HEAD `e918f4b`. Their production bytes are identical.
- The exp_04 transaction runs on the main thread; only its SIGTERM sender runs on a background thread, at [tests/test_exp04_launcher.py:381](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp04_launcher.py:381). Unchanged source, the isolated pass and the successful rerun support an unrelated timing-sensitive flake. The logs do not establish co-tenancy as its cause.
- “Every commit is under 200 changed lines” applies to the six implementation/test commits. The two bookkeeping commits add **624** and **538** lines.

The retained full-suite results are accurately reported as:

| Log | Result |
|---|---|
| First run | **2 failed, 3987 passed, 55 skipped** |
| Rerun | **1 failed, 3988 passed, 55 skipped** |

The additional first-run failure is the exp_04 capture assertion. No exp_04 source or test file changed since `9f98bbb`. The rerun’s sole failure is the expected stale exp_06 `summarize_haa` approval digest. I reviewed those logs and replayed the isolated failure; I did not repeat the full CPU suite.

Production changes total **+153/−21**; tests total **+179/−0**. The eight-commit history and record-only changes after `e918f4b` match the stated scope.

Closure hygiene passes:

- All **28 exp_06/exp_11 code digests** agree between `e918f4b` and `2e109d0`.
- Against exp_06’s approval record at `9f98bbb`, only `code.summarize_haa` moves; the other **19 keys** match.
- Five exp_11 keys move versus `f966461`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged.
- Sim-eval closures remain unchanged; all three closure tests pass.
- Existing frozen sources and all 12 exp_03 pinned source records match their reviewed bytes. Pinned oracle comparisons pass.
- `ckpt/exp11` is entirely absent after testing.
- Git status remains unchanged, containing only the pre-existing untracked close17 prompt.

The frozen identities are:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479

BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

The following is the **exact fill manifest for a content-identical merge of `2e109d0`**. It does not authorize merging this candidate; affected digests must be recomputed after fixes.

Replace only exp_06 `code.summarize_haa` with:

```text
0f5fdeab9c57de5702ab3e773209bbaaf95536b131eb57140dece834f21c05df
```

Preserving every other byte of the 3,190-byte exp_06 approval record produces this record SHA256:

```text
c79538299609ebb78b0d79cb82fbb2fdfe9c2959dcedb89b1a8bd432faf24e2d
```

The exp_11 manifest is:

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
    "finalize": "d30f729992ff3690ced21fd268f1af6cb147d095b245f8a2f07bcd18bc3ff8cf",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "2149526de3524f5602a1f3e5775f676e8b10fda309010be744fa8089085cb8a8",
    "smoke": "e81f8389019770cf88fc4193bda223a95e1e5c22d497fedb72aa773d3f5ed0da",
    "summarize_haa": "0f5fdeab9c57de5702ab3e773209bbaaf95536b131eb57140dece834f21c05df",
    "train": "be309c629c72a541e636075c4dbeec9c80687e22bbcf075597870b39f73d747a"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "c79538299609ebb78b0d79cb82fbb2fdfe9c2959dcedb89b1a8bd432faf24e2d",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one trailing newline, this exp_11 record is **1,316 bytes**, SHA256:

```text
64063b13b14d99467e18fde994956d81b58d1fc951f5d9969559e2465a7380f8
```

No review file or approval record was saved.

