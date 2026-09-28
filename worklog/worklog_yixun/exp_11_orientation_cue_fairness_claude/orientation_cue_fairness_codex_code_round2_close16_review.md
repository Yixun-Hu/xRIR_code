**Close review 16 — exp_11 `orientation_cue_fairness`, round 2**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized execution for `/tmp` fixtures and owned helpers |
| Worktree | `/home/yixunhu/codespace/xRIR_code_wt` |
| Branch | `exp11-cue` |
| HEAD | `f96646133975def3404669bea8699961534a218b` |
| Last production commit | `b9315aac7e6a5d8f3481433ecc3ad88668ad732c` |
| Review time | 2026-09-28, approximately 02:41 EDT |
| Execution | Non-root UID 1013; CUDA disabled; one CPU thread per test worker, at most four concurrent workers |

**Verdict: request changes. Round 2 cannot be CLOSED, and `f966461` should not be merged.**

The three close15 schedules are repaired, but malformed path-probe fields still authorize destructive lifecycle decisions.

**Blocking finding**

1. **[P1] A byte-exact line containing an invalid path field is accepted as a valid verdict.**

   [tools/exp11_launch.sh:382](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:382) accepts any nonempty canonical-path suffix. [tools/exp11_launch.sh:407](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:407) likewise accepts any nonempty link target. Neither requires an absolute path.

   With a test `$PYTHON` wrapper changing only the selected probe, while other probes and the sourced lifecycle remained real:

   | Injected response | Real operation | Observed result |
   |---|---|---|
   | `link garbage\n` for `final` | `resolve_unregistered`, with publication and registration locks; `final` actually publishes the target | Exit **0**; tombstone written; target renamed `_ABORTED_unregistered`; `final` left dangling |
   | `present 41ed relative\n` for `launching` | `require_quiet_arm` under the publication lock | Exit **0**, despite the actual launching marker |
   | `link garbage\n` for `final` | Isolated `abort_recovery` under the publication lock, against a registered live stub’s published attempt | Attempt and log renamed while the stub remained alive; `final` left dangling |

   The last replay tests the guard directly; it does **not** establish that the complete recovery entry point bypasses its preceding live-process scan. The resolution replay does reach the real scan, locks, tombstone and rename.

   `one_line` establishes byte preservation, not semantic validity. The mode check also accepts values such as `ffffffff`, without validating the file-type bits.

   **Required:** validate the extracted mode and path fields before granting a known verdict. Add regressions for relative canonical paths, relative link targets and invalid modes, including resolution of the actual published target. Malformed responses must produce unknown and preserve the attempt and publication.

**Additional findings**

- **[P2] The parent check can inspect a different pathname after lexical normalization.**  
  [tools/exp11_pathprobe.py:46](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pathprobe.py:46) applies `abspath` before deriving the parent. For `/tmp/base/missing/../leaf` and `/tmp/base/dangling/../leaf`, the actual parent lookup fails, but the helper checks `/tmp/base` and returns `absent`; the link probe returns `nolink`, and the pid reader returns `norecord`. A dangling link with a trailing slash similarly returns absence. Preserve the lookup’s path-component semantics when checking its parent. I did not establish destructive exposure through the ordinary canonicalized lifecycle paths.

- **[P2] Timeout validation still accepts overflowing decimal strings.**  
  At [tools/exp11_launch.sh:59](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:59), `999999999999999999999999999999999999` causes both comparisons to report `integer expression expected`; the enclosing condition is false and sourcing succeeds with that value. Bound the decimal string before arithmetic, as `check_ceiling` already does.

**Replay results**

All lifecycle fixtures were under `/tmp`, using the actual sourced launcher and real locks. Live stubs registered through `register_trainer`; only owned helper processes were terminated.

| Requested check | Result |
|---|---|
| Dangling `train.pid` after registration | Resolution exits **2**; attempt and marker preserved; no tombstone or retirement; stub remains alive |
| Looping `train.pid` | Unknown; resolution refuses |
| Missing name under an existing searchable parent | Absent |
| Actual ENAMETOOLONG and inaccessible parent | Unknown |
| Injected EIO in target `stat`, target `lstat`, or parent `stat` | Unknown |
| Dangling `final` | Unknown |
| Cross-arm attempt alias, default trainer call | Exit **3**, no wrong-arm lock and no registration |
| Attempt through its own arm’s symlink | Registers in the canonical arm |
| `--no-save` | No lock or sidecars; tombstone checks retained |
| TERM-ignoring pid reader, timeout 1 s | Unknown after **6.02 s** |
| TERM-ignoring path probe, timeout 1 s | Unknown after **6.03 s** |

The exact-path errno-injection matrix exercised **ENAMETOOLONG, EACCES and EIO** separately through the real path-probe module. For each errno, these seven operations returned **2**:

- Scan encountering `launching`.
- Resolution inspecting `launching`.
- Resolution inspecting `completion.json`.
- Resolution inspecting `final`, which actually published the target.
- `already_published`.
- `check_final_conflict`, including `REPLACE_FINAL=1`.
- `abort_recovery`.

All **21** cases preserved the attempt, marker and existing publication, with no tombstone or abort.

An unreadable but searchable parent is sufficient for absence of a **known name**: mode `0300` returned absent for a missing child. Directory enumeration separately requires read permission; `list_attempts` enforces that distinction. A parent without search permission returned unknown/EACCES. The normalization defect above prevents claiming that the parent rule is universally correct.

For verdict framing, missing terminal newlines, second lines, embedded newlines and NULs are rejected. Thus a canonical path containing a newline becomes unknown. Padded keywords such as `absent ` also fail parsing. However, relative path fields pass, as demonstrated. Spaces or carriage returns inside a path field survive byte comparison; POSIX filenames can contain them, so blanket claims that byte comparison rejects all “padding” are inaccurate.

The timeout checks reject empty, zero, negative, nonnumeric and ordinary out-of-range values; `1`, `30` and `300` pass. The overflow case remains. The documented approximately 60 seconds covers two TERM-responsive hangs: two TERM-ignoring helpers can consume approximately **70 seconds**, three consecutive sidecar probes **105 seconds**, and the four health-check questions **140 seconds** at the default. These are per-question bounds, not an overall scan deadline.

**Audit-table reconciliation**

I searched filesystem predicates and metadata operations in the launcher and trainer registration path and compared them with the rewritten table.

| Sites | Assessment |
|---|---|
| Launcher path/link wrappers and lifecycle consumers | Routed, but verdict-field validation is incomplete: blocker 1 |
| Launcher pid reader | Delegates existence to the shared probe; verdict-size failure becomes unknown |
| `list_attempts`, lines 533–536 | Canonicalization, directory/read/search checks and enumeration failures refuse |
| Per-entry `stat`, line 566 | Failure becomes unknown; successful mode determines type |
| Resolver canonicalization and directory inspection, lines 675, 678, 698 | Failures refuse |
| Scan identity fallbacks, lines 549 and 576 | The report overstates that failures necessarily make spellings unequal; equal literal fallbacks remain possible. PID checks precede exclusion |
| Marker timestamp, line 771 | Omitted from the table; failure substitutes “now”, conservatively preventing retirement with the positive default grace |
| Test barrier and data-root checks, lines 815 and 983 | Test-only barrier; invalid data root refuses |
| Locks, preflight and attempt validation | Inspection/open failures refuse |
| Trainer tombstone, directory, root, marker and child-record checks | Routed through the shared probe/reader |
| Trainer canonicalization | Happens at line 309 **before** the first directory probe at 317, contrary to the table’s wording; subsequent containment checks correctly reject the reviewed alias layout |
| Trainer exit recording | Failure does not establish successful completion |

The report’s assertion that no inspection failure can yield absent/dead/quiet/unpublished is therefore too broad.

**RED evidence and tests**

I replayed the dangling-record schedule with the same registered live stub and current sourced lifecycle, first using HEAD’s reader and then dispatching only pid-record reads to the extracted `ca42b42` implementation:

```text
HEAD reader:
exit 2; attempt preserved; no tombstone; no retirement; stub alive

ca42b42 reader:
exit 0; TOMBSTONE + ABORT + RESOLVED; attempt retired; stub alive
```

Independent selected tests used `PYTHONPATH` equal to the worktree, `PYTHONDONTWRITEBYTECODE=1`, CUDA disabled, `/tmp` fixtures and `-p no:cacheprovider`.

| Tests | Result |
|---|---:|
| `test_exp11_shell.py` | **180 passed** |
| `test_exp11_pathprobe.py` | **12 passed** |
| `test_exp11_train.py` | **56 passed** |
| Pid-record and lock-holder tests | **82 passed** |
| `test_exp09_sim_eval_closures.py` | **3 passed** |
| Pinned outputs, registry and historical-output checks | **8 passed** |
| **Total** | **341 passed** |

The additional adversarial replays expose gaps beyond those passing tests. Both launcher syntax checks and `git diff --check` pass.

Production **+285/−130** and tests **+301/−2** match the report. Its retained full-suite log records **1 failed, 3951 passed, 55 skipped**; the failure is the stale exp_06 summarizer approval. I did not rerun that full suite.

Two reporting corrections:

- The log’s [line 64](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_01:33_suite_full_cpu_b9315aa.log:64) records HEAD `cad60185a63b6247c422831bd508e8cadd08da73`, despite the report’s exact-HEAD claim of `b9315aa`. Those commits have identical production bytes.
- “Every commit is under 200 changed lines” applies to the 14 implementation/test commits, not all 16: the two bookkeeping commits add 707 and 254 lines.

**Closure hygiene**

All 28 exp_06/exp_11 code digests agree between `b9315aa` and `f966461`.

Against the exp_06 approval record at `9f98bbb`, only `code.summarize_haa` moves; the other 19 keys match. Against `2ca784a`, exp_11’s moving keys are `train`, `finalize`, `launch_sh`, `smoke` and `summarize_haa`. Every moving closure contains `tools/exp11_pathprobe.py`; all three HAA closures remain unchanged. All eight values appear in the manifest below.

Sim-eval closure checks pass, including all ten real manifests. Pinned oracles remain byte-identical. Frozen sources remain untouched; the baseline comparison retains the previously authorized summarizer change and new exp_11 model files.

```text
tools/exp06_launch.sh:
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py:
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479

BACKBONES_EXP06:
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

`ckpt/exp11/pretrain` is absent, including the stray `.publish.lock`. The repaired timeout regression explicitly redirects its roots to `tmp_path`, and the dry-run regression verifies that no production lock is created. The selected suite left those paths absent.

The production worktree is unchanged. Git status contains only the pre-existing untracked close16 prompt. No review file was saved.

**Exact fill manifest for the unchanged candidate tree**

These values describe a content-identical merge of `f966461`; they do **not** authorize merging it. Recompute affected closures after fixing the findings.

Replace only exp_06 `code.summarize_haa`:

```text
720862918e131449e111602b0e3684b4c29be1a36dae9a44542de47472e83b45
```

Preserving every other byte of the exp_06 record yields this record SHA256:

```text
ef3fac454ff762d7fd61e3185e1972d02f5873db2b97c65d68f0dfddd43a2702
```

The exp_11 fill manifest is:

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
    "finalize": "ee39e005182368d999f9d7824ab805ae5cd22e51f1a6d31e5d96fa514813a6e7",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "a9fd96978609ab038ea68aed51971de69ccd09d2a2ad01af9ef1f1de14b68955",
    "smoke": "3fd783ccc24daec89e88445fe2cdf2814d3da7c0e8663f583e32e28a943529b6",
    "summarize_haa": "720862918e131449e111602b0e3684b4c29be1a36dae9a44542de47472e83b45",
    "train": "21e08ffe412c1adaf80888fde36f0faca7227563da72d7e062d37c0f4229aae1"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "ef3fac454ff762d7fd61e3185e1972d02f5873db2b97c65d68f0dfddd43a2702",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one trailing newline, this exp_11 record is 1316 bytes with SHA256 `fa862ac9913c69a440f04cf813ba9d82767c66e7d56c0abfeb12f0e365d14e46`.
tokens used
77,346
**Close review 16 — exp_11 `orientation_cue_fairness`, round 2**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized execution for `/tmp` fixtures and owned helpers |
| Worktree | `/home/yixunhu/codespace/xRIR_code_wt` |
| Branch | `exp11-cue` |
| HEAD | `f96646133975def3404669bea8699961534a218b` |
| Last production commit | `b9315aac7e6a5d8f3481433ecc3ad88668ad732c` |
| Review time | 2026-09-28, approximately 02:41 EDT |
| Execution | Non-root UID 1013; CUDA disabled; one CPU thread per test worker, at most four concurrent workers |

**Verdict: request changes. Round 2 cannot be CLOSED, and `f966461` should not be merged.**

The three close15 schedules are repaired, but malformed path-probe fields still authorize destructive lifecycle decisions.

**Blocking finding**

1. **[P1] A byte-exact line containing an invalid path field is accepted as a valid verdict.**

   [tools/exp11_launch.sh:382](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:382) accepts any nonempty canonical-path suffix. [tools/exp11_launch.sh:407](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:407) likewise accepts any nonempty link target. Neither requires an absolute path.

   With a test `$PYTHON` wrapper changing only the selected probe, while other probes and the sourced lifecycle remained real:

   | Injected response | Real operation | Observed result |
   |---|---|---|
   | `link garbage\n` for `final` | `resolve_unregistered`, with publication and registration locks; `final` actually publishes the target | Exit **0**; tombstone written; target renamed `_ABORTED_unregistered`; `final` left dangling |
   | `present 41ed relative\n` for `launching` | `require_quiet_arm` under the publication lock | Exit **0**, despite the actual launching marker |
   | `link garbage\n` for `final` | Isolated `abort_recovery` under the publication lock, against a registered live stub’s published attempt | Attempt and log renamed while the stub remained alive; `final` left dangling |

   The last replay tests the guard directly; it does **not** establish that the complete recovery entry point bypasses its preceding live-process scan. The resolution replay does reach the real scan, locks, tombstone and rename.

   `one_line` establishes byte preservation, not semantic validity. The mode check also accepts values such as `ffffffff`, without validating the file-type bits.

   **Required:** validate the extracted mode and path fields before granting a known verdict. Add regressions for relative canonical paths, relative link targets and invalid modes, including resolution of the actual published target. Malformed responses must produce unknown and preserve the attempt and publication.

**Additional findings**

- **[P2] The parent check can inspect a different pathname after lexical normalization.**  
  [tools/exp11_pathprobe.py:46](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pathprobe.py:46) applies `abspath` before deriving the parent. For `/tmp/base/missing/../leaf` and `/tmp/base/dangling/../leaf`, the actual parent lookup fails, but the helper checks `/tmp/base` and returns `absent`; the link probe returns `nolink`, and the pid reader returns `norecord`. A dangling link with a trailing slash similarly returns absence. Preserve the lookup’s path-component semantics when checking its parent. I did not establish destructive exposure through the ordinary canonicalized lifecycle paths.

- **[P2] Timeout validation still accepts overflowing decimal strings.**  
  At [tools/exp11_launch.sh:59](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:59), `999999999999999999999999999999999999` causes both comparisons to report `integer expression expected`; the enclosing condition is false and sourcing succeeds with that value. Bound the decimal string before arithmetic, as `check_ceiling` already does.

**Replay results**

All lifecycle fixtures were under `/tmp`, using the actual sourced launcher and real locks. Live stubs registered through `register_trainer`; only owned helper processes were terminated.

| Requested check | Result |
|---|---|
| Dangling `train.pid` after registration | Resolution exits **2**; attempt and marker preserved; no tombstone or retirement; stub remains alive |
| Looping `train.pid` | Unknown; resolution refuses |
| Missing name under an existing searchable parent | Absent |
| Actual ENAMETOOLONG and inaccessible parent | Unknown |
| Injected EIO in target `stat`, target `lstat`, or parent `stat` | Unknown |
| Dangling `final` | Unknown |
| Cross-arm attempt alias, default trainer call | Exit **3**, no wrong-arm lock and no registration |
| Attempt through its own arm’s symlink | Registers in the canonical arm |
| `--no-save` | No lock or sidecars; tombstone checks retained |
| TERM-ignoring pid reader, timeout 1 s | Unknown after **6.02 s** |
| TERM-ignoring path probe, timeout 1 s | Unknown after **6.03 s** |

The exact-path errno-injection matrix exercised **ENAMETOOLONG, EACCES and EIO** separately through the real path-probe module. For each errno, these seven operations returned **2**:

- Scan encountering `launching`.
- Resolution inspecting `launching`.
- Resolution inspecting `completion.json`.
- Resolution inspecting `final`, which actually published the target.
- `already_published`.
- `check_final_conflict`, including `REPLACE_FINAL=1`.
- `abort_recovery`.

All **21** cases preserved the attempt, marker and existing publication, with no tombstone or abort.

An unreadable but searchable parent is sufficient for absence of a **known name**: mode `0300` returned absent for a missing child. Directory enumeration separately requires read permission; `list_attempts` enforces that distinction. A parent without search permission returned unknown/EACCES. The normalization defect above prevents claiming that the parent rule is universally correct.

For verdict framing, missing terminal newlines, second lines, embedded newlines and NULs are rejected. Thus a canonical path containing a newline becomes unknown. Padded keywords such as `absent ` also fail parsing. However, relative path fields pass, as demonstrated. Spaces or carriage returns inside a path field survive byte comparison; POSIX filenames can contain them, so blanket claims that byte comparison rejects all “padding” are inaccurate.

The timeout checks reject empty, zero, negative, nonnumeric and ordinary out-of-range values; `1`, `30` and `300` pass. The overflow case remains. The documented approximately 60 seconds covers two TERM-responsive hangs: two TERM-ignoring helpers can consume approximately **70 seconds**, three consecutive sidecar probes **105 seconds**, and the four health-check questions **140 seconds** at the default. These are per-question bounds, not an overall scan deadline.

**Audit-table reconciliation**

I searched filesystem predicates and metadata operations in the launcher and trainer registration path and compared them with the rewritten table.

| Sites | Assessment |
|---|---|
| Launcher path/link wrappers and lifecycle consumers | Routed, but verdict-field validation is incomplete: blocker 1 |
| Launcher pid reader | Delegates existence to the shared probe; verdict-size failure becomes unknown |
| `list_attempts`, lines 533–536 | Canonicalization, directory/read/search checks and enumeration failures refuse |
| Per-entry `stat`, line 566 | Failure becomes unknown; successful mode determines type |
| Resolver canonicalization and directory inspection, lines 675, 678, 698 | Failures refuse |
| Scan identity fallbacks, lines 549 and 576 | The report overstates that failures necessarily make spellings unequal; equal literal fallbacks remain possible. PID checks precede exclusion |
| Marker timestamp, line 771 | Omitted from the table; failure substitutes “now”, conservatively preventing retirement with the positive default grace |
| Test barrier and data-root checks, lines 815 and 983 | Test-only barrier; invalid data root refuses |
| Locks, preflight and attempt validation | Inspection/open failures refuse |
| Trainer tombstone, directory, root, marker and child-record checks | Routed through the shared probe/reader |
| Trainer canonicalization | Happens at line 309 **before** the first directory probe at 317, contrary to the table’s wording; subsequent containment checks correctly reject the reviewed alias layout |
| Trainer exit recording | Failure does not establish successful completion |

The report’s assertion that no inspection failure can yield absent/dead/quiet/unpublished is therefore too broad.

**RED evidence and tests**

I replayed the dangling-record schedule with the same registered live stub and current sourced lifecycle, first using HEAD’s reader and then dispatching only pid-record reads to the extracted `ca42b42` implementation:

```text
HEAD reader:
exit 2; attempt preserved; no tombstone; no retirement; stub alive

ca42b42 reader:
exit 0; TOMBSTONE + ABORT + RESOLVED; attempt retired; stub alive
```

Independent selected tests used `PYTHONPATH` equal to the worktree, `PYTHONDONTWRITEBYTECODE=1`, CUDA disabled, `/tmp` fixtures and `-p no:cacheprovider`.

| Tests | Result |
|---|---:|
| `test_exp11_shell.py` | **180 passed** |
| `test_exp11_pathprobe.py` | **12 passed** |
| `test_exp11_train.py` | **56 passed** |
| Pid-record and lock-holder tests | **82 passed** |
| `test_exp09_sim_eval_closures.py` | **3 passed** |
| Pinned outputs, registry and historical-output checks | **8 passed** |
| **Total** | **341 passed** |

The additional adversarial replays expose gaps beyond those passing tests. Both launcher syntax checks and `git diff --check` pass.

Production **+285/−130** and tests **+301/−2** match the report. Its retained full-suite log records **1 failed, 3951 passed, 55 skipped**; the failure is the stale exp_06 summarizer approval. I did not rerun that full suite.

Two reporting corrections:

- The log’s [line 64](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_01:33_suite_full_cpu_b9315aa.log:64) records HEAD `cad60185a63b6247c422831bd508e8cadd08da73`, despite the report’s exact-HEAD claim of `b9315aa`. Those commits have identical production bytes.
- “Every commit is under 200 changed lines” applies to the 14 implementation/test commits, not all 16: the two bookkeeping commits add 707 and 254 lines.

**Closure hygiene**

All 28 exp_06/exp_11 code digests agree between `b9315aa` and `f966461`.

Against the exp_06 approval record at `9f98bbb`, only `code.summarize_haa` moves; the other 19 keys match. Against `2ca784a`, exp_11’s moving keys are `train`, `finalize`, `launch_sh`, `smoke` and `summarize_haa`. Every moving closure contains `tools/exp11_pathprobe.py`; all three HAA closures remain unchanged. All eight values appear in the manifest below.

Sim-eval closure checks pass, including all ten real manifests. Pinned oracles remain byte-identical. Frozen sources remain untouched; the baseline comparison retains the previously authorized summarizer change and new exp_11 model files.

```text
tools/exp06_launch.sh:
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py:
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479

BACKBONES_EXP06:
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

`ckpt/exp11/pretrain` is absent, including the stray `.publish.lock`. The repaired timeout regression explicitly redirects its roots to `tmp_path`, and the dry-run regression verifies that no production lock is created. The selected suite left those paths absent.

The production worktree is unchanged. Git status contains only the pre-existing untracked close16 prompt. No review file was saved.

**Exact fill manifest for the unchanged candidate tree**

These values describe a content-identical merge of `f966461`; they do **not** authorize merging it. Recompute affected closures after fixing the findings.

Replace only exp_06 `code.summarize_haa`:

```text
720862918e131449e111602b0e3684b4c29be1a36dae9a44542de47472e83b45
```

Preserving every other byte of the exp_06 record yields this record SHA256:

```text
ef3fac454ff762d7fd61e3185e1972d02f5873db2b97c65d68f0dfddd43a2702
```

The exp_11 fill manifest is:

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
    "finalize": "ee39e005182368d999f9d7824ab805ae5cd22e51f1a6d31e5d96fa514813a6e7",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "a9fd96978609ab038ea68aed51971de69ccd09d2a2ad01af9ef1f1de14b68955",
    "smoke": "3fd783ccc24daec89e88445fe2cdf2814d3da7c0e8663f583e32e28a943529b6",
    "summarize_haa": "720862918e131449e111602b0e3684b4c29be1a36dae9a44542de47472e83b45",
    "train": "21e08ffe412c1adaf80888fde36f0faca7227563da72d7e062d37c0f4229aae1"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "ef3fac454ff762d7fd61e3185e1972d02f5873db2b97c65d68f0dfddd43a2702",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one trailing newline, this exp_11 record is 1316 bytes with SHA256 `fa862ac9913c69a440f04cf813ba9d82767c66e7d56c0abfeb12f0e365d14e46`.

