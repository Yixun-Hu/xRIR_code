**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15 · log `seen_protocol_2026-09-15_10:40:49_codex_code_round1.log`

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

## Verdict: approve with changes

**Blocking findings: none.** Round 2 implementation can proceed. GPU parity remains unverified and must pass before any launch.

## Findings

1. **Should-fix — legacy cache hits omit the promised protocol field.**  
   [tools/provenance.py:315](/home/yixunhu/codespace/xRIR_code_wt07/tools/provenance.py:315) returns legacy records unchanged. I confirmed that the live unseen inventory is accepted but its returned record lacks `protocol`. This makes the new API’s schema depend on whether its cache predates exp_07.

   **Minimal fix:** return an in-memory copy with `protocol='unseen'` after successful legacy validation. Preserve the existing cache key and on-disk file. Update [tests/test_provenance.py:339](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_provenance.py:339) accordingly. This is a schema inconsistency; I found no cross-protocol acceptance vulnerability.

## Verified

### Trainer and closure

Read every changed line in `git diff f492613..d7511fb`.

The unseen path preserves dataset classes, constructor arguments, train-before-test construction, loader options and order, seeding, model construction, augmentation, loss, optimizer, scheduler, banner, and counter/overflow logic. The added class selector consumes no RNG; importing the seen module neither constructs its dataset nor reads its pickle.

Additional CPU checks ran both backbones through initialization using real datasets, seed 0, `--epochs 0 --no-save`, and CPU substitution for `.cuda()`. They confirmed:

- Identical initial model state, dataset order, loader configuration, CPU RNG checkpoints, and first sampled batch.
- Identical runtime-argument values and keys except added `protocol='unseen'`.
- Identical remaining stdout, including the disabled banner.

`args.json` still serializes the same `vars(args)` used by `XRIR_RUNTIME_ARGS`.

Import-derived closure comparison found **11 → 12 files**, with exactly one addition:

`treble_multi_room_dataset/treble_xRIR_seen_dataset.py`

No files disappeared; `tools/provenance.py` remains absent and `utils/lr_scheduler.py` remains present. Imported source bytes matched their respective Git revisions.

These checks do **not** establish GPU forward/backward parity.

### Real datasets

Constructed both protocols on CPU and exhausted their batch samplers:

| Protocol | Train | Test | Batches at 32 | Final batch |
|---|---:|---:|---:|---:|
| unseen | 296,334 | 6,337 | 9,261 | 14 |
| seen | 296,454 | 6,217 | 9,265 | 6 |

Both have zero train/test target overlap. Training-set differences are **5,244 seen-only** and **5,124 unseen-only** files. Both seen real-batch readback tests passed, including shapes, float32 types, and finite values.

The unchanged accumulation rule yields **4,633 updates/epoch**, or **55,596 over 12 epochs**. The seen augmentation counter uses B=9,265 as intended.

### Provenance and audit

The legacy unseen key is safe under the existing stat-validated cache model:

- The live inventory was accepted for unseen and refused for seen.
- In-memory adversarial checks verified bidirectional refusal even with equal file counts.
- Removing a seen record’s protocol, corrupting key/count/digest/root/stamps, and relabeling a different seen inventory as unseen all produced refusal.
- Acceptance checks the complete selected file list and stamps, not merely protocol or count.

`seen_split_identity` matched `sha256sum`:

`bc97850b090198cf956dc7975a2da611dd058f8b0f83adee7f93f7b7fad7720f`

Valid binding passed `revalidate`; simulated changed content was rejected. Training inventory coverage remains WAV-only, as explicitly permitted.

The audit dispatched both protocols correctly in CPU checks with output captured in memory. Artifacts recorded the correct protocol and loader length. Seen without `--out` exited before dataset construction. The unseen default path and existing exclusive-write protection remain intact. `cohort_sha256` continues to bind ordered **[query path, offset] pairs**.

### Hygiene and test evidence

| Commit | Added | Deleted | Total changed | Opus trailer |
|---|---:|---:|---:|---|
| `6ed7bef` | 187 | 3 | 190 | Present |
| `e122355` | 126 | 17 | 143 | Present |
| `d7511fb` | 76 | 6 | 82 | Present |

- Exactly the six specified files changed: **389 insertions / 26 deletions**.
- Pinned-path diff: empty.
- `git diff --check f492613..d7511fb`: exit 0.
- Python 3.8 `py_compile.compile(..., doraise=True)`: all six passed, with bytecode writes suppressed.
- Tracked working tree remained clean. The pre-existing untracked `checkpoints` symlink remains.

The requested five-file pytest suite could not reproduce **188 passed / 14 skipped** unmodified: read-only restrictions blocked temporary files and dependency caches.

With in-memory capture and dependency-cache accommodations, it produced **119 passed, 14 skipped, 68 setup errors, 1 failure**. The failure was the subprocess closure import encountering the same plotting-cache restriction; its isolated retry passed. Thus **120 distinct cases passed**, while **68 remained blocked by temporary-file requirements**.

Blocked cases include trainer serialization/dispatch, provenance cache/mutation tests, audit output tests, and inherited trainer filesystem fixtures. Their implementations were reviewed; supplementary read-only checks are described above. No GPU was used.

## Coder decisions

| Decision | Assessment |
|---|---|
| **1 — Keep `xRIR_Dataset` unseen alias** | **Accept.** Preserves existing patches and callers; dispatch honors it. |
| **2 — Preserve unseen cache key** | **Accept the key.** Isolation checks remain effective and the live cache stays valid. Normalize the returned legacy record per finding 1. |
| **3 — Defer `seen_split` allowlist** | **Accept.** Explicitly permitted by the round-1 prompt; the TODO identifies the launcher-owned constant. |
| **4 — Stub real-root hashing** | **Accept.** Tests actual file selection/counts without rereading 33 GB; synthetic tests cover hashing behavior. |
| **5 — Duplicate `PROTOCOLS`** | **Accept.** The two definitions agree and provenance stays outside the training closure. |
| **6–7** | **Not assessable:** neither the supplied prompt nor available notebook specifies them. |

## Fitness for round 2

The implementation provides the required foundation. Round 2 must connect protocol-aware identity and `seen_split` binding to launch, recovery, and probe receipts; enforce seen roots, B=9,265, and protocol-specific refusals; and normalize historical unseen arguments.

Before any launch, complete the **four deferred GPU parity cases**—both backbones × yaw off/on—plus the prescribed seen finite-step, real alignment-audit, smoke/probe, and final-review gates.
132,667
**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

## Verdict: approve with changes

**Blocking findings: none.** Round 2 implementation can proceed. GPU parity remains unverified and must pass before any launch.

## Findings

1. **Should-fix — legacy cache hits omit the promised protocol field.**  
   [tools/provenance.py:315](/home/yixunhu/codespace/xRIR_code_wt07/tools/provenance.py:315) returns legacy records unchanged. I confirmed that the live unseen inventory is accepted but its returned record lacks `protocol`. This makes the new API’s schema depend on whether its cache predates exp_07.

   **Minimal fix:** return an in-memory copy with `protocol='unseen'` after successful legacy validation. Preserve the existing cache key and on-disk file. Update [tests/test_provenance.py:339](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_provenance.py:339) accordingly. This is a schema inconsistency; I found no cross-protocol acceptance vulnerability.

## Verified

### Trainer and closure

Read every changed line in `git diff f492613..d7511fb`.

The unseen path preserves dataset classes, constructor arguments, train-before-test construction, loader options and order, seeding, model construction, augmentation, loss, optimizer, scheduler, banner, and counter/overflow logic. The added class selector consumes no RNG; importing the seen module neither constructs its dataset nor reads its pickle.

Additional CPU checks ran both backbones through initialization using real datasets, seed 0, `--epochs 0 --no-save`, and CPU substitution for `.cuda()`. They confirmed:

- Identical initial model state, dataset order, loader configuration, CPU RNG checkpoints, and first sampled batch.
- Identical runtime-argument values and keys except added `protocol='unseen'`.
- Identical remaining stdout, including the disabled banner.

`args.json` still serializes the same `vars(args)` used by `XRIR_RUNTIME_ARGS`.

Import-derived closure comparison found **11 → 12 files**, with exactly one addition:

`treble_multi_room_dataset/treble_xRIR_seen_dataset.py`

No files disappeared; `tools/provenance.py` remains absent and `utils/lr_scheduler.py` remains present. Imported source bytes matched their respective Git revisions.

These checks do **not** establish GPU forward/backward parity.

### Real datasets

Constructed both protocols on CPU and exhausted their batch samplers:

| Protocol | Train | Test | Batches at 32 | Final batch |
|---|---:|---:|---:|---:|
| unseen | 296,334 | 6,337 | 9,261 | 14 |
| seen | 296,454 | 6,217 | 9,265 | 6 |

Both have zero train/test target overlap. Training-set differences are **5,244 seen-only** and **5,124 unseen-only** files. Both seen real-batch readback tests passed, including shapes, float32 types, and finite values.

The unchanged accumulation rule yields **4,633 updates/epoch**, or **55,596 over 12 epochs**. The seen augmentation counter uses B=9,265 as intended.

### Provenance and audit

The legacy unseen key is safe under the existing stat-validated cache model:

- The live inventory was accepted for unseen and refused for seen.
- In-memory adversarial checks verified bidirectional refusal even with equal file counts.
- Removing a seen record’s protocol, corrupting key/count/digest/root/stamps, and relabeling a different seen inventory as unseen all produced refusal.
- Acceptance checks the complete selected file list and stamps, not merely protocol or count.

`seen_split_identity` matched `sha256sum`:

`bc97850b090198cf956dc7975a2da611dd058f8b0f83adee7f93f7b7fad7720f`

Valid binding passed `revalidate`; simulated changed content was rejected. Training inventory coverage remains WAV-only, as explicitly permitted.

The audit dispatched both protocols correctly in CPU checks with output captured in memory. Artifacts recorded the correct protocol and loader length. Seen without `--out` exited before dataset construction. The unseen default path and existing exclusive-write protection remain intact. `cohort_sha256` continues to bind ordered **[query path, offset] pairs**.

### Hygiene and test evidence

| Commit | Added | Deleted | Total changed | Opus trailer |
|---|---:|---:|---:|---|
| `6ed7bef` | 187 | 3 | 190 | Present |
| `e122355` | 126 | 17 | 143 | Present |
| `d7511fb` | 76 | 6 | 82 | Present |

- Exactly the six specified files changed: **389 insertions / 26 deletions**.
- Pinned-path diff: empty.
- `git diff --check f492613..d7511fb`: exit 0.
- Python 3.8 `py_compile.compile(..., doraise=True)`: all six passed, with bytecode writes suppressed.
- Tracked working tree remained clean. The pre-existing untracked `checkpoints` symlink remains.

The requested five-file pytest suite could not reproduce **188 passed / 14 skipped** unmodified: read-only restrictions blocked temporary files and dependency caches.

With in-memory capture and dependency-cache accommodations, it produced **119 passed, 14 skipped, 68 setup errors, 1 failure**. The failure was the subprocess closure import encountering the same plotting-cache restriction; its isolated retry passed. Thus **120 distinct cases passed**, while **68 remained blocked by temporary-file requirements**.

Blocked cases include trainer serialization/dispatch, provenance cache/mutation tests, audit output tests, and inherited trainer filesystem fixtures. Their implementations were reviewed; supplementary read-only checks are described above. No GPU was used.

## Coder decisions

| Decision | Assessment |
|---|---|
| **1 — Keep `xRIR_Dataset` unseen alias** | **Accept.** Preserves existing patches and callers; dispatch honors it. |
| **2 — Preserve unseen cache key** | **Accept the key.** Isolation checks remain effective and the live cache stays valid. Normalize the returned legacy record per finding 1. |
| **3 — Defer `seen_split` allowlist** | **Accept.** Explicitly permitted by the round-1 prompt; the TODO identifies the launcher-owned constant. |
| **4 — Stub real-root hashing** | **Accept.** Tests actual file selection/counts without rereading 33 GB; synthetic tests cover hashing behavior. |
| **5 — Duplicate `PROTOCOLS`** | **Accept.** The two definitions agree and provenance stays outside the training closure. |
| **6–7** | **Not assessable:** neither the supplied prompt nor available notebook specifies them. |

## Fitness for round 2

The implementation provides the required foundation. Round 2 must connect protocol-aware identity and `seen_split` binding to launch, recovery, and probe receipts; enforce seen roots, B=9,265, and protocol-specific refusals; and normalize historical unseen arguments.

Before any launch, complete the **four deferred GPU parity cases**—both backbones × yaw off/on—plus the prescribed seen finite-step, real alignment-audit, smoke/probe, and final-review gates.
