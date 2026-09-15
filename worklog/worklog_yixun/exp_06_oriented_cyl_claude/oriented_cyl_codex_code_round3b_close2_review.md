**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# exp_06 round 3b — CLOSE review 2

**Verdict: request changes.** Findings **1, 4, and 7 are resolved**. Findings **2, 3, 5, and 6 are partially resolved** and remain closure-blocking.

Reviewed `df7ea24..07001a0` on `exp06-round2b` in `/home/yixunhu/codespace/xRIR_code_wt2b`, against the required SOP, plan v4, prior reviews, experiment records, notebook, fix2 prompt, and Coder report. Numbering below preserves close review 1.

## Findings

### 1. Resolved — H3 registered dataset and references

[tools/exp06_compare.py:49](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:49)

Admission now checks the registered inventory digest, seed-specific K=8 reference identity, canonical ordered query digest, and 17-room population.

Independent mutations of the real exp_04 seed-42 record were refused:

- Empty inventory with recomputed digest: registered inventory mismatch.
- `refs: []`: eight-reference requirement.
- Consistently invented query names: registered reference mismatch.

All five real exp_04 control runs still admit as arm A. Publishing these registered identities in `result["split"]` is appropriate.

### 2. Partial — **blocker** — Evidence retention and snapshot consistency remain incomplete

[tools/exp06_summarize_haa.py:510](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:510), [verification sequence:546](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:546)

The new binding helper, exact-byte JSON reader, child-artifact retention, and producer/approvals bindings are useful fixes. Two gaps remain.

**2a. Child provenance dependencies disappear from the publication input map.**

`bind_child` retains the provenance file itself, but omits dependencies named inside its data inventory, source closures, and mutable-input records. The finalizer checks these during admission; publication subsequently rechecks only the retained map.

Using a synthetic nine-child job produced and verified through the real wrappers/finalizer:

```text
CHILD_CACHE_BOUND rirs.npy False
CHILD_CACHE_BOUND depth.npy False
PUBLISHED_PRIMARY_AFTER_MUTATION rirs.npy True
PUBLISHED_PRIMARY_AFTER_MUTATION depth.npy True
```

The publication test changed each cache file after admission and called the real `analyse`/`write_outputs` functions. Both output files were published. Adding the summarizer’s producer closure did not cover the missing child-source dependencies either; examples included `tools/exp06_haa_finetune.py`, `tools/exp06_haa_eval.py`, and `sim_to_real/finetune_haa.py`.

**Why it matters:** the published evidence does not preserve the inputs whose identities admission certified.

**Minimal fix:** retain every validated child-provenance dependency, with its validated digest, in the shared input map. Revalidate that complete map before publication and reject conflicting bindings.

**2b. Parent-bound hashes are checked before validation but are not enforced against the later snapshot.**

For child completions, line 557 checks the parent’s digest, line 559 validates the child, and line 560 rereads and binds the completion. That final binding can describe different bytes. The job-spec sequence has the same missing comparison.

I changed the stage-1 completion’s log binding immediately after `verify_child` returned. Admission and final input revalidation both succeeded, although:

```text
inputs[stage1/completion.json] != job_completion["children"]["stage1"]
```

**Why it matters:** a published evidence map can contradict the parent completion that supposedly certifies it.

**Minimal fix:** bind the parent’s expected digest before delegation and require every subsequent snapshot to match it. Parse and hash the same bytes; do not replace a previously validated identity with a fresh digest.

### 3. Partial — **blocker** — Primary admission does not register validation rooms

[tools/exp06_summarize_haa.py:455](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:455)

The new profile correctly rejects shortened training and seeded-phase evaluation from primary admission. It checks training rooms but ignores `val_rooms`.

I constructed a finalized job with the registered 1000/10 and 200/2 schedules and other registered settings, while restricting stage-1 validation to `["hallway"]`. Primary admission returned:

```text
PRIMARY_WITH_STAGE1_VAL_HALLWAY_ADMITTED []
```

The empty list means no recipe deviation was recorded.

**Why it matters:** validation loss selects `best.pth`, which initializes the next stages. Changing that selection population changes the experiment even when training rooms and epoch counts match.

**Minimal fix:** compare the effective validation rooms—explicit `val_rooms`, otherwise training rooms—with exp_02’s registered selection population for each stage. Reject deviations in primary mode and report them in sensitivity mode. Add a finalized-job regression.

**Sensitivity choice:** labeling rather than suppressing verdicts is acceptable for explicitly descriptive sensitivity output. The document-wide mode/banner and H1/H1b prefixes distinguish it sufficiently. Primary production must omit `--sensitivity`.

### 4. Resolved — H3 model, registry, metadata, and epoch identities

[tools/exp06_compare.py:230](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:230)

Roles now bind backbone and registered epoch; C’s approved artifact must itself specify epoch 12. The exp_06 route checks the actual registry digest, corresponding model class, checkpoint role, frame/heading, and output metadata.

The new refusal tests passed. Historical baseline records without an explicit epoch receive their registered epoch, while contradictory declarations are refused.

### 5. Partial — **should-fix** — exp_05 tier parsing can describe different bytes from its bound digest

[tools/exp06_compare.py:394](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:394)

The substantive route checks are implemented: adjacent args path, both declared digests, registered M configuration/counts, and matching tier metadata. Wrong args digests and contradictory tier declarations are refused.

However, `bind(path)` hashes the file before `path.read_text()` parses it. The report’s assertion that publication catches a change is incomplete: a temporary replacement followed by restoration passes.

Independent reproduction:

1. Keep a legacy M args file as the bound bytes.
2. Declare `legacy_M=False`; ordinary admission correctly refuses it.
3. During the later parsing read, temporarily substitute explicit M-tier args, then restore the original bytes.
4. Admission returns `legacy_M=False`, binds the original file’s digest, and final revalidation succeeds.

```text
EXP05_BOUND_ORIGINAL_BYTES True
EXP05_TRANSIENT_ARGS_RECHECK_ACCEPTED
```

**Why it matters:** the published tier interpretation need not describe the file identified by its hash.

**Minimal fix:** read args bytes once, derive both JSON and SHA-256 from that buffer, and reconcile that digest with existing and declared bindings.

**Registration choice:** registering arm B as exp_05’s `M_cyl` in `ROLES`/`EXP05_M` is appropriate. Evaluating the same registered weights through exp_06 legitimately uses the exp_06 route; exp_05-specific tier metadata is then inapplicable.

### 6. Partial — **should-fix** — A3 owner parsing and hashing use different reads

[tools/exp06_summarize_haa.py:339](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:339)

Missing, unreadable, and persistently mismatching `launch.pid` files are now refused. Avoiding retrospective PID-liveness checks is correct.

The owner is parsed first and independently hashed afterward. I changed the marker between those operations:

```text
OWNER_RACE_ADMITTED 3 DISK_PID 4
```

Admission returned owner PID 3 while binding the bytes containing PID 4; final input revalidation passed.

**Why it matters:** the retained ownership evidence can contradict `owner_pid`.

**Minimal fix:** read `launch.pid` once, parse that buffer, hash that same buffer, compare the parsed PID with `owner_pid`, and retain that digest.

### 7. Resolved — Production approvals must match committed bytes

[tools/exp06_compare.py:686](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:686), [tools/exp06_summarize_haa.py:938](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:938)

Both producers now require committed-byte binding outside exploratory mode.

I independently tested a fully populated, schema-valid approval record:

- Outside-repository uncommitted record: refused by both producers.
- Modified bytes presented at the tracked template path: refused by both against HEAD’s blob.
- Historical commit where the path was untracked: refusal covered by the replayed tests.

`--approved-commit` is a reasonable override for later analysis. Planner wiring should supply the full reviewed commit SHA; the override still requires blob equality and applicable producer identities.

## Verification

All checks used the requested Python/environment, CPU only, bytecode disabled, and pytest cache provider disabled. No filesystem file was modified, no process was signalled, and no GPU work ran.

Tests requiring temporary files used a **process-local RAM filesystem**. Finalizer fixtures used the current repository instead of creating a writable clone; the shell’s unchanged job-spec Python was executed in-process. Numba JIT was disabled to avoid cache writes. Filesystem durability was therefore not tested.

| Check | Result |
|---|---|
| `git diff df7ea24..07001a0` across all four changed files | Changed lines reviewed |
| `git diff --check df7ea24..07001a0` | Clean |
| Commit log, numstats, and trailers | 18 commits; 9 red/implementation pairs; maximum 109 changed lines; both trailers on all 18 |
| Protected files compared with `git show main:<path>` | 42 distinct protected paths byte-identical; exp_03 and trainer closure digests matched |
| `git diff main...07001a0 --diff-filter=M --name-only` | Only eight earlier exp_06-owned files |
| In-memory compilation of the four changed Python files | Passed |
| Compare + summarizer + approvals API pytest replay, excluding finalizer-backed cases | **174 passed, 8 deselected** |
| Remaining finalizer-backed summarizer cases | **8 passed, 59 deselected** |
| Broader `tests/test_exp06_*.py tests/test_provenance.py tests/test_exp03_record_tools.py`, excluding filesystem-fixture cases | **435 passed, 8 skipped, 772 deselected** |
| Real exp_02 canonical regression | **All eleven cells exact** |
| Real exp_04 control admission | **All five seeds admitted as A** |
| Repeated H3 analysis and global RNG checks | Identical results; Python, NumPy, and Torch RNG states unchanged |
| Final branch/HEAD/status | `exp06-round2b`, `07001a0cf6e248d9640a9f2f20c15a4eb1609a78`; clean |

Pytest replays used `-q --capture=sys -p no:cacheprovider`; the broader selection also used `--tb=short`. Counts above overlap across selections.

The Coder reports **38 historical red failures**. I verified the commit structure and final tests, but did not independently replay every historical red revision or the complete reported 1,207-test suite.

## Exact closure-blocking list

The Coder must fix:

- **2a:** retain and revalidate all child-provenance dependencies.
- **2b:** enforce parent-bound completion/spec digests across validation and subsequent reads.
- **3:** enforce registered effective validation rooms in primary admission.
- **5:** parse and hash one exp_05 args snapshot.
- **6:** parse and hash one `launch.pid` snapshot.

**Round 3b must remain open.**
