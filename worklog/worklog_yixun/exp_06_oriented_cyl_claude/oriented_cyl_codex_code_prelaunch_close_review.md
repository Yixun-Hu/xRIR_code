**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-16.

**Verdict: approve for the post-training merge.**

Reviewed `8bc7cf8..c867e24` on `exp06-prelaunch`, including every changed line and both commits, against the required records, prior review, and Coder prompt/report.

1. **Previous should-fix — resolved: HAA diagnostic artifact confinement.**  
   [tools/exp06_finalize.py:863](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:863), [artifact validation:916](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:916), [regressions:517](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_finalize.py:517).

   `confined()` rejects symlinks in every component below the resolved attempt directory and rejects paths changed by resolution. The artifact directory and `args.json` are checked before reading arguments; every registered artifact is checked before hashing. The reproduced escapes now refuse without publishing completion. No further fix is required.

**No new blocker, should-fix, or nit findings.**

The scope decision is acceptable. `full`, `haa_train`, and `haa_eval` retain their existing contracts; A4’s disposable-tree restriction applies specifically to the two HAA diagnostic types. Widening confinement is not required before HAA execution.

Hard links are accepted in both diagnostic types. This satisfies the stated regular-file/resolved-location contract: a hard link is a regular directory entry inside the attempt, even when another name for its inode exists outside. The contract does not require exclusive inode ownership or establish artifact authorship through location alone.

Verification performed:

- `git diff 8bc7cf8..c867e24`: only `tests/test_exp06_finalize.py` and `tools/exp06_finalize.py` changed. Commits contain **40** and **41** changed lines respectively; both required trailers are present, and the red commit precedes implementation.
- `git diff main...exp06-prelaunch --diff-filter=M --name-only`: exactly the five previously reviewed exp_06 files. **41 pinned files are byte-identical to `main`**, including exp_03’s twelve-file closure and the trainer’s eleven-file closure.
- `git diff --check 8bc7cf8..c867e24`, `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh`, and in-memory `compile()` of both changed Python files: **passed**.
- Replayed the **five unchanged added regression bodies** using an in-memory filesystem and extracted production functions: **five expected failures at `9381b6a`; five passes at `c867e24`**. The directory case reproduces the original wrong-cause failure.
- A **67-case supplemental matrix** covered both diagnostic types: every registered filename, including `args.json`, `provenance.json`, and `metrics_all.json`; external/internal/relative symlinks; dangling links; loops; symlinked directories; hard links; directory/FIFO substitutions; and escaping attempt paths. Results matched the contract.
- Additional probes rejected artifact-level `..` escapes. Non-strict `resolve()` can leave a missing suffix unchanged, but `is_file()` subsequently refuses it. Existing real `/proc/self` symlinks, a real `..` escape, and double-leading-slash spelling also refused without creating filesystem fixtures.
- Confirmed `root=None` preserves previous hash-through-symlink behavior. AST comparison shows `full_evidence`, `haa_train_evidence`, `haa_eval_evidence`, `load_job_spec`, and `finalize` unchanged.
- Native pytest with the supplied interpreter/environment and `--capture=sys -p no:cacheprovider`:
  - Five selected approvals schema/binding tests: **5 passed, 19 deselected**.
  - `tests/test_exp06_bootstrap.py tests/test_exp03_record_tools.py`: **72 passed, 1 skipped, 12 setup errors**, all from unavailable writable temporary directories.
  - Requested full suite: **18 collection errors** from unwritable Numba/Matplotlib caches; no full replay was possible.

The supplemental harness used Python 3.8’s actual `pathlib` resolution with memory-backed filesystem operations. Unrelated approval/lifecycle dependencies were simulated; this is targeted verification, not a full integration replay.

**Approvals-drift qualification:** I independently reproduced the approvals guard’s failing assertion and the training approval gate’s refusal. Exactly four code keys differ; the other sixteen match. The Coder reports **1,252 passed, 7 skipped, one failure**, naming only `test_every_filled_record_digest_is_the_one_this_checkout_computes`. The failure’s cause is corroborated; the sandbox prevents me from independently certifying that full-suite total or the absence of other failures.

The independently recomputed closure digests at **`c867e245186c2e72eb481ad9e515e54098369f11`** are:

| Key | Classification | SHA-256 |
|---|---|---|
| `finalize` | TRAINING_KEYS | `5ef318e8da734e6076f89237014d1e2366078692b1da83ef93b304d1dc609399` |
| `smoke` | TRAINING_KEYS | `00e3b91b8289cc0f1fda8c2ab2ce3b151ca897f4a4c746f93863e3d338091d00` |
| `launch_sh` | TRAINING_KEYS | `c64b60df4632b83617c1343f0d0a56c6ac7cbab24db95719a2aa8734f3977773` |
| `summarize_haa` | Non-training | `2ad63b6dea015f996f1568a4e6e8d1513d356c83246b96dd19cc98c3fa4cefb4` |

These use statically derived import-closure membership and the actual `provenance.closure_record` hashing. All four match the Coder’s report.

**Exact blocking findings for the Coder: none.** Preserve A6: merge only after pretraining completes and finalizes using its original code. The Planner’s reviewed approvals refill, green suite, and integrative review remain prerequisites before HAA execution.

No files were modified, no processes were signalled, and no GPU work was started.
