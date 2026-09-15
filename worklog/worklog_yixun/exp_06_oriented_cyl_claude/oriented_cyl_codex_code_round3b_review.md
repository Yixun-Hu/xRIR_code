**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Verdict: request changes

Reviewed `7f29877..0ca806c` on `exp06-round2b` in `/home/yixunhu/codespace/xRIR_code_wt2b`, including every changed line, the required briefing documents, and the composed tooling.

The canonical exp_02 regression passes exactly. The main problems are admission and provenance: valid HAA runs encounter an impossible closure requirement, while incomplete or inconsistent evidence can pass other admission checks. **Round 3b must remain open.**

## Findings

1. **Blocker — valid HAA training and evaluation children cannot share the required closure.**  
   [tools/exp06_summarize_haa.py:388](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:388)

   `load_new_arm` collects every child’s `source_closure_sha256` into one set and requires exactly one digest. The finalizer records the executed entry point’s closure: training and evaluation have different entry points and different closures. On this HEAD, their computed digests start `9288884b…` and `1f20e36b…`. Supplying distinct training/evaluation digests causes refusal.

   Consequently, genuine completed arms cannot reach analysis. The fixture’s single constant `CLOSURE` hides this incompatibility.

   **Minimal fix:** validate closures by entry point/run type against the corresponding approvals; require consistency among children executing that entry point. Add a positive admission test using the actual distinct closures.

2. **Blocker — new-arm admission bypasses the finalizer’s evidence, lineage, and heading validation.**  
   [tools/exp06_summarize_haa.py:323](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:323)

   `child_completion` is a shallow reader. The substantive checks reside in `verify_child`, which this consumer does not call. Hashing whichever artifacts a completion happens to enumerate does not establish that required artifacts exist. Likewise, job admission checks the syntax of `job_spec_sha256` without reading the specification.

   An in-memory tree containing four jobs and 31 child completions was admitted without provenance, logs, exit receipts, checkpoints, histories, summaries, metrics files, heading JSONs, or job specifications. Additional accepted mutations included:

   - `admissible_arm=False` on a child.
   - A stage-1 initialization contradicting the approved job initialization.
   - Different heading identities between seeds.
   - Per-sample heading identity contradicting the child/arguments.
   - `num_shot=1` in child arguments.

   The report’s assertion that per-sample heading identity agrees with the child is incorrect: `child_per_sample` checks its shape and roll, not equality.

   **Minimal fix:** rehash and validate the A3 specification, invoke the full child verifier, validate initialization/checkpoint lineage and the frozen evaluation protocol, and compare each room’s heading identity across the arm and against approvals. Re-read heading JSONs through the existing cache-aware validator. Strengthen fixtures to exercise these validators.

3. **Blocker — producer approvals mostly establish populated fields, not approved execution.**  
   [tools/exp06_summarize_haa.py:636](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:636), [tools/exp06_compare.py:359](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:359)

   `require_producer` checks for null leaves. The producers must subsequently compare actual identities, but those comparisons are missing. The summarizer does not compare its producer/child closures, heading artifacts, G1 artifact, or canonical exp_02 identities with their required pins. Its `approved` analysis parameter is unused. The comparer records its producer identity without checking `code.compare`; the reused approvals-file hashes are not consumed.

   Moreover, `--write-legacy-receipt` branches before the approvals gate, bypassing that producer’s required `code` approvals altogether.

   **Minimal fix:** retain the adapter’s completeness check, then enforce actual-versus-approved identities in each producer. Gate receipt production too. Include the summarizer’s verified producer identity in its output. Add wrong-but-well-formed digest tests.

4. **Blocker — H3 accepts missing output hashes and incomplete completion evidence.**  
   [tools/exp06_compare.py:85](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:85)

   `bind(..., expected=None)` treats a declared null hash as permission to skip comparison. Setting the completion’s per-sample hash to null and changing EDT to `9999` was accepted.

   Admission also accepted a missing completion log, `completion.confirmatory=False`, a missing bound training-completion input, and an incorrect aggregate EDT mean. It omits the inherited input revalidation, data identity, source-file revalidation, and metrics reconciliation.

   **Minimal fix:** use a distinct sentinel for intentionally unbound inputs; require valid hashes for declared bindings. Compose the applicable `paired_compare` admission checks, including completion flags/logs, mutable inputs, source/data evidence, and aggregate reconciliation.

5. **Blocker — H3 does not establish the registered K=8 reference/query protocol.**  
   [tools/exp06_compare.py:107](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:107)

   Admission never requires `num_shot == 8`. It hashes the reference file but does not parse it, verify its seed/K/semantic hash, or compare its entries with the output queries. Array length and `index == range(6337)` are insufficient.

   Overlays of a real control run were accepted with `num_shot=1`, reversed query names while retaining the indices, and an invented semantic manifest hash. Exp_06 model/registry identity and baseline checkpoint epoch checks are also incomplete.

   **Minimal fix:** validate the reference using the existing manifest helpers and registered exp_04 identities; require exact query/index order, K=8, applicable grids/protocol fields, and role-specific model/checkpoint identities. Test these refusals with internally consistent hashes so an earlier checksum failure cannot mask them.

6. **Should-fix — a legacy receipt can omit an arm that admission still loads.**  
   [tools/exp06_summarize_haa.py:189](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:189)

   Expected enumeration is derived from the receipt’s own `arms` field. A receipt listing only `control`, with a matching approved receipt hash, passes verification; `load_legacy` nevertheless returns both `control` and `cyl`.

   I verified this against the complete real exp_02 root using an in-memory receipt. It enumerated 64 files, and changing an unlisted cylindrical hallway C50 observation to `9999` remained admissible.

   **Minimal fix:** require exactly the registered legacy arms and construct the expected enumeration independently from `LEGACY_ARMS`. Test partial/empty arm declarations and tampering in either arm.

7. **Should-fix — the claimed exp_05 M-tier admission route rejects its actual entry-point identity.**  
   [tools/exp06_compare.py:141](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:141)

   Absence of `writer_exp06` selects an exp_04-only entry-point check. Actual `exp05_eval.build_fields` delegates with `module=tools.exp05_eval`; its entry-point closure is therefore exp_05’s, although its writer is exp_04’s. Report discrepancy 9 incorrectly claims this route admits those runs.

   **Minimal fix:** implement explicit exp_04, exp_05, and exp_06 identity routes. Read the hash-approved exp_05 approvals record for its evaluator identity and validate the M-tier metadata/bindings. Add an exp_05-shaped positive test.

8. **Should-fix — an empty HAA cohort aborts instead of producing the required void decision.**  
   [tools/exp06_summarize_haa.py:432](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:432)

   `cell_rows` raises before exclusion counts reach the void policy. Setting every treatment hallway C50 observation invalid produced `ValueError: … cohort … is empty`, rather than an H1/H1b `void` result. For nonempty void cohorts, bootstrap execution can also fail before `verdict_of` applies its advertised void precedence.

   **Minimal fix:** compute cohort/exclusion diagnostics first and return a reportable void cell when the rule fires. Handle unavailable intervals in rendering and downstream tables. Preserve the bootstrap helper’s registered zero-width refusal behavior.

9. **Should-fix — H3 verdict eligibility is not enforced.**  
   [tools/exp06_compare.py:250](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:250), [tools/exp06_compare.py:336](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:336)

   `--runs-b` is optional, and analysis silently skips the primary contrast. A C/A-only analysis produced a `CONFIRMATORY H3` report without H3’s comparator.

   Separately, exploratory analysis retains decision verdicts despite admission deviations. With an explicit missing-approvals deviation and default bootstrap count, it reported **`non-inferior`** for C−B.

   **Minimal fix:** require the complete A/B/C run set for production reporting and suppress decision-bearing verdicts whenever admission is exploratory/unapproved. Partial analyses must identify H3 as unavailable.

10. **Should-fix — publication does not revalidate the inputs used during analysis.**  
    [tools/exp06_summarize_haa.py:619](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:619), [tools/exp06_compare.py:321](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:321)

    Both producers can spend substantial time bootstrapping after admission, then publish without a final input check. The summarizer even hashes job completions later than their original read, allowing the recorded hash to describe different bytes. Its side-split cache inputs are not included in the input bindings.

    **Minimal fix:** capture hashes with the bytes consumed, include the cache files used by the side split, and revalidate all bindings immediately before publication. Reuse the existing `paired_compare.recheck_inputs` pattern. Add mutation-during-analysis tests.

11. **Nit — older refusal tests still have the reported temporary-path regex weakness.**  
    [tests/test_exp06_haa.py:196](/home/yixunhu/codespace/xRIR_code_wt2b/tests/test_exp06_haa.py:196)

    I audited the other exp_06 files. Direct candidates include `test_exp06_eval_launch.py:139`, `test_exp06_finalize.py:1515`, `:1519`, `:1871`, and `test_exp06_haa.py:158`, `:196`.

    The existing missing-cache-root test passed when `prepare` was replaced with an unrelated `ValueError` containing only the pytest-style temporary path: `match='root'` matched the directory name.

    **Minimal fix:** assert a distinctive diagnostic phrase or structured reason, independently of paths. Ensure mutated records have valid upstream bindings before testing downstream refusals.

12. **Nit — output-pair publication leaves partial results on failure.**  
    [tools/exp06_summarize_haa.py:740](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:740), [tools/exp06_compare.py:321](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:321)

    Report discrepancy 17 is accurate: exclusive creation protects existing files, but failure creating JSON leaves the newly written summary behind.

    **Minimal fix:** preflight distinct/absent output paths and adopt staged publication with cleanup, as the composed writer already does.

## Verification performed

All checks were CPU-only. I modified no files, signaled no processes, and did not access the excluded training worktree.

### Repository and provenance

- Read `git diff 7f29877..0ca806c`: **9 changed files**, 21 commits.
- `git diff --check 7f29877..0ca806c`: **clean**.
- `git diff main...HEAD --diff-filter=MD --name-only`: **empty**.
- Compared **44 protected paths** with `main`: **no mismatches**.
- Computed the trainer’s **11-file import closure**: byte-identical to `main`, digest `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`.
- Computed all **19 adapter code digests**, including the distinct HAA training/evaluation closures.
- Compiled all nine changed Python files **in memory**, without creating bytecode.
- Verified all 21 commit trailers. Confirmed the four oversized commits: `c2d4c8a1` 204, `d887e3fc` 251, `3ba18720` 258, `e79a8265` 207 changed lines.

### Test replay and limitations

Normal collection encountered writable-cache failures. The read-only environment also prevents `tmp_path` fixtures. For runnable subsets, I disabled Numba JIT and used the existing Matplotlib cache through an in-process writability-check shim. Repository source remained unchanged.

All pytest calls used `-p no:cacheprovider`; adapted calls also used `-s`.

| Check | Observed result |
|---|---|
| `pytest.main` over `test_exp06_profiles.py`, `test_exp06_compare.py`, `test_exp06_bootstrap.py`, with writable-fixture tests and the closure-subprocess test deselected | **91 passed, 28 deselected** |
| HAA subset covering constants, arm table, real regression, and missing-module refusal | **14 passed, 27 deselected, 5 setup errors**; errors were selected `tmp_path` cases |
| `tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged` | **1 passed** |
| `pytest.main` over mirror-probe and HAA-pipeline tests, excluding writable fixtures | **45 passed, 2 skipped, 52 deselected** |
| Five existing HAA statistical assertion bodies, supplied equivalent in-memory arrays | **5 passed** |

I did **not** independently reproduce the reported full **980 passed / 7 skipped** suite or historical red-run counts.

### Real artifacts and adversarial checks

- **Real exp_02:** all **11 cells** reproduced the canonical difference and both endpoints of **both interval schemes exactly**, through `exp06_bootstrap`; cohort counts matched.
- **Real exp_04 arm A:** all five retained control runs admitted, **6337 queries each**, using a constructed approval mapping for the available branch.
- **H2:** exercised all 11 cells with the actual default **50,000** adjusted draws; adjusted alpha was `0.05/11`, giving the specified tails. All three direction-label branches and the zero boundary were checked.
- **H1/H1b/D:** exercised pairing refusals, frozen margins, verdict boundaries, and the 33 descriptive cells.
- **Bootstrap:** replay covered endpoint convergence, seeds 0/1, tolerance, one quadrupling, and withheld verdicts.
- **H3:** repeated analysis produced identical serialized JSON; room-cluster intervals and inherited statistical functions were exercised.
- **RNG isolation:** Python, NumPy’s global generator, and Torch RNG states remained unchanged.
- Read-only filesystem overlays demonstrated the acceptance failures in findings 2, 4–6; direct analysis demonstrated findings 8–9. Numerical/admission functions were not replaced in those checks.
- The zero-shot side-split implementation and cache-agreement assertion were inspected; its writable-fixture integration tests were not replayed.

## Judgment on the Coder’s discrepancies

The report contains **19 numbered discrepancies**.

| Report item | Judgment |
|---|---|
| 1, 14 — absent module/lazy adapter | Appropriate for this branch. Ordinary production analysis refuses as intended; receipt production needs finding 3’s fix. |
| 2 — aliases/template additions | Explicit normalization and duplicate-key refusal are sound. Add `code.probe_align` during wiring; optional `code.profiles` is reasonable. |
| 3 — A3 assumptions | The stated shape follows the briefing. Reconcile actual merged field names and verify the bound specification; syntax alone is insufficient. |
| 4 — synthetic completions | Insufficient validation coverage; directly contributes to findings 1–2. |
| 5 — repo-relative legacy root | Accepted; verified against real artifacts. |
| 6 — heading verification depth | Rejected as implemented; the claimed identity agreement is incomplete. |
| 7 — zero-shot side split | Correct scope. |
| 8 — split/role test seams | Reasonable, but do not establish production protocol admission. |
| 9 — exp_05 dispatch | Incorrect; finding 7. |
| 10 — exploratory initialization relaxation | Acceptable for clearly suppressed draft decisions. |
| 11 — oversized commits | Confirmed; disclosed exception to the SOP’s “generally <200” guidance, not a closure blocker. |
| 12 — false-passing patterns | Confirmed residual weakness; finding 11. |
| 13 — fixed fixture PIDs | Valid post-merge concern. Preserve guaranteed-dead fixtures; I did not inspect the training branch’s fix. |
| 15 — computation cost | Unbenchmarked full-data cost remains disclosed; default adjusted-count execution was verified. |
| 16 — unconverged H2 labels | Correct. |
| 17 — publication atomicity | Accurate; finding 12. |
| 18 — `pkill` incident | Process incident requiring notebook follow-up; no assertion about affected peers can be established from this review. |
| 19 — baseline totals | Reported accounting, not an independently replayed full-suite result. Protected-file identity was verified directly. |

## Exact blocking list

**Findings 1–10 must be fixed and re-verified before round 3b closes:**

1. Role-specific HAA closure validation.
2. Complete HAA/A3 evidence, lineage, heading, and protocol verification.
3. Actual producer/artifact approval enforcement.
4. H3 required hashes and completion/input evidence.
5. H3 K=8 reference/query and identity admission.
6. Complete two-arm legacy receipt enumeration.
7. Working exp_05 M-tier admission.
8. Reportable H1/H1b void outcomes.
9. Complete-arm and approved-execution eligibility for H3 verdicts.
10. Input binding and final revalidation before publication.

Findings **11–12** may be batched as nonblocking follow-ups. **Round 3b may not close yet.**
