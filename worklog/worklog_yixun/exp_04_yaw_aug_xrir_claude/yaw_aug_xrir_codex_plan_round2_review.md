# Codex plan review — yaw_aug_xrir (exp_04), plan v2, round 2

**Reviewer:** OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra) · **Date:** 2026-09-12 11:31:03 -0400 · prompt: review_prompts/plan_round2_prompt.md

---

The core design is now substantially stronger: the seed mixer and counts are correct, fresh-run-only training is appropriate, the five-seed H1/H2 estimand is sound, and the registered H1/H2 quantiles are correct. The plan is still not approvable because several confirmatory and launch contracts cannot yet be executed as written.

## Round-1 closure table

| # | Status | Round-2 assessment |
|---|---|---|
| 1 | **Partially closed** | The counter and literal FLAC mixer are correct. Independent verification gives seed-0 outputs `{1254532375, 2068746652, 2466316700, 368013681}` and first W=512 offsets `{474,395,358,401}` for `t={0,1,9261,111131}`; all 111,132 reportable rank-0 seeds are unique. The requested invalid type/range tests are still not explicitly listed. |
| 2 | **Closed in design** | `--resume` is rejected, mid-epoch saves are prohibited, and interruption requires a fresh epoch-1 run. Operational handling of the abandoned run directory remains defective; see finding 5. |
| 3 | **Closed** | Seeds 42–46, K-specific references plus matched GL seed, within-seed arm pairing, per-query seed averaging, fresh comparator evaluation, and a fail-on-incomplete living table are specified. The separate TOST matrix is incomplete; see finding 1. |
| 4 | **Closed** | H1 correctly uses the strict one-sided \(Q_{0.975}\) upper bound for family 2; H2 uses \(Q_{0.99375}\) for family 8. |
| 5 | **Partially closed** | The written H2 and TOST rules are now complete: C50 primary, EDT supportive, exact angles, family 8/14, and no omnibus verdict. The registered TOST cannot be executed by the run matrix. |
| 6 | **Not closed** | The field list improved, but the manifest is not operationally immutable or bound into evaluator outputs, and the required training source closure is absent. |
| 7 | **Not closed** | Most refusal checks are listed, but the CLI/schema, external checkpoint authentication, one-sided convergence gate, TOST input contract, and confirmatory-constant enforcement remain incomplete. |
| 8 | **Closed in scientific wording** | The plan now correctly says “single intended delta,” enumerates cadence/data-order/reference differences, and disclaims bitwise trajectory pairing and training-seed inference. The mechanical args-diff gate needs repair; see finding 7. |
| 9 | **Partially closed** | A pre-result 1% rule was added, but “alignment gain changes” is numerically undefined and neither possible branch is fully wired into the trainer contract. |
| 10 | **Partially closed** | Most missing tests and ladder rungs were added. The smoke/full argv, evaluation launch integrity, resource thresholds, and launcher test specification remain defective. |
| 11 | **Partially closed** | Counts, updates, runtime, epoch-12 selection, and removal of the seen split are corrected. The matrix and ≈14 GPU-hour budget omit required TOST work. |
| 12 | **Closed** | The existing-test count is no longer hard-coded. |

## New findings

1. **Blocker — §§3–5 — The five-seed TOST family is not covered by the run matrix.**

   **What is wrong:** TOST requires augmented K=8 results at `k={0,32,64,128,256,384,448,480}` for all five seeds. Section 4 supplies five-seed augmented results only at `{0,32,64,448,480}`; `{128,256,384}` appears only in the seed-42 descriptive run. Six of fourteen metric-angle cells therefore lack seeds 43–46.

   **Why it matters:** The family-14 equivalence claim cannot be calculated under the registered five-seed estimand, and the evaluation budget is understated.

   **Minimal change:** Expand all five augmented P runs to the eight-angle union above. Either permit a registered augmented-arm grid superset when H2 uses only four angles, or expand both arms. Make TOST explicitly one-arm and update the run count and GPU-hour budget, likely to roughly 16–17 hours rather than 14.

2. **Blocker — §§3, 5 — The convergence gate is undefined for the new one-sided bounds.**

   **What is wrong:** Exp_03’s gate compares `{lo,hi}` endpoints and divides their maximum movement by the first seed’s interval width. The proposed `one_sided_upper` produces only \(U\). The separately mentioned two-sided adjusted interval cannot silently supply the width because its upper quantile differs: H1 would use \(Q_{0.9875}\), not decision bound \(Q_{0.975}\); H2 would use \(Q_{0.996875}\), not \(Q_{0.99375}\).

   **Why it matters:** A mechanical port could test convergence of the wrong endpoint while leaving the actual decision bound ungated.

   **Minimal change:** Pin bootstrap seeds 0 and 1 and define the companion interval whose upper endpoint is the decision bound:

   - H1: \([Q_{0.025},Q_{0.975}]\)
   - H2: \([Q_{0.00625},Q_{0.99375}]\)
   - TOST: \([Q_{0.05/14},Q_{1-0.05/14}]\)

   Apply exp_03’s maximum-endpoint-movement divided by seed-0 width ≤0.10. Gate every decision-driving query-level bound and suppress canonical artifacts until increased `n_boot` passes.

3. **Blocker — §§4–5, 8 — The evaluation manifest is not operationally immutable or execution-bound.**

   **What is wrong:** No planned change makes `eval_yaw_rotation.py` require and echo an evaluation-manifest digest, and no evaluation launcher exclusively reserves the output directory. The current evaluator accepts an existing directory and replaces result files. A failed rerun could therefore leave stale JSON that is subsequently rebound to a new, post-hoc manifest.

   In addition:

   - The proposed manifest binds `args.json` but omits the training source closure required by round 1.
   - Reusing exp_03’s `closure_record` verbatim is invalid because it hard-codes reviewed commit `62c9107…`, while exp_04 changes `tools/yaw_rotation.py`, a member of that closure.
   - Producer provenance is not bound.

   **Why it matters:** Internally plausible results can be admitted without proof that the declared checkpoint, code, data, and protocol were the ones executed.

   **Minimal change:** Add a reviewed evaluation launcher/evaluator handshake that exclusively creates the run directory and manifest, validates its digest before model loading, echoes the digest into both evaluator outputs, rechecks mutable inputs after execution, and atomically creates a completion sidecar binding manifest, log, and output hashes. Generalize the exp_03 helpers with an explicit reviewed exp_04 commit/data root and bind training, evaluator, manifest-writer, and producer closures. Add stale-directory and changed-manifest/source/checkpoint tests.

4. **Blocker — §§3–5 — `paired_compare.py` is not yet independently fail-closed.**

   **What is wrong:**

   - `--arm-a <label>=<runs…>` is not an exact executable CLI grammar.
   - “Expected” role/checkpoint hashes come from the same manifests being authenticated; an internally consistent wrong checkpoint could pass.
   - `--angles`, `--family`, and `--margin` remain caller-controlled without an explicit requirement to reject confirmatory reparameterization.
   - The two-arm contract is wrong for augmented-arm-only TOST.
   - The finite-mask wording does not say whether all five seed values must be finite, and “across arms and angles” conflicts with §3’s per-cell mask.
   - “Hash every output” lacks a non-self-referential sidecar scheme.

   **Why it matters:** Wrong but mutually consistent runs, post-hoc thresholds, or inconsistent valid-query sets could receive confirmatory verdicts.

   **Minimal change:** Use an immutable external analysis specification, or hard-coded confirmatory profiles, that pins arm/backbone/checkpoint hashes, seed-to-reference-manifest hashes, GL coupling, K, grids, margins, family sizes, tails, bootstrap seeds and count. Deviations must be exploratory and verdict-free. Define:

   - H1/H2 as two-arm modes.
   - TOST as a one-arm mode.
   - H2’s per-cell mask as finite for every seed, both arms, and `{0,k}`.
   - TOST’s mask as finite for every augmented-arm seed at `{0,k}`.
   - Ordinary five-value averaging only—never `nanmean`.

5. **Blocker — §§2, 4, 6–7 — The smoke/full launch contract cannot pass as written.**

   **What is wrong:** The trainer currently defaults `--save-every` to 500; v2 says yaw augmentation rejects any value above zero, but the smoke command omits `--save-every 0`. The full launch also does not explicitly pin it. Likewise, `--epoch-ckpt-every` defaults to 5, while the plan requires every-epoch snapshots and confirmatory epoch 9/12 files; `--epoch-ckpt-every 1` is absent. The current native filenames would be `epoch_009.pth` and `epoch_012.pth`.

   An interrupted attempt also leaves a nonempty canonical checkpoint directory, making the mandatory fresh restart incompatible with the empty-directory gate. A late failure plus restart may exceed the unexplained 43-hour ceiling.

   **Why it matters:** The mandatory validation smoke refuses before training, required checkpoints may not exist, and restart risks either overwriting evidence or violating the gate/budget.

   **Minimal change:** Add the launcher and guard tests to §5’s planned files. Pin exact smoke/full argv, including `--save-every 0`, full `--epoch-ckpt-every 1`, accumulation, and exact checkpoint names. Use immutable attempt directories; rename/preserve the whole failed attempt as `_ABORTED_<timestamp>` and promote only a completed attempt. State whether 43 hours is per-attempt or cumulative and require renewed authorization before exceeding a cumulative ceiling.

6. **Should-fix — §§2, 5–6 — The alignment-audit branch remains numerically undefined and unwired.**

   **What is wrong:** “Alignment gain changes” has no tolerance. The gain is a continuous float32 expression, `||ref||/(||src||+1e-7)`; exact inequality can count harmless ulp-scale changes as widespread flips. The plan also specifies no trainer flag/artifact selecting rotated versus fixed alignment and no integration test for the fixed branch.

   **Why it matters:** Negligible numerical noise could silently select a materially different training path.

   **Minimal change:** Preferably pin unrotated alignment unconditionally and keep the audit descriptive. Otherwise define the exact device/op sequence, absolute or relative gain tolerance, delay/gain denominator, deterministic 50-batch cohort and epoch-1 offset counters; record a hashed alignment-mode decision before launch and test both branches, including nonzero-yaw byte identity of target/reference audio.

7. **Should-fix — §§2, 5–7 — Several mechanical preflight gates remain non-executable.**

   **What is wrong:**

   - Seed-domain tests do not explicitly cover invalid `t`, rank, epoch, batch, and type boundaries.
   - The type-strict args diff excludes only save/yaw/cadence fields, but the proposed record also adds `no_save` and `PYTHONHASHSEED`.
   - The treatment `args.json` does not exist until after launch, while the diff is a pre-launch rung.
   - `PYTHONHASHSEED` is recorded but no value is pinned before Python starts.
   - “Free GPU and disk” has no measurable thresholds.
   - Historical epoch 1 was 2.315 hours; +5% is 2.431 hours, not 2.6 hours.

   **Why it matters:** Gates will either fail mechanically or permit configuration/resource drift.

   **Minimal change:** Add explicit counter boundary tests, including `t=-1`, `t=2^20`, rank `-1/4096`, epoch 0, and batch `-1/9261`. Generate a pre-spawn `effective_args` record with a complete normalization allowlist and require runtime `args.json` to match it. Pin `PYTHONHASHSEED=0` in the launcher environment. Set numeric GPU ownership/free-VRAM and disk-floor rules. Use 2.431 hours for the 5% epoch gate, or label 2.6 hours as a distinct operational ceiling with a separate rationale.

## Verdict

**Request changes.**

Before the user is asked to approve, v3 must:

1. Supply five-seed data for every TOST cell and update the run matrix/budget.
2. Define the one-sided-bound convergence calculation exactly.
3. Make evaluation manifests pre-run, immutable, evaluator-bound, and backed by current training/evaluator/producer closures.
4. Replace the producer’s self-declared, configurable contract with exact externally pinned confirmatory profiles, CLI schemas, and finite masks.
5. Provide executable smoke/full argv, checkpoint cadence/names, immutable attempt handling, and a cumulative restart-budget rule.
6. Fully define and wire the alignment policy.
7. Repair the seed-boundary, effective-args, `PYTHONHASHSEED`, resource, and timing gates.

No files were modified.
codex exit 0 at 11:44:43
