# Codex plan review — yaw_aug_xrir (exp_04), plan v1

**Reviewer:** OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra) · **Date:** 2026-09-12 11:12:00 -0400 · prompt: review_prompts/plan_prompt.md

---

The core geometry is sound: per-sample column roll plus active \(R_z(+2\pi d/W)\) on `depth_coord`, `src_loc`, and `ref_locs` is correct for xRIR’s camera-frame coordinates; target and raw reference RIRs should remain untouched. The plan is not yet approvable because its reproducibility and confirmatory-analysis contracts are incomplete.

## Findings

1. **Blocker — §§2, 5 — The seed contract is not yet a faithful exp_17 port.**
   **Wrong/missing:** exp_17 packs `(global_step, global_rank)` as `(step << 12) | rank`; the plan instead proposes an unspecified packing of `(epoch, batch_idx)` while calling it the same construction. Gradient accumulation makes the meaning of “step” especially important. See the [reference implementation](/home/yixunhu/codespace/exp-17-yawaug-a6000/src/training/diffusion.py:82).
   **Why it matters:** the claimed reference parity, injectivity, and deterministic yaw sequence cannot be audited from the current specification.
   **Minimal change:** explicitly define the xRIR adaptation—for example, `t=(epoch−1)*9261+batch_idx`, followed by the literal exp_17 `_yaw_aug_step_seed(seed,t,0)` mixer. Add hard-coded exp_17-derived seed/offset golden vectors, type/range boundary tests, and a collision check over the complete armed domain.

2. **Blocker — §§2, 7 — “Resume reproduces an uninterrupted run” is false for the current trainer.**
   **Wrong/missing:** checkpoints omit gradients, sampler state, global RNG state, and worker NumPy state. Mid-epoch checkpoints are saved after backward, often between accumulation boundaries, and resume restarts the entire epoch rather than skipping completed batches. See [checkpoint and resume handling](/home/yixunhu/codespace/xRIR_code/train_xRIR_backbone.py:93).
   **Why it matters:** resuming duplicates optimization, loses a pending gradient, and changes shuffled samples, randomly selected references, and their assigned yaws. Counter-based augmentation alone cannot repair this.
   **Minimal change:** make the reportable arm fresh-run-only: disable mid-epoch saving, reject `--resume`, and restart from epoch 1 after interruption. Alternatively, specify and test a substantially larger exact-resume implementation covering accumulation, sampler, worker-reference draws, and RNG states.

3. **Blocker — §§4, 8, 9 — The evaluation violates adopted directive 04.**
   **Wrong/missing:** only reference-manifest seed 0 and Griffin–Lim seed 0 are planned, with no generator-backed living results table.
   **Why it matters:** reference selection and phase reconstruction are evaluation nuisances; a seed-0 row or pass/fail screen is explicitly disallowed. Reusing the seed-0 exp_03 control realization also reuses the data that selected H2’s four angles.
   **Minimal change:** preregister five fresh evaluation seeds—preferably exp_17’s `{42,43,44,45,46}`—and define how each seed selects both the K-specific reference manifest and Griffin–Lim phase seed. Pair control and treatment within every seed, rerun both using the same final evaluator closure, report full-split mean ± sample SD, and aggregate paired per-query errors across the five seeds before bootstrap inference. Add a raw-JSON-driven living-table generator that refuses incomplete or protocol-mismatched five-seed inputs; single-seed runs may be technical diagnostics only.

4. **Blocker — §§3, 5 — The one-sided bounds are not executable through the reused statistics API.**
   **Wrong/missing:** [the existing helper](/home/yixunhu/codespace/xRIR_code/tools/paired_stats.py:210) produces equal-tail two-sided intervals. Passing `alpha=.05/2` would give H1 an upper endpoint at \(Q_{0.9875}\), not the intended Bonferroni one-sided \(Q_{0.975}\). H2 does not resolve whether it uses a directional one-sided bound or exp_03’s conservative two-sided interval.
   **Why it matters:** implementation could silently change the registered test and its power.
   **Minimal change:** state H1 as \(H_{0j}:\rho_j\ge .03\), \(H_{Aj}:\rho_j<.03\), with \(U_j=Q_{1-.05/2}=Q_{.975}\), passing only when \(U_j<.03\). For H2, choose explicitly between directional \(Q_{1-.05/8}=Q_{.99375}\) and the upper endpoint \(Q_{.996875}\) of an exp_03-style two-sided adjusted interval. Add a dedicated one-sided-bound helper with exact-quantile and strict-boundary tests.

5. **Blocker — §3 — H2 and TOST do not yet form complete preregistered decisions.**
   **Wrong/missing:** the H2 family contains eight EDT/C50 cells, but the aggregate verdict depends only on four C50 cells. “Patch-aligned angles” for TOST could mean the four H2 angles or all seven nonzero acoustic-grid multiples of 32; its family and per-test alpha are absent.
   **Why it matters:** the aggregate claim and multiplicity correction could be chosen after results are seen.
   **Minimal change:** either make H2 explicitly C50-primary with EDT supportive, or require all eight cells for the aggregate verdict. State that H1 and H2 are separate family-local claims with no omnibus success rule. List the exact TOST angles, condition P, K=8, separate family size and alpha; if all seven angles and two metrics are used, the family is 14 and each TOST uses `.05/14`. Label every other angle descriptive-only, and call failed TOST “equivalence not established.”

6. **Blocker — §§4, 8 — Directive 05 is not satisfied by the reference-selection manifests.**
   **Wrong/missing:** the reference manifests bind query/reference choices, not the full evaluation protocol; existing K=1 comparator outputs also lack equivalent provenance sidecars.
   **Why it matters:** K, checkpoint, code closure, GL seed, grid, TF32, or truncation could differ while still appearing paired.
   **Minimal change:** define an immutable evaluation manifest written before every run, binding checkpoint path/hash, training and evaluator closure, dataset identity, unseen split and 6,337 count, K, reference-manifest path/hash/seed, Griffin–Lim seed and parameters, batch/canonical-padding policy, TF32, P/E condition and grids, no TTA, command/environment, and output hashes. Validate outputs against it. Preserve the already-hashed reference manifests; bind or rerun comparator artifacts.

7. **Blocker — §§5, 8 — `paired_compare.py` is not specified as a fail-closed confirmatory producer.**
   **Wrong/missing:** it is described first as a two-run k=0 tool and later as the H2 sweep producer; its only stated rejection is mismatched query lists.
   **Why it matters:** stale, truncated, wrong-K, wrong-grid, wrong-seed, or unauthenticated runs could receive confirmatory verdicts.
   **Minimal change:** specify its exact schema and CLI. Require expected role/checkpoint hashes, five-seed completeness, identical query/index ordering, 6,337 queries/17 rooms, K and protocol agreement, exact grids and P condition, `max_samples=0`, metric reconciliation, and accepted provenance. H2 must use one common finite mask across both arms at k=0 and k, report invalidity categories, share resamples, rerun bounds with exp_03’s second-bootstrap-seed convergence gate, and bind every input/output digest. Add malformed-input, mask, tail, family, and convergence tests.

8. **Should-fix — §§2, 4, 9 — The “single delta” wording overstates historical pairing.**
   **Wrong/missing:** the control used `save_every=500` and `epoch_ckpt_every=5`, not per-epoch checkpoints; see its [actual arguments](/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_simple_8_shot/args.json:17). More importantly, training paths are built through an unordered `set`, while references use worker-local `np.random.choice`; exp_01 did not record `PYTHONHASHSEED`. See [dataset ordering](/home/yixunhu/codespace/xRIR_code/treble_multi_room_dataset/treble_xRIR_dataset.py:113).
   **Why it matters:** the historical control and treatment cannot share the exact sample/reference trajectory, and there is only one training seed.
   **Minimal change:** call this a “single intended algorithmic/config delta with disabled-path evidence, not bitwise trajectory pairing.” Add a type-strict args diff excluding only output and declared yaw/artifact keys, enumerate checkpoint-cadence differences, and list data-order/reference realization plus single-training-seed noise as causal limitations. An exact causal single-delta claim would require a contemporaneous unaugmented retrain.

9. **Should-fix — §§2, 6, 9 — `shift_and_align` is algebraically but not numerically invariant.**
   **Wrong/missing:** float32 rotation, norm evaluation, and rounded delays can change aligned references; exp_03 observed several delay flips. A norm tolerance of `1e-6` does not test this discontinuity.
   **Why it matters:** augmentation can introduce a second, unintended audio-alignment perturbation even though raw RIR tensors are unchanged.
   **Minimal change:** add a real-batch audit of integer delays and alignment gains under assigned yaws. Either pin alignment to the unrotated geometry or preregister thresholds and disclose the quantified finite-precision effect as part of the treatment.

10. **Should-fix — §§5–7 — Tests, parity, smoke, and launch gates are too narrow.**
    **Wrong/missing:** absent checks include exact batch shapes/dtypes/common dimensions, width matching `depth.shape[-1]`, empty/nonmutation cases, strict flag validation, training-only dispatch, complete untouched RIR identity, and disabled-path preservation of all losses, gradients, buffers, and Python/NumPy/Torch RNG. `--save-every 0` still writes `best.pth`, `last.pth`, and history, while direct `nohup` does not enforce the banner or clean-output gate.
    **Why it matters:** the most load-bearing invariants could fail without the proposed tests noticing.
    **Minimal change:** add those per-function and integration tests, prove `test_epoch` never augments, add a true no-save smoke mode, make >5% overhead block launch, perform the SOP’s final integrative review, and use a preflight/launcher that validates exact args, source/data identity, empty output directory, GPU/storage state, and the flushed banner before step one.

11. **Should-fix — §§3, 4, 7–9 — The run matrix and budget are internally inconsistent.**
    **Wrong/missing:** the actual training split is 296,334 samples, yielding 9,261 batches and 4,631 optimizer steps per epoch: 41,679 through epoch 9 and 55,572 through epoch 12. The historical control took 28.215 hours, not 43 hours. The plan also leaves epoch 12, epoch 9, `best.pth`, and the unsupported seen-split evaluation insufficiently separated.
    **Why it matters:** checkpoint selection can become post-hoc, and the stated evaluation cost omits five seeds and fresh-control runs.
    **Minimal change:** pin epoch 12 versus epoch 12 as the only confirmatory/table comparison; make epoch 9 diagnostic-only and prohibit `best.pth` or test-selected epochs from verdicts. Drop the seen diagnostic or specify its evaluator and manifest. Correct the counts, filenames/cadence, and evaluation matrix. Use approximately 28.2 hours times measured augmentation overhead as the central training estimate; retain 43 hours only as a reservation ceiling and recalculate evaluation GPU-hours.

12. **Nit — §6 — The existing-test count is stale.**
    **Wrong/missing:** the plan says 237 existing tests; the repository currently contains 242.
    **Minimal change:** update the count or avoid hard-coding it.

## Verdict

**Request changes.**

Before the user is asked to approve, the revised plan must:

1. Pin the exact xRIR counter-to-exp_17 seed mapping and golden parity tests.
2. Remove trajectory-exact resume claims and make reportable training fresh-only, unless true exact resume is fully designed.
3. Define the five-seed paired evaluation block, seed aggregation, living-table generator, and fresh same-closure control evaluations.
4. Fully specify H1/H2 one-sided quantiles, H2 aggregation, TOST angles/family, and descriptive-only outputs.
5. Define immutable per-run evaluation manifests and a fail-closed confirmatory producer.
6. Qualify the historical-control single-delta claim and enumerate all operational differences and training-realization risks.
7. Add the numerical alignment audit and the missing schema, disabled-path, evaluation-dispatch, parity, smoke, and launch tests.
8. Pin the confirmatory checkpoint/run matrix and correct sample, update, checkpoint, runtime, and multi-seed evaluation budgets.

No files were modified.
codex exit 0 at 11:26:59
