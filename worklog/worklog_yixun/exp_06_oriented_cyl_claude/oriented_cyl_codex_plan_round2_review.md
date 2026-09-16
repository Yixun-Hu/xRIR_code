**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-14.

## Verdict: request changes

V2 substantially improves the design. The revised heading rule accepts all four rooms, the epoch-12 checkpoint now has a valid production path, and the proposed bootstrap can reproduce the historical intervals exactly.

However, admission, training-completeness, and CPU-probe contracts still need correction. **The plan is not yet ready to hand to the Coder for round 1.** The exact blocking findings are **R2.1, R2.2, and R2.3** below.

This reviews [plan v2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/plan_oriented_cyl.md), not an implementation. No files were modified, GPU work started, or processes signaled.

## Round-1 findings: resolution status

| Original finding | Assessment of v2 |
|---|---|
| **1. Heading estimator** | **Resolved operationally.** All four rooms select −90°, including every leave-one-out refit. Singleton-candidate semantics need clarification: R2.7. |
| **2. G1 cohort/backend mismatch** | **Resolved in design.** Legacy reproduction and full-cohort gating are separated; failed bracketing gives “inconclusive.” The CPU implementation issue belongs to R2.3. |
| **3. Wrapper interfaces** | **Partly resolved.** Explicit parsers, serializers, factories, and dataset subclassing are feasible without namespace mutation. CPU alignment and launcher adapters remain unspecified: R2.3. |
| **4. Provenance/admission** | **Partly resolved.** H3’s protocol checklist is substantially complete, but historical admission and completion/approval lifecycle remain inconsistent: R2.1. |
| **5. Epoch-12 checkpoint** | **Resolved.** Cadence 1 produces `epoch_012.pth`; final `last.pth` uses epoch 12, batch index 0. Coverage validation remains a separate recipe issue. |
| **6. H2 interpretation/invalidity** | **Core issue resolved.** H2 is descriptive. Extend validity protection to the newly decision-bearing H1b: R2.4. |
| **7. Estimands/bootstrap** | **Core issue resolved and numerically verified.** H3 matches the imported statistic; alpha-aware HAA intervals are implementable with exact legacy reproduction. Convergence needs strengthening: R2.5. |
| **8. Controls/mechanism claims** | **Resolved at the intended-intervention level.** F supplies the missing fine-tuned cylindrical heading-frame control; G1 is explicitly budget triage. C−F remains conditional on unpaired pretraining realizations. |
| **9. Phase policy** | **Resolved.** Historical unseeded primary evaluation and fresh shared-phase sensitivity are clearly separated. |
| **10. Symmetry tests** | **Partly resolved.** Tolerances and independent camera checks are appropriate, but Python rounding breaks the promised boundary case: R2.6. |
| **11. Recipe/validation ladder** | **Partly resolved.** Full-batch fit probing and operational disclosures were added; field coverage and smoke bounds remain incomplete: R2.2 and R2.8. |
| **12. Arithmetic/parameter wording** | **Resolved.** The approximate margin wording and **526,336** parameter increment are correct. New schedule arithmetic is discussed in R2.9. |

## Numbered findings

### R2.1. **Blocker — the admission and completion lifecycle cannot be applied literally**

**Plan:** §§5, 6.4, 8 rounds 2–3.

**What is wrong:** The summarizer loads historical A/B while requiring completion presence and new provenance fields. I checked **19 relevant directories per arm; neither arm has any `completion.json`**. Historical evaluation files also lack the new heading/closure records.

The new-run lifecycle is unclear too:

- §5 puts training completion inside the training entry point, immediately after its loop.
- §6.4 requires successful child exit and a closed log—facts that require an external finalizer.
- Approvals include future checkpoint/heading hashes without clearly separating pre-run code approvals from post-run artifact approvals.
- The inherited HAA queue invokes four room evaluations into the same `eval/` directory. Exclusive per-invocation directories and one shared `args.json`/completion cannot simply be added to that layout.

**Why it matters:** Literal enforcement rejects the historical controls or makes new completion impossible; informal exceptions weaken the promised provenance.

**Minimal fix:** Specify:

1. A **legacy A/B admission branch**, using exp_02 completeness/pairing checks and an exp_06-owned hash receipt over retained artifacts, explicitly labelled reconstructed.
2. Code/input approval before execution; checkpoint/heading hash approval after artifact creation.
3. An external finalizer owning child-exit, closed-log, input-revalidation, and completion checks.
4. The HAA completion unit: separate room directories, or one orchestrator-owned completion after all room children finish.
5. Explicit approved exp_04/exp_05 evaluator/writer identities for reused M evaluations.

Keep historical artifacts unchanged.

### R2.2. **Blocker — recipe normalization can certify truncated training as complete**

**Plan:** §§5, 8 round 2.

**What is wrong:** Both historical `args.json` files and the current trainer contain:

```text
max_train_batches
max_test_batches
test_subset
```

None appears in the proposed recipe or operational classifications. With `max_train_batches=3`, the trainer can still produce twelve history entries, `last.pth` at epoch 12/batch 0, and an exactly matching `epoch_012.pth`.

Other fields need explicit treatment: `yaw_aug_seed`, `yaw_aug_width`, `tier`, and `param_counts`. The parser resolves an unspecified yaw seed to **0**, not `None`. Excluding `resume` without a production constraint also permits a different training history.

**Why it matters:** The completion contract proves checkpoint consistency, not the registered training budget.

**Minimal fix:** Require production limits to be zero, `resume is None`, and `no_save is False`; validate epochs **1–12**, dataset/loader coverage, and the expected **9,261 micro-batches per epoch**. Classify every written field, normalize historical missing fields to checked defaults, and validate derived dimensions/counts separately.

### R2.3. **Blocker — CPU legacy reproduction still has no compatible numerical path**

**Plan:** §§6.1, 8; associated launcher adapters in §§6.3/8.

**What is wrong:** The required CPU legacy forward reaches:

```text
side_probe.fwd
→ model.shift_and_align
→ model/xRIR.py::apply_delay
→ torch.zeros_like(signal).cuda()
```

The original diagnostic works only by globally replacing Torch’s `.cuda()` and `.to()` behavior. Importing it also immediately runs the diagnostic. Neither is compatible with v2’s isolation requirements.

Two launcher details also require explicit adapters:

- `exp04_eval_launch.build_fields` rejects `heading` in `--bind-input`.
- Even with `module=exp06_eval`, it records `tools.exp04_eval_launch` as the writer closure, not the new launcher.

**Why it matters:** Correct cohort selection does not make the CPU reproduction executable, and direct launcher delegation cannot produce the promised bindings.

**Minimal fix:** Register an exp_06-owned device-preserving diagnostic alignment helper or local probe subclass, with delay/gain/dtype parity tests. Do not import the executable diagnostic script. Specify that the launcher separates validated new bindings before delegation and adds/revalidates its actual exp_06 writer closure afterward.

### R2.4. **Should-fix — H1b lacks the validity protection required by the overall verdict**

**Plan:** §7.

**What is wrong:** The ≥99% cohort and excess-invalidity rules explicitly void **H1 only**, while H1b is also required for the positive conclusion. C−A could remain complete while C−F is evaluated on a small favorable subset because F produces invalid metrics.

The text also says the positive conclusion rests on “H1 alone,” then requires H1b in the overall rule.

**Why it matters:** Survivor selection in F could support the channel-effect claim despite inadequate coverage.

**Minimal fix:** Apply minimum-cohort and validity requirements to both H1 and H1b; define counts as distinct queries or seed-query rows. Make the overall wording consistent and scope the conclusion to the **hallway C50 deficit**, with other cells described through H2.

### R2.5. **Should-fix — half-width stability does not establish endpoint stability**

**Plan:** §7; §8 bootstrap row.

**What is wrong:** Intervals `[-0.1, 0.1]` and `[0.1, 0.3]` have identical half-widths but produce different H1 decisions at margin 0.23.

**Why it matters:** The proposed convergence check can accept an unstable decision bound.

**Minimal fix:** Use `paired_compare.convergence`: maximum endpoint movement divided by the seed-0 interval width, bootstrap seeds **0/1**, tolerance **0.10**, explicit zero-width handling. Specify increasing `n_boot` or withholding the verdict on failure.

Also freeze the legacy RNG order and NumPy linear-percentile convention documented below.

### R2.6. **Should-fix — rounding ties violate the stated joint-rotation test**

**Plan:** §§2.1–2.2, 8 round 1.

**What is wrong:** Python’s ties-to-even `round` is not translation-equivariant at half-column headings.

Verified at width 512:

```text
φ = 0.3515625°:                  k = 0
joint scene rotation +1 column:
φ′ = 1.0546875°:                k′ = 510
net scene roll = 1 + 510 = 511, not 0
```

**Why it matters:** This is a one-column discrepancy, not float32 rounding noise. It contradicts the promised arbitrary-heading boundary test.

**Minimal fix:** Define a consistent translation-equivariant tie convention, or canonicalize heading to an integer column before joint rotations. Test exact ties explicitly. The four actual axis headings are unaffected.

### R2.7. **Should-fix — singleton heading candidates need explicit semantics**

**Plan:** §4.

**What is wrong:** Classroom and complex each have only one evaluable candidate. The requirement to exceed “every other evaluable candidate” passes vacuously; there is no measured runner-up margin.

The sample-window conversion is also unstated operationally: the literal half-open intervals contain **111/1,103 samples**, whereas `int(w*sr)` gives **110/1,102**.

**Why it matters:** A Coder could implement an empty maximum, produce NaN, or silently refuse rooms that the literal plan accepts.

**Minimal fix:** Explicitly allow singleton candidates with `competitor_count=0` and margin `null/N/A`, documenting the limitation; otherwise register refusal and resolve classroom/complex separately. Freeze sample rounding and test both cases. These are inferred acoustic axes, not independently verified physical orientations.

### R2.8. **Should-fix — the co-tenant smoke commands are not concretely bounded**

**Plan:** §§8–9.

**What is wrong:** The displayed trainer smoke flags inherit **200 epochs, batch 64, and an unlimited test loop**. HAA’s two-epoch stage-1 smoke inherits a full **36-target batch**.

**Why it matters:** Neither establishes the promised ≤3 GB footprint or short co-tenant execution.

**Minimal fix:** Provide exact commands with explicit small micro-batches, one training epoch, finite train/test limits, and a small HAA batch. Verify the memory bound before co-tenancy.

### R2.9. **Nit — separate expected runtime from conservative reservation**

**Plan:** §11.

The listed components including S1 total **45⅓–55⅓ GPU-hours**; 46–56 is reasonable rounding upward. Pretraining is **31.16 hours**, placing a Sep 16 13:30 start at approximately Sep 17 20:40, consistent with the schedule.

However, completed canonical control evaluations take **8.02–8.13 wall minutes per seed**, totaling **40.25 minutes for five seeds**, versus five hours budgeted. The proposed ten-hour A/B fallback is similarly conservative.

**Minimal fix:** Label those figures as reservation ceilings or update expected runtime from the matching evaluations. The ≈50-minute cylindrical HAA job estimate is supported.

## Actual heading decisions

Only the twelve training RIR rows per room were accessed, using memory mapping and float64 energy accumulation. Results below use the literal half-open windows.

Sector counts are ordered **0°, +90°, 180°, −90°**.

| Room | Sector counts | Winning contrast: 5/50 ms, dB | Runner-up margin: 5/50 ms, dB | Decision |
|---|---|---:|---:|---|
| Classroom | 2, 1, 2, 7 | 12.929 / 9.037 | N/A: sole candidate | −90°, `k=128` |
| Dampened | 4, 0, 3, 5 | 11.381 / 8.669 | 17.415 / 13.144 | −90°, `k=128` |
| Hallway | 0, 5, 0, 7 | 18.412 / 11.291 | 36.824 / 22.583 | −90°, `k=128` |
| Complex | 2, 1, 1, 8 | 11.963 / 10.806 | N/A: sole candidate | −90°, `k=128` |

Other evaluable contrasts:

- Dampened 0°: **−7.356/−5.706 dB**; 180°: **−6.035/−4.475 dB**.
- Hallway +90°: **−18.412/−11.291 dB**.

**No room is refused under the written rule.** Every room retains −90° in **12/12 leave-one-out refits for both windows**, including when thresholds are reapplied within each refit. The smallest leave-one-out winning contrast is **7.882 dB**. Floor-length windows produce the same decisions.

## Bootstrap and helper verification

### Bootstrap

An in-memory alpha-generalization of `summarize_haa.paired` reproduced **all eleven canonical cells exactly**, including both interval schemes: maximum absolute difference **0.0**.

Exact reproduction requires:

- `np.random.default_rng(0)` and float64 inputs.
- Historical row order and `np.unique` label ordering.
- Query-bootstrap draws first, then two-way draws using the **same advancing generator**.
- Query/seed multiplicity products and weighted denominators.
- NumPy’s **linear percentile** convention, not `paired_compare`’s empirical order-statistic quantiles.

H2’s tails are implementable: **0.0022727273 / 0.9977272727**, approximately **114 draws per tail** at 50,000 resamples.

Historical hallway C50 adjusted two-way intervals were:

| Bootstrap seed | Interval |
|---|---|
| 0 | `[0.6772420705, 1.1938191609]` |
| 1 | `[0.6785358872, 1.1811890479]` |

Endpoint movement/width was **2.445%**, passing the stronger convergence rule.

H3’s revised estimand matches `five_seed_mean`, `cell_mask`, and `rho_bootstrap(C,B)`. `one_sided_upper(samples, .025)` supplies its required bound; `h1_verdict` applies the strict `< .03` criterion.

### Composed helpers

All named numerical helpers exist:

| Helpers | Compatibility |
|---|---|
| Trainer `seed_everything`, `seed_worker`, `train_epoch`, `test_epoch`, `save_checkpoint` | Compatible with an explicitly constructed model/namespace; training/loss paths require CUDA. |
| `finetune_haa.compute_loss`, `evaluate`, `load_model_state` | Available, including imported aliases; suitable for the composed GPU loop. |
| `run_exp04(args, model_factory=None, metadata=None, manifest_validator=None)` | Compatible with the proposed factory and validator. |
| `execute_run(args, command, fields_factory, repo)` | Reusable for simulated evaluation’s existing two-output contract. |
| `build_fields(args, command, repo, module=None)` | Requires the adapters in R2.3. |
| `child_environment(repo, data_root, gpu='1')` | Exists. |
| `eval_unseen.Evaluator`, `griffin_lim` | Available; per-query phase seeding can wrap the CPU inversion call. |
| `HeadingFrameDataset(HAADataset)` | Feasible while preserving `.items`, `.data`, tuple order, and reference draws. |
| `source_closure(entry_module, repo)` | Requires a **dotted module name and explicit repo argument**; slash-form one-argument calls in §5 are pseudocode, not valid calls. |

Dynamic imports of trainer/HAA/evaluator modules encountered read-only Numba/Matplotlib cache requirements. These are review-environment limitations, not demonstrated production import defects.

## Verification record

**Files read, including parallel read-only subreviews:**

- `CLAUDE.md`, `worklog/experiment_SOP.md`; no announcement directives.
- Exp_06 v2 plan, round-1 review, query, and notebook.
- Exp_02 plan, results, analysis, complete notebook including September 14 entries, and all six diagnostic scripts.
- Exp_03 analysis; exp_04/exp_05 plans and amendments.
- Both exp_01 `args.json` files.
- `model/{cylindrical_vit,xRIR_cyl,xRIR}.py`, `tools/yaw_rotation.py`, and `treble_multi_room_dataset/treble_xRIR_dataset.py`.
- `sim_to_real/{haa_dataset,finetune_haa,eval_haa,summarize_haa}.py`, its pipeline shell script, and `train_xRIR_backbone.py`.
- `tools/{exp04_eval,exp04_eval_launch,exp05_eval,exp05_profiles,exp05_gates,param_curve,paired_compare,provenance}.py`, `tests/test_provenance.py`, and relevant referenced helpers.
- Four HAA cache metadata/coordinate files and training RIR rows.
- `ckpt/sim2real/{summary.txt,stats.json}`, historical A/B per-sample records, and completed timing records.

**Commands:** `rg`, `cat`, `nl`, `sed`, `wc`, and inline CPU checks. Every Python invocation used:

```text
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=/home/yixunhu/codespace/xRIR_code
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=4
/home/yixunhu/miniconda3/envs/xRIR/bin/python
```

Additional verified outputs:

- Canonical summary hash matches; historical completeness: **`(True, [])`**.
- All required historical A/B metrics finite.
- Hallway test coordinates: **185 +y / 238 −y**; legacy cohort IDs and deterministic reference draws reconstructed.
- −y maps to **`[1, −6.12e−17, 0]`** at `k=128`.
- Parent gauge versus independent camera conversion: maximum error **`5.96e−8`**.
- All twelve enumerated exp_03 closure files match reviewed bytes/history; digest **`5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`**.

Neural G1 anchors were not rerun, and write-requiring pytest tests were not executed.

**Final verdict: request changes. Blocking set: R2.1, R2.2, R2.3.**
