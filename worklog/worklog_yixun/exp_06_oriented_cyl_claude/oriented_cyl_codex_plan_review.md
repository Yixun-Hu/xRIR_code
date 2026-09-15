**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-14.

## Verdict: request changes

The heading-frame construction is mathematically sound, and the proposed channel is a reasonable intervention. However, the default heading estimator already refuses the hallway, G1’s anchors do not match its proposed population, and several implementation and statistical contracts need correction before approval.

This reviews **plan v1**, not absent exp_06 code. No files were modified, GPU work started, or processes signaled.

## Numbered findings

### 1. **Blocker — the heading estimator fails on the primary room and its consistency check is sampling-biased**
**Plan:** §§4, 8 round 1, 9.

Applying the specified estimator to only the 12 training RIRs per room gives:

| Room | Fitted heading | Weighted mean | Disagreement | Contrast | Result |
|---|---:|---:|---:|---:|---|
| Classroom | −82.0° | −93.28° | 11.28° | 12.57 dB | Pass |
| Dampened | −96.0° | −99.26° | 3.26° | 18.04 dB | Pass |
| **Hallway** | **−44.0°** | **−90.43°** | **46.43°** | **17.13 dB** | **Refuse** |
| Complex | −93.0° | −95.87° | 2.87° | 21.00 dB | Pass |

Hallway training directions concentrate near ±y. Leave-one-out fits range from approximately −147° to −23°. Its cache metadata contains no physical heading.

Moreover, the energy-weighted mean conflates directivity with microphone sampling density. On the actual hallway angles, noiseless synthetic data satisfying the exact proposed model with true heading 0° produce a weighted mean of −81.1°. Thus the consistency rule can reject a perfect fit.

**Why it matters:** The default experiment cannot proceed as written. A 50 ms energy fit also estimates an acoustic preferred direction influenced by reflections; it does not automatically identify physical loudspeaker orientation.

**Minimal fix:** Resolve the hallway heading before approval through independently supported orientation information or a revised, pre-registered estimator with sampling-aware uncertainty/stability checks. Preserve refusal for ambiguity. Add real training-only readback, nonuniform-angle synthetic cases, and leave-one-out stability tests.

### 2. **Blocker — G1 requires reproduction of anchors measured on a different cohort and backend**
**Plan:** §§6.1, 8 round 3.

The historical `0.975/0.85` and `0.435/0.24` anchors came from **24 selected +y queries**, with queries sorted by y and sampled at evenly spaced ranks. The forward probe used CPU batches of eight. The plan instead requires agreement within ±0.03 over **all 185 +y queries on GPU**. See [side_probe.py](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_02_sim2real_transfer_claude/diagnostics_2026-09-14_hallway_side/side_probe.py:24).

**Why it matters:** Correct implementation can fail this gate simply because the population changed.

**Minimal fix:** Separate exact legacy-cohort reproduction from the full-population diagnostic. Freeze the legacy query IDs and numerical settings; report fresh baseline values for the 185-query cohort. Preserve original room-frame ±y labels through rotation—transformed `y < 0` no longer identifies the original corridor side.

### 3. **Blocker — the namespace-injection approach needs explicit interfaces and isolation**
**Plan:** §8 round 2.

The proposed wrappers omit necessary integration details:

- The trainer looks up `BACKBONES` and `build_xrir`; adding globals under the new names alone does not redirect it.
- The exp_04 parser accepts only the existing backbones.
- Its launcher accepts only `exp04`/`exp05` entry points.
- HAA parsers do not accept `--heading-json-dir`.
- HAA evaluation accesses `ds.items`, writes no `args.json`, and constructs per-sample metadata through a whitelist excluding heading. Registry/dataset injection alone cannot produce the promised records. See [eval_haa.py](/home/yixunhu/codespace/xRIR_code/sim_to_real/eval_haa.py:99).

Unscoped mutation also risks making the supposedly untouched baseline in a parity test use the injected factory, yielding a vacuous pass.

**Minimal fix:** Specify experiment-owned parsers, serializers, and thin entry points that compose existing numerical helpers. Use explicit factory/dataset arguments where available. If retaining trainer namespace substitution, perform it only inside a dedicated child’s execution, patch the actual symbols, avoid import-time mutation, and restore bindings in `finally`.

Add tests for actual CLI routing, `.items`/reference-order preservation, every train/validation/test room, metadata emission, and unchanged original bindings after import, success, and failure. No pinned source file needs editing.

### 4. **Blocker — recording a registry digest does not establish the promised provenance**
**Plan:** §§5, 6.3, 8 rounds 2–3.

`run_exp04` validates the frozen evaluator closure and merges supplied metadata into its outputs. The new factory/encoder is not certified merely because its digest appears there. Exp_05 adds validation and launcher bindings beyond its `model_factory` argument; exp_06 needs the equivalent contract.

The proposed H3 admission list—manifest digests, five seeds, K and k—is insufficient to admit historical M runs safely.

**Minimal fix:** Add a new exp_06 launcher/manifest validator and independently approved closure bindings for the new entry points, encoder, heading code, and producer. Bind the final checkpoint, training completion, heading/data inputs, closed logs, and outputs; require exclusive output directories and successful child completion.

H3 admission must additionally enforce:

- Correct checkpoint role/hash and epoch.
- `gl_seed == manifest_seed`, canonical batch 16, TF32 off.
- Condition P, k=0, full 6,337-query ordered split.
- Valid completion/output hashes and approved producer/evaluator closures.
- The registered finite-data cohort and convergence rules.

Add changed-input, stale-output, wrong-role, and incomplete-run refusal tests.

### 5. **Blocker — the registered command does not produce the registered epoch-12 checkpoint**
**Plan:** §§5, 8, 10.

With `--epoch-ckpt-every 5`, the unchanged trainer writes `epoch_005.pth` and `epoch_010.pth`. It does **not** write `epoch_12.pth` or `epoch_012.pth`. Epoch 12 is retained in `last.pth["model"]` after successful completion. See [train_xRIR_backbone.py](/home/yixunhu/codespace/xRIR_code/train_xRIR_backbone.py:262).

**Why it matters:** G1, HAA initialization, and H3 otherwise depend on a nonexistent artifact.

**Minimal fix:** Either register cadence 1 and the native `epoch_012.pth`, declaring that operational difference, or add a reviewed final-export helper. Export must require epoch 12, `batch_idx == 0`, complete history, and matching recipe; bind the exported hash. Test incomplete/mid-epoch refusal and exact exported state-dict equality.

### 6. **Blocker — H2’s “success” rule does not establish noninferiority**
**Plan:** §7 H2 and overall reading.

“No adjusted lower bound above zero” means **no detected harm**. Large but uncertain degradation can pass. Bonferroni adjustment makes this screen less likely to detect harm, rather than establishing parity.

The inherited HAA completeness rule also requires only one finite observation per required metric. Without a new invalidity policy, a treatment could exclude difficult queries and still obtain a success verdict.

**Minimal fix:** Either:

- Keep H2 descriptive and restrict the positive conclusion to hallway C50 noninferiority under H1; or
- Register practical margins and require adjusted **upper** bounds below those margins for every cell claimed noninferior.

Specify exclusions, paired-cohort sizes, and a failure policy for newly invalid treatment metrics. Test that missing metric validity cannot create success.

### 7. **Blocker — the statistical specification and proposed helpers disagree**
**Plan:** §§7, 8 round 3.

Two discrepancies need resolution:

1. H3 is described as the “five-seed mean of per-seed relative changes.” The imported machinery instead averages each query’s errors across seeds, then computes the relative difference of means. These operations differ.
2. `summarize_haa.paired()` hard-codes 95% percentile intervals and returns neither configurable tails nor bootstrap draws. It cannot directly produce H2’s Bonferroni intervals. See [summarize_haa.py](/home/yixunhu/codespace/xRIR_code/sim_to_real/summarize_haa.py:54).

**Minimal fix:** Freeze H3 explicitly as the exp_04 estimand:

\[
\rho=\frac{\operatorname{mean}_{q}(\operatorname{mean}_{s}e_{C,s,q})
-\operatorname{mean}_{q}(\operatorname{mean}_{s}e_{B,s,q})}
{\operatorname{mean}_{q}(\operatorname{mean}_{s}e_{B,s,q})},
\]

with the ratio recomputed inside each paired resample.

Add an exp_06-owned alpha-aware HAA bootstrap helper. Specify seeds, quantile convention, finite-data handling, and convergence checks. Test exact legacy 95% reproduction and H2 tails at \(0.05/(2·11)\) and \(1-0.05/(2·11)\). At 10,000 resamples, each H2 tail contains only about 23 draws.

### 8. **Should-fix — the controls and G1 do not support channel-specific causal claims**
**Plan:** §§1, 2.3, 6.1, 7, 11.

C versus B changes both the encoder and the frame. D versus A measures the frame effect on **SimpleViT**, while E measures the cylindrical frame effect only zero-shot. Exp_03 already shows that the full cylindrical xRIR is not invariant.

Likewise, mirror cosine and unsigned mixing-weight share do not directly measure channel use. Failure before fine-tuning does not establish that real-room fine-tuning could not learn to use the channel.

**Minimal fix:** Report C versus D in the common heading frame. Either add fine-tuned cylindrical heading-frame controls or attribute improvement to the **complete orientation-conditioned pipeline**. Make G1’s mechanism thresholds descriptive, or explicitly call its stopping rule budget triage and a stopped experiment inconclusive for the fine-tuned hypothesis. A channel intervention is needed to claim measured channel use.

Also narrow the inherited mechanism claims: the oracle used 16 queries per side, spectral proxies, and K=5/7 versus K=8. It supports reference-side sensitivity, not an isolated encoder diagnosis or a waveform-performance ceiling. Describe this as test-informed, pre-specified follow-up evaluation on these rooms and checkpoints.

### 9. **Should-fix — freeze the phase policy while preserving the valid pairing distinction**
**Plan:** §§2.3, 6.2.

**Reusing exp_02 A/B is valid query/reference pairing.** Reference selection uses a local generator keyed by `(eval_seed, room, idx)`. Independent, unseeded Griffin–Lim phases add nuisance variation; they do not invalidate those pairs.

Seeding only new-arm phases also does **not mechanically break reference pairing**, but cannot reproduce historical phases or create shared-phase comparisons. The proposed “only if shown not to change pairing” condition therefore lacks a precise acceptance criterion.

**Minimal fix:** Freeze the primary policy now. The simplest historical comparison retains exp_02’s phase behavior and discloses its nuisance. A fully shared-phase sensitivity comparison requires fresh evaluation-only copies of all compared checkpoints under exp_06 paths; historical outputs remain untouched.

### 10. **Should-fix — strengthen the symmetry tests without asserting bitwise identity**
**Plan:** §§2.1–2.2, 8–9.

The requested sign checks **pass**:

- Active `k=128` maps heading −90° to +x.
- Parent `cos_theta/sin_theta` match the normalized horizontal look direction from `convert_equirect_to_camera_coord`.

However, “identical heading-frame input” needs a numerical qualification. Composed float32 rotations differed by up to \(9.54\times10^{-7}\) in the CPU check.

**Minimal fix:** Test joint scene-and-heading rotations end to end over integer-column offsets, using justified tolerances, arbitrary headings, wrap-around, and rounding boundaries. Keep k=0 identity separately bit-exact. Test against the camera conversion independently, not only against the buffers used to construct the channels. State that continuous rotations are quantized and that E/B output agreement is descriptive, not guaranteed by token equivariance.

### 11. **Should-fix — recipe parity and the validation schedule need executable definitions**
**Plan:** §§5, 8–9, 11–12.

Raw `args.json` equality cannot pass the stated allowlist: the current trainer adds `vit_*`, tier/count fields, yaw defaults, `no_save`, loader length, and environment fields absent from exp_01.

The ladder also jumps from micro-batch four to the full 32×2 recipe. HAA’s existing two-epoch smoke writes checkpoints; it has no `--no-save`.

**Minimal fix:** Adopt exp_05’s normalized, type-strict recipe schema and explicit operational exclusions. Pin interpreter, `PYTHONHASHSEED`, and thread settings; disclose unpaired historical training realizations. Add a bounded 32×2 fit/timing probe and explicit storage-light HAA smoke handling. Keep scheduling contingent on actual handover and reviewed probe results.

Clarify that immutable-source restrictions permit the authorized notebook/table updates, and allow multiple small commits per reviewed round.

### 12. **Nit — correct rounded arithmetic and parameter wording**
**Plan:** §§5, 7 H1.

Canonical values give:

- Control hallway C50: **1.1354768442 dB**.
- Historical deficit: **0.9161867492 dB**.
- 20% of control: **0.2270953688 dB**.
- 25% of deficit: **0.2290466873 dB**.
- Historical two-way half-width: **0.1793697774 dB**.

Thus +0.23 dB is a reasonable rounded margin, but implies approximately **74.896%** deficit removal, not strictly at least 75%. The historical interval width is not a power guarantee for C versus A.

The parameter formula adds **526,336**, not approximately one million, parameters.

**Minimal fix:** Use approximate equalities/wording, or choose the exact quarter-gap margin; correct the parameter count.

## Verification performed

### Files read

Review coverage, including parallel read-only subreviews:

- `CLAUDE.md`, `worklog/experiment_SOP.md`; no announcement files found.
- Exp_06 plan, query, and notebook.
- Exp_02 plan, results, analysis, and complete notebook, including both 2026-09-14 entries.
- All six diagnostic scripts: `hall_diag.py`, `hall_diag2.py`, `hall_diag3.py`, `gt_sides.py`, `side_probe.py`, `oracle_probe.py`.
- Exp_03 analysis; exp_04 and exp_05 plans and amendments.
- Both exp_01 `args.json` files.
- `model/{cylindrical_vit,xRIR_cyl,xRIR}.py`, `tools/yaw_rotation.py`, and `treble_multi_room_dataset/treble_xRIR_dataset.py`.
- `sim_to_real/{haa_dataset,finetune_haa,eval_haa,summarize_haa}.py`, `run_haa_pipeline.sh`, and `train_xRIR_backbone.py`.
- `tools/{exp04_eval,exp04_eval_launch,exp05_eval,exp05_params,paired_compare,provenance}.py`, `tests/test_provenance.py`; relevant Griffin–Lim implementation.
- All four cache `meta.json`, `xyzs.npy`, `speaker_xyz.npy`, and `depth.npy`; RIRs opened lazily and only training rows accessed.
- Canonical `ckpt/sim2real/{summary.txt,stats.json}` and A/B per-sample outputs for all rooms and three fine-tuning seeds.

### Commands and outputs

Reads used `pwd`, `rg --files`, `rg`, `cat`, `nl -ba`, `sed`, and `wc -l`. Every Python check used:

```text
PYTHONPATH=/home/yixunhu/codespace/xRIR_code CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/yixunhu/miniconda3/envs/xRIR/bin/python
```

Inline CPU checks produced:

| Check | Output |
|---|---|
| Actual training-only heading fits, leave-one-out fits, nonuniform-angle synthetic cases | Results in finding 1; hallway refuses |
| `rotate_scene_yaw`, parent buffers, AST-extracted camera conversion | −y → `[1, −6.12e−17, 0]`; look-direction maximum error `5.96e−8`; joint-composition maximum error `9.54e−7` |
| Canonical summary hash | Matches `stats.json` |
| `summarize_haa.completeness(load_runs(...))` | `(True, [])` |
| Recomputed hallway C50 paired statistics | Exact canonical reproduction; two-way CI `[0.7433052605, 1.1020448152]` |
| A/B pairing and AST-extracted `_pick_refs` | Matching indices/paths/seeds/K; reproducible ordered eight-reference draws for every test query |
| Required A/B metric validity | All finite; dampened T60 correctly omitted |
| Cache schema | Depth `(256,512)`, RIR length 22,050, SR 22,050; test counts `463/198/423/198`; hallway sides `238/185` |
| Hallway half-turn depth difference | `0.090882` |

The pinned test was attempted with:

```text
-m pytest tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged -q -p no:cacheprovider
```

and again with `-s`. Writable temporary/cache requirements prevented completion. Direct dataset imports likewise encountered read-only Numba caching restrictions; AST extraction avoided those imports for the relevant checks.

A separate read-only `closure_record` check verified all 12 enumerated exp_03 files against the reviewed commit: working bytes matched, no later commits touched them, and the digest was:

```text
5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
```

This verifies the enumerated bytes/history; it does **not** claim that the dynamic import-closure test passed.

## Blocking findings to resolve before user approval

The exact blocking set is **1–7**:

1. Resolve the hallway heading and revise the sampling-biased estimator check.
2. Separate legacy G1 anchor reproduction from the full-cohort gate.
3. Specify executable, isolated wrapper interfaces and tests.
4. Complete the new-code provenance and run-admission contract.
5. Define production and validation of the epoch-12 checkpoint.
6. Correct H2’s success interpretation and invalidity policy.
7. Freeze one statistical estimand and implementable interval contract.

**Final verdict: request changes.**
