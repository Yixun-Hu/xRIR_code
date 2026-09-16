**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-14.

## Verdict: approve with changes

**V3 is ready for Coder round 1: encoder, factory, and heading tool.** Include the test clarification in R3.1 in the handoff. The remaining findings can be amended in §10a before their respective implementation rounds close and before affected runs launch. The SOP’s user-approval step still applies.

**Blocking findings: none.**

This is a plan review, not implementation approval. No files were modified, GPU work started, or processes signaled.

## R2.1–R2.9 disposition

| Finding | V3 assessment |
|---|---|
| **R2.1 — admission/completion lifecycle** | **Resolved structurally.** Separate approval phases, an external finalizer, per-child/job HAA completion, and reconstructed legacy admission address the original blockers. Concrete contract amendments remain: R3.4–R3.6. |
| **R2.2 — recipe normalization** | **Resolved in design.** All **33 currently written trainer fields** are classified. Production constraints and budget checks exclude the truncated-training counterexample. |
| **R2.3 — CPU legacy path and launcher adapters** | **Resolved for the cited defects.** Device-preserving alignment is feasible; binding separation and the additional writer closure compose with the existing launcher. Another GPU composition detail needs correction: R3.2. |
| **R2.4 — H1b validity** | **Resolved.** Both H1 and H1b now require the distinct-query finite cohort and validity checks; the overall claim consistently requires both. |
| **R2.5 — bootstrap convergence** | **Resolved.** Endpoint movement, seeds 0/1, tolerance, zero-width refusal, retry, and withholding are explicit. Historical bootstrap conventions are frozen correctly. |
| **R2.6 — quantization ties** | **Resolved and checked.** The stated example returns **0 and 511**; canonicalized joint rotations compose correctly. |
| **R2.7 — singleton/window semantics** | **Resolved and checked.** Singleton acceptance and null margins are explicit. Floor windows contain **110/1102 samples**; all four rooms remain stable. |
| **R2.8 — bounded smoke commands** | **Partially resolved.** Batch sizes and iteration limits are now bounded, but checkpoint prerequisites, evaluation argv, and memory reporting need R3.3. |
| **R2.9 — runtime versus reservation** | **Resolved.** Including S1 and B fallback gives **41.87 GPU-hours expected**, with approximately **50.33 hours reserved**. |

## Numbered findings

### R3.1. **Should-fix — align the round-1 test brief with the plan’s active rotation**

**Plan:** §§2.1–2.2, §8 round 1; associated [draft Coder prompt](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/coder_prompts/round1_prompt.md).

**What is wrong:** The draft asks for parent equivariance on a random input and its “32-column roll.” Parent equivariance requires the **active scene yaw: column roll plus rotation of xyz vectors**. A plain tensor roll does not satisfy it.

Its proposed joint-invariance test also compares `rotate_scene_yaw` with an independently constructed roll-plus-rotation, without explicitly exercising heading cancellation.

**Why it matters:** One test can reject the correct parent; the other can pass while missing a broken heading transform.

**Verified:** On a small CPU parent encoder, maximum token discrepancy after the expected permutation was:

- Plain column roll: **1.21245**.
- Active scene yaw: **1.07 × 10⁻⁶**.

**Minimal fix:** Specify active yaw for the scene-only equivariance test. For joint invariance, compare the heading-frame result for `(scene, φ)` against that for `(rotate_scene_yaw(scene, j), φ_c + j·360/512)`, including ties, wrap-around, and k=0. Keep the independent rotation-construction check separately.

### R3.2. **Should-fix — the mirror probe needs the pinned forward’s time-tensor device transfer**

**Plan:** §§6.1, 8 round 3.

**What is wrong:** “Exactly as `side_probe.fwd` did” includes passing `model.times` directly into the time embedder. `times` is an ordinary CPU tensor, not a registered buffer; moving the model to CUDA does not move it. The actual [pinned forward](/home/yixunhu/codespace/xRIR_code/model/xRIR.py:184) explicitly transfers it to `ref_ir_locs.device`.

**Why it matters:** Correcting alignment makes legacy CPU reproduction feasible, but literal diagnostic composition still fails during GPU G1.

**Minimal fix:** Preserve the pinned forward’s:

```python
times.unsqueeze(0).to(ref_ir_locs.device)
```

Add a GPU-skipped parity test comparing the composed probe’s prediction and target spectrogram with ordinary model forward, while returning weights/features as additional outputs.

### R3.3. **Should-fix — finish making the smoke commands executable**

**Plan:** §9, rung 4; §8 HAA entry points.

**What is wrong:**

1. The oriented HAA smoke uses `--init <smoke or exp_01 checkpoint>`. Earlier training smokes use `--no-save`, producing no checkpoint. An exp_01 checkpoint has incompatible patch-embedding shapes: **1536 → 2560** LayerNorm elements and **[512,1536] → [512,2560]** Linear weights. The inherited HAA loading path is strict.
2. The evaluation command still contains `…`, omitting its concrete backbone, checkpoint, and heading arguments.
3. The immutable baseline trainer does not print `torch.cuda.max_memory_allocated()`, although §9 requires every displayed command to do so. An exit report alone also does not implement the stated abort rule.

**Why it matters:** The oriented HAA smoke cannot reach an optimizer step as specified, and the baseline command cannot produce the promised memory evidence.

**Minimal fix:** Register a reviewed, disposable CPU-generated **oriented state-dict fixture** for the HAA smoke. Supply complete evaluation argv using that fixture or the resulting HAA checkpoint. Specify an exp_06-owned, same-process memory-reporting harness and bounded runner around the unchanged baseline trainer.

All displayed **inherited flags exist**. In particular, trainer `--save-every 0` is valid, and `--no-save` suppresses args/history and all checkpoints. New exp_06 parser additions remain implementation work.

### R3.4. **Should-fix — dispatch completion/provenance rules by run type**

**Plan:** §§5, 6.4, 8 round 2.

**What is wrong:** The explicit finalizer contract describes full twelve-epoch pretraining, but the finalizer also serves no-save smoke/probe runs, HAA training/evaluation children, and whole HAA jobs.

The blanket requirement that entry points write `provenance.json` also conflicts with delegated simulated evaluation:

- `exp04_eval.validate_manifest` permits only `eval_manifest.json` before execution.
- `execute_run` requires exactly that manifest and its two evaluation outputs before completion.

**Why it matters:** Literal enforcement rejects successful runs or requires undocumented exceptions.

**Minimal fix:** Add a small run-type matrix defining:

| Run type | Required completion evidence |
|---|---|
| Full pretraining | Existing twelve-epoch contract |
| Smoke/probe | No-save contract and explicitly non-arm diagnostic receipt |
| HAA training | Stage args/history/summary and checkpoint hashes |
| HAA evaluation | Room args/metrics/per-sample hashes |
| HAA job | Required child-completion hashes; explicit zero-shot layout |
| Simulated evaluation | Inherited `eval_manifest.json`/completion contract, without an extra in-directory provenance file |

The per-child HAA directory design itself is sound.

### R3.5. **Should-fix — make approval dependencies explicit and approve the legacy receipt**

**Plan:** §6.4; §8 profiles and summarizer rows.

**What is wrong:** The three approval sections have sensible fill times, but “code-only … refuses producers” and the blanket refusal rule remain ambiguous for heading and G1 producers, which create artifacts subsequently approved.

The explicit template also omits the **legacy receipt’s own hash**. Canonical stats/summary hashes do not themselves bind the retained per-sample files enumerated by that receipt.

**Why it matters:** Requiring every artifact before every producer creates a dependency cycle. An unpinned receipt cannot provide the promised tamper-refusal guarantee.

**Minimal fix:** List required approval keys per producer, excluding its own future outputs. Add reviewed `legacy_receipt: {path, sha256}` approval, and verify both the receipt hash and every enumerated artifact hash during admission. Preserve the `reconstructed` label.

For legacy completeness, load the complete historical exp_02 root before extracting A/B: the inherited checker expects the historical run set.

### R3.6. **Should-fix — bind shell orchestration as files, not Python modules**

**Plan:** §6.4 code-approval roster.

**What is wrong:** `source_closure` imports Python modules; `tools.exp06_launch.sh` is not an importable entry point. The roster also omits `tools/exp06_haa_pipeline.sh`, despite its ownership of job completion.

**Why it matters:** Literal closure capture fails or leaves orchestration bytes outside approval.

**Minimal fix:** Append both shell paths explicitly to the appropriate `closure_record` file lists after collecting Python import closures. [Exp_04’s launcher](/home/yixunhu/codespace/xRIR_code/tools/exp04_launcher.py:286) already demonstrates this pattern.

### R3.7. **Nit — synchronize the arm table with the settled baseline sources**

**Plan:** §2.3 versus §§6.3, 11.

**What is wrong:** The table still assigns A to exp_05/fallback evaluation and B “as A.” Later sections correctly assign A to completed exp_04 control evaluations, with the separate exp_05-or-exp_06 fallback for B.

**Minimal fix:** Update those two table cells.

## Verification record

### Files read

Coverage includes my reads and parallel read-only subreviews:

- `CLAUDE.md`, `worklog/experiment_SOP.md`; no announcement files found.
- Exp_06 v3 plan, including §§12–13; both prior reviews; full notebook; draft round-1 Coder prompt.
- Relevant exp_01 results/recipe records; exp_02 plan, results, analysis, September 14 notebook diagnostics and `side_probe.py`; exp_03 analysis; relevant exp_04/05 plan amendments.
- `train_xRIR_backbone.py`; both exp_01 `args.json` files.
- `model/{cylindrical_vit,xRIR_cyl,xRIR}.py`, `tools/yaw_rotation.py`, dataset camera-conversion code.
- `sim_to_real/{haa_dataset,finetune_haa,eval_haa,summarize_haa}.py` and `run_haa_pipeline.sh`.
- `tools/{exp04_eval,exp04_eval_launch,exp04_launcher,exp05_eval,exp05_params,exp04_profiles,paired_compare,provenance}.py` and relevant loss code.
- HAA metadata/coordinates and **only the twelve training RIR rows per room**, accessed through memory mapping.
- **Directory listing only** for the exp_02 run layout under `ckpt/sim2real`.

### Commands and outputs

Read commands used `rg`, `cat`, `nl`, `sed`, `wc`, directory listings, and inline CPU Python checks. Every Python check used:

```text
PYTHONPATH=/home/yixunhu/codespace/xRIR_code
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=4
PYTHONDONTWRITEBYTECODE=1
/home/yixunhu/miniconda3/envs/xRIR/bin/python
```

Key results:

- **Recipe:** 33 written fields versus 33 classified fields; no unclassified or phantom fields. Unspecified `yaw_aug_seed` resolves to **0**. `tier`, all four parameter-count categories, `env`, and `train_batches_per_epoch` are covered.
- **Budget:** 296,334 samples imply **9,261 micro-batches** at batch 32. Cadence 1 produces `epoch_012.pth`; final `last.pth` stores epoch 12/batch 0 and its args.
- **Launcher:** `heading` requires the adapter; `train_completion` is already accepted, and stripping/restoring it is harmless. Additional records under `source_closures["writer_exp06"]` and `mutable_inputs` are revalidated generically.
- **Legacy layout:** each A/B arm has 19 relevant directories, **zero completion files**, and **zero evaluation args files**—supporting the separate legacy branch.
- **Quantization:** `0.3515625° → 0`; `1.0546875° → 511`; **18,441** canonicalized heading/offset combinations had no composition failures.
- **Parameter increment:** **526,336**.
- **Bootstrap:** independent synthetic nominal intervals reproduced historical `paired` exactly, maximum difference **0.0**. H2 tails are **0.0022727273/0.9977272727**, approximately **114 draws per tail** at 50,000.
- **Convergence:** shifted equal-width intervals correctly fail endpoint convergence. The imported helper accepts identical zero-width intervals, so exp_06 must retain its explicitly specified additional refusal.

Fresh floor-window heading readback:

| Room | Contrast, 5/50 ms (dB) | Runner-up margin, 5/50 ms | Stable refits |
|---|---:|---:|---:|
| Classroom | 12.921 / 9.036 | Singleton | 12/12 |
| Dampened | 11.388 / 8.669 | 17.420 / 13.144 | 12/12 |
| Hallway | 18.453 / 11.293 | 36.905 / 22.586 | 12/12 |
| Complex | 11.953 / 10.806 | Singleton | 12/12 |

All select **−90°, k=128**; minimum leave-one-out winning contrast is **7.882 dB**.

No neural G1 anchors, GPU smokes, training runs, or write-requiring test suites were executed.

**Final verdict: approve with changes. Exact blocking set: empty.**
