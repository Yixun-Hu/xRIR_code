# Plan — oriented_cyl (exp_06): an orientation-relative azimuth channel for the CylindricalViT, tested on the real-room (HAA) benchmark

**Status:** v1, 2026-09-14, written by the Planner (Claude Fable 5.1, session `xrir-code-6e`) for the Codex plan review; **not yet approved by the user.** Builds on exp_01 (backbone comparison), exp_02 (sim-to-real, `worklog/worklog_yixun/exp_02_sim2real_transfer_claude/`), exp_03 (yaw rotation; pinned closure), exp_04/exp_05 (evaluation tooling `tools/exp04_eval.py`, `tools/provenance.py`, `tools/paired_compare.py`) and the 2026-09-14 post-hoc diagnostic recorded in exp_02's notebook (`diagnostics_2026-09-14_hallway_side/`). Yixun approved the direction on 2026-09-14: "physically add an orientation-relative channel to the cylindrical encoder as the next experiment" (`oriented_cyl_yixun_query.md`, query 4).

## 1. Question and the mechanism being fixed

**Question.** If the CylindricalViT geometry encoder is given each panorama column's azimuth *relative to the source's orientation* (a heading), does the cylindrical xRIR match or beat the SimpleViT xRIR after the paper's two-stage real-room fine-tuning on Hearing Anything Anywhere (HAA), while keeping its simulated-room accuracy?

**Mechanism (established descriptively on exp_02's runs, 2026-09-14).** The HAA loudspeaker is directional and faces −y in every room; in the hallway the measured C50 is ≈ +2.3…+3.8 dB for mics at −y and ≈ −6.4…−8.2 dB for mics at +y (≈ 10 dB gap at every distance, 2× energy). The hallway panorama at the speaker is nearly 180°-rotation symmetric (mean relative pixel difference 0.09 under a 256-column roll vs 0.57 under a 128-column roll). The cylindrical encoder is azimuth-equivariant by construction (exp_03: tokens equivariant to 2e-7) and its learned position pool leaves the pooled geometry feature nearly invariant (exp_03: 1.7 % change under a one-patch roll), so a mic at +y and its mirror image at −y produce the same geometry feature (cosine 0.975; SimpleViT control 0.435, released 0.325). The reference attention then selects references by distance regardless of corridor end (85 % of the |mixing weight| on −y references for +y queries; control 24 %) and predicts a −y-like RIR for +y mics (signed spectral-C50 error +7.4 dB; control +1.7 dB). Zero-shot hallway C50 error: cylindrical 1.04 dB at −y / 8.23 dB at +y; control 2.37 / 2.42 dB. With references restricted to the query's own side (oracle) the cylindrical model reaches 0.99 dB vs the control's 1.04 dB and has the lower log-spectrogram L1 on both sides — the encoder's room representation is fine; only the front/back ambiguity is broken. The same sign pattern appears weakly in the other three rooms, whose panoramas are asymmetric.

**Why the fix should be a heading input, not a different pool.** In AcousticRooms (omnidirectional sources and receivers, receiver frame = world frame) absolute azimuth is physically irrelevant, so an equivariant encoder is the *right* prior there; real rooms break the symmetry through the directional loudspeaker, which no geometry input represents. The physically correct invariance is a **joint** rotation of the scene *and* the source orientation. Giving the encoder the column azimuth relative to the heading makes it equivariant to exactly that joint rotation and no longer to a scene-only rotation.

**Expected outcome (stated before the evidence).** Parity or a small advantage on HAA as released (each room has one fixed loudspeaker heading, so the SimpleViT's absolute-azimuth prior is already the right one there; the oracle caps the gain at a few percent of log-spectrogram error), with the hallway deficit removed. Not a decisive win. The experiment can also fail in a specific way: a network pretrained on omnidirectional simulation may not *use* the new channel (gate G1 below tests this before any fine-tuning compute is spent).

## 2. Design

### 2.1 Frames and the heading transform (data level)

- **Receiver frame** (unchanged): the query receiver's camera frame with rotation 0, i.e. the world frame translated to the receiver (`treble_xRIR_dataset.get_3d_point_camera_coord`). Column `c` of the 256×512 panorama looks along azimuth `θ_c = (c + 0.5)·2π/512 − π` (`convert_equirect_to_camera_coord`; the same buffer the CylindricalViT uses for its gauge).
- **Heading `φ`**: the azimuth (room/receiver frame) of the source's orientation axis. AcousticRooms: `φ ≡ 0` by convention (omnidirectional; the world +x axis is the heading — this makes the new channel the absolute column azimuth, i.e. the same information the SimpleViT's 2-D positional code carries). HAA: `φ_room` estimated from the room's 12 *training* RIRs (§4) — in the model's role swap the loudspeaker is the "receiver" whose panorama we hold, so the heading is a property of the panorama location.
- **Heading-frame transform** (applied to the data tuple, model forward unchanged): `k = round(−φ · 512 / 2π) mod 512`; `(depth_coord, src_local, ref_src_local) ← rotate_scene_yaw(depth_coord, src_local, ref_src_local, k)` from the pinned `tools/yaw_rotation.py` (active yaw by `Δ = 2πk/512`: the panorama is rolled by `k` columns and every xyz vector is rotated by `Rz(Δ)`). After the transform the heading points along +x. Quantisation error ≤ 0.35°; `k` and `φ` are recorded in every stage `args.json`. For `φ = −90°`, `k = 128` (exactly 4 patch columns).
- **Why data level.** It reuses exp_03's tested rotation, keeps `xRIR.forward` and every evaluator untouched, records the heading as an explicit input, and gives the SimpleViT heading-frame arm (D) for free. Applying `k = 0` is the identity (tested bit-exact).

### 2.2 Encoder: `CylindricalViTOriented`

`model/cylindrical_vit_oriented.py`, subclass of the pinned `model.cylindrical_vit.CylindricalViT`:

- Input `[B, 3, H, W]` xyz as before; per-column gauge alignment `Rz(−θ_c)` exactly as the parent.
- **Two constant channels** appended after alignment: `cos θ_c` and `sin θ_c`, the column's unit look direction in the (heading-aligned) receiver frame, broadcast over rows and batch (`register_buffer("az_channels", [2, H, W])`, built from the parent's `cos_theta`/`sin_theta` buffers so the two conventions cannot drift).
- Patch embedding rebuilt for `patch_dim = 5·16·32 = 2560` (LayerNorm → Linear → LayerNorm as the parent); elevation-only positional code, circular relative-position bias transformer, output `[B, 256, 512]` tokens — all inherited, so everything downstream (`src_proj`, `lin_proj_0`, …) is byte-identical to `xRIR_Cyl`.
- Symmetry statement (tested): tokens are **not** equivariant to a scene-only roll any more (the channels do not roll) and **are** invariant to a joint scene+heading rotation (the heading-frame input is identical by construction). With the two channels zeroed, the embedding of the xyz part is the parent's construction (same layer shapes for the first 1536 inputs; LayerNorm statistics differ, so no bit-equality claim is made).
- Parameters: `+ 2·16·32·512` Linear weights `+ 2·(2560 − 1536)` LayerNorm affine parameters relative to `CylindricalViT` (exact count asserted in a test).

`model/xRIR_cyl_oriented.py`: `xRIR_CylOriented(xRIR)` swapping `source_network` (mirror of `xRIR_Cyl`), `BACKBONES_EXP06 = {**BACKBONES, "cylindrical_oriented": xRIR_CylOriented}` and `build_xrir_exp06(backbone, num_shot, **kw)`; the `simple`/`cylindrical` routes must return the pinned classes.

### 2.3 Arms

| Arm | Model / init | Frame for HAA | Pretraining | HAA fine-tuning (3 seeds) + zero-shot | Sim k = 0 eval |
|---|---|---|---|---|---|
| A `control` | SimpleViT, exp_01 `ckpt/xRIR_simple_8_shot/epoch_12.pth` | room frame | exists | **exists** (exp_02, `ckpt/sim2real/control/`) | exp_05 M-tier evaluations if complete, else exp_06 run |
| B `cyl` | CylindricalViT, exp_01 `ckpt/xRIR_cyl_8_shot/epoch_12.pth` | room frame | exists | **exists** (exp_02, `ckpt/sim2real/cyl/`) | as A |
| C `cyl_or` | `cylindrical_oriented`, **new pretraining** (§5), seed 0 | heading frame | **new** (≈ 31 h) | **new** | **new** (5 seeds) |
| D `control_hf` | SimpleViT, exp_01 checkpoint | heading frame | exists | **new** (cheap ablation: separates "knowing the heading frame" from "the oriented encoder") | none |
| E (sanity, not an arm) | `cyl` in heading frame | heading frame | exists | zero-shot eval only | none |

A and B are reused read-only: the exp_06 pairing is valid because exp_02 drew references deterministically per `(eval_seed 0, room, idx)` and uses the same DiffRIR test indices; `tools/exp06_summarize_haa.py` asserts identity of `index`/`ir_path`/`eval_seed`/`K` on every pair, as exp_02's summariser does.

## 3. What is *not* changed

- exp_03's pinned closure (`eval_yaw_rotation.py`, `eval_unseen.py`, `eval_xRIR_backbone.py`, `model/{cylindrical_vit,simple_vit,xRIR,xRIR_cyl}.py`, `tools/{per_sample_metrics,reference_manifest,yaw_rotation}.py`, `treble_multi_room_dataset/treble_xRIR_dataset.py`, `utils/spec_utils.py`): never edited (`tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged`).
- exp_05's training-closure pin: `train_xRIR_backbone.py` and every file in `tools.provenance.source_closure('train_xRIR_backbone')` (digest `5b2da250…`) stay byte-identical while exp_05's L runs launch/finalise; exp_06 therefore trains through a new entry point (§8, `tools/exp06_train.py`).
- exp_02's pipeline (`sim_to_real/{haa_dataset,finetune_haa,eval_haa,summarize_haa,prepare_haa}.py`, `run_haa_pipeline.sh`) and record: unchanged; exp_06 entry points import them.
- exp_04/exp_05 tooling (`tools/exp04_*.py`, `tools/exp05_*.py`, `tools/paired_compare.py`, `tools/results_table.py`, `tools/provenance.py`): import only.
- The HAA cache `~/data_cache/HAA_xrir/` and the AcousticRooms mirror: read only.

## 4. Heading estimation for HAA (`tools/exp06_heading.py`)

Inputs per room (cache): `rirs.npy`, `xyzs.npy`, `speaker_xyz.npy`, `meta.json` (`train` indices, `sr`). **Only the 12 training mics are used** (no validation/test leakage). For each training mic `i`: `θ_i = atan2(y_i − y_s, x_i − x_s)`, `d_i = ‖(x_i − x_s, y_i − y_s)‖`, onset `n_i` = first sample with `|h_i| > 0.2·max|h_i|`, early energy `E_i = Σ_{n ∈ [n_i, n_i + 0.05·sr)} h_i[n]²`, level `L_i = 10 log10(E_i d_i²)` (spherical-spreading compensated).

- **Primary estimator**: least-squares fit `L_i ≈ a + b cos(θ_i − φ)` with `b ≥ 0`, `φ` on a 0.5° grid (closed-form `a, b` per grid point); `φ̂ = argmin` residual. Front/back contrast `2b` dB and residual RMS are reported.
- **Consistency check**: energy-weighted mean direction `φ̄ = atan2(Σ w_i sin θ_i, Σ w_i cos θ_i)`, `w_i = E_i d_i²`. The estimator **refuses** (fail-closed) if `|wrap(φ̂ − φ̄)| > 20°` or `2b < 3 dB` (no usable directivity) unless `--heading-deg <value> --override-reason "<text>"` is supplied, in which case the override and reason are recorded.
- Output `ckpt/exp06/heading/<room>.json`: `phi_rad`, `phi_deg`, `k`, `contrast_db`, `phi_mean_deg`, `residual_db`, `n_train`, sha256 of the four inputs, the estimator's source digest (`tools.provenance.source_closure`), timestamp. The diagnostic predicts `φ ≈ −90°` in all four rooms (higher energy/C50 at negative y); the plan does **not** assume it — the estimates are reported and used as found. If Yixun knows the true loudspeaker orientation (DiffRIR metadata), it is supplied through the override and the estimate is reported alongside (decision §11).

## 5. Pretraining recipe (= exp_01, one permitted difference: the backbone)

`tools/exp06_train.py --backbone cylindrical_oriented --save-dir ckpt/exp06/pretrain/xRIR_cylor_8_shot --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 500 --epoch-ckpt-every 5` with `XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, one A6000. Identical to `ckpt/xRIR_{simple,cyl}_8_shot/args.json` (asserted by a test on the recorded `args.json`, allowing only `backbone`, `save_dir` and the exp_06 provenance fields to differ). Wall time: exp_01's cylindrical run recorded 155.8 min/epoch → ≈ 31 h; the extra 2 input channels add ~1 M parameters to the patch embedding and negligible compute. The model at **epoch 12** is the arm (as exp_01/exp_02 used `epoch_12.pth`); `epoch_05`, `epoch_10`, `best.pth`, `last.pth` kept. `provenance.json` (source closure of `tools/exp06_train`, HEAD, environment, data inventory via `tools/provenance.py`) written at start, `completion.json` at the end; the log is teed to the record folder.

## 6. Evaluation protocols

### 6.1 Gate G1 — does the pretrained encoder use the channel? (`tools/exp06_mirror_probe.py`, GPU, minutes, **before any fine-tuning**)

On the hallway cache panorama, for every +y DiffRIR test mic (185) and its exact mirror `(−x, −y, z)` about the speaker: cosine of the pooled geometry feature (`lin_proj_0(src_proj(source_network((src − depth_coord)/5)))`). With the `eval_seed 0` references (K = 8): share of `Σ_t |w_i(t)|` on −y references for +y queries, and the signed spectral C50 error (predicted vs target magnitude spectrogram at the geometric onset). Models: `cyl_or` epoch 12 in the heading frame; anchors `cyl` and `control` in the room frame (must reproduce the 2026-09-14 values within ±0.03: 0.975/0.85 and 0.435/0.24), plus E (`cyl`, heading frame; descriptive). Output `ckpt/exp06/gate_g1.json` (hash-bound, includes the checkpoint sha256 and the heading JSON sha256).

**G1 passes iff** mirror cosine(`cyl_or`) < 0.80 **and** −y weight share for +y queries(`cyl_or`) < 0.50. Fail → stop before fine-tuning (§11 fallbacks); the pretraining still yields H3.

### 6.2 HAA (exactly exp_02's protocol)

Four rooms, DiffRIR splits (12 training RIRs; validation for checkpoint selection; test sizes classroom 463, dampened 198, hallway 423, complex 198). Stage 1 on classroom + hallway + complex (`--epochs 1000 --val-every 10 --tf32`, full-batch, lr 1e-4, wd 1e-4), stage 2 per room (`--epochs 200 --val-every 2 --tf32`), eval on the DiffRIR test split with K = 8, `eval_seed 0`, Griffin-Lim inversion, EDT / C50 / T60 on the 9600-sample waveforms, T60 skipped for the dampened room; **per-query seeded Griffin-Lim phase** is added in the exp_06 evaluation wrapper only if it can be shown not to change exp_02's pairing (default: unchanged, as exp_02 — decision §11). Fine-tuning seeds 0, 1, 2 per new init; `depth_variant default` (cache renders for complex/dampened, as exp_02). New inits `cyl_or` (heading frame, `cylindrical_oriented`, epoch-12 pretraining checkpoint) and `control_hf` (heading frame, `simple`, exp_01 control checkpoint); zero-shot evals of `cyl_or`, `control_hf` and E. Outputs `ckpt/exp06/sim2real/<init>/{seed<s>/{stage1,stage2_<room>,eval}, zeroshot}` mirroring exp_02, each stage/eval `args.json` carrying `heading` (φ, k, JSON sha256 per room) and `backbone`.

### 6.3 Simulated-room parity at k = 0 (exp_04 protocol)

`tools/exp06_eval.py` = `tools/exp04_eval.run_exp04` with `model_factory=build_xrir_exp06`, manifests `ckpt/yaw_aug/reference_manifest_k8_seed{42,43,44,45,46}.json`, `--conditions P`, k = 0 only, TF32 off, unseen split (6337 queries per seed), outputs under `ckpt/exp06/sim_eval/cyl_or/seed<S>/`. For `simple`/`cylindrical` the exp_05 M-tier evaluations (same evaluator, same manifests) are reused if complete when needed; otherwise exp_06 runs them itself (identical protocol, +10 h). No heading transform in simulation (`φ ≡ 0`).

## 7. Pre-registered hypotheses and decision rules

Statistics as exp_02 for HAA (per-sample paired differences pooled over the three fine-tuning seeds; two-way query × seed percentile bootstrap, 10 000 resamples, fixed seed; verdicts on the two-way interval) and as exp_04 for simulation (five-seed mean of per-seed relative changes; one-sided 97.5 % bootstrap upper bound at the query level; room-cluster interval reported).

- **G1 (gate, before fine-tuning)** — as §6.1. Anchors must reproduce.
- **H1 (primary, confirmatory)** — hallway C50 error, `cyl_or − control`: the two-way 95 % upper bound is **< +0.23 dB**. The margin is frozen from exp_02's canonical values: 0.23 dB = 20 % of the control's hallway C50 mean (1.135 dB) = 25 % of the exp_02 gap (+0.916 dB), i.e. at least three quarters of the deficit removed; it exceeds exp_02's realised two-way half-width (0.18 dB) so parity can pass. One cell, nominal level.
- **H1b (mechanism)** — hallway C50, `cyl_or − cyl`: two-way 95 % upper bound < 0 (the channel removes the deficit).
- **H2 (family)** — all 11 room × metric cells, `cyl_or − control`, Bonferroni-adjusted two-way intervals (level 1 − 0.05/11): success = **no cell** with lower bound > 0 (cyl_or significantly worse). Cells with upper bound < 0 are counted and reported (superiority, descriptive).
- **H3 (sim parity, k = 0, K = 8)** — `cyl_or` vs `cyl`: relative increase in EDT and in C50 error, one-sided 97.5 % upper bound (query level, five-seed mean) **< +3 %** for both (exp_04's H1 rule). `cyl_or` vs `simple` reported descriptively with the same machinery (exp_03/04 found cyl EDT ≈ −4 % vs simple).
- **D (descriptive)** — `control_hf − control` on all 11 cells; zero-shot side-split table (−y / +y) for every arm and room; E's zero-shot agreement with B.

**Overall reading.** "The orientation channel fixes the real-room deficit" iff G1 and H1 pass and H2 has zero worse cells; H3 states whether it costs simulated accuracy. Anything else is reported as found; no post-hoc margin changes (amendments go to §10a with reasons before the relevant runs).

## 8. Code (Coder = OpenAI Codex `gpt-6-astra`; TDD; Fable 5.1 review per round; worktree branch `exp06-window`)

Constraint for every round: **no existing tracked file is modified** (§3). Python 3.8 (`xRIR` env). New tests in `tests/` (`tests/test_exp06_*.py`), CPU-only unless marked GPU (skipped when no CUDA).

**Round 1 — encoder, factory, heading transform**

| File | Functions / classes | Tests (write first) |
|---|---|---|
| `model/cylindrical_vit_oriented.py` | `azimuth_channels(H, W, dtype)`; `CylindricalViTOriented(CylindricalViT)` (`__init__` rebuilds `to_patch_embedding` for 5 channels, registers `az_channels`; `forward`) | shapes `[B,256,512]`; `az_channels == stack(cos_theta, sin_theta)` broadcast (uses the parent's buffers); exact parameter count vs parent; **non-equivariance**: tokens under a 32-column scene roll are not a row permutation (parent: are, to 1e-6); **joint invariance**: heading-frame input identical → identical tokens; mirror discrimination on a synthetic 180°-symmetric scene: parent tokens for a source and its mirror are an 8-patch roll of each other, child tokens are not; dtype/device propagation; `in_channels != 3` refused |
| `model/xRIR_cyl_oriented.py` | `xRIR_CylOriented(xRIR)`, `BACKBONES_EXP06`, `build_xrir_exp06` | `simple`/`cylindrical` return the pinned classes (`is`); forward shapes `[B,63,310,1]`/`[B,1,63,310]` (GPU test, skipped on CPU); param count = `xRIR_Cyl` + expected delta; unknown name refused; kwargs (`dim`, `depth`…) pass through |
| `tools/exp06_heading.py` | `early_level(rir, sr)`, `fit_heading(theta, level)` (grid LS), `mean_direction(theta, w)`, `estimate_heading(room_dir, override=None)`, `heading_roll_k(phi, W=512)`, `HeadingFrameDataset(base, k_by_room)`, `write/read_heading_json`, CLI | synthetic cardioid recovers φ within 1° (several φ, noise); refusal on disagreement > 20° and on contrast < 3 dB; override recorded; `heading_roll_k(−π/2) == 128`, `heading_roll_k(0) == 0`, wrap-around; `HeadingFrameDataset` with k = 0 is bit-identical to the base; k = 128 equals `rotate_scene_yaw` on the unsqueezed tuple and leaves audio untouched; JSON round-trip + sha256 of inputs; training-only indices asserted |

**Round 2 — entry points (wrappers; existing scripts unchanged)**

| File | Behaviour | Tests |
|---|---|---|
| `tools/exp06_train.py` | imports `train_xRIR_backbone` as a module, injects `BACKBONES_EXP06`/`build_xrir_exp06` into its namespace, adds provenance (`provenance.json`, `completion.json`, `exp06` marker + registry digest in `args.json`), then runs its `main()` unchanged | `--backbone cylindrical_oriented` accepted; `simple` builds a state_dict bit-identical to the untouched trainer's `build_model` (same seed); recipe fields of a written `args.json` equal exp_01's except the allowed keys; `py_compile`; GPU smoke (`--epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --num-workers 4 --no-save`) recorded on the ladder, not in pytest |
| `tools/exp06_eval.py` | `run_exp04(args, model_factory=build_xrir_exp06, metadata={model class, registry digest, heading: none})`, launcher bindings as `tools/exp05_eval.py` does | factory routing; metadata fields; manifest mismatch refused (reuse exp_04 test fixtures) |
| `tools/exp06_haa_finetune.py`, `tools/exp06_haa_eval.py` | wrap `sim_to_real.finetune_haa` / `eval_haa`: extended registry; `--heading-json-dir` (per-room JSON → `HeadingFrameDataset`); **required** for `cylindrical_oriented`, optional for `simple`/`cylindrical`; `heading` recorded in `args.json` and in per-sample `meta` | fail-closed without heading JSON for the oriented backbone; default path (no heading, pinned backbones) yields bit-identical dataset items and the same model class as the exp_02 scripts; heading recorded |
| `tools/exp06_haa_pipeline.sh` | exp_02's queue script for inits `cyl_or`, `control_hf`, `zeroshot` (incl. E) writing under `ckpt/exp06/sim2real/`; `--dry-run` prints the commands | `bash -n`; dry-run output golden (paths, flags, seeds) |

**Round 3 — analysis tools**

| File | Behaviour | Tests |
|---|---|---|
| `tools/exp06_mirror_probe.py` | G1 probe (§6.1) → `ckpt/exp06/gate_g1.json`; decision function `g1_decision(stats)`; anchors check | mirror/side bookkeeping on synthetic tiny models; decision thresholds (boundary values); anchor tolerance; JSON hash binding |
| `tools/exp06_summarize_haa.py` | reuses `summarize_haa.load_runs/paired/completeness` under exp_06 tables (EXPECTED_RUNS: `cyl_or` seeds 0–2 + zero-shot, `control_hf` seeds 0–2 + zero-shot, `cyl_hf` zero-shot; exp_02's `control`/`cyl` read from `ckpt/sim2real/`), pairs H1/H1b/H2/D, Bonferroni for H2, side-split table, `--json`/`--summary` hash-bound as exp_02 | completeness refusals (missing seed, wrong heading k, wrong backbone, wrong test indices); pairing assertions; **regression**: exp_02's own `cyl − control` numbers reproduced from `ckpt/sim2real/` (skipped if absent); margin constants frozen (0.23 dB) |
| `tools/exp06_compare.py` | H3 from per-sample outputs of `exp06_eval` (and exp_05 M evals for `simple`/`cyl`), reusing `paired_compare.{five_seed_mean, cell_mask, rho_bootstrap, h1_verdict}`; fail-closed admission (manifest digests, 5 seeds, k = 0, K = 8) | admission refusals; verdict boundaries; determinism |
| `oriented_cyl_results_assets/make_results_{html,md}.py` (Planner one-offs) | render only from `ckpt/exp06/stats.json` + `gate_g1.json` | structural checks; refuse anything else (as exp_03/exp_02) |

Each round: red → green → one small commit (< 200 lines) with trailer `Co-Authored-By: OpenAI Codex gpt-6-astra <noreply@openai.com>`; per-round Fable review saved as `oriented_cyl_fable_code_round<N>_review.md`; blocking findings fixed before the next round; `full` review before the pretraining launch.

## 9. Validation ladder, parity audit, acceptance criteria

1. Static: `py_compile` all new files, `bash -n`, `git diff --check`, `python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q` (pinned-closure test must stay green).
2. Tiny synthetic forward of `CylindricalViTOriented` and `xRIR_CylOriented` (CPU; the full xRIR forward needs a GPU because `apply_delay` is `.cuda()`-only).
3. Real-data readback: `HeadingFrameDataset` on the hallway cache with k = 0 (bit-identical) and k = 128 (equals `rotate_scene_yaw`); heading JSONs for all four rooms written and inspected (φ, contrast, residual).
4. GPU smoke of `tools/exp06_train.py` (3 batches, `--no-save`) and of `tools/exp06_haa_finetune.py` (`--epochs 2`) and `exp06_haa_eval.py` (`--max-samples 4`) — on a free card only (§11).
5. Parity audit (recorded in the notebook): (i) `build_xrir_exp06('cylindrical')` output bit-identical to the pinned factory on a fixed batch; (ii) `az_channels` derived from the parent's gauge buffers; (iii) recipe `args.json` identical to exp_01 except allowed keys; (iv) heading-frame transform: k = 0 identity, k = 128 = `rotate_scene_yaw`; (v) HAA wrappers with no heading reproduce exp_02's dataset items for one query bit-exactly.
6. Pre-launch acceptance criteria (written in the notebook before each launch): commit SHA, GPU index, batch 32 × 2, TF32 on, 12 epochs, reaches ≥ 1 optimizer step, epoch-1 time ≤ 1.1 × 156 min, loss finite; fine-tuning: every stage writes `summary.json`, every eval `metrics_<room>.json` + `per_sample_<room>.json` with heading recorded.

## 10. Deliverables

`ckpt/exp06/{pretrain/xRIR_cylor_8_shot, heading/*.json, gate_g1.json, sim2real/**, sim_eval/**, stats.json, summary.txt, h3.json}`; record: `oriented_cyl_params_set_up.md`, `oriented_cyl_command.md`, logs, `oriented_cyl_results.md`, `oriented_cyl_analysis.md`, `oriented_cyl_01_results.html` (+ assets), `commits_oriented_cyl.md`; `worklog/worklog_yixun/model_comparison.md` row if the table format admits it.

### 10a. Amendments during implementation (logged in the notebook)
(none yet)

## 11. Cost, schedule, risks, decisions

**GPU cost.** Pretraining ≈ 31 h (one A6000). G1 ≈ 5 min. HAA: 6 fine-tuning jobs × ≈ 50 min + zero-shot evals ≈ 5.5 h (2.8 h on two cards). Sim eval: 5 × ≈ 1 h for `cyl_or` (+ 10 h if the M pair must be re-evaluated). Total ≈ 42–52 GPU-hours.

**Schedule (contingent on the peer session's exp_04/exp_05 plan, notebook of exp_05, 2026-09-14).** GPU 1 runs exp_05 L_cylindrical until ≈ Sep 16 10:00 then its evaluations; GPU 0 runs the exp_04 evaluation chain, then exp_05 L_simple (≈ Sep 15 04:00 → 22:00) and evaluations (≈ 6.5 h). Earliest exp_06 pretraining start ≈ Sep 16 morning on GPU 0 → ends ≈ Sep 17 evening; G1 + HAA + sim eval Sep 18; results/analysis/page Sep 19. ICLR deadline 2026-09-24: ~4 days of slack, no room for a second pretraining.

**Risks.** (1) G1 fails — the channel is unused after omnidirectional pretraining; the fine-tuning is then not run and the fallbacks are exp_07 candidates: directivity-aware pretraining augmentation (a synthetic source directivity gain applied to targets and references as a function of the angle to a random heading, making the heading acoustically relevant) or the fine-tune-only angle feature. (2) The channel is used but fine-tuning on 12 RIRs still under-uses it — H1 fails with G1 passed; reported as such. (3) The heading estimator is ambiguous in a room (low contrast) — refused, decision needed. (4) Schedule slip from the running experiments. (5) One pretraining seed per backbone (as exp_01/02).

**Decisions for Yixun (before approval).** (a) Include arm D `control_hf` (recommended: yes, 2.5 GPU-hours, it attributes any change to the encoder rather than to the frame). (b) Heading source: estimator (default) or a known DiffRIR loudspeaker orientation supplied as override. (c) GPU hand-over: exp_06 pretraining starts on the first free card after exp_05's L runs and their evaluations, or takes precedence over the L evaluations (saves ≈ 6 h). (d) Whether the 5-minute GPU smokes (ladder rung 4) may run co-tenant with the running exp_04 evaluation chain on GPU 0 now, or wait for a free card. (e) H1 margin +0.23 dB (20 % of control) — accept or tighten.

## 12. Handoff notes

Codex works in a separate worktree `/home/yixunhu/codespace/xRIR_code_wt` (branch `exp06-window`, base = current `main`), `ckpt/` symlinked read-only, `CUDA_VISIBLE_DEVICES=''`, never signals processes; the branch is merged into `main` only at a launch-free window agreed with the peer session `xrir-code-25` (its exp_05 receipts pin a commit; a `main` commit between its probe and `full` launch would refuse the launch). The record folder is written in the main tree under `worklog/` and committed at the same windows. All long jobs are `nohup setsid`-detached with teed timestamped logs and watched by pid/GPU occupancy, never by a `pgrep -f` of their own text.
