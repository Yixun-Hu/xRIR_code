# Plan — oriented_cyl (exp_06): an orientation-relative azimuth channel for the CylindricalViT, tested on the real-room (HAA) benchmark

**Status:** v2, 2026-09-14, revised by the Planner (Claude Fable 5.1, session `xrir-code-6e`) after the round-1 Codex plan review (`oriented_cyl_codex_plan_review.md`: **request changes**, blockers 1–7, should-fix 8–11, nit 12 — all addressed, §12 changelog; v1 kept as `plan_oriented_cyl_v1_superseded.md`). **Not yet approved by the user.** Builds on exp_01 (backbone comparison), exp_02 (sim-to-real), exp_03 (yaw rotation; pinned closure), exp_04/exp_05 (evaluation tooling: `tools/exp04_eval.py`, `tools/exp04_eval_launch.py`, `tools/provenance.py`, `tools/paired_compare.py`, `tools/exp05_eval.py`) and the 2026-09-14 post-hoc diagnostic in exp_02's notebook (`diagnostics_2026-09-14_hallway_side/`). Yixun approved the direction on 2026-09-14 (`oriented_cyl_yixun_query.md`, query 4).

## 1. Question and the mechanism being fixed

**Question.** If the CylindricalViT geometry encoder receives each panorama column's azimuth *relative to the source's orientation* (a heading), does the cylindrical xRIR match or beat the SimpleViT xRIR after the paper's two-stage real-room fine-tuning on Hearing Anything Anywhere (HAA), while keeping its simulated-room accuracy?

**Mechanism (descriptive diagnosis on exp_02's realised runs, 2026-09-14; the follow-up evaluation below is test-informed and pre-specified on the same rooms and checkpoints).** The HAA loudspeaker is directional and its acoustic preferred direction is −y in every room; in the hallway the measured C50 is ≈ +2.3…+3.8 dB for mics at −y and ≈ −6.4…−8.2 dB at +y (≈ 10 dB gap at every distance, 2× energy). The hallway panorama at the speaker is nearly 180°-rotation symmetric (mean relative pixel difference 0.09 under a 256-column roll vs 0.57 under 128). The cylindrical encoder is azimuth-equivariant by construction (exp_03: tokens equivariant to 2e-7) and its learned position pool leaves the pooled geometry feature nearly invariant (exp_03: 1.7 % under a one-patch roll). On 24 rank-sampled +y hallway test mics, a mic and its mirror image (−x, −y) gave pooled-feature cosine 0.975 for the cylindrical model vs 0.435 for the SimpleViT control (released 0.325); 85 % of the |mixing weight| went to −y references for +y queries (control 24 %); the signed spectral-C50 error was +7.4 dB (control +1.7 dB). Zero-shot hallway C50 error over all test mics: cylindrical 1.04 dB at −y / 8.23 dB at +y; control 2.37 / 2.42 dB. An oracle with same-side references (16 queries per side, spectral proxies, K = 5/7 instead of 8) brought the cylindrical model to 0.99 dB vs the control's 1.04 dB with a lower log-spectrogram L1 — evidence of **reference-side sensitivity**, not an isolated encoder diagnosis or a waveform-performance ceiling. The same sign pattern appears weakly in the other rooms, whose panoramas are asymmetric.

**Why a heading input.** In AcousticRooms (omnidirectional sources and receivers; receiver frame = world frame) absolute azimuth is physically irrelevant, so an equivariant encoder is the right prior there; real rooms break the symmetry through the directional loudspeaker, which no geometry input represents. The physically correct invariance is a **joint** rotation of scene *and* source orientation. Expressing the scene in the heading frame and giving the encoder the column azimuth in that frame makes the whole pipeline orientation-conditioned. Any improvement is attributed to the **complete orientation-conditioned pipeline** (frame + channel + fine-tuning); the contrasts in §2.3 separate the frame effect from the channel effect.

**Expected outcome (stated before the evidence).** Parity or a small advantage on HAA as released (each room has one fixed loudspeaker heading), with the hallway deficit removed; not a decisive win. A specific failure mode exists: a network pretrained on omnidirectional simulation may not use the channel (gate G1, a budget-triage rule, §6.1).

## 2. Design

### 2.1 Frames and the heading transform (data level)

- **Receiver frame** (unchanged): the query receiver's camera frame with rotation 0 (`treble_xRIR_dataset.get_3d_point_camera_coord`). Column `c` of the 256×512 panorama looks along azimuth `θ_c = (c + 0.5)·2π/512 − π` (`convert_equirect_to_camera_coord`; identical to the CylindricalViT's gauge buffers — verified in the round-1 review, max error 6e-8).
- **Heading `φ`**: azimuth (receiver frame) of the source-orientation axis. AcousticRooms: `φ ≡ 0` (world +x is the heading; the channel is then the absolute column azimuth, i.e. the information the SimpleViT's 2-D positional code carries). HAA: per-room decision `φ_room ∈ {0, +90°, 180°, −90°}` from the 12 training RIRs (§4) — in the role swap the loudspeaker is the model's "receiver", so the heading is a property of the panorama location.
- **Heading-frame transform** (data tuple; `xRIR.forward` unchanged): `k = round(−φ·512/2π) mod 512`; `(depth_coord, src_local, ref_src_local) ← rotate_scene_yaw(·, ·, ·, k)` (pinned `tools/yaw_rotation.py`; active yaw by `Δ = 2πk/512`: panorama rolled by `k` columns, every xyz rotated by `Rz(Δ)`). Afterwards the heading points along +x; verified: `k = 128` maps (0, −1, 0) to (1, −6e-17, 0). Continuous headings are quantised to columns (≤ 0.35°); the four axis headings are exact and patch-aligned (`k ∈ {0, 128, 256, 384}`). `k`, `φ` and the heading-JSON sha256 are recorded in every stage `args.json` and every per-sample `meta`. `k = 0` is the identity (bit-exact, tested). **Original room-frame side labels (±y about the speaker) are computed before the transform and carried alongside**; transformed coordinates never define "side".
- **Why data level.** Reuses exp_03's tested rotation, keeps evaluators unchanged, records the heading as an explicit input, and yields the heading-frame control arms (D, F) with no new model code.

### 2.2 Encoder: `CylindricalViTOriented` (`model/cylindrical_vit_oriented.py`, subclass of the pinned `CylindricalViT`)

- Input `[B, 3, H, W]` xyz; per-column gauge alignment `Rz(−θ_c)` exactly as the parent.
- **Two constant channels** appended after alignment: `cos θ_c`, `sin θ_c` (the column's unit look direction in the heading-aligned receiver frame), broadcast over rows and batch; `register_buffer("az_channels", [2, H, W])` built from the parent's `cos_theta`/`sin_theta` **and tested independently against `convert_equirect_to_camera_coord`'s normalised horizontal look direction**.
- Patch embedding rebuilt for `patch_dim = 5·16·32 = 2560` (LayerNorm → Linear → LayerNorm); elevation-only positional code, circular relative-bias transformer, `[B, 256, 512]` tokens — inherited, so everything downstream is byte-identical to `xRIR_Cyl`.
- Symmetry (tested): tokens are **not** equivariant to a scene-only roll (the channels do not roll); under a **joint** scene-and-heading rotation by integer column offsets the two input tensors agree to float32 rounding (composed rotations differ by ≤ ~1e-6; tests use a justified tolerance, k = 0 tested bit-exact separately, arbitrary headings incl. wrap-around and rounding boundaries). Agreement of arm F with arm B outputs is descriptive, not guaranteed (the pool and coordinate branches are not equivariant; exp_03).
- Parameters: `+2·16·32·512 + 2·(2560 − 1536) = 526 336` relative to `CylindricalViT` (exact count asserted).

`model/xRIR_cyl_oriented.py`: `xRIR_CylOriented(xRIR)` swapping `source_network` (mirror of `xRIR_Cyl`); `BACKBONES_EXP06 = {**BACKBONES, "cylindrical_oriented": xRIR_CylOriented}`; `build_xrir_exp06(backbone, num_shot, **kw)`; the `simple`/`cylindrical` routes return the pinned classes (`is`), bit-identical outputs on a fixed batch.

### 2.3 Arms and contrasts

| Arm | Model / init | Frame (HAA) | Pretraining | HAA fine-tuning (seeds 0,1,2) + zero-shot | Sim k = 0 eval |
|---|---|---|---|---|---|
| A `control` | SimpleViT, exp_01 `ckpt/xRIR_simple_8_shot/epoch_12.pth` | room | exists | **exists** (exp_02, `ckpt/sim2real/control/`) | exp_05 M-tier runs if admissible (§6.3), else exp_06 run |
| B `cyl` | CylindricalViT, exp_01 `ckpt/xRIR_cyl_8_shot/epoch_12.pth` | room | exists | **exists** (exp_02, `ckpt/sim2real/cyl/`) | as A |
| C `cyl_or` | `cylindrical_oriented`, **new pretraining** (§5), seed 0 | heading | **new** (≈ 31 h) | **new** | **new** (5 seeds) |
| D `control_hf` | SimpleViT, exp_01 checkpoint | heading | exists | **new** | none |
| F `cyl_hf` | CylindricalViT, exp_01 checkpoint | heading | exists | **new** | none |

Contrasts: **C − A** (does the new model beat the published baseline as used; H1); **C − F** (channel effect at a fixed frame; H1b); **C − D** (encoder comparison at a fixed frame; descriptive); **D − A**, **F − B** (frame effects; descriptive, F − B expected small). A and B are reused read-only; the pairing is valid because exp_02 drew references per `(eval_seed 0, room, idx)` with a local generator and used the same DiffRIR test indices (verified in the round-1 review); the exp_06 summariser asserts identity of `index`/`ir_path`/`eval_seed`/`K` on every pair.

## 3. What is *not* changed

- exp_03's pinned closure (12 files; `tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged`): never edited.
- exp_05's training-closure pin: `train_xRIR_backbone.py` and every file in `tools.provenance.source_closure('train_xRIR_backbone')` stay byte-identical (exp_05's L runs launch/finalise against digest `5b2da250…`); exp_06 trains through its own entry point (§8).
- exp_02's pipeline (`sim_to_real/*.py`, `run_haa_pipeline.sh`) and record; exp_04/exp_05 tooling (`tools/exp04_*.py`, `tools/exp05_*.py`, `tools/paired_compare.py`, `tools/results_table.py`, `tools/param_curve.py`, `tools/provenance.py`): import only.
- Data caches: read only.

"No existing tracked **source** file is modified" is the rule; the notebook, `commits_*.md`, `model_comparison.md`, `approved_digests.json` fills and other record files are updated as the SOP requires, in several small commits per reviewed round.

## 4. Heading decision for HAA (`tools/exp06_heading.py`)

The round-1 review showed that a continuous cosine fit is ill-posed on the hallway (training mics lie along ±y: fit −44°, weighted mean −90°, leave-one-out range −147°…−23°) and that an energy-weighted mean conflates directivity with microphone sampling density. The estimator is therefore a **discrete axis decision with stability checks**, and the continuous fits are reported descriptively only.

Inputs per room (cache `~/data_cache/HAA_xrir/<room>/`): `rirs.npy` (training rows only), `xyzs.npy`, `speaker_xyz.npy`, `meta.json` (`train`, `sr`). **Only the 12 training mics are used.** For training mic `i`: `θ_i = atan2(y_i − y_s, x_i − x_s)`, `d_i = ‖(x_i − x_s, y_i − y_s)‖`, onset `n_i` = first sample with `|h_i| > 0.2·max|h_i|`, compensated level `L_i^(w) = 10 log10(d_i² · Σ_{n∈[n_i, n_i + w·sr)} h_i[n]²)` for two windows `w ∈ {5 ms, 50 ms}`.

- **Candidates**: the four axis headings `φ ∈ {0, +90°, 180°, −90°}` (physical prior: the loudspeaker was placed facing along a room axis; the model quantises headings anyway). A candidate is **evaluable** if ≥ 3 training mics lie within ±45° of it and ≥ 3 lie outside that sector. Its **contrast** = mean `L^(w)` inside − mean `L^(w)` outside.
- **Decision**: the evaluable candidate with the largest contrast, required to satisfy all of: contrast ≥ 3 dB (both windows); ≥ 3 dB above every other evaluable candidate (both windows); the same winner for both windows; the same winner when any single training mic is left out (12 leave-one-out refits, both windows). Otherwise **refuse** (fail-closed).
- **Descriptive extras** (reported, never decisive): the continuous cosine fit `L ≈ a + b cos(θ − φ)` on a 0.5° grid with its leave-one-out range, the energy-weighted mean direction, and per-mic `(θ_i, d_i, L_i)`.
- **Override**: `--heading-deg <value> --override-reason "<text>"` (e.g. a documented DiffRIR loudspeaker orientation) is recorded verbatim alongside the estimator's output; the decision then uses the override.
- Output `ckpt/exp06/heading/<room>.json`: `phi_deg`, `k`, `decision` (`estimated`/`override`/`refused`), contrasts per candidate and window, leave-one-out winners, the descriptive fits, sha256 of the four inputs, the tool's source-closure digest, timestamp.

The diagnostic predicts −y (`k = 128`) in all four rooms (round-1 review, continuous fits: classroom −82°, dampened −96°, hallway −90.4° by weighted mean, complex −93°). The plan does not assume it; refusals stop the affected room and are escalated (§11).

## 5. Pretraining recipe (= exp_01; declared operational differences)

`tools/exp06_train.py --backbone cylindrical_oriented --save-dir ckpt/exp06/pretrain/xRIR_cylor_8_shot --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 500 --epoch-ckpt-every 1`, `XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=8`, GPU 1 (§11).

- **Recipe parity** is checked with a normalised, type-strict schema (as exp_05): the *recipe fields* (`num_shot, max_len, lr, weight_decay, decay_epochs, lr_gamma, epochs, batch_size, accum_steps, seed, tf32`, and the M-tier ViT dims `dim 512 / depth 12 / heads 8 / mlp_dim 512` implied by the factory) must equal exp_01's; *operational fields* are excluded and listed: `backbone`, `save_dir`, `num_workers`, `log_interval`, `save_every`, `epoch_ckpt_every` (**1** here vs 5 in exp_01, so that the trainer's native `epoch_012.pth` exists — exp_01's `epoch_12.pth` was exported outside the trainer's cadence), `resume`, `no_save`, `env`, `train_batches_per_epoch`, provenance fields; yaw augmentation must be off. Disclosure: one pretraining realisation per backbone (as exp_01/02); the A/B/C training runs are unpaired realisations.
- **Arm checkpoint** = `epoch_012.pth` (state dict written by the unchanged `train_epoch`/checkpoint code at the end of epoch 12). Completion contract (`completion.json`, written by the entry point after the loop): `history.jsonl` has 12 epochs, `last.pth["epoch"] == 12 and ["batch_idx"] == 0`, `epoch_012.pth` state dict equals `last.pth["model"]` exactly, sha256 of `epoch_012.pth` recorded; the approvals (§6.4) bind that hash. Wall time: exp_01's cylindrical run took 155.8 min/epoch → ≈ 31 h; the channel adds 526 336 parameters and negligible compute.
- `provenance.json` at start: `source_closure('tools/exp06_train')` records + digest (the closure contains `train_xRIR_backbone.py` and its imports unchanged), HEAD, environment (`tools.provenance.environment()`), data inventory of the AcousticRooms mirror, the full argv and the runtime args.

## 6. Evaluation protocols

### 6.1 Gate G1 — budget triage before fine-tuning (`tools/exp06_mirror_probe.py`, GPU, minutes)

Two parts on the hallway cache panorama, run with the pretrained `epoch_012.pth` and the two exp_01 checkpoints:

1. **Legacy reproduction** (validity of the probe code): the frozen 24-query cohort per side selected by the 2026-09-14 rule (test mics sorted by room-frame y, 24 evenly spaced ranks; ids frozen in the tool), CPU, batch 8, same reference draw (`eval_seed 0`), same numerics as `side_probe.py` → `cyl` and `control` mirror cosine / −y weight share must reproduce 0.975 / 0.85 and 0.435 / 0.24 within ±0.01.
2. **Full-cohort gate**: all 185 +y test mics (room-frame label) and their exact mirrors (−x, −y, z) about the speaker; GPU; models `cyl_or` (heading frame), `cyl`, `control` (room frame), `cyl_hf` = `cyl` in the heading frame (descriptive). Statistics: pooled-feature mirror cosine; share of `Σ_t |w_i(t)|` on −y references for +y queries; signed spectral-C50 error. **Validity precondition** (anchors bracket): cosine(`cyl`) > 0.90 and cosine(`control`) < 0.60; share(`cyl`) > 0.70 and share(`control`) < 0.35 — otherwise G1 is *inconclusive* and escalated. **G1 passes iff** cosine(`cyl_or`) < 0.80 **and** share(`cyl_or`) < 0.50.

G1 is a **stopping rule for compute**, not a test of the fine-tuned hypothesis: mirror cosine and weight share are indirect indicators of channel use, and a pretrained network that ignores the channel could still learn to use it during fine-tuning. If G1 fails, fine-tuning is not launched, the experiment is recorded as *inconclusive for H1/H2* and the fallbacks in §11 go to the user. Output `ckpt/exp06/gate_g1.json` (hash-bound: checkpoint sha256s, heading-JSON sha256, cohort ids, code closure).

### 6.2 HAA (exp_02's protocol, exp_06-owned entry points)

Four rooms; DiffRIR splits (12 training RIRs; validation for checkpoint selection; test sizes 463 / 198 / 423 / 198); stage 1 on classroom + hallway + complex (`--epochs 1000 --val-every 10 --tf32`, full-batch, lr 1e-4, wd 1e-4), stage 2 per room (`--epochs 200 --val-every 2 --tf32`); test evaluation with K = 8, `eval_seed 0`, Griffin-Lim inversion, EDT / C50 / T60 on 9600-sample waveforms, T60 skipped for the dampened room; `depth_variant default`. Seeds 0, 1, 2 for each new init (C, D, F); zero-shot evaluations of C, D, F.

- **Phase policy (frozen).** Primary comparison: Griffin-Lim phase initialisation **unseeded, as in exp_02** (A/B outputs are historical; the nuisance has zero expected effect on paired means and is disclosed). **S1 sensitivity (optional, ≈ 1 h GPU):** evaluation-only re-runs of the fine-tuned checkpoints of A, B, C, D, F (3 seeds each) and their zero-shot inits with **per-query seeded** phases under `ckpt/exp06/sim2real_seeded_phase/`; historical outputs untouched; H1 recomputed there descriptively.
- **Records**: every stage/eval writes `args.json` (recipe, init sha256, `heading` {room: φ, k, json sha256}, source-closure digest of the entry point, HEAD) and every per-sample file a `meta` including `heading`, `backbone`, `checkpoint_sha256`, `frame`; room-frame side labels stored per sample.
- Outputs `ckpt/exp06/sim2real/<init>/{seed<s>/{stage1,stage2_<room>,eval}, zeroshot}` (inits `cyl_or`, `control_hf`, `cyl_hf`), mirroring exp_02.

### 6.3 Simulated-room parity at k = 0 (exp_04 protocol)

`tools/exp06_eval.py` = `tools/exp04_eval.run_exp04(args, model_factory=build_xrir_exp06, metadata=…, manifest_validator=…)` with an exp_06 parser (backbone choices from `BACKBONES_EXP06`), launched by `tools/exp06_eval_launch.py` (own `parse_args`; composes `exp04_eval_launch.execute_run`, `child_command`-equivalent for the exp_06 module, and `build_fields(module=exp06_eval)` extended with `checkpoint_role`, `checkpoint_epoch`, heading = none, registry digest — as `tools/exp05_eval.py` does). Manifests `ckpt/yaw_aug/reference_manifest_k8_seed{42..46}.json`, `--conditions P`, k = 0 only, TF32 off, canonical batch 16, per-query Griffin-Lim seed = manifest seed, full ordered unseen split (6337 queries). Outputs `ckpt/exp06/sim_eval/<arm>/seed<S>/`. For A/B: exp_05's M-tier evaluations are admitted only if they satisfy the same admission list (§7 H3); otherwise exp_06 evaluates the exp_01 checkpoints itself (+ ≈ 10 h).

### 6.4 Approvals and admission (mirrors exp_04/exp_05)

`tools/exp06_profiles.py` + `oriented_cyl_results_assets/approved_digests.json` (null template until the second reviewed commit) pin: source-closure digests of `tools.exp06_train`, `model.cylindrical_vit_oriented` + `model.xRIR_cyl_oriented`, `tools.exp06_heading`, `tools.exp06_eval`, `tools.exp06_eval_launch`, `tools.exp06_haa_finetune`, `tools.exp06_haa_eval`, `tools.exp06_mirror_probe`, `tools.exp06_summarize_haa`, `tools.exp06_bootstrap`, `tools.exp06_compare`; the pinned exp_03 evaluator digest; the `epoch_012.pth` sha256 (path, epoch); the four heading JSON sha256s. Every producer refuses (fail-closed, deviations listed) unless the recorded digests match the approved ones, the run directories are complete (`completion.json`, closed logs, exclusive output dirs, successful child exit) and inputs are bound.

## 7. Pre-registered hypotheses and decision rules

**Estimands and intervals.** HAA: per-query paired differences `Δ_{s,q} = e_{X,s,q} − e_{Y,s,q}` for fine-tuning seed `s` and test query `q` of one room and metric; point estimate = mean over `(s, q)`; intervals from `tools/exp06_bootstrap.py`, an exp_06-owned, **alpha-aware** percentile bootstrap implementing exp_02's two schemes (query-cluster: the rows of one query across seeds resampled together; two-way: queries and seeds resampled with multiplicity weights), fixed seed, `n_boot = 10 000` for nominal 95 % (must reproduce exp_02's canonical intervals exactly — regression test) and `n_boot = 50 000` for adjusted tails; convergence check with a second bootstrap seed (half-width change ≤ 10 %). Verdicts use the two-way interval. Simulation: exp_04's estimand `ρ = [mean_q(mean_s e_{C,s,q}) − mean_q(mean_s e_{B,s,q})] / mean_q(mean_s e_{B,s,q})` (queries averaged over the five evaluation seeds first; the ratio recomputed inside each paired query-level resample; room-cluster interval reported), one-sided 97.5 % upper bound, via `tools/paired_compare.{five_seed_mean, cell_mask, rho_bootstrap, h1_verdict}`.

**Paired cohort and invalidity policy.** For each cell, the cohort is the set of queries with finite values in every compared run (both arms, all seeds); excluded counts are reported per arm and seed. A non-inferiority verdict (H1) is **void** if the treatment arm's invalid count exceeds the control's by more than 1 % of the room's test size, or if the cohort is smaller than 99 % of the test split.

- **G1** — §6.1 (pass / fail / inconclusive).
- **H1 (primary, confirmatory)** — hallway C50 error, **C − A**: two-way 95 % upper bound **< +0.23 dB**. Margin frozen from the canonical exp_02 values: 0.23 dB ≈ 20 % of the control's hallway C50 mean (1.1355 dB → 0.227) ≈ 25 % of the exp_02 gap (0.9162 dB → 0.229), i.e. about three quarters of the deficit removed; it exceeds exp_02's realised two-way half-width (0.179 dB), which is a reference for scale, not a power guarantee for C − A.
- **H1b (channel effect)** — hallway C50, **C − F**: two-way 95 % upper bound < 0.
- **H2 (descriptive screen, not a parity claim)** — all 11 room × metric cells for C − A: nominal and Bonferroni-adjusted two-way intervals (two-sided level 1 − 0.05/11, tails at 0.05/22 and 1 − 0.05/22, `n_boot = 50 000`); cells with adjusted lower bound > 0 are reported as **detected harm**, cells with adjusted upper bound < 0 as **detected improvement**, all others as **no detected difference** — never as "non-inferior". The positive conclusion of the experiment rests on H1 alone.
- **H3 (sim parity, k = 0, K = 8)** — C vs B: `ρ` upper bound **< +3 %** for EDT and for C50 (exp_04's H1 rule). C vs A descriptive with the same machinery. **Admission** (fail-closed): checkpoint role/hash/epoch bound to the approvals; `gl_seed == manifest_seed`; batch 16; TF32 off; condition P; k = 0; full ordered 6337-query split; valid `completion.json` and output hashes; approved evaluator/entry/producer closures; the registered finite-data cohort rule (`cell_mask`) and the convergence rule.
- **D (descriptive)** — C − D, D − A, F − B on all cells; zero-shot side-split table (room-frame −y / +y) for every arm and room; G1 statistics for `cyl_hf`.

**Overall reading.** "The orientation-conditioned cylindrical pipeline removes the real-room deficit" iff G1 passes, H1 holds (not void) and H1b holds; H2 describes the remaining cells; H3 states whether it costs simulated accuracy. Anything else is reported as found; margins and rules change only through §10a amendments logged before the affected runs.

## 8. Code (Coder = OpenAI Codex `gpt-6-astra`; TDD; Fable 5.1 review per round; worktree branch `exp06-window`)

Constraint for every round: **no existing tracked source file is modified** (§3); exp_06-owned parsers, serialisers and thin entry points compose the existing numerical helpers; **no module-namespace mutation**. Python 3.8 (`xRIR` env). Tests in `tests/test_exp06_*.py` (CPU unless marked GPU; GPU tests skip without CUDA). Several small commits per round are fine.

**Round 1 — encoder, factory, heading tool**

| File | Content | Tests (first) |
|---|---|---|
| `model/cylindrical_vit_oriented.py` | `azimuth_channels(H, W, dtype)`; `CylindricalViTOriented(CylindricalViT)` (`__init__` rebuilds `to_patch_embedding` for 5 channels, registers `az_channels`; `forward`) | token shape; `az_channels` equals the parent buffers **and** the camera conversion's normalised horizontal look direction (independent AST-free reimplementation in the test); exact parameter count (+526 336); non-equivariance under a 32-column scene roll (parent: permutation to 1e-6; child: not); joint scene+heading rotation over integer offsets incl. wrap-around/rounding boundaries within a float32 tolerance, k = 0 bit-exact; mirror discrimination on a synthetic 180°-symmetric scene; dtype/device propagation; `in_channels != 3` refused |
| `model/xRIR_cyl_oriented.py` | `xRIR_CylOriented`, `BACKBONES_EXP06`, `build_xrir_exp06` | pinned classes returned by identity for `simple`/`cylindrical` with bit-identical outputs on a fixed batch (CPU encoder path; full forward GPU-skipped); forward shapes (GPU); param delta; unknown name refused; kwargs pass-through |
| `tools/exp06_heading.py` | `early_level(rir, sr, window_s)`, `candidate_contrasts(theta, levels, candidates, half_width=45°, min_in=3, min_out=3)`, `decide_heading(...)` (two windows, margins, leave-one-out), `continuous_fit(theta, level)` (descriptive), `mean_direction`, `heading_roll_k(phi_deg, W=512)`, `HeadingFrameDataset(HAADataset)` (subclass; preserves `.items`, `.data`, reference draw and order; applies `rotate_scene_yaw` per item; exposes room-frame side labels), `write/read_heading_json`, CLI with `--heading-deg/--override-reason` | real training-only readback on the four rooms (decision, contrasts, leave-one-out winners; refusal path exercised on a synthetic layout); non-uniform-angle synthetic cases where the weighted mean is biased but the axis decision is correct; refusal when contrast < 3 dB, when windows disagree, when a leave-one-out flips; override recorded; `heading_roll_k(−90) == 128`, `(0) == 0`, `(180) == 256`, `(90) == 384`, wrap-around; dataset k = 0 bit-identical to `HAADataset`; k = 128 equals `rotate_scene_yaw` on the unsqueezed tuple, audio untouched, `.items` and reference indices unchanged, side labels from room-frame coordinates; JSON round-trip with input sha256s |

**Round 2 — entry points (compose, never mutate)**

| File | Content | Tests |
|---|---|---|
| `tools/exp06_train.py` | own `parse_args` (the trainer's flags with `--backbone` from `BACKBONES_EXP06`; `--yaw-aug 1` refused); own `main` composed from `train_xRIR_backbone.{seed_everything, seed_worker, train_epoch, test_epoch, save_checkpoint}`, `xRIR_Dataset`, the trainer's optimizer/scheduler construction, model from `build_xrir_exp06`; writes `provenance.json` at start and `completion.json` (§5) at the end | shared flag defaults equal the trainer's (introspected); `simple` model state bit-identical to the trainer's `build_model` for the same seed; recipe schema check (§5) passes on a written `args.json` and fails on a changed recipe field; completion contract refusals (missing epoch, `batch_idx ≠ 0`, state-dict mismatch); GPU parity smoke on the ladder (not pytest): both entry points, `--backbone simple --max-train-batches 3 --no-save`, TF32 off → identical printed losses |
| `tools/exp06_eval.py` | exp_06 parser (`build_parser`), `run_exp06` = `run_exp04(model_factory=build_xrir_exp06, metadata, manifest_validator)`, `build_fields` extension (as exp_05) | factory routing; metadata fields present; manifest mismatch refused (reuse exp_04 fixtures) |
| `tools/exp06_eval_launch.py` | own `parse_args`; composes `exp04_eval_launch.{execute_run, build_fields(module=exp06_eval), child_environment}`; `--bind-input` for `train_completion`/`heading` | command serialisation golden; refusal of an unlisted binding; dirty-tree refusal |
| `tools/exp06_haa_finetune.py` | own parser (exp_02's flags + `--backbone` from `BACKBONES_EXP06`, `--heading-json-dir`, required for `cylindrical_oriented`, optional otherwise); dataset = `HeadingFrameDataset` when a heading dir is given else `HAADataset`; loop composed from `finetune_haa.{compute_loss, evaluate, load_model_state}` and the same optimizer/scheduler; own serialiser for `args.json`/`history.jsonl`/`summary.json` (exp_02 fields + heading/provenance) | fail-closed without heading for the oriented backbone; with no heading and a pinned backbone the dataset items are bit-identical to `HAADataset` and the model class is the pinned one; `args.json` carries heading/k/sha256; 2-epoch CPU-free dry run of the serialiser |
| `tools/exp06_haa_eval.py` | own parser; loop composed from `eval_unseen.{Evaluator, griffin_lim}`, `utils.spec_utils` losses, `HeadingFrameDataset`; per-sample `meta` incl. heading/side labels; `--gl-seed-per-query` flag for S1 (default off) | routing; meta fields; `.items` order preserved; per-query seeding reproducible when on and absent when off |
| `tools/exp06_haa_pipeline.sh` | exp_02's queue for inits `cyl_or`, `control_hf`, `cyl_hf`, `zeroshot`; `--dry-run` | `bash -n`; dry-run golden (paths, flags, seeds, heading dir) |

**Round 3 — probe, statistics, producers, approvals**

| File | Content | Tests |
|---|---|---|
| `tools/exp06_mirror_probe.py` | legacy-cohort reproduction (frozen ids, CPU batch 8) + full-cohort gate (§6.1); `g1_decision(stats, anchors)` → pass/fail/inconclusive; `ckpt/exp06/gate_g1.json` | bookkeeping on tiny synthetic models; decision boundaries incl. inconclusive; side labels from room-frame coordinates after transform; hash binding |
| `tools/exp06_bootstrap.py` | alpha-aware two-scheme paired bootstrap (§7); convergence check | exact reproduction of exp_02's canonical 95 % intervals (skipped if `ckpt/sim2real` absent); adjusted tails at 0.05/22; multiplicity weights; degenerate inputs refused |
| `tools/exp06_summarize_haa.py` | `load_runs` for `ckpt/sim2real` (A, B) and `ckpt/exp06/sim2real` (C, D, F); exp_06 tables (EXPECTED_RUNS, BACKBONE_OF, INIT_OF, FRAME_OF, heading k per room); completeness (exp_02's checks + heading equality within an arm + closure equality + `completion` presence); pairs H1/H1b/H2/D; cohort/invalidity policy; side-split table; `--json/--summary` hash-bound | completeness refusals (missing seed, wrong k, wrong backbone, wrong frame, wrong indices, missing heading); pairing assertions; invalidity-void rule; margin constant frozen; regression against exp_02's canonical `cyl − control` rows |
| `tools/exp06_compare.py` | H3 from `exp06_eval` outputs (and admissible exp_05 M evaluations), admission list §7 | refusals: changed input, stale output, wrong role/epoch, incomplete run, wrong `gl_seed`, wrong split size; verdict boundaries; determinism |
| `tools/exp06_profiles.py` + `oriented_cyl_results_assets/approved_digests.json` | §6.4 (null template) | null template refused in production and listed under `--exploratory`; filled template validated (64-hex) |
| `oriented_cyl_results_assets/make_results_{html,md}.py` (Planner one-offs, reviewed) | render only from `ckpt/exp06/stats.json`, `h3.json`, `gate_g1.json` | structural checks; refuse anything else |

## 9. Validation ladder, parity audit, acceptance criteria

1. Static: `py_compile`, `bash -n`, `git diff --check`, `pytest tests/test_exp06_*.py tests/test_provenance.py -q` (pinned-closure test green).
2. Tiny synthetic forward of the encoder and factory (CPU).
3. Real-data readback: heading decisions for the four rooms (inspected, logged); `HeadingFrameDataset` k = 0 / k = 128 checks on the hallway cache.
4. GPU smokes (GPU 0, co-tenant with the peer's evaluation chain, ≤ 3 GB, ≤ 8 CPU threads, never 03:00–04:30 Sep 15, never GPU 1 — peer session's rules): trainer parity smoke (§8 round 2), `exp06_train --backbone cylindrical_oriented` 3 batches `--no-save`; HAA smokes write to `ckpt/exp06/_smoke/` (2 epochs stage 1, `--max-samples 4` eval) and are deleted afterwards.
5. **Bounded fit/timing probe** on the target card when it is free: 200 batches at the full 32 × 2 recipe, `--no-save`, peak memory and s/iter recorded; full launch only if the extrapolated epoch time ≤ 1.15 × 156 min.
6. Parity audit (notebook): factory identity for pinned backbones; `az_channels` vs camera conversion; recipe schema vs exp_01; heading transform identities; HAA wrapper default path vs exp_02 items; legacy-cohort G1 reproduction.
7. Pre-launch acceptance criteria written in the notebook (commit SHA, GPU 1, batch 32 × 2, TF32 on, 12 epochs, `epoch_ckpt_every 1`, ≥ 1 optimizer step, epoch-1 time ≤ 1.15 × 156 min, finite loss; for HAA: every stage `summary.json`, every eval `metrics_/per_sample_` with heading recorded; `completion.json` present).

## 10. Deliverables

`ckpt/exp06/{pretrain/xRIR_cylor_8_shot, heading/*.json, gate_g1.json, sim2real/**, [sim2real_seeded_phase/**], sim_eval/**, stats.json, summary.txt, h3.json}`; record: `oriented_cyl_params_set_up.md`, `oriented_cyl_command.md`, logs, `oriented_cyl_results.md`, `oriented_cyl_analysis.md`, `oriented_cyl_01_results.html` (+ assets), `commits_oriented_cyl.md`, filled `approved_digests.json` (second reviewed commit); a `model_comparison.md` row if the table format admits it.

### 10a. Amendments during implementation (logged in the notebook)
(none yet)

## 11. Cost, schedule, risks, decisions

**GPU cost.** Pretraining ≈ 31 h. Fit/timing probe ≈ 10 min. G1 ≈ 10 min. HAA: 9 fine-tuning jobs (C, D, F × 3 seeds) × ≈ 50 min + zero-shots ≈ 8 h; S1 sensitivity ≈ 1 h. Sim: 5 × ≈ 1 h for C (+ ≈ 10 h if A/B must be re-evaluated). Total ≈ 46–56 GPU-hours.

**Schedule (agreed with the peer session `xrir-code-25` on 2026-09-14 20:30; contingent on actual hand-over).** GPU 1 frees ≈ Sep 16 13:00 (after exp_05 L_cylindrical and its evaluations; the peer messages when) → fit probe → pretraining Sep 16 ≈ 13:30 → Sep 17 ≈ 21:00 → G1 → HAA (9 jobs, ≈ 8 h; GPU 0 frees ≈ Sep 17 04:00 and can take half) → sim eval → results/analysis/page ≈ Sep 19. Code merges into `main` only in the peer's windows (W1 Sep 15 04:30–20:00; Sep 16 14:00–20:00), each preceded by a message. ICLR deadline 2026-09-24.

**Risks.** (1) G1 fails or is inconclusive → no fine-tuning; fallbacks (user decision): directivity-aware pretraining augmentation (synthetic source-directivity gain on targets and references vs a random heading, making the heading acoustically relevant) or a fine-tune-only orientation feature. (2) G1 passes but H1 fails (12 RIRs do not exploit the channel) → reported. (3) A heading refusal in some room → override or exclusion, escalated. (4) Schedule slip upstream. (5) One pretraining realisation per backbone.

**Decisions for Yixun (before approval).** (a) Arms D and F (recommended: both; ≈ 5 GPU-hours; they attribute the effect to frame vs channel). (b) Heading source: the axis decision of §4 (default), or a documented DiffRIR loudspeaker orientation supplied as override. (c) GPU 0 smokes co-tenant now under the peer's rules (recommended: yes). (d) H1 margin +0.23 dB (recommended). (e) S1 shared-phase sensitivity pass (recommended: yes, ≈ 1 h). (f) If exp_05's M-tier evaluations are not admissible in time, spend ≈ 10 h re-evaluating A/B for H3, or report H3 against exp_03's paired k = 0 numbers descriptively (recommended: re-evaluate only if GPU time is free before Sep 19).

## 12. v2 changelog (responses to the round-1 Codex plan review)

1. **Heading estimator (blocker 1)** → §4 rewritten: discrete axis decision with sector contrasts, two windows, 3 dB margins, leave-one-out stability, refusal; continuous fit and weighted mean descriptive only; identifiability caveat; override path; real-readback + non-uniform synthetic + leave-one-out tests.
2. **G1 anchors (blocker 2)** → §6.1 split into legacy-cohort reproduction (frozen 24 ids, CPU, batch 8, ±0.01) and a full-cohort gate with a bracketing precondition and an *inconclusive* outcome; room-frame side labels carried through the transform (§2.1).
3. **Wrapper interfaces (blocker 3)** → §8 rewritten: exp_06-owned parsers/serialisers/entry points composing the existing helpers; no namespace mutation; `HeadingFrameDataset` subclass preserving `.items`/reference order; explicit tests for CLI routing, metadata emission, default-path identity.
4. **Provenance/admission (blocker 4)** → §6.4 approvals JSON + profiles; `tools/exp06_eval_launch.py`; H3 admission list (§7); completion contracts; refusal tests.
5. **Epoch-12 checkpoint (blocker 5)** → §5: `--epoch-ckpt-every 1`, native `epoch_012.pth`, completion contract (equality with `last.pth["model"]`, epoch 12, `batch_idx 0`), hash bound in approvals; declared as an operational difference from exp_01.
6. **H2 semantics (blocker 6)** → §7: H2 descriptive ("detected harm / improvement / no detected difference"), positive conclusion on H1 (+ H1b) only; paired-cohort and invalidity-void policy.
7. **Estimands (blocker 7)** → §7: exp_04's `ρ` written out and frozen; exp_06-owned alpha-aware bootstrap with exact legacy reproduction test, 50 000 resamples for adjusted tails, convergence check.
8. **Controls (should-fix 8)** → arm F `cyl_hf` added; contrasts C − F (H1b), C − D, D − A, F − B; G1 declared a compute stopping rule; mechanism claims narrowed in §1 and attribution to the whole pipeline.
9. **Phase policy (should-fix 9)** → §6.2 frozen: primary unseeded as exp_02; optional S1 shared-phase sensitivity with fresh evaluation-only copies.
10. **Symmetry tests (should-fix 10)** → §2.2/§8: tolerance-based joint-rotation tests, bit-exact k = 0, independent camera-conversion check, quantisation stated, F/B agreement descriptive.
11. **Recipe parity / ladder (should-fix 11)** → §5 normalised schema with operational exclusions, `PYTHONHASHSEED`/threads pinned, unpaired realisations disclosed; §9 bounded 32 × 2 fit/timing probe, storage-light HAA smokes, schedule contingent on hand-over; §3 clarifies permitted record updates and multiple commits per round.
12. **Arithmetic (nit 12)** → margin wording (≈ 20 %, ≈ 75 %), parameter delta 526 336.
