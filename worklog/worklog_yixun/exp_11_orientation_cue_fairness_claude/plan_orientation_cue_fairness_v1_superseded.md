# Plan — orientation_cue_fairness (exp_11): does CERPA's advantage on HAA survive when the vanilla xRIR gets the same loudspeaker-orientation cue?

Version 1, 2026-09-26. Planner: Claude Fable 5.1. Requested by Yixun 2026-09-26 (query log #1): "Fine-tune and evaluate xRIR / xRIR + orientation cue / Yaw-Augmented xRIR + orientation cue / CylindricalViT + orientation cue under the same xRIR protocol; the key controlled comparison is xRIR + orientation cue vs CylindricalViT + orientation cue."

## 1. Question

In the paper's HAA table the oriented CylindricalViT (exp_06 arm C, "CERPA / CylindricalViT") is fine-tuned with an explicit loudspeaker-facing cue while the vanilla xRIR baseline is not. Is the hallway improvement attributable to the extra conditioning information rather than to the encoder? The controlled test holds the conditioning information fixed (every arm receives the loudspeaker heading in the same form) and varies only the model.

## 2. Design

### 2.1 What "the orientation cue" is, mechanically (from exp_06, `plan_oriented_cyl.md` §2, code survey 2026-09-26)

The cue has two parts, both applied at fine-tuning and evaluation on HAA:
1. **Heading frame (data level).** The whole scene (depth panorama, query-source and reference-source coordinates) is rotated by the roll `k = 128` so that the loudspeaker's inferred acoustic axis (−y in all four rooms, `ckpt/exp06/heading/*.json`, decision `estimated`, admissibility `confirmatory`) points along +x (`tools/exp06_heading.py::HeadingFrameDataset` → pinned `tools/yaw_rotation.py::rotate_scene_yaw`). Any model fine-tuned in this frame receives the heading: "the loudspeaker faces the panorama's +x column".
2. **Azimuth channels (model level, `CylindricalViTOriented`).** Two constant input planes `cos θ_c, sin θ_c` of the camera-frame column azimuth, concatenated after the cylindrical gauge alignment. They are a pure function of the column index (no batch, scene or heading dependence; non-persistent buffer). They restore *absolute* azimuth to an encoder whose gauge alignment and circular relative bias otherwise make its tokens roll-equivariant; that is what lets the heading frame act as a cue for the cylindrical model (exp_06 H1b: C − F −2.01 dB hallway C50; mirror cosine 0.944 → −0.156).

For the **SimpleViT** the second part is already present: its fixed 2-D sinusoidal positional code (`model/simple_vit.py::posemb_sincos_2d`) encodes the absolute token column, i.e. the same azimuth the channels carry (the patch embedding is LayerNorm → Linear over the flattened patch, so two constant per-pixel planes reduce to a per-column constant in the patch vector — an additional absolute column code, redundant with the positional code up to the first LayerNorm's statistics). Hence, information-wise, **"xRIR + orientation cue" = the SimpleViT fine-tuned and evaluated in the heading frame**, which exp_06 ran as arm D (`control_hf`), and **"CylindricalViT + orientation cue" = arm C** (`cyl_or`). Architecturally literal versions (SimpleViT with the two planes, pretrained with them) are Phase 2 below; their predicted outcome is pre-registered.

A second consequence, stated up front: yaw augmentation (exp_04/exp_09's E) trains the SimpleViT to be *invariant* to the scene roll relative to its pinned position code, i.e. to ignore exactly the absolute azimuth through which the heading frame delivers the cue. A perfectly roll-invariant model cannot use the heading frame; the yaw-augmented model is approximately invariant, so its heading-frame arm is expected to differ little from its room-frame arm. The cylindrical encoder resolves this tension (relative-geometry tokens + an explicit anchor that does not roll with the scene); that resolution is the CERPA claim under test.

### 2.2 Arms (HAA protocol identical to exp_02/exp_06/exp_09: stage 1 class_room + hallway + complex_room 1000 epochs / val 10, stage 2 per room 200 / val 2, AdamW 1e-4, wd 1e-4, ×0.1 per 50 epochs, full batch, TF32, selection on the DiffRIR validation split, K = 8, `eval_seed` 0, unseeded Griffin-Lim; three fine-tuning seeds 0/1/2 + one zero-shot job per arm)

| Arm | Requested variant | Model / init (pretraining) | Frame | Status |
|---|---|---|---|---|
| A `control` | **xRIR** | SimpleViT, exp_01 same-budget control `ckpt/xRIR_simple_8_shot/epoch_12.pth` | room | exists (exp_02) |
| A′ `released` | xRIR (authors' checkpoint) | released `checkpoints/xRIR_unseen.pth` | room | exists (exp_02; the table's current "xRIR" row) — re-read if the legacy receipt admits it, else reported from exp_02's canonical JSON as a labelled external row |
| D `control_hf` | **xRIR + orientation cue** | SimpleViT, exp_01 control | heading | exists (exp_06) |
| E `yawaug` | yaw-augmented xRIR (no cue) | yaw-augmented SimpleViT, exp_04 `ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth` | room | exists (exp_09) |
| **G `yawaug_hf`** | **Yaw-Augmented xRIR + orientation cue** | yaw-augmented SimpleViT, exp_04 checkpoint (same resolver as exp_09) | **heading** | **new (Phase 1, ≈ 3 GPU-h)** |
| C `cyl_or` | **CylindricalViT + orientation cue** | `cylindrical_oriented`, exp_06 pretraining `ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth` | heading | exists (exp_06) |
| B `cyl`, F `cyl_hf` | CylindricalViT without / with the frame (no channels) | exp_01 cylindrical checkpoint | room / heading | exist (exp_02 / exp_06); supplementary |
| **H `simple_or`** | xRIR + literal azimuth channels | **new** `simple_oriented` (SimpleViT + the two planes, `channels = 5`), pretrained from scratch with exp_06's recipe (= exp_01's; 12 epochs, seed 0, ≈ 31 h) | heading | **Phase 2 (optional, ≈ 34 GPU-h)** |
| **I `simple_or_yaw`** | Yaw-Augmented xRIR + literal azimuth channels | **new** `simple_oriented` pretrained with `--yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512` (exp_04's recipe; ≈ 28 h) | heading | **Phase 2 (optional, ≈ 31 GPU-h)** |

Zero-shot jobs evaluate each init without fine-tuning in the arm's frame. Every new arm is produced by `tools/exp06_haa_pipeline.sh` with `EXP06_HAA_OUT=ckpt/exp11/sim2real` and `EXP06_HAA_RECORD=<this record>`, each child finalised by `tools/exp06_finalize.py` (admissible_arm, blob-at-commit approvals, heading binding); the summariser re-reads A, B, C, D, E, F from their existing completions exactly as exp_09 re-read A, B, C.

### 2.3 Why Phase 1 answers the question and Phase 2 is a confirmation

- The key controlled comparison **C vs D** (both in the heading frame; identical conditioning information; only the encoder differs) already exists as exp_06's descriptive contrast `cyl_or − control_hf` (hallway C50 −0.629 dB, nominal two-way [−0.764, −0.500]). exp_10 re-reports it as the key row; **no new inference is claimed for it** (the data were seen before this plan was written).
- The genuinely new evidence is arm **G**: the strongest SimpleViT that can be given the cue without an architecture change (roll-robust by pretraining, heading delivered by the frame). Its contrasts are pre-registered below.
- Phase 2 tests the redundancy argument of §2.1 directly with the literal channel arms; predictions: **H ≈ D** and **I ≈ G** (equivalence within ±0.23 dB on hallway C50). If either prediction fails, the literal arms replace D/G in the paper's table and the analysis says why.

## 3. Pre-registered contrasts and decision rules (exp_06 §7 machinery; `tools/exp06_bootstrap.py`; two-way (seed × query) and query-cluster percentile bootstraps, `n_boot` 10 000, adjusted tails 50 000, seeds 0/1, convergence tolerance 0.10 with one quadrupling, void rule 1 % / 99 %, alpha 0.05)

Estimand per cell: paired per-query differences over the three fine-tuning seeds and the room's DiffRIR test queries; point estimate = mean; primary interval = converged seed-0 two-way 95 %.

**Phase 1 (arm G).** Primary cell hallway C50 unless stated.
- **N1** — G − D (yaw-augmented + cue vs vanilla + cue): `category` ∈ {detected harm, detected improvement, no detected difference} (two-sided). Reading: does roll-robust pretraining let the SimpleViT use the heading frame better than the vanilla one (D's deficit vs A being a frame-shift effect)?
- **N2 (key new fairness test)** — C − G (CERPA + cue vs the best cue-equipped SimpleViT): `category` plus `non_inferior_at_margin` of G against C at +0.23 dB (G non-inferior iff the upper bound of G − C < +0.23 dB). CERPA's advantage "survives the fairness control" iff N2 = detected improvement for C **and** G is not non-inferior.
- **N3** — G − E (the frame's effect on the yaw-augmented model): `category`; predicted no detected difference (invariance argument, §2.1).
- **S1** — screens over all 11 room × metric cells for G − D, C − G and G − E, Bonferroni-11 adjusted two-way intervals (labels only).
- **R1** — re-reported existing contrasts, labelled "from exp_06 / exp_09 canonical JSON, no new inference": C − D (key controlled comparison), D − A (frame alone, vanilla), C − A, C − F, E − A.
- **Descriptive**: zero-shot room-frame side split (−y / +y) of G, E, D, A, C on every room; G's per-seed values.

**Phase 2 (arms H, I), if run.**
- **P1** — H − D and **P2** — I − G: `equivalent_at_margin` (two one-sided tests: the two-way interval lies within [−0.23, +0.23] dB) on hallway C50; screens as in S1.
- **P3 (literal key comparison)** — C − I and C − H: `category` + non-inferiority of the SimpleViT arm at +0.23 dB, same rule as N2.
- Sim k = 0 parity for H and I (exp_04 protocol, five seeds, `tools/exp06_eval_launch.py` route) is **not** required for the HAA question; it is run only if the arms enter the paper's simulated table (Yixun's decision (e)).

**Overall reading.** The reviewer's premise ("the improvement comes from the extra conditioning") is refuted if D − A is not an improvement (it is a detected harm in exp_06) and N2 is an improvement for C. Absolute numbers remain non-comparable with the paper's Table 2 (test-split definition and selection leakage there; exp_02).

## 4. Code (Coder = Claude Opus 5, worktree `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`; TDD red-first; commits ≤ 200 lines; Codex code review at xhigh; the SOP's closure rules)

**Frozen (never edited; the closure-pin tests and the per-arm identity checks would fail):** every `model/*.py` file, `sim_to_real/haa_dataset.py`, `tools/exp06_heading.py`, `tools/yaw_rotation.py`, `tools/yaw_aug.py`, `tools/provenance.py`, `tools/exp04_profiles.py`, `tools/exp06_approvals_api.py`, `tools/exp06_profiles.py`, `tools/exp06_{eval,eval_launch,compare,mirror_probe,probe_align,bootstrap,haa_finetune,haa_eval,train,recipe,smoke}.py`, `tools/exp06_launch.sh`, `train_xRIR_backbone.py`, the exp_03-pinned evaluators and dataset module. `BACKBONES_EXP06` is never mutated (its digest is part of every completed child's registry check).

**Round 1 (Phase 1; small).**
1. `tools/exp06_haa_pipeline.sh`: init `yawaug_hf` (backbone `simple`, checkpoint `${EXP09_YAWAUG_CKPT}` resolved and admitted through the existing `exp04_aug_checkpoint` branch of `APPROVALS_PY`, frame **heading**), `yawaug_hf:zeroshot`; log prefix for records containing `exp_10`. Tests extend `tests/test_exp06_haa_pipeline.py::INITS` and the golden dry-run.
2. `tools/exp06_summarize_haa.py`: `ARMS['yawaug_hf']` (experiment `exp11`, root `ckpt/exp11/sim2real/yawaug_hf`, frame heading, init identity = exp_04's approved aug checkpoint), `EXPERIMENTS["exp11"]` (arms A, B, C, D, E, F, G; decisions N1/N2/N3 with the classified two-sided category and N2's margin; screen S1; descriptive R1 rows sourced from the arms' completions with the "no new inference" label; outputs `ckpt/exp11/{stats.json,summary.txt}`), `--exp11-root`, `check_output_paths` for exp10. Tests extend the EXPERIMENTS freeze and add exp10 cases (arm G admitted only in the heading frame with the exp_04 identity; a room-frame G refused; the exp06/exp09 outputs byte-identical to before).
3. `tools/exp06_finalize.py`: nothing new expected (the resolver and the heading binding exist); if a run-type or resolver change is needed it goes here (exp_09 precedent), followed by a `finalize` re-fill.
4. Record generators `orientation_cue_fairness_results_assets/make_results_{md,html}.py` (render only from `ckpt/exp11/stats.json`; paper-facing table with the four requested rows + supplementary rows) and `planner_probes/exp11_runbook.sh` (merge | refill | suite | dryrun | smoke | launch | summarize | results).

**Round 2 (Phase 2; larger; may proceed while GPUs are busy).**
5. `model/simple_vit_oriented.py`: `SimpleViTOriented(SimpleViT)` — `channels = 5`, the two azimuth planes re-derived locally from the column grid (tested equal to `convert_equirect_to_camera_coord`'s horizontal look direction and to `CylindricalViTOriented.az_channels`), `forward = cat([img, az]) → SimpleViT.forward`; parameter delta asserted (`2·16·32·512 + 2·2·16·32 = 526 336`… computed for `patch_dim` 2560 vs 1536 with LayerNorm affine terms — the test states the exact count).
6. `model/xRIR_simple_oriented.py`: `xRIR_SimpleOriented(xRIR)` swapping `source_network` with the 256-token pool guard; **new registry** `BACKBONES_EXP11 = {**BACKBONES_EXP06, 'simple_oriented': …}` and `build_xrir_exp11`; the three inherited routes return the pinned classes (`is`), bit-identical outputs on a fixed batch.
7. `tools/exp11_train.py` (the exp_06 loop on `BACKBONES_EXP11`, `--yaw-aug 0|1` with the exp_04 constraints, own `registry_sha256`), `tools/exp11_recipe.py` (exp_06's schema with `yaw_aug ∈ {0, 1}` as a *declared* field: arm H must equal exp_01's recipe with `yaw_aug 0`; arm I must equal exp_04's with `yaw_aug 1, yaw_aug_seed 0, yaw_aug_width 512`), `tools/exp11_launch.sh` (smoke | probe | full | finalize; attempt roots `ckpt/exp11/pretrain/xRIR_simpor_8_shot`, `…/xRIR_simpor_yawaug_8_shot`; promotion `final → attempt_<UTC>`; exclusive-card census; 36 h ceiling), `tools/exp11_haa_finetune.py` / `tools/exp11_haa_eval.py` (thin wrappers of the exp_06 entry points on the new registry, own run types), finalizer run types for them (in `tools/exp06_finalize.py`), pipeline inits `simple_or` / `simple_or_yaw` (frame heading; checkpoints admitted by an exp_10 artifact key), summariser arms H, I and the Phase-2 decisions (P1–P3, `equivalent_at_margin`), `tools/exp11_profiles.py` for the exp_10 code/artifact keys read by the finalizer (the exp_06 approvals API stays byte-identical), record approvals `orientation_cue_fairness_results_assets/approved_digests.json` (all-null until the reviewed commit).
8. Tests for every acceptance case in §5; the full CPU suite green.

## 5. Validation ladder and acceptance

1. Static + unit tests (`CUDA_VISIBLE_DEVICES=''`). Round 1: G admitted only with the exp_04 identity in the heading frame; exp06/exp09 summaries byte-identical; golden dry-run of `yawaug_hf:0 yawaug_hf:zeroshot`. Round 2: encoder equalities and counts; registry isolation (`BACKBONES_EXP06` unchanged, its digest unchanged); recipe schema accepts exactly arms H and I; launcher golden argv; wrappers refuse the exp_06 run types.
2. Pipeline dry run and a bounded heading-frame smoke of G on the card (2 epochs class_room + 4-sample hallway eval, exploratory receipts) before the queue — as in exp_09.
3. Codex code review → merge (peer-coordinated) → approvals re-fill (only the moved keys, listed in advance) → CPU suite → smoke → launch.
4. Phase 2 pretraining: fit/timing probe (10 min) → `full` on an exclusive card → finalizer completion contract (exit 0, log closed, revalidation, recipe schema, `epoch_012.pth == last.pth["model"]`) → approvals pin → HAA queue.
5. Acceptance of a run = exp_06's (every stage `summary.json`, every eval with frame `heading` and the bound heading, `completion.json` admissible, recipe fields equal to the registered recipe).

## 6. Deliverables

`ckpt/exp11/{sim2real/**, stats.json, summary.txt}` (+ `pretrain/**` if Phase 2 runs); record `worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/` (this plan, query log, worklog, command record, commit index, params set-up, Codex reviews, Coder reports, logs, `orientation_cue_fairness_results.md`, `orientation_cue_fairness_analysis.md`, `orientation_cue_fairness_01_results.html` + generators); a paper-ready LaTeX block with the four requested rows (xRIR; xRIR + cue; Yaw-Aug xRIR + cue; CylindricalViT + cue) and the supplementary rows, sourced from `ckpt/exp11/stats.json`.

## 7. Cost, schedule, risks

**GPU cost.** Phase 1: 3 × ≈ 55 min + zero-shot ≈ 6 min ≈ 2.9 h [4 h]; smoke 3 min. Phase 2: pretraining H ≈ 31 h [36 h] and I ≈ 28 h [36 h] (exp_04's yaw-augmented run took 27.4 h), HAA 2 × 2.9 h [8 h], probes 20 min. Total Phase 1 ≈ 3 GPU-h; Phase 2 ≈ 65 GPU-h expected [≈ 80 h].

**Schedule (2026-09-26 17:45, owner's ETAs from session cylindrical-dinov3-0a).** Both cards are held by the FLAC exp_13 tier-L param-curve trainings, commissioned by Yixun to run to 40 k steps: the first card frees ≈ Sep 29 12:30, the second ≈ Oct 3; only co-tenant work (≈ 4 GB fine-tunes, checkpoint-evaluation watcher) is queued behind them; the owner messages when a card frees. Phase 1 needs an exclusive card too (stage-1 fine-tuning takes ≈ 36 GB). Round 1 code + review Sep 26–27 (no GPU); Phase 1 queue on the first free card (≈ Sep 29) → results the same day. Round 2 code + review Sep 27–29. Phase 2: H on GPU 0 after Phase 1 (≈ Sep 29 → Oct 1), I on GPU 1 when free (≈ Oct 3 → Oct 4) or sequentially on GPU 0 (→ Oct 2); HAA queues after each; final results ≈ Oct 3–5. Merges into `main` only after messaging the peer session (xrir-code-25) and only with a clean tree outside `worklog/` (three untracked files at the repo root belong to someone else and must be moved by their owner before any launcher runs).

**Risks.** (1) G ≈ E (predicted): the fairness table then shows that no SimpleViT variant uses the cue — the intended conclusion, but Phase 2 becomes the only literal evidence. (2) exp_02's legacy receipt may not admit A′ into a new summariser run → A′ reported as an external row. (3) GPU hand-over slips. (4) One pretraining realisation per Phase-2 arm (as in exp_06). (5) A reviewer may still want the channel arms → Phase 2.

## 8. Decisions for Yixun (defaults in bold; the Planner proceeds with the defaults unless told otherwise)

(a) Run Phase 2 (literal channel arms H and I, ≈ 65 GPU-h, results ≈ Oct 3–5): **yes, after Phase 1, on whichever card frees first**. (b) The "xRIR" row of the paper table: **A′ released-checkpoint arm as now, with A (same-budget control) as a supplementary row**. (c) Margin for N2/P3 non-inferiority and P1/P2 equivalence: **+0.23 dB** (exp_06's). (d) Heading source: **exp_06's axis decision (k = 128)**. (e) Sim k = 0 parity evaluations for H/I: **no** (HAA question only). (f) Exploratory mirror-probe diagnostics (exp_06 G1 protocol) on E, G, D after Phase 1: **yes** (10 min, `ckpt/exp11/_diag/`, not evidence).

## 9. Changelog

- v1 (2026-09-26): initial version.
