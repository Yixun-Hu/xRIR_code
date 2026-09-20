# Plan — yawaug_haa (exp_09): the yaw-augmented xRIR on the real-room (HAA) benchmark

**Status:** v1, 2026-09-19, Planner (Claude Fable 5.1, session `xrir-code-6e`), awaiting the Codex plan review. Requested by Yixun on 2026-09-19 ("run the yaw-augmented xRIR fine-tuned and evaluated on HAA using the xRIR paper's recipe"; `yawaug_haa_yixun_query.md`). Builds on exp_02 (HAA protocol, historical arms A/B), exp_04 (the yaw-augmented checkpoint and its approvals), exp_06 (the reviewed HAA pipeline, finalizer, summariser and approvals machinery). Roles per `CLAUDE.md`: Planner = this session, Coder = Claude Opus 5 (worktree), Reviewer = OpenAI Codex `gpt-6-astra` at xhigh.

## 1. Question

exp_04 trained the SimpleViT xRIR with per-sample active yaw of the whole scene (`train_xRIR_backbone.py --yaw-aug 1`, offsets uniform over the 512 panorama columns) and showed, on the simulated unseen split, non-inferior EDT but a ≈ 4.5 % C50 cost at the canonical heading and a modest reduction of the diagonal-heading degradation. Nothing is known about this model on real rooms. Two mechanisms pull in opposite directions on HAA: yaw augmentation makes the encoder less dependent on the absolute heading of the panorama, which exp_02's diagnosis identified as exactly the cue the SimpleViT uses to resolve the hallway's mirror ambiguity under a directional loudspeaker (the CylindricalViT, which lacks that cue, loses 0.92 dB of C50 there); on the other hand the augmentation may improve generalisation to an unseen real geometry. **Question:** after the paper's few-shot fine-tuning, does the yaw-augmented SimpleViT match, beat or lose to the same-budget SimpleViT control on the four HAA rooms, and in particular on the hallway C50 cell?

## 2. Design

One new fine-tuning arm, run through exp_06's reviewed HAA pipeline in the **room frame** (no heading rotation, no orientation channel — the paper's setting, identical to exp_02's arms A and B):

| Arm | Init | Backbone | Frame | HAA runs |
|---|---|---|---|---|
| A `control` | exp_01 SimpleViT `ckpt/xRIR_simple_8_shot/epoch_12.pth` | simple | room | exists (exp_02, legacy receipt) |
| B `cyl` | exp_01 CylindricalViT | cylindrical | room | exists (exp_02) |
| C `cyl_or` | exp_06 oriented CylindricalViT | cylindrical_oriented | heading | exists (exp_06) |
| **E `yawaug`** | exp_04 yaw-augmented SimpleViT `ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth` (sha256 f8e64052…, exp_04 approvals `checkpoints.aug`) | simple | room | **new: seeds 0, 1, 2 + zero-shot** |

The init is the twelfth-epoch checkpoint of exp_04's certified run (same budget as exp_01's control: 12 epochs, batch 32 × 2, AdamW 1e-3 decayed ×0.1 per 3 epochs, wd 1e-4, TF32), so E − A isolates the augmentation. Recipe (frozen, = exp_02 = exp_06 §6.2): stage 1 on classroom + hallway + complex (`--epochs 1000 --val-every 10 --tf32`, full batch, AdamW lr 1e-4, wd 1e-4, ×0.1 per 50 epochs), stage 2 per room (`--epochs 200 --val-every 2 --tf32`), checkpoint selection on the DiffRIR validation split, K = 8, `eval_seed 0` (identical query/reference pairs across all arms), unseeded Griffin-Lim (primary phase policy), EDT / C50 / T60 on the 9 600-sample waveforms, T60 omitted for the dampened room, `depth_variant default`. Test sizes 463 / 198 / 423 / 198. Three fine-tuning seeds; zero-shot evaluation of the init in every room. Outputs `ckpt/exp09/sim2real/yawaug/{seed<s>/{stage1,stage2_<room>,eval/<room>},zeroshot/eval/<room>}` with exp_06's completion units and finalizer.

**What is not changed:** every pinned file (exp_03's twelve, `train_xRIR_backbone.py`, `tools/exp04_*.py`, `tools/exp05_*.py`, `tools/paired_compare.py`, `tools/provenance.py`, `sim_to_real/*`), exp_06's model modules, heading tool, recipe, trainer, launcher, smoke runner, mirror probe, eval launcher and comparer; exp_02's and exp_06's completed runs and canonical outputs.

## 3. Pre-registered hypotheses and decision rules (exp_06 §7 machinery, unchanged)

Estimand per cell: paired per-query differences `Δ_{s,q} = e_{E,s,q} − e_{A,s,q}` over the three fine-tuning seeds and the test queries of one room and metric; point estimate = mean over (s, q); intervals from `tools/exp06_bootstrap.py` (query-cluster and two-way schemes; `n_boot` 10 000 nominal, 50 000 for adjusted tails; convergence rule seeds 0/1, tolerance 0.10, one quadrupling; verdicts on the two-way interval); paired cohort = queries finite in every compared run; void rule as in exp_06 (excess invalid > 1 % of the room or cohort < 99 %).

- **E1 (primary)** — hallway C50, **E − A**: report the two-way 95 % interval and classify: *detected harm* if the lower bound > 0, *detected improvement* if the upper bound < 0, otherwise *no detected difference*; additionally state whether the upper bound is below **+0.23 dB** (exp_06's H1 margin, kept for comparability; "non-inferior at the exp_06 margin" if so). The mechanism predicts harm (loss of the absolute-heading cue); the rule is two-sided.
- **E2 (screen)** — all 11 room × metric cells, E − A: nominal and Bonferroni-11 adjusted two-way intervals; labels *detected harm* / *detected improvement* / *no detected difference*, never "non-inferior".
- **E3 (descriptive)** — E − C (yaw-augmented SimpleViT vs the oriented CylindricalViT) on all cells with nominal intervals; zero-shot room-frame side split (−y / +y) of E in every room next to A, B, C; E's per-seed values.
- No simulated-parity test is needed: exp_04's canonical `TABLE_V1` / `H1_K8` already give E's k = 0 numbers against A on the same manifests.
- **Optional diagnostic (not evidence):** the exp_06 mirror probe with E in the `--control-checkpoint` slot, `--exploratory`, to read E's hallway mirror cosine and wrong-side attention share zero-shot; reported as a labelled diagnostic if run.

**Overall reading.** E1's category is the headline; E2 describes the rest; E3 places the yaw-augmented model relative to the oriented model. Absolute numbers remain non-comparable with the paper's Table 2 (test-split definition and selection leakage there; see the exp_02 analysis).

## 4. Code (Coder = Claude Opus 5; TDD red-first; commits ≤ 200 lines; Codex code review; worktree branch `exp09-yawaug`)

1. **`tools/exp06_haa_pipeline.sh`** — a fourth init `yawaug` (backbone `simple`, checkpoint `${EXP09_YAWAUG_CKPT:-ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth}`, **frame `room`**): its children get no `--heading-json-dir`; the job spec records `frame: room` and no `heading`; the approvals gate compares the init sha256 with `exp04_profiles.AUG['sha256']` (as it does for the exp_01 inits) and calls `enforce_producer('haa_children', …)` with `headings=None`-equivalent semantics for a room-frame job (the `HEADING_COMPLETE` rule applies to heading-frame jobs only — the Coder states how; the current rule refuses `haa_children` without four headings, so either the gate passes the four headings anyway (they exist and are approved; harmless) or the producer map gets a documented room-frame exception). Output root selectable (`EXP06_HAA_OUT`, default unchanged) so exp_09 writes under `ckpt/exp09/sim2real`. Golden dry-run test for `yawaug:0` and `zeroshot` with the room-frame argv.
2. **`tools/exp06_finalize.py`** — (a) room-frame HAA children already validated (`_heading_binding` refuses a heading in the room frame); confirm with a test that a `yawaug` stage/eval completion certifies; (b) **should-fix from exp_06:** `haa_approvals(closure, run_type, commit, repo)` must read the approvals **blob at the child's reviewed commit** (`git show <commit>:<path>`), not the working-tree file, so that a later re-fill of the approvals for another producer does not make historical children unverifiable (today it required restoring the working file). Regression: a child certified under approvals version X stays verifiable after the working file changes to version Y.
3. **`tools/exp06_summarize_haa.py`** — arm E registered (`branch new`, root `ckpt/exp09/sim2real/yawaug`, backbone simple, frame room, `init_sha256 = exp04_profiles.AUG['sha256']`); room-frame new arms skip the heading checks; new outputs for exp_09: decision cell E1 (E − A hallway C50, two-sided classification + the +0.23 dB statement), screen E2 (E − A, 11 cells, Bonferroni-11), descriptive E3 (E − C), side split including E; a `--experiment exp09` (or a separate thin producer `tools/exp09_summarize_haa.py` that imports exp_06's functions — the Coder chooses and states why) writing `ckpt/exp09/stats.json` + `summary.txt` with the same hash-bound, exclusive-create, revalidate-before-publish behaviour; exp_06's outputs are never rewritten. The approvals producer for it is `summarize_haa` (its own closure key) plus the reused exp_04 approvals identity that already pins `checkpoints.aug`.
4. **`tools/exp06_approvals_api.py`** — only if needed for item 1 (room-frame producer rule); no new approvals key: E's init identity comes from exp_04's approved record, already bound by `reused.approved_digests_exp04`.
5. Tests for every item; the full suite green except the known exp_07 guard; digests to re-fill after merge: `haa_pipeline_sh`, `summarize_haa`, `finalize` (and any propagation), listed in the Coder's report.

## 5. Validation ladder and acceptance

1. Static + unit tests (CPU, `CUDA_VISIBLE_DEVICES=''`).
2. Pipeline dry run `tools/exp06_haa_pipeline.sh <gpu> yawaug:0 zeroshot --dry-run` printed and inspected (room-frame argv, job spec, output root).
3. Codex code review → merge → approvals re-fill at the merge tip (the changed keys only) → CPU suite green (except the known exp_07 guard) → Codex verification of the re-fill (may be folded into the code review's close pass).
4. Launch on an exclusive card negotiated with the peer session (stage-1 children need ≈ 36 GB); queue `yawaug:0 yawaug:1 yawaug:2 zeroshot` (≈ 3.3 h); every child/job `admissible_arm: true`.
5. Summariser (approvals at the launch tip) → `ckpt/exp09/{stats.json,summary.txt}` → results table + analysis + page generated only from the canonical JSON.

Acceptance of a run: exp_06's (every stage `summary.json`, every eval `metrics_/per_sample_` with frame `room` and no heading, `completion.json` present and admissible; recipe fields equal to the registered recipe).

## 6. Deliverables

`ckpt/exp09/{sim2real/yawaug/**, stats.json, summary.txt}`; record `worklog/worklog_yixun/exp_09_yawaug_haa_claude/` (`plan_yawaug_haa.md`, `yawaug_haa_yixun_query.md`, `yawaug_haa_worklog.md`, `yawaug_haa_command.md`, `commits_yawaug_haa.md`, `yawaug_haa_params_set_up.md`, Codex reviews, Coder reports, logs, `yawaug_haa_results.md`, `yawaug_haa_analysis.md`, `yawaug_haa_01_results.html` + generators); an added row in exp_06's results context where the arms are compared (E next to A, B, C).

## 7. Cost, schedule, risks

- CPU: Codex plan review ≈ 40 min; Coder round ≈ 2 h; Codex code review ≈ 45 min (+ fix cycle ≈ 1 h if needed); merge/re-fill/suite ≈ 1 h → code ready ≈ 03:00 Sep 20.
- GPU: 3 × ≈ 58 min + zero-shot ≈ 12 min ≈ 3.3 h exclusive (peer's exp_07 arms hold both cards at plan time; slot to be agreed). Results ≈ 1 h after the queue ends.
- Risks: (i) the approvals re-fill changes the working file → exp_06's children must remain verifiable (item 4.2 fixes the binding; until it lands the summariser cannot be run for either experiment without the restore workaround); (ii) a room-frame job through a pipeline written for the heading frame — covered by the dry-run golden and the finalizer tests; (iii) the yaw-augmented model's hallway behaviour may be harmed (E1 category "detected harm"), which is a valid finding, not a failure.

## 8. Decisions for Yixun

None blocking: the arm, recipe and hypotheses follow the request. Optional: (a) also run E in the heading frame (E_hf, +3.3 h GPU) to separate augmentation from frame effects as exp_06 did for A and B — not planned unless asked.
