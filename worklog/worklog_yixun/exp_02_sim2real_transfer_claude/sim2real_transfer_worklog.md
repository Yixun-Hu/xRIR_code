# Lab notebook — sim2real_transfer (retrospective for 2026-09-05 15:20–22:59; live afterwards)

## 2026-09-05T15:25:00-04:00 — scope the request; inventory data and reference code
- **Goal** — Answer Q1: derive the paper's sim-to-real protocol and find the HAA data locally before proposing configs.
- **Result** — `passed`: repo scripts read (`finetune_all_rooms.py`: 3 rooms, 12 targets each, batch 128, lr 1e-4, 1000 epochs, selects on test loss; `finetune_classroom.py`: 12 targets, 200 epochs; `eval_classroom.py`: metrics on full waveforms, imports missing modules). Data found: `/media/diskstation/yixunhu/HAA_processed/{classroomBase,dampenedBase,hallwayBase,complexBase}/mono_rirs_22050Hz` + `metadata/{scenes,poses}_metadata.json` + `train/val/test_base.json`; raw zips and the full HAA set also on the share.
- **Analysis** — Repo train indices match DiffRIR's `train_indices`; repo hard-coded receiver = `speaker_xyz`; repo test split (odd indices) differs from DiffRIR's test split; repo selects checkpoints on test loss whereas the paper says validation.
- **Next** — checks on delay preservation and depth-map agreement, then the plan.

## 2026-09-05T15:30:00-04:00 — read-only checks
- **Result** — Onset of the 12 training RIRs tracks source distance in every room (corr 0.99–1.00; ≈7-sample constant offset). Depth maps: repo vs HAA_processed differ by 3–10 m max.
- **Next** — present the plan.

## 2026-09-05T15:35:00-04:00 — plan presented; 16:00 approved
- **Result** — Plan in `plan_sim2real_transfer.md`; user: "Approved, run the sim2real experiments after training finishes".

## 2026-09-05T16:05:00-04:00 — cache builder; depth-map audit
- **Change** — `sim_to_real/prepare_haa.py` (new).
- **Result** — cache built (162 MB). Audit: horizon depth toward ±x/±y vs mic-grid extents shows repo `complex_room.npy` ≡ `hallway.npy` (`array_equal` True) and repo `dampened_room.npy` puts walls at 0.56 m in −y while mics extend 1.78 m; HAA_processed renders consistent for all rooms and equal to the repo maps at the horizon for classroom/hallway. Decision: repo maps for classroom/hallway, local renders for complex/dampened; both stored; `--depth-variant repo` reproduces the released assets.
- **Analysis** — *Asset bug in the released repo*, not in our code. Backbone comparison unaffected (identical inputs).

## 2026-09-05T16:12:00-04:00 — dataset / fine-tune / eval scripts
- **Change** — `sim_to_real/haa_dataset.py`, `sim_to_real/finetune_haa.py`, `sim_to_real/eval_haa.py` (new).
- **Command / Validation** — dataset self-test (48/560/1282; shapes; deterministic refs); smoke fine-tunes (stage 1 released init 2 epochs batch 4×9: val 0.0450→0.0394; stage 2 cyl dampened: 0.0564→0.0474); smoke evals 5 samples/room incl. dampened (T60 skipped); paired compare works.
- **Result** — `passed`.

## 2026-09-05T16:20:00-04:00 — orchestration + summarizer; launcher staged
- **Change** — `sim_to_real/run_haa_pipeline.sh`, `sim_to_real/summarize_haa.py`; `tools/compare_eval.py` tolerant of optional/empty metrics.
- **Acceptance criteria (pre-launch)** — launcher starts only after both exp_01 epoch-12 per-sample files exist and no eval process is running; each stage writes `summary.json`; each eval writes `metrics_<room>.json`; released fine-tuned rows near Table 2; zero-shot released classroom near 0.204 / 3.43.
- **Result** — `launched` (waiting). Smoke artifacts removed.

## 2026-09-05T21:01:39-04:00 — pipeline started
- **Result** — `in_progress`: zero-shot released: classroom 0.1667 / 3.373 / 7.87, dampened 0.0749 / 4.757 / –, hallway 0.1523 / 2.096 / 6.82, complex 0.1324 / 2.567 / 6.29 (paper pretrained classroom 0.204 / 3.427 → consistent). Zero-shot control: classroom 0.1179 / 1.933 / 7.35, hallway 0.1869 / 2.389 / 7.47, complex 0.1561 / 2.765 / 6.92.

## 2026-09-05T21:47:44-04:00 — first fine-tuned job (control seed 1)
- **Result** — stage 1 best val 0.0347 @ ep 920 (init 0.0436), 26 min; stage 2 best epochs 34/54/190/78. Test: classroom 0.0818 / 1.492 / 4.47, dampened 0.0384 / 2.812 / –, hallway 0.0615 / 1.144 / 2.68, complex 0.0766 / 1.658 / 4.96 — at or better than Table 2 except hallway C50.
- **Analysis** — Pipeline reproduces the paper's regime; job time ≈46 min.

## 2026-09-05T22:00:48-04:00 — cyl seed 0
- **Result** — classroom 0.0897 / 1.602 / 4.26, dampened 0.0381 / 2.862 / –, hallway 0.0867 / 1.990 / 4.14, complex 0.0817 / 1.729 / 5.10.
- **Analysis** — Unpaired, single seed; control ahead, notably in the hallway. Judgement deferred to the pooled paired comparison.

## 2026-09-05T23:38:22-04:00 — results-page generator reviewed (REJECT) → draft mode, canonical stats
- **Result** — Batched Codex review (`sim2real_transfer_codex_code_results_pages_review.md`): the page had been generated from a partial run set with final-looking verdicts; bootstrap treated repeated evaluations of the same queries across seeds as independent; paper complex-room T60 (4.33) was missing; `released_repomaps` row omitted; Griffin-Lim phases unseeded (references *are* identical across models here: deterministic per query via `eval_seed`).
- **Change** — `sim_to_real/summarize_haa.py`: `PAPER` complex T60 = 4.33; query-cluster bootstrap (rows of one query across seeds resampled together); per-seed effect sizes; pairing assertions on index/path/eval_seed/K; `EXPECTED_RUNS` completeness check (refuses to summarise a partial set without `--allow-partial`); `--json` canonical stats. Page generator: reads only `ckpt/sim2real/stats.json`, asserts exact agreement of per-run means, renders a red DRAFT banner with verdicts suppressed while the set is incomplete, includes the side row, filters non-finite values, states the phase-init caveat and that per-cell CIs are unadjusted.
- **Analysis** — *Real bug (statistics/presentation)*; no run artefacts affected. Final page and `_results.md`/`_analysis.md` are produced only after all 10 jobs finish.
- **Next** — Codex round 2 on the fixes; results/analysis when the queues complete (3 jobs left at 2026-09-05T23:38:22-04:00).

## 2026-09-05T23:51:09-04:00 — round 2 (REJECT) → two-way bootstrap, strict completeness gate, full draft suppression
- **Result** — Round-2 review (`sim2real_transfer_codex_code_results_pages_r2_review.md`): query-cluster bootstrap alone is conditional on the realised training seeds (a two-way query×seed bootstrap moves the interim classroom-T60 interval across zero); completeness gate only counted runs; page recomputed means; draft still showed stars and exported verdicts.
- **Change** — `sim_to_real/summarize_haa.py`: `paired(..., clusters, seeds)` adds a two-way (query × training-seed) percentile bootstrap via multiplicity weights alongside the query-cluster one; strict `completeness()` (exact seed ids per init, DiffRIR test-split sizes, split == test, ≥1 finite value for each required metric per room, unexpected runs/groups rejected, cyl and control seed sets must match); required paired cells with no runs or zero valid pairs abort the final summary; per-seed valid counts, n_queries, n_seeds, boot seed recorded in the JSON. Page: per-run means/std and paper values taken from `stats.json` rows (no recomputation); asserts the on-disk missing list equals the JSON's; seed labels must be the planned ones; final mode requires every planned run in every room/metric; verdicts use the two-way interval; in draft mode stars, verdicts and the exported `data.json` verdicts are all suppressed (`"draft": true`). `_command.md` updated.
- **Command / Validation** — `summarize_haa.py --allow-partial --json ckpt/sim2real/stats.json` (interim, 2 seeds): classroom EDT +0.0070 [+0.0019, +0.0121] two-way (control lower), classroom C50 +0.148 [+0.053, +0.253], classroom T60 −0.42 [−0.97, +0.08] (no detected difference), hallway EDT +0.0245 [+0.0165, +0.0323]; page regenerated as DRAFT.
- **Next** — Codex round 3; final summary when the queues finish.

## 2026-09-06T00:01:27-04:00 — round 3 → strict gate v2, hash binding, draft legend
- **Change** — `sim_to_real/summarize_haa.py`: `completeness()` now checks exact DiffRIR test indices per room (order and uniqueness, from the cache `meta.json`), every metric/ir_path array length, ir_path room membership, per-sample meta against the protocol (K=8, eval_seed 0, split test, backbone per init), and for fine-tuned runs the stage `args.json` seed, init checkpoint and depth variant (`released_repomaps` → `repo`); `paired()` validates finite/non-empty inputs; `--summary` + sha256 in the JSON; protocol/init tables stored in the JSON. Page: verifies the summary hash; checks exact per-run agreement between disk and canonical rows; chart uses the canonical seed mean; significance legend hidden in draft; `_command.md` records the `--allow-partial` draft invocation. Zero-shot evals do not record `depth_variant` (eval_haa.py is not changed mid-sweep; they use the default variant by construction of the pipeline script) — a known gap to close after the sweep.
- **Command / Validation** — the page's staleness guard fired once because in-flight jobs wrote files between summary and page; regenerated back-to-back (DRAFT).
- **Next** — Codex round 4; final artefacts after the queues finish.

## 2026-09-06T00:02:00-04:00 — fix: NameError in the page chart helper (stats not passed) introduced by the round-3 edit; regenerated DRAFT

## 2026-09-06T00:08:01-04:00 — results-page review round CLOSED (round 4: one narrow blocker → fixed, Planner-verified)
- **Result** — Round 4 (`..._results_pages_r4_review.md`): wording, canonical binding (hash-bound summaries, contiguous verbatim block, exact per-run agreement, canonical chart means, `--allow-partial` recorded) and draft suppression verified fixed; completeness gate v2 verified against synthetic probes except that a *missing* `depth_variant` key was defaulted.
- **Change** — `sim_to_real/summarize_haa.py`: `depth_variant` must be present and equal to the planned variant in every stage `args.json`. exp_02 page: asserts the canonical `mean`/`std` equal the aggregate of their own `per_run` entries.
- **Command / Validation** — probe: real run set → only in-flight/missing problems, no seed/init/depth problems; synthetic manifest with `depth_variant` removed → rejected with "must be present". Draft regenerated.
- **Analysis** — All blocking findings from rounds 1–4 fixed and re-verified (rounds 2–4 by the Reviewer, the last one-liner by the Planner). Open, deferred with reason: regression tests for the new functions (tests folder pending user decision at the exp_03 approval); zero-shot per-sample meta lacks `depth_variant` (eval_haa.py unchanged mid-sweep; to be added after the sweep).
- **Next** — final exp_02 artefacts when the queues finish; commits once the base-commit decision is made.

## 2026-09-06T01:13:34-04:00 — pipeline complete; final artefacts
- **Result** — `passed`: QUEUE_DONE GPU 1 00:57:58, GPU 0 01:10:31; 10/10 jobs, no failures. Final `summarize_haa.py --json --summary` passed the strict gate (draft false, no missing); page regenerated as FINAL. Headline: control lower on 6/11 cells (two-way CI), cylindrical lower on none; released checkpoint reproduces Table 2 on classroom/dampened but not on hallway/complex (as-released maps give the same), our control matches/beats Table 2 on EDT everywhere. See `_results.md`, `_analysis.md`, `sim2real_transfer_01_results.html`.
- **Acceptance criteria check** — every stage wrote `summary.json`; every eval wrote metrics + per-sample; released fine-tuned classroom near Table 2 (EDT 0.096 vs 0.093); zero-shot released classroom 0.167/3.37 vs paper 0.204/3.43. Passed, with the hallway/complex reproduction gap recorded as a finding.
- **Next** — commits (base-commit decision pending); seed Griffin-Lim phases and record `depth_variant` in `eval_haa.py` for future runs (not applied to this sweep).
