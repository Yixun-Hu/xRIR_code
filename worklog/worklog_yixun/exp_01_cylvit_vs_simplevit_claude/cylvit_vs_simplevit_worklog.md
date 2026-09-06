# Lab notebook — cylvit_vs_simplevit (RETROSPECTIVE)

> Reconstructed on 2026-09-05 from the session transcript, log birth/modification times and `history.jsonl`. Timestamps are local (America/New_York, −04:00); entries marked ≈ are accurate to a few minutes. No plan/code review loop existed at the time (pre-SOP).

## 2026-09-04T12:12:00-04:00 — repo survey, CLAUDE.md
- **Goal** — Understand the repo before touching it; write CLAUDE.md (`/init`).
- **Version Control** — branch `main` at `629b836` (user switched from `receiver-frame-ablation` mid-survey).
- **Result** — `passed`: CLAUDE.md rewritten (old one described a deleted PairRIR experiment). Environment facts recorded: conda `xRIR` (py3.8/torch 2.0.1) is the only compatible env; `data/` symlinks to a CIFS share.
- **Next** — baseline reproduction (Q1).

## 2026-09-04T12:19:00-04:00 — launch baseline reproduction
- **Goal** — Reproduce paper Table 1 (xRIR K=8) with the released checkpoints via the untouched `eval_unseen.py` / `eval_seen.py`.
- **Hypothesis** — Numbers land within run-to-run noise of Table 1 (seen 0.038/0.940/8.13, unseen 0.055/1.457/~10).
- **Command / Validation** — see `_command.md` §Baseline; logs `cylvit_vs_simplevit_2026-09-04_12:21:10_baseline_eval_{unseen,seen}.log`. GPU 1 unseen, GPU 0 seen. Split sizes: unseen test 6337, seen test 6217.
- **Acceptance criteria** — full splits complete without exception; averages printed.
- **Result** — `launched`; throughput ≈1.6–1.8 samples/s (CIFS-bound). A stray `find /` from the CLAUDE.md step was hogging the disk and was killed (12:24).
- **Next** — read the paper's protocol; build the cylindrical variant while evals run.

## 2026-09-04T12:24:00-04:00 — read paper protocol; new query (Q2)
- **Goal** — Extract Table 1 targets, metric definitions and the reference-selection protocol from `xRIR_pdf.md`; scope Q2 (swap SimpleViT → CylindricalViT).
- **Analysis** — Paper: 22.05 kHz, 9600 samples, STFT 124/62/31 → 63×310, panorama 256×512 with 16×32 patches, 10 candidate sources per room, K∈{1,4,8} sampled at train and test; λ=0.01 decay loss. Paper says a 6-layer ViT but code/checkpoints use depth 12. No training budget/hardware stated.
- **Next** — implement `xRIR_Cyl`, backbone-selectable train/eval scripts.

## 2026-09-04T12:29:00-04:00 — implement cylindrical swap + scripts
- **Goal** — Minimal, provenance-clean swap of the geometry encoder plus scripts that run both backbones through identical data/loss/metrics.
- **Change** — `model/cylindrical_vit.py` +1 line (`from __future__ import annotations`; it failed to import on 3.8: `TypeError: 'type' object is not subscriptable`); new `model/xRIR_cyl.py`, `train_xRIR_backbone.py`, `eval_xRIR_backbone.py`.
- **Command / Validation** — `python model/xRIR_cyl.py`: simple 32.08 M / cylindrical 32.12 M params, outputs `[2,63,310,1]` finite; smoke train (3 batches) writes `best.pth`/`last.pth`; new eval on released ckpt (5 samples) matches the original script's running averages; smoke eval on the smoke ckpt loads and runs.
- **Result** — `passed`.
- **Analysis** — Design fact recorded: `lin_proj_0` is a learned 256→1 linear pool over the token axis; it works only because 16×16 patches = 256 = `intermediate_ch`. CylindricalViT keeps the token count, so nothing else changes.
- **Next** — memory/speed probe, then launch.

## 2026-09-04T12:33:00-04:00 — fit / batch-size probe
- **Goal** — Find the largest batch for one A6000 and the per-iteration cost (validation-ladder rung 6).
- **Result** — `passed`: B=64 OOM for both; B=32 fits at 30.7 GB peak; fp32 2.34 s/iter (cyl) / 1.94 (simple); TF32 1.39 / 1.28 s/iter → 23 / 25 samples/s. Decision: batch 32 × 2 accumulation, TF32, 12 epochs with `decay_epochs 3` (same 4:1 ratio as the authors' 200/50). ≈3.6 h/epoch estimated.
- **Next** — launch both runs.

## 2026-09-04T12:34:30-04:00 — FAILED launch (stalled on CIFS)
- **Goal** — Start both trainings (cyl GPU 0, simple GPU 1) with 16 loader workers.
- **Acceptance criteria** — ≥1 optimizer step within minutes; no OOM/NaN.
- **Result** — `failed` (infrastructure): after 6 minutes no batch had been produced; 43 processes in D state; iowait 12 %; the baseline evals slowed from 1.6 to 0.7 samples/s.
- **Analysis** — *Infrastructure, not a bug.* `xRIR_Dataset.__getitem__` does a 1600-entry `os.listdir` plus ~20 small file reads per sample; over the CIFS mount (`//mae-cab-nas.princeton.edu/shared_data`, SMB 3.0) 32 workers saturate the share. Single cold reads: 0.14 s per file. Killed both runs at 12:40 (lesson recorded: never `pkill -f` with a pattern that appears in the same shell command; it killed the shell twice).
- **Next** — mirror the dataset to local NVMe.

## 2026-09-04T12:41:25-04:00 — local data mirror
- **Goal** — Copy the ~24 GB the code reads (`single_channel_ir_1`, `depth_map`, `metadata`; 261 rooms) to `~/data_cache/AcousticRooms` (113 GB free).
- **Command / Validation** — 30 rsync jobs (dir × category), 6 in parallel; log `cylvit_vs_simplevit_2026-09-04_12:41:24_data_mirror.log`. Completed 13:24:12 (≈43 min, ~27 MB/s).
- **Acceptance criteria** — dataset split sizes on the mirror equal those on the share; rsync dry run reports nothing left to transfer.
- **Result** — `passed`: 296334/6337 unseen, 6217 seen; dry run: 0 regular files to transfer for IR (302,671) and metadata (302,925), only a stray `.DS_Store` in depth_map; loader throughput 873 samples/s (12 workers) vs 1.7 on the share.
- **Next** — relaunch training with `XRIR_DATA_PATH` set.

## 2026-09-04T13:25:01-04:00 — LAUNCH training (the reported runs)
- **Goal** — Train both backbones with identical flags (see `_params_set_up.md`).
- **Version Control** — uncommitted working tree on `main` (`629b836` + untracked files listed in `commits_*.md`).
- **Command / Validation** — `_command.md` §Training; logs `cylvit_vs_simplevit_2026-09-04_13:24:58_train_{cylindrical,simple}.log`; pids 2900604 / 2900605.
- **Acceptance criteria** — batch 32 × accum 2 on one GPU each, 12 epochs, finite losses, `history.jsonl` one line per epoch, ≥1 optimizer step within 2 minutes.
- **Result** — `launched`; at batch 100: cyl 18 samples/s (sharing GPU 0 with the seen eval), simple 33 samples/s; losses 0.026 / 0.026. After the baseline evals finished both ran at ≈32 samples/s (cyl 156 min/epoch, simple 141).
- **Next** — wait for baseline results; build the per-epoch paired evaluation.

## 2026-09-04T13:44:00-04:00 — baseline reproduction result
- **Result** — `passed`: unseen (n=6337) EDT 0.0549 s / C50 1.358 dB / T60 9.69 %; seen (n=6217) 0.0389 / 1.029 / 7.27 (paper: 0.055/1.457/~10 and 0.038/0.940/8.13). Saved `ckpt/baseline_reproduction.json`, logs copied to `ckpt/baseline_eval_logs/`.
- **Analysis** — Within the noise of random reference selection; the pipeline is calibrated. Noise floor for later comparisons taken from adjacent-epoch variation (EDT ≈ ±0.001 s, T60 ≈ ±0.3 pts).

## 2026-09-04T14:59:00-04:00 — per-sample dumps, paired comparison tool, preliminary read
- **Change** — `eval_xRIR_backbone.py --save-per-sample`; new `tools/compare_eval.py` (paired mean difference, 10k-bootstrap 95 % CI, fraction A<B). Verified on an 8-sample smoke run that the two evals see identical sample indices/paths (loader seeding is deterministic, so references are paired too).
- **Result** — `passed`. Preliminary 500-sample subset (mid-epoch snapshots, cyl batch 5000 vs control batch 6000): EDT 0.0531/0.0545, C50 1.466/1.374, T60 8.98/10.18 — mixed and within noise; noted that both were already near the released checkpoint after ~160k samples.
- **Next** — queue full-split evals after each epoch (`epoch_compare.sh N`).

## 2026-09-04T16:20:00-04:00 — epoch 1 (full split, paired)
- **Result** — cyl vs control: test loss 0.0179 vs 0.0184*, T60 11.77 vs 12.11 %*, EDT 0.0577 vs 0.0586 (n.s.), C50 1.497 vs 1.466* (control), log-STFT MSE 0.462 vs 0.428* (control). Log `..._16:02:40_eval_unseen_epoch01_{cyl,simple}.log`.
- **Analysis** — Split decision; cylindrical better on tails (means), control on medians. Both well behind the released checkpoint.

## 2026-09-04T18:39:00-04:00 — epoch 2: pipeline race (orchestration bug), recovery
- **Result** — `partial`: the pipeline waited for *both* runs before snapshotting `last.pth`; the control (20 min ahead) had advanced to epoch 3 batch 2000, so its epoch-2 snapshot failed (`AssertionError: (3, 2000)`). Its epoch-2 weights were recovered from `best.pth` (epoch 2 was its best and epoch 3 unfinished). Cyl eval ran normally; control eval re-launched by hand; compare at 18:56.
- **Analysis** — *Real bug, orchestration only* (no effect on models/metrics). Fixed by snapshotting each run the moment its `history.jsonl` logs the epoch (`snap cyl & snap simple & wait`). Also: a harness background waiter was killed by a "low memory" heuristic (18:41) while 208 GB were available (forked loader workers inflate RSS); re-armed as a Monitor.
- **Epoch 2 result** — T60 8.91 vs 10.27 %* (cyl), EDT 0.0526 vs 0.0502* (ctrl), C50 1.428 vs 1.350* (ctrl), loss 0.0177 vs 0.0169* (ctrl).

## 2026-09-04T21:34:00-04:00 — epoch 3
- **Result** — EDT 0.0458 vs 0.0491* (cyl), T60 9.95 vs 10.28 %* (cyl), loss 0.0162 vs 0.0166* (cyl), C50 1.276 vs 1.252* (ctrl, margin shrank 3×), MSE 0.435 vs 0.427* (ctrl). Both now beat the released checkpoint on EDT and C50.

## 2026-09-05T00:10:00-04:00 … 07:59 — epochs 4–7 (routine)
- **Result** — ep4: EDT 0.0453/0.0502*, C50 1.244/1.225*, T60 10.00/9.55* (ctrl), MSE 0.415/0.437*, loss 0.0160/0.0162*. ep5: cyl ahead on all five (EDT, C50, MSE significant). ep6: cyl ahead on all five (four significant). ep7: EDT −0.0008*, loss*, T60 +0.34* (ctrl), C50/MSE ties.
- **Analysis** — T60 flips sign between epochs for both models; at lr ≤ 1e-5 epoch-to-epoch movement equals the between-model gap, so the final judgment must average over epochs.

## 2026-09-05T10:31:00-04:00 — epoch 8: compare step crashed (script edited while running)
- **Result** — `partial`: both epoch-8 evals completed but the pipeline's final compare printed `syntax error: unexpected EOF` — I had patched `epoch_compare.sh` (the per-run snapshot fix) while the epoch-8 instance was still executing it (bash reads scripts incrementally). Compare re-run by hand 12:50: C50 1.217 vs 1.257* (cyl, largest C50 win), EDT 0.0454 vs 0.0467*, loss*, T60 tie, MSE +0.004* (ctrl, borderline).
- **Analysis** — *Real bug (process discipline)*: violates "never edit a script while it is running". Epochs 9–12 pipelines were launched only after the patch, from the patched file, and the file was not touched again. Also snapshotted the control's epoch-9 weights by hand at 10:34 because the epoch-9 pipeline was not yet queued when the control finished (then queued 9–12 together with an "already exists" guard).

## 2026-09-05T13:11:00-04:00 — epoch 9
- **Result** — EDT 0.0454 vs 0.0475*, T60 9.50 vs 9.66 %*, MSE 0.407 vs 0.423*, loss 0.0158 vs 0.0160* (all cyl), C50 tie. Best test losses so far: cyl 0.01579 (ep 9) vs ctrl 0.01597 (ep 10).

## 2026-09-05T14:59:00-04:00 — seen-split evaluation (Q8), epoch-9 checkpoints
- **Goal** — Evaluate both on the seen test split (6217) as requested.
- **Result** — `passed` (15:15): cyl vs ctrl EDT 0.0337 vs 0.0347*, C50 0.873 vs 0.895*, T60 6.84 vs 6.92 %*, MSE 0.410 vs 0.417*, loss 0.0152 vs 0.0153*. `--by-test-rooms` breakdown: held-out rooms (1093 clean queries) cyl better on all, only MSE significant; training rooms (5124 queries whose target RIRs were training samples) cyl better on EDT/C50/MSE/loss*.
- **Analysis** — Absolute seen numbers are contaminated (our models trained on all receivers of the 244 non-test rooms); the backbone comparison remains fair (shared contamination). Not comparable to the paper's seen model, which was trained on the seen split.

## 2026-09-05T15:35:00-04:00 — budget/provenance questions (Q7, Q9–Q12)
- **Result** — Answered from logs: 4,630 steps/epoch (eff. batch 64), 41,670 steps at epoch 9; schedule compressed 200/50 → 12/3 with identical LR endpoints; paper gives no epochs/hardware, the authors' script hints at an epoch-20 best; profiled one step (batch 8): ViT 191 ms, audio enc 126 ms, per-sample STFT loop 118 ms (35 % of forward, does not scale with GPU) → one H200 ≈10 days for 200 epochs as-is, ≈6 with a vectorized STFT; decided not to touch the baseline STFT code for this experiment.

## 2026-09-05T15:53:00-04:00 … 18:28 — epochs 10–11
- **Result** — ep10: EDT 0.0454/0.0469*, MSE*, loss*, C50/T60 ties. ep11: EDT 0.0456/0.0472*, MSE*, loss*, C50/T60 ties. Control finished 12 epochs at 17:38 (final test 0.01618, best 0.01597 @10).

## 2026-09-05T21:00:52-04:00 — epoch 12 and final summary
- **Result** — `passed`: cyl finished 20:46 (final test 0.01590, best 0.01579 @9). Epoch 12 paired: EDT 0.0451 vs 0.0473*, T60 9.46 vs 9.59*, loss 0.0159 vs 0.0162*, MSE 0.419 vs 0.425*, C50 tie. `tools/summarize_epochs.py --avg-epochs 5-12` → `ckpt/backbone_comparison_summary.txt` (see `_results.md`).
- **Analysis** — see `_analysis.md`.
- **Next** — exp_02 sim-to-real (already approved and running); exp_03 yaw-rotation study.

## 2026-09-05T23:10:30-04:00 — retrospective record + results page (post-SOP bookkeeping)
- **Goal** — Bring this pre-SOP experiment into the SOP record format (query, plan-as-executed, notebook, params, commands, results, analysis, logs, commits) and produce the HTML results page.
- **Change** — `cylvit_vs_simplevit_results_assets/make_results_html.py` (Planner one-off; reads the per-epoch per-sample files and `history.jsonl`, writes `data.json` and `../cylvit_vs_simplevit_01_results.html` with inline SVG charts: per-epoch means with the released-checkpoint reference line, per-epoch paired differences with 95 % CIs, verdict blocks, tables, method/loss).
- **Command / Validation** — `python .../make_results_html.py` → 44 KB page, HTML parses (670 tags); numbers cross-checked against `_results.md` (EDT −3.73 %, C50 −0.67 %, T60 −0.40 %, loss −1.22 % over epochs 5–12, 2000-resample bootstrap).
- **Result** — `passed` with two deviations recorded: (1) the dataviz palette validator (`validate_palette.js`) cannot run on this host's node v12 (needs `??=`, node ≥ 15); the two series use reference-palette slots 1 (blue) and 2 (orange), documented as passing adjacent-pair CVD ΔE 9.1 light / 8.4 dark and normal-vision 19.6 / 19.3; (2) no browser on this host, so the page was checked structurally, not rendered.
- **Analysis** — Per the SOP's universal review coverage, `make_results_html.py` still needs a Codex review; queued as a batched review round together with exp_02's page generator.
- **Next** — Codex review of the page generators; commit with the base commit decided at exp_03 approval.

## 2026-09-05T23:37:33-04:00 — Codex review of the results-page generators: REJECT → corrections
- **Goal** — Close the batched review round for `make_results_html.py` (exp_01 + exp_02).
- **Result** — Review saved as `cylvit_vs_simplevit_codex_code_results_pages_review.md` (gpt-5.6-sol, effort ultra). Five blocking findings; two of them correct claims in this experiment's record: (1) references were not identical across models (loader seed drawn after model construction; cyl constructor consumes more RNG) and Griffin-Lim phases were unseeded; (2) with the 17 rooms as bootstrap unit the EDT and loss intervals include 0. Also: bootstrap replicate counts differed between page and producer (2000 vs 10000); exp_02 page was built from partial runs with verdicts; paper complex-room T60 4.33 was missing.
- **Command / Validation** — Replay of the reference draws for both model builds (seed 0, 6 workers, first 3000 queries): 722 / 3000 = 24.1 % differ; candidate counts {6: 91, 7: 429, 8: 2259, 9: 221}; 91.4 % of 9-candidate queries differ. Room-cluster bootstrap (10k) on epochs 5–12: EDT [−0.0039, +0.0001], loss [−0.00039, +0.00002], MSE [−0.0180, +0.0028], C50/T60 include 0; per-room EDT diff negative in 11/17 rooms.
- **Change** — `tools/summarize_epochs.py`: room-cluster CIs, per-room differences, `--json` canonical stats (`ckpt/backbone_comparison_stats.json`); page generator now reads only that JSON (no own bootstrap), shows both CIs, "no detected difference" wording, zoomed-axis labels, pairing caveat, unadjusted-CI note, per-room table; `_results.md` and `_analysis.md` corrected (claim retracted, both CIs, revised outcome statement). `sim_to_real/summarize_haa.py`: paper T60 4.33, query-cluster bootstrap, completeness check, per-seed effects, `--json`; exp_02 page reads the canonical JSON and renders as a watermarked DRAFT with verdicts suppressed until the run set is complete.
- **Analysis** — *Real bug (evaluation pairing) + statistics.* Direction of the exp_01 conclusion unchanged for this split; the population-of-rooms claim is withdrawn. The properly paired re-evaluation is folded into exp_03's k=0 gate. Non-blocking items (pairing assertions in the page, isfinite filtering, axis labels, unadjusted note) applied.
- **Next** — Codex round 2 on the fixed generators and producer changes; then this round closes.

## 2026-09-05T23:51:09-04:00 — round 2 (REJECT, narrower) → round-2 fixes
- **Result** — Round-2 review saved (`..._results_pages_r2_review.md`): exp_01 room-cluster bootstrap verified independently by the reviewer (200k resamples reproduce the stored intervals); remaining: producer header still said "identical references"; my correction wrongly said receivers with ≤8 candidates get the same set (fewer than 8 sample *with replacement* and differ — the replay's 520 such queries all differed); the page still re-read history/baseline and duplicated constants; no assertion tied `_results.md` to the JSON; `_command.md` lacked the `--json` invocation.
- **Change** — `tools/summarize_epochs.py`: header wording fixed; JSON now also carries lr history, baseline reproduction, paper/released references, bootstrap seed, n_rooms/n_queries. Page generator reads only the JSON and asserts that `_results.md` quotes sections 2–3 of the same-run `summary.txt` verbatim (fails otherwise). `_results.md` candidate-count statement corrected; "no systematic direction" qualified as an expectation. `_command.md` records the `--json` invocation and the page command.
- **Command / Validation** — regenerated `ckpt/backbone_comparison_{stats.json,summary.txt}` and the page; assertion passes.
- **Analysis** — No numbers changed; wording, provenance and enforcement changed. Tests for the new bootstrap/consistency functions are deferred until the user fixes the tests folder (exp_03 approval).
- **Next** — Codex round 3 (verification).

## 2026-09-06T00:01:27-04:00 — round 3 (REJECT, mechanical items) → round-3 fixes
- **Result** — Round 3 verified the two-way bootstrap (independent Cartesian resampling reproduced all 11 interval pairs to 1e-16) and the room-cluster CIs; remaining: page wording still said "without a systematic direction"; `_results.md` block check was per-line not contiguous and `summary.txt` was not bound to `stats.json`; multiplicity note said four metrics (producer evaluates five); no index/path identity assertion in the producer.
- **Change** — `tools/summarize_epochs.py`: asserts index/ir_path/meta equality between the two per-sample files for every epoch; `--summary` writes the summary text and stores its sha256 in the JSON. Page: verifies the hash, requires sections 2–3 as contiguous verbatim blocks in `_results.md`, wording changed to "expected effect zero; realised contribution unknown", multiplicity note says five metrics. `_command.md` updated with `--summary`.
- **Command / Validation** — regenerated `ckpt/backbone_comparison_{stats.json,summary.txt}` and the page; hash and block assertions pass.
- **Next** — Codex round 4 (verification only).

## 2026-09-06T00:08:01-04:00 — results-page review round CLOSED (round 4: one narrow blocker → fixed, Planner-verified)
- **Result** — Round 4 (`..._results_pages_r4_review.md`): wording, canonical binding (hash-bound summaries, contiguous verbatim block, exact per-run agreement, canonical chart means, `--allow-partial` recorded) and draft suppression verified fixed; completeness gate v2 verified against synthetic probes except that a *missing* `depth_variant` key was defaulted.
- **Change** — `sim_to_real/summarize_haa.py`: `depth_variant` must be present and equal to the planned variant in every stage `args.json`. exp_02 page: asserts the canonical `mean`/`std` equal the aggregate of their own `per_run` entries.
- **Command / Validation** — probe: real run set → only in-flight/missing problems, no seed/init/depth problems; synthetic manifest with `depth_variant` removed → rejected with "must be present". Draft regenerated.
- **Analysis** — All blocking findings from rounds 1–4 fixed and re-verified (rounds 2–4 by the Reviewer, the last one-liner by the Planner). Open, deferred with reason: regression tests for the new functions (tests folder pending user decision at the exp_03 approval); zero-shot per-sample meta lacks `depth_variant` (eval_haa.py unchanged mid-sweep; to be added after the sweep).
- **Next** — final exp_02 artefacts when the queues finish; commits once the base-commit decision is made.
