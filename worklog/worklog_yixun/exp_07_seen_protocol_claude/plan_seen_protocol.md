# Plan — seen_protocol (exp_07), v1 (2026-09-15)

Planner: Claude Fable 5.1 (main session). Coder: OpenAI Codex `gpt-6-astra` via `codex exec`. Code Reviewer: Claude Fable 5.1 subagent. Plan Reviewer: Codex.

## 1. Question

What are the seen-split errors (EDT, C50, T60; K = 1 and K = 8) of the three same-budget xRIR variants — SimpleViT (xRIR), CylindricalViT and YawAug-xRIR — when each is trained under the authors' **seen protocol**, so that the paper-style table has both split columns for all three rows? No hypothesis is pre-registered beyond the table; the paired differences between arms on the seen split are reported descriptively with intervals.

## 2. Protocol and arms

- **Seen protocol** (authors' `treble_xRIR_seen_dataset.py`, pinned by exp_03 — never edited): test = the 6 217 source–receiver pairs of `seen_test_split.pkl` (131 rooms); train = every other RIR of the dataset = 296 454 files (9 265 micro-batches of 32 per epoch; 12 epochs = 55 590 updates, the same budget as the unseen-protocol runs to within 0.05 %). Of the 6 217 test pairs, 5 124 lie in rooms whose other receivers are trained on ("seen"), 1 093 lie in the 17 unseen-protocol test rooms (also trained on under this protocol).
- **Arms** (three new 12-epoch trainings, one realisation each; the M tier, dim 512 / depth 12 / heads 8 / mlp 512): `seen_simple` (xRIR control, `--backbone simple`), `seen_cyl` (`--backbone cylindrical`), `seen_aug` (`--backbone simple --yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512`). Recipe otherwise identical to exp_01/exp_04/exp_05: `--num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1`, plus the new `--protocol seen`. `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=2`, interpreter `/home/yixunhu/miniconda3/envs/xRIR/bin/python`. Single-delta contract per arm vs its unseen-protocol twin: the training set (protocol) is the only intended difference; the trainer's `--protocol unseen` default reproduces the current behaviour bit for bit (parity test as exp_05 §7 item 6).
- **Evaluation**: the seen split, all 6 217 queries, K = 8 and K = 1, at k = 0 only, condition P, batch 16, TF32 off; **five evaluation seeds {42, 43, 44, 45, 46}** with seen-split reference manifests `ckpt/exp07/reference_manifest_seen_k{8,1}_seed{s}.json` built by `tools.reference_manifest.build_manifest(seen_dataset, seed=s, num_shot=K)` (the seen dataset exposes the same `get_ir_and_location_for_other_sources` interface) and `--gl-seed s`; the evaluator is `tools/exp04_eval.py` unchanged (its `ManifestDataset` reads the manifest's file list; the data identity is content-hashed over the referenced files), through `tools/exp04_eval_launch.py` with the run-level provenance (immutable manifest, completion, `--bind-input train_manifest/train_completion`). 3 arms × 2 K × 5 seeds = 30 runs ≈ 4 h. The released `xRIR_seen.pth` is evaluated under the same five-seed protocol as a fourth, reference-only row (K = 8 and K = 1; no training).
- **Estimand**: per arm × K, the mean ± sample SD over the five seeds of the mean error over the 6 217 queries (as TABLE_V1); paired per-query differences between arms (five-seed means, 20 000-resample query bootstrap, seed 0, 95 % percentile interval, 131-room cluster interval secondary) reported descriptively, no verdict.

## 3. Code (Coder rounds; TDD; < 200 lines per commit; Fable review per round)

| file | change | tests |
|---|---|---|
| `train_xRIR_backbone.py` | `--protocol {unseen,seen}` (default `unseen` = today's behaviour; `seen` imports `treble_multi_room_dataset.treble_xRIR_seen_dataset.xRIR_Dataset` for both splits); recorded in `args.json`; `train_batches_per_epoch` is whatever the loader gives (9 265 for `seen`) and the exp_04 A4 overflow assertion still applies | default path: state-dict/init/fixed-seed-step bit-identical to the pre-change script for both backbones and both yaw paths (GPU 1 if free, else deferred and stated); `seen` path: dataset sizes 296 454 / 6 217, a real batch loads, one step finite |
| `tools/exp04_launcher.py` (+ `exp04_launch.sh`) | `--protocol seen` selects arm roots `ckpt/exp07/<arm>/`, golden argv files `seen_protocol_results_assets/argv_golden_seen_{simple,cyl,aug}.txt`, the control-parity diff allows exactly `protocol` (and, for `seen_aug`, the three yaw flags) to differ from exp_01's control, `train_batches_per_epoch == 9265` asserted for `full`, banner `yaw_aug ENABLED …` for `seen_aug` / `yaw_aug DISABLED` for the others, per-arm ledgers, the exp_05 probe/receipt protocol reused (`probe --protocol seen --backbone … [--yaw-aug 1]`, T_run ≤ 60 h, 1.5 × ceiling, epoch-1 ≤ 1.05 × T_epoch), the exp_04 M path and the exp_05 tier path byte-for-byte unchanged | guard tests as exp_05 round 1; `refuse-test` incl. a seen argv under an unseen root and vice versa |
| `tools/exp07_profiles.py` + `seen_protocol_results_assets/approved_digests.json` | profile `TABLE_SEEN_V1` (four roles: seen_simple, seen_cyl, seen_aug, released_seen; checkpoint paths; the released checkpoint's sha256 pinned now, the three new ones after training; seen dataset identity: 6 217 queries / 131 rooms, query digest, content inventory; the ten seen manifest hashes; K, grid [0], condition P, batch 16, TF32 off) and `PAIRS_SEEN_V1` (descriptive pairs: cyl − simple, aug − simple, at K = 8 and K = 1); approval JSON as exp_04 A9 (`closures.{evaluator, writer, training_launcher (list), training, producer_results_table, producer_pairs}`, `checkpoints.{seen_simple, seen_cyl, seen_aug}`) | literal, immutable, hashes equal the manifest files |
| `tools/results_table.py` | accepts `--profile TABLE_SEEN_V1` from `tools.exp07_profiles` with exp_07's approval JSON (profile registry keyed by name → module; the exp_04 behaviour unchanged) | exp_04 tests unchanged; a seen-profile synthetic fixture |
| `tools/exp07_pairs.py` | descriptive paired producer (reuses `paired_compare` admission and bootstrap helpers; outputs canonical JSON + summary + sidecar; `decision_driving: false`) | synthetic fixture; refusals |
| record | `seen_protocol_results_assets/{make_results_md,make_results_html,bind_provenance,check_record}.py` = exp_04's generators parameterised by profile (copy with the exp_04 modules loaded by path, or a thin wrapper) rendering the combined seen + unseen table (unseen rows read from exp_04's bound `TABLE_V1.json`) and the LaTeX block | byte-stable; every number from JSON |

## 4. Validation ladder and acceptance

1 static; 2 `pytest tests/` (full, both GPUs visible when they free); 3 default-protocol parity (bit-identical, GPU); 4 seen-protocol readback (sizes, a real batch, finite step) and a 3-batch alignment audit for `seen_aug` through `tools.yaw_aug --audit` on the seen loader (new `--protocol seen` option on the audit CLI); 5 smoke through the launcher for each arm (`--no-save`, 3 micro-batches, banner check); 6 fit/timing probe per arm on a free GPU (receipt; T_run ≤ 60 h; expected ≈ 27.5 h each); 7 launch with the exp_05 acceptance criteria (epoch-1 ≤ 1.05 × T_epoch, `epoch_NNN.pth` × 12, no resume, certified completion). Evaluations: 30 runs + 10 released-checkpoint runs; the reference manifests are built and hashed before any evaluation.

## 5. Schedule and budget

Training 3 × ≈ 27.5 h ≈ 83 GPU-h; evaluations ≈ 5 h; total ≈ 88 GPU-h. GPU availability (as of 2026-09-15 10:30): GPU 0 busy with exp_05 L_simple until ≈ 2026-09-16 23:30 then exp_05 evaluations until ≈ 2026-09-17 08:00; GPU 1 busy with exp_05 L_cylindrical until ≈ 2026-09-16 10:00, its evaluations until ≈ 13:00, then promised to exp_06 (31 h pretraining + 6 h HAA). Two orderings, both finishing before the 2026-09-24 deadline:
- **A (exp_07 before exp_06 on GPU 1)**: GPU 1 `seen_simple` 2026-09-16 13:00 → 09-17 17:00, then `seen_aug` → 09-18 21:00; GPU 0 `seen_cyl` 09-17 08:00 → 09-18 12:00; evaluations 09-18 21:00 → 09-19 02:00; table 09-19. exp_06 then takes GPU 0 from 09-18 12:00 (→ ≈ 09-20 01:00).
- **B (keep the exp_06 promise)**: GPU 1 exp_06 09-16 13:00 → 09-18 02:00; GPU 0 `seen_cyl` 09-17 08:00 → 09-18 12:00, `seen_simple` → 09-19 16:00; GPU 1 `seen_aug` 09-18 02:00 → 09-19 06:00; evaluations → 09-19 21:00; table 09-20.
Decision requested from Yixun (§7). Cumulative ceiling per arm 1.5 × probe projection; one retry at most.

## 6. Deliverables

`seen_protocol_results.md` (tables from canonical JSON: seen rows ± SD, the released reference row, paired differences with intervals), the combined LaTeX table (seen + unseen, both K), `seen_protocol_01_results.html`, `seen_protocol_analysis.md`, `_params_set_up.md`, `_command.md`, `_worklog.md`, `commits_seen_protocol.md`, bound record (`ckpt/exp07/binding_report_*.json`, `check_record.py` exit 0). The living table `worklog/worklog_yixun/model_comparison.md` gains the seen rows (a second generated block).

## 7. Decisions requested

1. Ordering A or B for GPU 1 (default if no answer by 2026-09-16 12:00: **B**, the promise to exp_06 stands).
2. Include the released `xRIR_seen.pth` reference row (recommended; five-seed protocol, 10 runs ≈ 1.3 h).
3. K = 1 rows for the seen split (recommended, needed for the table's K = 1 rows; cost ≈ 1.5 h of evaluation).

## 8. Risks

Contamination of the released checkpoint's row is the authors' own protocol (not ours). `train_batches_per_epoch` 9 265 ≠ 9 261: the yaw-aug counter changes (A4 makes it the loader length), so the `seen_aug` offsets differ from exp_04's at the same (epoch, batch) — intended (different data order anyway) and recorded. Host contention with exp_05/exp_06 evaluations is the only timing risk; no gate beyond epoch 1 depends on it.
