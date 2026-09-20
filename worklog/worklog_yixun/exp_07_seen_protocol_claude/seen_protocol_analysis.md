# exp_07 seen_protocol — analysis (Planner, 2026-09-20)

Status: written after the end-game at main `0e47341`. Every number is taken from the generated `seen_protocol_results.md` / `seen_protocol_01_results.html` / `seen_protocol_results_assets/table_seen_unseen.tex` (rendered by the reviewed generators from the canonical `ckpt/exp07/results/{TABLE_SEEN_V1, PAIRS_SEEN_V1_seen_cyl, PAIRS_SEEN_V1_seen_aug, PAIRS_SEEN_V1_released_seen}.json` and exp_04's bound unseen `TABLE_V1.json`), bound by `binding_report_20260920T061550320247Z.json` (sha256 e5432f7fb526b8dd8cbcb34f2aa94a8a5b350f31cc5c88d405412e1e17a544d5, `check_record.py ckpt/exp07` exit 0). Seen split = the authors' seen protocol (6 217 test queries from rooms present in training), five evaluation seeds 42–46, K = 8 and K = 1, epoch-12 checkpoints, one training seed per arm. The paired profile `PAIRS_SEEN_V1` is registered as DESCRIPTIVE (checkpoint-conditional intervals; no pre-registered hypothesis, no verdict, no superiority claim).

## 1. What was tested

The unseen-protocol models of exp_04 cannot fill the seen columns of the paper's Table 1 (5 124 of the 6 217 seen-test files are in the unseen training set), so three arms were retrained under the seen protocol with the identical recipe and 12-epoch budget: xRIR (SimpleViT, `seen_simple`), CylindricalViT xRIR (`seen_cyl`) and YawAug-xRIR (`seen_aug`). Gate (plan §2): the released `xRIR_seen.pth` evaluated under our seen protocol must reproduce the historical seen row — it did (K = 8: EDT 0.0384 ± 0.0001 s, C50 1.024 ± 0.002 dB, T60 7.27 ± 0.005 % vs 0.0389 / 1.029 / 7.27; `CALIBRATION_SEEN_V1.json` passed), which validates the pipeline before any new-arm number was read.

## 2. Results (seen split; five-seed means, SDs in the LaTeX footnote)

| Arm | K = 8: EDT (ms) | C50 (dB) | T60 (%) | K = 1: EDT (ms) | C50 (dB) | T60 (%) |
|---|---|---|---|---|---|---|
| xRIR (seen_simple) | 34.41 | 0.908 | 6.89 | 57.55 | 1.490 | 10.61 |
| CylindricalViT (seen_cyl) | 34.10 | 0.890 | 6.92 | 60.60 | 1.527 | 11.12 |
| YawAug-xRIR (seen_aug) | 35.07 | 0.917 | 6.84 | 56.75 | 1.472 | 10.27 |
| released seen checkpoint (reference) | 38.40 | 1.024 | 7.27 | 62.76 | 1.613 | 10.37 |

**Descriptive paired intervals (relative change vs seen_simple, query-level bootstrap; room-cluster intervals in the canonical JSON).** seen_cyl at K = 8: EDT −0.9 % [−2.2, +0.4], C50 −2.0 % [−3.2, −0.9], T60 +0.5 % [−0.5, +1.5]; at K = 1: EDT +5.3 % [+4.7, +6.0], C50 +2.5 % [+2.0, +3.0], T60 +4.7 % [+4.0, +5.5]. seen_aug at K = 8: EDT +1.9 % [+0.6, +3.2], C50 +1.0 % [−0.2, +2.1], T60 −0.6 % [−1.4, +0.2]; at K = 1: EDT −1.4 % [−1.7, −1.1], C50 −1.2 % [−1.5, −0.9], T60 −3.2 % [−3.6, −2.8]. Released vs seen_simple at K = 8: EDT +11.6 % [+9.4, +13.8], C50 +12.8 % [+10.8, +14.8], T60 +5.5 % [+3.8, +7.2].

Reading: (i) all three retrained arms are better than the released seen checkpoint on every metric at K = 8 by 5–13 %, so the same recipe reproduces and exceeds the paper's seen row; (ii) at K = 8 the cylindrical encoder shows a small C50 advantage (−2.0 %, interval excluding 0 at query level, not at room level) and an EDT interval that includes 0 — the seen split, where every room is in training, leaves less room for the encoder's geometry prior than the unseen split (where exp_04 measured EDT −4.2 %); (iii) yaw augmentation costs EDT at K = 8 (+1.9 %) and is best at K = 1 on all three metrics (−1.2 to −3.2 %), while the cylindrical model is worst at K = 1 (+2.5 to +5.3 %) — the single-reference reversal of exp_03/exp_04/exp_05 appears on the seen split as well; (iv) the unseen columns of the combined table are exp_04's bound five-seed values, unchanged.

## 3. Reading for the paper

Table `tab:seen_unseen` gives the seen and unseen protocols side by side for the three recipe arms with the released checkpoint as an external reference (unknown training budget, not compared). Claims must stay descriptive for the seen split: no hypothesis was pre-registered for the seen pairings, the intervals are conditional on one training realisation per arm, and the bolding marks the smallest mean per column and K among the three recipe arms only.

## 4. Limitations and disclosures

One training seed per arm; seed SDs are over evaluation seeds only. Timing co-tenancy (wall time only): seen_simple shared GPU 0 with an unattributed measurement job on 2026-09-17 from ≈ 14:20 to ≈ 01:30 (+13–35 % on epochs 5–8) and with a stuck test suite (≈ 0.9 GB) 19:22–21:14; seen_cyl and seen_aug ran alone. Evaluations: seen_simple's ten ran at reviewed commit `ddfcb66`; seen_cyl's first run at `38a6727` and its other nine at `02bf1e2` after a stray untracked helper script made the launcher refuse them (worklog-only commits, closures unchanged); seen_aug's ten at `e9a0ecf`; the released K = 8 runs at `8273943` (gate) and K = 1 at `f5e1cf6`. exp_05's attempt relocation showed the exp_07 binder needed the same fix; it was made and reviewed (rounds 11–12) before this bind. Pre-launch integrative review: Codex round 9 (`full`) approved the branch for merge and GPU work before the gate.
