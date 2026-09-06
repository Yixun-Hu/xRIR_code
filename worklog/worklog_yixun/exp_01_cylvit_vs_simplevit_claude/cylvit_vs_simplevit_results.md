# Results — cylvit_vs_simplevit

All numbers: AcousticRooms **unseen** test split unless stated, 6337 queries from 17 held-out rooms, K=8. Lower is better everywhere. `*` = nominal 95 % paired bootstrap CI (10,000 resamples) excludes 0; no multiplicity adjustment.

> **Correction (2026-09-05, from the Codex review of the results pages).** The original record claimed identical reference RIRs for both models. That is false: `eval_xRIR_backbone.py` draws the DataLoader base seed from the torch RNG *after* constructing the model, and `xRIR_Cyl.__init__` consumes more RNG state than `xRIR.__init__`, so the worker NumPy seeds — and hence the `np.random.choice` reference draws — differ between the two models. A replay (`_worklog.md`, 23:35 entry) found different reference sets for 722 of the first 3000 queries (24.1 %): receivers with exactly 8 candidate sources (2259 queries) necessarily get the same set; of the 221 queries with 9 candidates, 91 % differed; the 520 queries with 6–7 candidates are drawn *with replacement* (`np.random.choice(..., replace=True)` fallback) and all of them differed. Griffin-Lim phase initialisation was also unseeded and therefore unpaired. Both are nuisance factors whose expected contribution to the paired difference is zero; the realised contribution in these arrays is unknown, which is why the properly paired re-evaluation is scheduled. Queries themselves are paired. **Two CIs are now reported:** query-level (inference about this fixed split) and room-cluster (17 rooms; inference about unseen rooms).

## 1. Baseline reproduction (released checkpoints, untouched eval scripts, 2026-09-04)

| Split | n | EDT (s) | C50 (dB) | T60 (%) |
|---|---|---|---|---|
| Unseen, paper Table 1 | | 0.055 | 1.457 | 10.x |
| Unseen, reproduced | 6337 | 0.0549 | 1.358 | 9.69 |
| Seen, paper Table 1 | | 0.038 | 0.940 | 8.13 |
| Seen, reproduced | 6217 | 0.0389 | 1.029 | 7.27 |

## 2. Per-epoch means, cylindrical / control

    epoch       lr |               EDT (s) |              C50 (dB) |               T60 (%) |          log-STFT MSE |             test loss
        1  4.6e-04 |       0.0577 / 0.0586 |         1.497 / 1.466 |         11.77 / 12.11 |       0.4618 / 0.4278 |     0.01790 / 0.01836
        2  2.2e-04 |       0.0526 / 0.0502 |         1.428 / 1.350 |          8.91 / 10.27 |       0.4266 / 0.4193 |     0.01772 / 0.01685
        3  1.0e-04 |       0.0458 / 0.0491 |         1.276 / 1.252 |          9.95 / 10.28 |       0.4350 / 0.4273 |     0.01619 / 0.01663
        4  4.6e-05 |       0.0453 / 0.0502 |         1.244 / 1.225 |          10.00 / 9.55 |       0.4153 / 0.4369 |     0.01597 / 0.01618
        5  2.2e-05 |       0.0448 / 0.0479 |         1.257 / 1.276 |           9.72 / 9.81 |       0.4087 / 0.4125 |     0.01601 / 0.01611
        6  1.0e-05 |       0.0455 / 0.0471 |         1.233 / 1.256 |           9.54 / 9.68 |       0.4111 / 0.4128 |     0.01595 / 0.01612
        7  4.6e-06 |       0.0462 / 0.0470 |         1.241 / 1.244 |           9.83 / 9.50 |       0.4193 / 0.4226 |     0.01583 / 0.01602
        8  2.2e-06 |       0.0454 / 0.0467 |         1.216 / 1.256 |           9.55 / 9.56 |       0.4164 / 0.4128 |     0.01586 / 0.01610
        9  1.0e-06 |       0.0454 / 0.0475 |         1.236 / 1.236 |           9.50 / 9.66 |       0.4065 / 0.4230 |     0.01581 / 0.01604
       10  4.6e-07 |       0.0454 / 0.0469 |         1.242 / 1.240 |           9.48 / 9.50 |       0.4144 / 0.4218 |     0.01582 / 0.01600
       11  2.2e-07 |       0.0456 / 0.0472 |         1.231 / 1.222 |           9.50 / 9.58 |       0.4087 / 0.4185 |     0.01588 / 0.01606
       12  1.0e-07 |       0.0451 / 0.0473 |         1.241 / 1.234 |           9.46 / 9.59 |       0.4189 / 0.4249 |     0.01590 / 0.01617
    paper          |                 0.055 |                 1.457 |                     - |                     - |                     -
    relsd          |                0.0549 |                 1.358 |                  9.69 |                     - |                     -
    

Test loss per epoch (from `history.jsonl`, batch-averaged): cyl 0.01790, 0.01768, 0.01621, 0.01599, 0.01599, 0.01596, 0.01581, 0.01588, **0.01579**, 0.01580, 0.01588, 0.01590; control 0.01835, 0.01686, 0.01664, 0.01619, 0.01612, 0.01610, 0.01603, 0.01610, 0.01602, **0.01597**, 0.01606, 0.01618. Cylindrical lower in every epoch from 3 on.

## 3. Paired difference averaged over epochs 5–12 (8 checkpoint pairs; per-sample differences averaged across epochs, then bootstrapped over queries or over rooms)

    metric               cyl      ctrl      diff   rel %       query-level 95% CI      room-cluster 95% CI  epochs cyl lower
    EDT (s)           0.0454    0.0472   -0.0018    -3.7 [  -0.0024,   -0.0011]* [  -0.0039,   +0.0001]         8/8
    C50 (dB)           1.237     1.246    -0.008    -0.7 [   -0.024,    +0.008]  [   -0.035,    +0.015]         5/8
    T60 (%)             9.57      9.61     -0.04    -0.4 [    -0.12,     +0.04]  [    -0.22,     +0.15]         7/8
    log-STFT MSE      0.4130    0.4186   -0.0056    -1.3 [  -0.0092,   -0.0022]* [  -0.0180,   +0.0028]         7/8
    test loss        0.01588   0.01608  -0.00020    -1.2 [ -0.00029,  -0.00010]* [ -0.00039,  +0.00002]         8/8

Query-level: EDT, log-STFT MSE and test loss favour cylindrical with CIs excluding 0; C50 and T60 show no detected difference. Room-cluster (17 rooms): every interval includes 0 (EDT [−0.0039, +0.0001], loss [−0.00039, +0.00002] just barely), so the advantage is established for this test split, not for the population of rooms. Per-room mean EDT difference (cyl − ctrl) is negative in 11/17 rooms; the two largest rooms drive most of it (Cafe_idx_1 −0.0082 over 922 queries, Auditorium_idx_1 −0.0026 over 1000).

## 4. Best-test-loss checkpoints (cyl epoch 9 vs control epoch 10)

    metric               cyl      ctrl      diff   rel %       query-level 95% CI      room-cluster 95% CI
    EDT (s)           0.0454    0.0469   -0.0016    -3.4 [  -0.0022,   -0.0010]* [  -0.0035,   +0.0000] 
    C50 (dB)           1.236     1.240    -0.004    -0.3 [   -0.020,    +0.013]  [   -0.032,    +0.023] 
    T60 (%)             9.50      9.50     -0.00    -0.0 [    -0.09,     +0.08]  [    -0.19,     +0.19] 
    log-STFT MSE      0.4065    0.4218   -0.0153    -3.6 [  -0.0188,   -0.0118]* [  -0.0292,   -0.0043]*
    test loss        0.01581   0.01600  -0.00019    -1.2 [ -0.00029,  -0.00010]* [ -0.00033,  -0.00003]*

## 5. Seen split, epoch-9 checkpoints (6217 queries)

| Subset | n | EDT cyl / ctrl | C50 cyl / ctrl | T60 cyl / ctrl | Significant (cyl better) |
|---|---|---|---|---|---|
| All | 6217 | 0.0337 / 0.0347 | 0.873 / 0.895 | 6.84 / 6.92 | EDT, C50, T60, MSE, loss |
| Held-out rooms (clean) | 1093 | 0.0587 / 0.0612 | 1.560 / 1.607 | 10.08 / 10.29 | MSE only |
| Training rooms, held-out receivers (contaminated) | 5124 | 0.0284 / 0.0291 | 0.727 / 0.744 | 6.15 / 6.20 | EDT, C50, MSE, loss |

Contamination: both models trained on all receivers of the 244 non-test rooms, so the 5124 "training rooms" queries were training targets. Released seen checkpoint (trained on the seen split) reproduced at 0.039 / 1.03 / 7.27 for reference only.

## 6. Cost

Training 12 epochs: cylindrical 31.4 h, control 28.2 h (one A6000 each, ≈32–36 samples/s). Per-epoch full-split eval: ≈18 min per model. Data mirror: 43 min.

Sources: `ckpt/backbone_comparison_summary.txt` and the canonical `ckpt/backbone_comparison_stats.json` (both from `tools/summarize_epochs.py --avg-epochs 5-12 --json`), `ckpt/xRIR_{cyl,simple}_8_shot/{history.jsonl, eval_unseen_epochNN.json, per_sample_unseen_epochNN.json, per_sample_seen_epoch09.json}`, `ckpt/baseline_reproduction.json`.
