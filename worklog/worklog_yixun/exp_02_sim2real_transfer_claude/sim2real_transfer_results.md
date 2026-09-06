# Results — sim2real_transfer

Hearing Anything Anywhere, four rooms, DiffRIR splits (12 training RIRs per room; validation split for checkpoint selection; per-room test split: classroom 463, dampened 198, hallway 423, complex 198). Two-stage fine-tuning per the paper (stage 1 on classroom + hallway + complex, stage 2 per room), K=8 references drawn deterministically per query (identical for every model and seed), Griffin-Lim inversion (phase initialisation unseeded — a nuisance with zero expected effect on paired differences, realised contribution unknown), EDT / C50 / T60 on the 9600-sample 22.05 kHz waveforms; T60 omitted for the dampened room (paper). Three fine-tuning seeds per init; means ± std over seeds. Canonical numbers: `ckpt/sim2real/stats.json` and the hash-bound `ckpt/sim2real/summary.txt` (`sim_to_real/summarize_haa.py --json --summary`, strict completeness gate passed 2026-09-06 01:20).

## 1. Per-room means (EDT s / C50 dB / T60 %), mean ± std over seeds

    model                       | class_room                             | dampened_room                          | hallway                                | complex_room                           
                                |       EDT (s)     C50 (dB)      T60 (%)|       EDT (s)     C50 (dB)      T60 (%)|       EDT (s)     C50 (dB)      T60 (%)|       EDT (s)     C50 (dB)      T60 (%)
    paper xRIR K=8              |         0.093        1.628         6.25|         0.044        3.302            -|         0.062        0.954          3.2|         0.077        1.688         4.33
    released zero-shot (n=1)    |        0.1667        3.373         7.87|        0.0749        4.757            -|        0.1523        2.096         6.82|        0.1324        2.567         6.29
    released fine-tuned (n=3)   | 0.0958±0.0015  1.975±0.177    4.32±0.09| 0.0380±0.0005  2.747±0.047            -| 0.0948±0.0031  1.702±0.049    4.42±0.13| 0.1072±0.0019  2.233±0.048    4.96±0.13
    released_repomaps fine-tuned (n=1)|        0.0951        1.822         4.03|        0.0374        2.712            -|        0.0998        1.793         4.60|        0.1022        2.157         4.96
    control zero-shot (n=1)     |        0.1179        1.933         7.35|        0.0723        4.670            -|        0.1869        2.389         7.47|        0.1561        2.765         6.92
    control fine-tuned (n=3)    | 0.0812±0.0007  1.501±0.008    4.31±0.13| 0.0384±0.0002  2.847±0.028            -| 0.0611±0.0009  1.135±0.008    2.68±0.00| 0.0765±0.0003  1.666±0.021    4.94±0.07
    cyl zero-shot (n=1)         |        0.1199        1.915         7.82|        0.0719        4.632            -|        0.1840        4.183         9.91|        0.1095        2.172         5.77
    cyl fine-tuned (n=3)        | 0.0887±0.0008  1.630±0.038    4.00±0.24| 0.0385±0.0021  2.863±0.132            -| 0.0874±0.0018  2.052±0.084    4.16±0.09| 0.0821±0.0006  1.736±0.030    5.05±0.22
    

## 2. Paired cylindrical − control (fine-tuned), pooled over the three seeds

Two intervals per cell: query-cluster (rows of one query across seeds resampled together; conditional on the realised training seeds) and two-way (queries and training seeds both resampled). Verdicts use the two-way interval. Nominal 95 %, no multiplicity adjustment across the 11 cells. Negative favours cylindrical.

    room           metric           cyl      ctrl      diff       query-cluster CI             two-way CI  cyl<ctrl  valid  (seeds seed0, seed1, seed2)
    class_room     EDT (s)       0.0887    0.0812   +0.0074 [  +0.0031,   +0.0119]* [  +0.0028,   +0.0122]*      0.46   1389  per-seed: seed0:+0.0081 (n=463), seed1:+0.0059 (n=463), seed2:+0.0082 (n=463)
    class_room     C50 (dB)       1.630     1.501    +0.129 [   +0.058,    +0.200]* [   +0.045,    +0.226]*      0.45   1389  per-seed: seed0:+0.104 (n=463), seed1:+0.192 (n=463), seed2:+0.091 (n=463)
    class_room     T60 (%)         4.00      4.31     -0.31 [    -0.50,     -0.12]* [    -0.78,     +0.07]       0.53   1389  per-seed: seed0:-0.05 (n=463), seed1:-0.79 (n=463), seed2:-0.08 (n=463)
    dampened_room  EDT (s)       0.0385    0.0384   +0.0001 [  -0.0031,   +0.0032]  [  -0.0040,   +0.0041]       0.51    594  per-seed: seed0:-0.0001 (n=198), seed1:+0.0028 (n=198), seed2:-0.0025 (n=198)
    dampened_room  C50 (dB)       2.863     2.847    +0.016 [   -0.141,    +0.170]  [   -0.227,    +0.256]       0.49    594  per-seed: seed0:+0.014 (n=198), seed1:+0.213 (n=198), seed2:-0.179 (n=198)
    hallway        EDT (s)       0.0874    0.0611   +0.0264 [  +0.0185,   +0.0345]* [  +0.0177,   +0.0351]*      0.40   1269  per-seed: seed0:+0.0248 (n=423), seed1:+0.0241 (n=423), seed2:+0.0301 (n=423)
    hallway        C50 (dB)       2.052     1.135    +0.916 [   +0.769,    +1.069]* [   +0.743,    +1.102]*      0.29   1269  per-seed: seed0:+0.865 (n=423), seed1:+0.850 (n=423), seed2:+1.035 (n=423)
    hallway        T60 (%)         4.16      2.68     +1.48 [    +1.15,     +1.83]* [    +1.12,     +1.85]*      0.32   1269  per-seed: seed0:+1.46 (n=423), seed1:+1.38 (n=423), seed2:+1.61 (n=423)
    complex_room   EDT (s)       0.0821    0.0765   +0.0057 [  +0.0029,   +0.0084]* [  +0.0022,   +0.0092]*      0.42    594  per-seed: seed0:+0.0056 (n=198), seed1:+0.0064 (n=198), seed2:+0.0049 (n=198)
    complex_room   C50 (dB)       1.736     1.666    +0.070 [   +0.013,    +0.125]* [   -0.014,    +0.150]       0.43    594  per-seed: seed0:+0.084 (n=198), seed1:+0.118 (n=198), seed2:+0.007 (n=198)

**Verdicts (two-way CI):** control lower on classroom EDT (+9 %), classroom C50 (+9 %), hallway EDT (+43 %), hallway C50 (+81 %), hallway T60 (+55 %), complex EDT (+7 %); no detected difference on classroom T60, dampened EDT, dampened C50, complex C50, complex T60. Cylindrical is not lower in any cell. The significant cells have the same sign in all three seeds.

## 3. Paper Table 2 reproduction (released checkpoint, our fine-tuning protocol)

| Room | Paper xRIR K=8 | Released ckpt, fine-tuned (3 seeds) | Released ckpt, as-released depth maps (1 seed) | Our SimpleViT control, fine-tuned |
|---|---|---|---|---|
| Classroom | 0.093 / 1.628 / 6.25 | 0.096 / 1.98 / 4.32 | 0.095 / 1.82 / 4.03 | 0.081 / 1.50 / 4.31 |
| Dampened | 0.044 / 3.302 / – | 0.038 / 2.75 / – | 0.037 / 2.71 / – | 0.038 / 2.85 / – |
| Hallway | 0.062 / 0.954 / 3.20 | 0.095 / 1.70 / 4.42 | 0.100 / 1.79 / 4.60 | 0.061 / 1.14 / 2.68 |
| Complex | 0.077 / 1.688 / 4.33 | 0.107 / 2.23 / 4.96 | 0.102 / 2.16 / 4.96 | 0.077 / 1.67 / 4.94 |

The released checkpoint reproduces the paper on the classroom (EDT) and dampened room and is *worse* than Table 2 on the hallway and complex room; the as-released depth maps give the same picture, so the depth-map substitution is not the cause. Our retrained SimpleViT control matches or beats Table 2 on EDT in every room and on C50 in three of four.

## 4. Zero-shot (no fine-tuning)

| Init | Classroom | Dampened | Hallway | Complex |
|---|---|---|---|---|
| Released | 0.167 / 3.37 / 7.87 | 0.075 / 4.76 / – | 0.152 / 2.10 / 6.82 | 0.132 / 2.57 / 6.29 |
| Control | 0.118 / 1.93 / 7.35 | 0.072 / 4.67 / – | 0.187 / 2.39 / 7.47 | 0.156 / 2.77 / 6.92 |
| Cylindrical | 0.120 / 1.92 / 7.82 | 0.072 / 4.63 / – | 0.184 / 4.18 / 9.91 | 0.110 / 2.17 / 5.77 |

Paper Table 6 (classroom, pretrained without fine-tuning): 0.204 / 3.43. Zero-shot, the cylindrical model is markedly worse than the control on hallway C50/T60 and better on the complex room; fine-tuning removes its complex-room advantage and leaves the hallway deficit.

## 5. Cost

Each fine-tuning job (stage 1: 1000 full-batch epochs with validation every 10; stage 2: 4 rooms × 200 epochs; 4 evals) took 46–51 min on one A6000; 10 jobs + 3 zero-shot evals over two GPUs: 21:01–01:10 (4.2 h).
