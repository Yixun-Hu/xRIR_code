# Results — yaw_rotation_degradation

AcousticRooms unseen test split, all 6337 queries in 17 rooms; references per query fixed by the hashed manifest `47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d` (identical for every model and angle; its seed and shot count are recorded in `_params_set_up.md`). Evaluation settings per run are in §6 (from the runs' own metadata). The whole scene (receiver-centred depth panorama and all source / reference-source coordinates) is rotated in yaw by k panorama columns; the true RIR and target are unchanged. Condition P = pinned k=0 alignment; condition E = end to end. r_k = (mean error at k - mean error at 0 deg) / mean error at 0 deg over the queries valid at both angles; the bootstrap resamples those pairs and recomputes the whole ratio inside each resample. n_boot=20000 seed=0 family=18 tests -> adjusted level 0.00278; the query-level and room-cluster intervals on r_k and D_k are both two-sided 99.72% (section 3's k=0 table is nominal 95%, unadjusted) (99.72 % query-level and 99.72 % room-cluster intervals, 17 rooms resampled; k = 0 at nominal 95.00 %). T60 is descriptive only. Canonical numbers: `ckpt/yaw_rotation/full_stats.json` (sha256 `5d4afce8b81c47e148b714f351c670c4bb32e30cd4cc1f07587a73ba05a3809c`, copy in `_results_assets/`) and the hash-bound `ckpt/yaw_rotation/full_summary.txt` (sha256 `52f1e62eced6cdc55417e97876d3ca85fbbac37832a3a7271dfa8ed745db45f9`, copy `yaw_rotation_degradation_full_summary.txt`); `tools/summarize_yaw.py --mode full` reported `valid_for_confirmatory: true`.

## 1. Pre-registered verdicts and distance to the threshold (producer fields h1 / h2 / h1.bounds)

- **H1: not supported** — rule: H1 (joint yaw rotation substantially degrades the model): supported if the adjusted lower bound of r exceeds +10% for EDT or C50 at any angle k != 0, condition P. Wording: Verdict wording (--mode full only; the other modes record "not evaluated (<mode> mode)"): the same-budget control (role primary) and the released checkpoint (role released) are reported separately and never pooled -- both pass: "supported and replicated"; the control only: "supported for the same-budget model only"; the released checkpoint only: "replication only"; neither: "not supported". A role with no run in the summary counts as not passing. Roles evaluated: primary, released (simple, checkpoints/xRIR_unseen.pth). The +10 % margins at 0° are control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth): 0.0047 s EDT error, 0.124 dB C50 error; released (simple, checkpoints/xRIR_unseen.pth): 0.0055 s EDT error, 0.136 dB C50 error.
- **H2: not evaluable (H1 has no passing cell)** — rule: H2 (the cylindrical backbone degrades less): D_k = r_cyl - r_control, paired on the same resamples; decided cell by cell at the H1-passing cells of the primary model, where a cell passes when D_k is negative and its adjusted upper bound is below zero; the aggregate is "supported" when every such cell passes, "partially supported" when some do, "not supported" when none does and "not evaluable (H1 has no passing cell)" when H1 has no passing cell. (0 of 0 H1-passing cells). D_k and the TOST results are in §4.
- Bootstrap convergence: every decision-driving bound recomputed with a second seed, as a fraction of its own interval width (limit 0.10). 72 bounds; max movement 0.0378 of the interval width (limit 0.1): pass.
- Distance to the +10 % threshold (condition P, non-zero acoustic angles):

    model                                                   metric          largest adjusted query-level upper bound   query-level bounds above threshold   largest adjusted room-cluster upper bound   room-cluster bounds above threshold
    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth)  EDT error         +3.76 % at +45.0°                    none                                    +5.65 % at +45.0°                    none
    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth)  C50 error         +7.52 % at +45.0°                    none                                   +11.37 % at +45.0°                    +22.5° (+10.99 %), +45.0° (+11.37 %)
    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth)    EDT error         +5.44 % at -45.0°                    none                                    +4.91 % at -90.0°                    none
    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth)    C50 error         +7.37 % at -45.0°                    none                                   +10.25 % at +45.0°                    +45.0° (+10.25 %)
    released (simple, checkpoints/xRIR_unseen.pth)          EDT error         +5.45 % at -45.0°                    none                                    +6.96 % at -45.0°                    none
    released (simple, checkpoints/xRIR_unseen.pth)          C50 error         +7.56 % at -45.0°                    none                                   +10.75 % at -45.0°                    -45.0° (+10.75 %)

## 2. Acoustic metrics per angle (EDT error [s], C50 error [dB], T60 error [%])

Columns: mean at 0°, mean at k, r_k, adjusted query-level interval (99.72 %), adjusted room-cluster interval (99.72 %), n paired-valid, fraction of usable baselines the angle broke (newly invalid). Condition P at the 10-angle grid, condition E at k = 0 and the 4 E angles.

    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) | condition P | EDT error (s)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0     0.0470     0.0473     +0.68 %     [-1.08, +2.40]     [-3.37, +2.77]   6337   0.0000
      -45.0     0.0470     0.0474     +0.84 %     [-1.51, +3.31]     [-1.02, +4.00]   6337   0.0000
      -22.5     0.0470     0.0472     +0.36 %     [-1.74, +2.51]     [-1.87, +2.30]   6337   0.0000
       -5.6     0.0470     0.0466     -0.82 %     [-1.98, +0.31]     [-1.94, +0.19]   6337   0.0000
       +0.0     0.0470     0.0470     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6     0.0470     0.0471     +0.34 %     [-0.77, +1.50]     [-0.76, +1.62]   6337   0.0000
      +22.5     0.0470     0.0475     +1.02 %     [-1.12, +3.12]     [-0.71, +3.79]   6337   0.0000
      +45.0     0.0470     0.0476     +1.34 %     [-1.01, +3.76]     [-0.84, +5.65]   6337   0.0000
      +90.0     0.0470     0.0476     +1.25 %     [-0.47, +2.96]     [-2.61, +3.06]   6337   0.0000
     +180.0     0.0470     0.0470     -0.06 %     [-1.74, +1.71]     [-4.87, +1.81]   6337   0.0000

    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) | condition P | C50 error (dB)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0      1.235      1.247     +0.94 %     [-0.81, +2.82]     [-2.70, +4.54]   6337   0.0000
      -45.0      1.235      1.291     +4.55 %     [+2.18, +7.09]     [-0.10, +9.42]   6337   0.0000
      -22.5      1.235      1.267     +2.58 %     [+0.44, +4.68]     [-1.09, +8.10]   6337   0.0000
       -5.6      1.235      1.235     -0.03 %     [-1.26, +1.21]     [-1.50, +0.50]   6337   0.0000
       +0.0      1.235      1.235     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6      1.235      1.245     +0.84 %     [-0.31, +2.01]     [-0.65, +2.05]   6337   0.0000
      +22.5      1.235      1.286     +4.12 %     [+1.86, +6.42]    [-1.07, +10.99]   6337   0.0000
      +45.0      1.235      1.298     +5.09 %     [+2.65, +7.52]    [-0.10, +11.37]   6337   0.0000
      +90.0      1.235      1.259     +1.95 %     [+0.26, +3.70]     [-3.13, +6.40]   6337   0.0000
     +180.0      1.235      1.232     -0.27 %     [-2.12, +1.56]     [-5.63, +3.49]   6337   0.0000

    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) | condition P | T60 error (%)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0       9.56       9.39     -1.79 %     [-2.95, -0.60]     [-4.78, +0.35]   6337   0.0000
      -45.0       9.56       9.41     -1.53 %     [-3.20, +0.17]     [-8.56, +2.82]   6337   0.0000
      -22.5       9.56       9.49     -0.74 %     [-2.14, +0.70]     [-3.11, +1.42]   6337   0.0000
       -5.6       9.56       9.49     -0.75 %     [-1.48, -0.03]     [-1.55, -0.06]   6337   0.0000
       +0.0       9.56       9.56     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6       9.56       9.65     +0.94 %     [+0.12, +1.73]     [-0.34, +1.63]   6337   0.0000
      +22.5       9.56       9.66     +1.04 %     [-0.44, +2.64]     [-1.95, +3.57]   6337   0.0000
      +45.0       9.56       9.59     +0.29 %     [-1.36, +2.01]     [-3.55, +3.59]   6337   0.0000
      +90.0       9.56       9.44     -1.25 %     [-2.53, +0.07]     [-4.32, +0.87]   6337   0.0000
     +180.0       9.56       9.22     -3.57 %     [-4.82, -2.34]     [-9.52, -0.49]   6337   0.0000

    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) | condition P | EDT error (s)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0     0.0449     0.0458     +1.98 %     [+0.51, +3.41]     [-1.06, +4.91]   6337   0.0000
      -45.0     0.0449     0.0463     +3.07 %     [+0.90, +5.44]     [+0.95, +4.78]   6337   0.0000
      -22.5     0.0449     0.0456     +1.54 %     [-0.22, +3.41]     [-0.54, +4.01]   6337   0.0000
       -5.6     0.0449     0.0446     -0.73 %     [-1.74, +0.29]     [-1.49, +0.07]   6337   0.0000
       +0.0     0.0449     0.0449     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6     0.0449     0.0450     +0.23 %     [-0.91, +1.41]     [-0.79, +1.22]   6337   0.0000
      +22.5     0.0449     0.0455     +1.22 %     [-0.75, +3.15]     [-1.17, +4.17]   6337   0.0000
      +45.0     0.0449     0.0461     +2.55 %     [+0.43, +4.75]     [+1.04, +4.22]   6337   0.0000
      +90.0     0.0449     0.0459     +2.05 %     [+0.58, +3.55]     [-0.52, +4.62]   6337   0.0000
     +180.0     0.0449     0.0454     +0.94 %     [-0.50, +2.36]     [-0.40, +1.96]   6337   0.0000

    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) | condition P | C50 error (dB)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0      1.226      1.248     +1.82 %     [+0.48, +3.22]     [-1.34, +5.39]   6337   0.0000
      -45.0      1.226      1.291     +5.34 %     [+3.26, +7.37]     [+0.52, +9.53]   6337   0.0000
      -22.5      1.226      1.265     +3.23 %     [+1.42, +5.10]     [-0.72, +7.97]   6337   0.0000
       -5.6      1.226      1.225     -0.01 %     [-1.09, +1.11]     [-1.00, +0.91]   6337   0.0000
       +0.0      1.226      1.226     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6      1.226      1.232     +0.49 %     [-0.63, +1.63]     [-0.81, +1.32]   6337   0.0000
      +22.5      1.226      1.267     +3.37 %     [+1.53, +5.22]     [-1.14, +8.55]   6337   0.0000
      +45.0      1.226      1.289     +5.19 %     [+3.09, +7.24]    [+0.43, +10.25]   6337   0.0000
      +90.0      1.226      1.258     +2.61 %     [+1.22, +4.10]     [-0.86, +8.04]   6337   0.0000
     +180.0      1.226      1.229     +0.27 %     [-1.22, +1.79]     [-1.35, +2.70]   6337   0.0000

    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) | condition P | T60 error (%)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0       9.44       9.36     -0.76 %     [-1.79, +0.26]     [-4.18, +1.42]   6337   0.0000
      -45.0       9.44       9.44     +0.08 %     [-1.58, +1.70]     [-6.33, +4.49]   6337   0.0000
      -22.5       9.44       9.42     -0.18 %     [-1.48, +1.13]     [-2.41, +1.66]   6337   0.0000
       -5.6       9.44       9.40     -0.43 %     [-1.13, +0.27]     [-1.20, +0.40]   6337   0.0000
       +0.0       9.44       9.44     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6       9.44       9.45     +0.14 %     [-0.58, +0.84]     [-1.20, +0.80]   6337   0.0000
      +22.5       9.44       9.59     +1.60 %     [+0.22, +2.97]     [-1.88, +4.87]   6337   0.0000
      +45.0       9.44       9.59     +1.66 %     [+0.06, +3.32]     [-1.65, +4.90]   6337   0.0000
      +90.0       9.44       9.39     -0.44 %     [-1.49, +0.61]     [-2.12, +0.66]   6337   0.0000
     +180.0       9.44       9.13     -3.23 %     [-4.26, -2.24]     [-9.51, +0.42]   6337   0.0000

    released (simple, checkpoints/xRIR_unseen.pth) | condition P | EDT error (s)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0     0.0551     0.0542     -1.70 %     [-4.46, +1.22]     [-3.23, +1.10]   6337   0.0000
      -45.0     0.0551     0.0564     +2.36 %     [-0.67, +5.45]     [+0.21, +6.96]   6337   0.0000
      -22.5     0.0551     0.0556     +0.76 %     [-1.86, +3.36]     [-1.15, +3.06]   6337   0.0000
       -5.6     0.0551     0.0549     -0.34 %     [-2.34, +1.67]     [-1.04, +1.13]   6337   0.0000
       +0.0     0.0551     0.0551     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6     0.0551     0.0548     -0.52 %     [-2.40, +1.30]     [-2.05, +1.20]   6337   0.0000
      +22.5     0.0551     0.0552     +0.16 %     [-2.52, +2.90]     [-2.74, +6.31]   6337   0.0000
      +45.0     0.0551     0.0546     -1.06 %     [-4.29, +2.29]     [-4.39, +6.48]   6337   0.0000
      +90.0     0.0551     0.0535     -2.88 %     [-5.85, +0.17]     [-5.17, +1.81]   6337   0.0000
     +180.0     0.0551     0.0539     -2.22 %     [-5.21, +1.06]     [-7.79, +1.51]   6337   0.0000

    released (simple, checkpoints/xRIR_unseen.pth) | condition P | C50 error (dB)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0      1.360      1.382     +1.59 %     [-1.14, +4.29]     [-3.02, +6.75]   6337   0.0000
      -45.0      1.360      1.424     +4.70 %     [+1.82, +7.56]    [-1.65, +10.75]   6337   0.0000
      -22.5      1.360      1.391     +2.23 %     [-0.15, +4.63]     [-1.36, +5.45]   6337   0.0000
       -5.6      1.360      1.365     +0.33 %     [-1.42, +2.05]     [-0.61, +1.62]   6337   0.0000
       +0.0      1.360      1.360     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6      1.360      1.360     -0.02 %     [-1.78, +1.73]     [-3.22, +2.39]   6337   0.0000
      +22.5      1.360      1.381     +1.54 %     [-0.97, +4.19]     [-2.54, +8.82]   6337   0.0000
      +45.0      1.360      1.394     +2.43 %     [-0.58, +5.55]     [-1.69, +8.85]   6337   0.0000
      +90.0      1.360      1.358     -0.20 %     [-3.28, +2.90]     [-2.56, +2.61]   6337   0.0000
     +180.0      1.360      1.369     +0.62 %     [-2.31, +3.60]     [-4.55, +6.05]   6337   0.0000

    released (simple, checkpoints/xRIR_unseen.pth) | condition P | T60 error (%)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0       9.70       9.87     +1.69 %     [-0.27, +3.66]     [-2.47, +5.33]   6337   0.0000
      -45.0       9.70      10.21     +5.18 %     [+2.91, +7.57]    [+0.14, +10.15]   6337   0.0000
      -22.5       9.70       9.86     +1.60 %     [-0.37, +3.64]     [-1.52, +6.46]   6337   0.0000
       -5.6       9.70       9.68     -0.25 %     [-1.48, +0.92]     [-1.18, +1.11]   6337   0.0000
       +0.0       9.70       9.70     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
       +5.6       9.70       9.85     +1.51 %     [+0.25, +2.81]     [-0.89, +2.72]   6337   0.0000
      +22.5       9.70      10.01     +3.20 %     [+1.12, +5.34]     [-2.20, +7.80]   6337   0.0000
      +45.0       9.70      10.18     +4.87 %     [+2.45, +7.29]     [-1.16, +9.48]   6337   0.0000
      +90.0       9.70       9.96     +2.67 %     [+0.67, +4.69]    [-2.11, +10.31]   6337   0.0000
     +180.0       9.70       9.47     -2.43 %     [-4.43, -0.46]     [-9.27, +1.48]   6337   0.0000

    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) | condition E | EDT error (s)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0     0.0470     0.0473     +0.68 %     [-1.08, +2.40]     [-3.37, +2.77]   6337   0.0000
      -22.5     0.0470     0.0472     +0.36 %     [-1.74, +2.51]     [-1.86, +2.30]   6337   0.0000
       +0.0     0.0470     0.0470     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5     0.0470     0.0475     +1.02 %     [-1.12, +3.12]     [-0.71, +3.79]   6337   0.0000
      +90.0     0.0470     0.0476     +1.25 %     [-0.47, +2.96]     [-2.61, +3.06]   6337   0.0000

    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) | condition E | C50 error (dB)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0      1.235      1.247     +0.94 %     [-0.81, +2.82]     [-2.70, +4.54]   6337   0.0000
      -22.5      1.235      1.267     +2.58 %     [+0.44, +4.68]     [-1.09, +8.10]   6337   0.0000
       +0.0      1.235      1.235     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5      1.235      1.286     +4.12 %     [+1.86, +6.42]    [-1.07, +10.99]   6337   0.0000
      +90.0      1.235      1.259     +1.95 %     [+0.26, +3.70]     [-3.13, +6.40]   6337   0.0000

    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) | condition E | T60 error (%)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0       9.56       9.39     -1.79 %     [-2.95, -0.60]     [-4.78, +0.35]   6337   0.0000
      -22.5       9.56       9.49     -0.74 %     [-2.14, +0.70]     [-3.11, +1.42]   6337   0.0000
       +0.0       9.56       9.56     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5       9.56       9.66     +1.04 %     [-0.44, +2.64]     [-1.95, +3.57]   6337   0.0000
      +90.0       9.56       9.44     -1.25 %     [-2.53, +0.07]     [-4.32, +0.87]   6337   0.0000

    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) | condition E | EDT error (s)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0     0.0449     0.0458     +1.98 %     [+0.51, +3.41]     [-1.06, +4.91]   6337   0.0000
      -22.5     0.0449     0.0456     +1.54 %     [-0.22, +3.41]     [-0.54, +4.01]   6337   0.0000
       +0.0     0.0449     0.0449     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5     0.0449     0.0455     +1.22 %     [-0.75, +3.15]     [-1.17, +4.17]   6337   0.0000
      +90.0     0.0449     0.0459     +2.05 %     [+0.58, +3.55]     [-0.52, +4.62]   6337   0.0000

    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) | condition E | C50 error (dB)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0      1.226      1.248     +1.82 %     [+0.48, +3.22]     [-1.35, +5.39]   6337   0.0000
      -22.5      1.226      1.265     +3.23 %     [+1.41, +5.10]     [-0.73, +7.97]   6337   0.0000
       +0.0      1.226      1.226     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5      1.226      1.267     +3.37 %     [+1.53, +5.22]     [-1.14, +8.55]   6337   0.0000
      +90.0      1.226      1.258     +2.61 %     [+1.22, +4.10]     [-0.86, +8.04]   6337   0.0000

    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) | condition E | T60 error (%)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0       9.44       9.36     -0.76 %     [-1.79, +0.25]     [-4.18, +1.42]   6337   0.0000
      -22.5       9.44       9.42     -0.18 %     [-1.48, +1.13]     [-2.41, +1.66]   6337   0.0000
       +0.0       9.44       9.44     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5       9.44       9.59     +1.60 %     [+0.22, +2.97]     [-1.88, +4.87]   6337   0.0000
      +90.0       9.44       9.39     -0.44 %     [-1.49, +0.61]     [-2.12, +0.66]   6337   0.0000

    released (simple, checkpoints/xRIR_unseen.pth) | condition E | EDT error (s)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0     0.0551     0.0542     -1.70 %     [-4.46, +1.22]     [-3.23, +1.10]   6337   0.0000
      -22.5     0.0551     0.0556     +0.76 %     [-1.86, +3.36]     [-1.15, +3.06]   6337   0.0000
       +0.0     0.0551     0.0551     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5     0.0551     0.0552     +0.16 %     [-2.52, +2.90]     [-2.74, +6.31]   6337   0.0000
      +90.0     0.0551     0.0535     -2.88 %     [-5.85, +0.17]     [-5.17, +1.81]   6337   0.0000

    released (simple, checkpoints/xRIR_unseen.pth) | condition E | C50 error (dB)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0      1.360      1.382     +1.59 %     [-1.14, +4.29]     [-3.02, +6.75]   6337   0.0000
      -22.5      1.360      1.391     +2.23 %     [-0.15, +4.63]     [-1.36, +5.45]   6337   0.0000
       +0.0      1.360      1.360     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5      1.360      1.381     +1.54 %     [-0.97, +4.19]     [-2.54, +8.82]   6337   0.0000
      +90.0      1.360      1.358     -0.20 %     [-3.28, +2.90]     [-2.56, +2.61]   6337   0.0000

    released (simple, checkpoints/xRIR_unseen.pth) | condition E | T60 error (%)
      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv
      -90.0       9.70       9.87     +1.69 %     [-0.27, +3.66]     [-2.47, +5.33]   6337   0.0000
      -22.5       9.70       9.86     +1.60 %     [-0.37, +3.64]     [-1.52, +6.46]   6337   0.0000
       +0.0       9.70       9.70     +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   6337   0.0000
      +22.5       9.70      10.01     +3.20 %     [+1.12, +5.34]     [-2.20, +7.80]   6337   0.0000
      +90.0       9.70       9.96     +2.67 %     [+0.67, +4.69]    [-2.11, +10.31]   6337   0.0000

## 3. Spectral metrics per angle (all 18 angles, conditions P and E)

loss = test loss = STFT L1 + 0.01 x decay; log_mse = log-STFT MSE; consistency = mean|log-spec(k) - log-spec(0)| (baseline 0, so no ratio). Intervals: adjusted query-level (99.72 %).

    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) | condition P
      deg   test loss =    r(loss)       adj. query CI  log-STFT MS   r(logmse)       adj. query CI   consistency =   (full names: loss = test loss = STFT L1 + 0.01 x decay; log_mse = log-STFT MSE; consistency = consistency = mean|log-spec(k) - log-spec(0)|)
     -135.0     0.01647    +2.02 %     [+0.94, +3.12]     0.42188    -0.10 %     [-1.43, +1.31]       0.13468
      -90.0     0.01620    +0.36 %     [-0.48, +1.17]     0.42395    +0.39 %     [-0.33, +1.09]       0.07682
      -67.5     0.01646    +1.94 %     [+0.98, +2.93]     0.42089    -0.34 %     [-1.49, +0.93]       0.11661
      -45.0     0.01641    +1.69 %     [+0.69, +2.71]     0.42245    +0.03 %     [-1.26, +1.41]       0.12521
      -22.5     0.01631    +1.06 %     [+0.23, +1.88]     0.42223    -0.02 %     [-1.03, +1.00]       0.10329
      -11.2     0.01616    +0.10 %     [-0.55, +0.76]     0.42201    -0.07 %     [-0.74, +0.61]       0.07262
       -5.6     0.01611    -0.19 %     [-0.63, +0.25]     0.42215    -0.04 %     [-0.45, +0.37]       0.04545
       -2.8     0.01612    -0.12 %     [-0.42, +0.17]     0.42208    -0.05 %     [-0.32, +0.21]       0.02744
       +0.0     0.01614    +0.00 %     [+0.00, +0.00]     0.42231    +0.00 %     [+0.00, +0.00]       0.00000
       +2.8     0.01619    +0.33 %     [+0.03, +0.62]     0.42138    -0.22 %     [-0.51, +0.07]       0.02771
       +5.6     0.01625    +0.68 %     [+0.21, +1.14]     0.42006    -0.53 %     [-1.01, -0.07]       0.04605
      +11.2     0.01630    +1.00 %     [+0.34, +1.64]     0.41912    -0.75 %     [-1.45, -0.08]       0.07404
      +22.5     0.01636    +1.35 %     [+0.48, +2.20]     0.42017    -0.51 %     [-1.50, +0.52]       0.10690
      +45.0     0.01649    +2.14 %     [+1.13, +3.11]     0.42431    +0.48 %     [-0.82, +1.83]       0.12743
      +67.5     0.01641    +1.69 %     [+0.72, +2.67]     0.42402    +0.41 %     [-0.74, +1.54]       0.11550
      +90.0     0.01622    +0.48 %     [-0.27, +1.25]     0.42368    +0.33 %     [-0.31, +0.95]       0.07693
     +135.0     0.01642    +1.75 %     [+0.67, +2.85]     0.42113    -0.28 %     [-1.60, +1.09]       0.13195
     +180.0     0.01611    -0.17 %     [-0.99, +0.63]     0.42036    -0.46 %     [-1.08, +0.13]       0.07116

    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) | condition P
      deg   test loss =    r(loss)       adj. query CI  log-STFT MS   r(logmse)       adj. query CI   consistency =   (full names: loss = test loss = STFT L1 + 0.01 x decay; log_mse = log-STFT MSE; consistency = consistency = mean|log-spec(k) - log-spec(0)|)
     -135.0     0.01628    +2.47 %     [+1.56, +3.38]     0.41666    +0.14 %     [-0.86, +1.20]       0.11877
      -90.0     0.01603    +0.88 %     [+0.32, +1.43]     0.41765    +0.38 %     [-0.15, +0.92]       0.06456
      -67.5     0.01629    +2.48 %     [+1.68, +3.30]     0.41476    -0.31 %     [-1.18, +0.57]       0.10233
      -45.0     0.01628    +2.47 %     [+1.57, +3.37]     0.41710    +0.25 %     [-0.75, +1.28]       0.11277
      -22.5     0.01612    +1.41 %     [+0.68, +2.11]     0.41686    +0.19 %     [-0.54, +0.93]       0.09174
      -11.2     0.01592    +0.18 %     [-0.45, +0.77]     0.41634    +0.07 %     [-0.52, +0.67]       0.06750
       -5.6     0.01588    -0.05 %     [-0.47, +0.37]     0.41692    +0.20 %     [-0.15, +0.58]       0.04305
       -2.8     0.01588    -0.07 %     [-0.34, +0.21]     0.41655    +0.12 %     [-0.11, +0.35]       0.02623
       +0.0     0.01589    +0.00 %     [+0.00, +0.00]     0.41607    +0.00 %     [+0.00, +0.00]       0.00000
       +2.8     0.01591    +0.13 %     [-0.15, +0.40]     0.41532    -0.18 %     [-0.42, +0.05]       0.02714
       +5.6     0.01594    +0.30 %     [-0.13, +0.73]     0.41483    -0.30 %     [-0.72, +0.12]       0.04514
      +11.2     0.01600    +0.70 %     [+0.09, +1.30]     0.41464    -0.34 %     [-0.98, +0.30]       0.07011
      +22.5     0.01616    +1.70 %     [+0.96, +2.45]     0.41516    -0.22 %     [-1.05, +0.63]       0.09548
      +45.0     0.01631    +2.61 %     [+1.74, +3.49]     0.41969    +0.87 %     [-0.14, +1.92]       0.11424
      +67.5     0.01611    +1.36 %     [+0.60, +2.13]     0.41898    +0.70 %     [-0.25, +1.61]       0.10196
      +90.0     0.01594    +0.27 %     [-0.26, +0.81]     0.41963    +0.86 %     [+0.26, +1.49]       0.06676
     +135.0     0.01622    +2.06 %     [+1.16, +2.94]     0.41799    +0.46 %     [-0.56, +1.52]       0.11712
     +180.0     0.01586    -0.23 %     [-0.73, +0.29]     0.41542    -0.16 %     [-0.67, +0.38]       0.06047

    released (simple, checkpoints/xRIR_unseen.pth) | condition P
      deg   test loss =    r(loss)       adj. query CI  log-STFT MS   r(logmse)       adj. query CI   consistency =   (full names: loss = test loss = STFT L1 + 0.01 x decay; log_mse = log-STFT MSE; consistency = consistency = mean|log-spec(k) - log-spec(0)|)
     -135.0     0.01746    +2.86 %     [+1.35, +4.40]     0.44915    +2.36 %     [+0.80, +3.95]       0.20811
      -90.0     0.01697    -0.02 %     [-1.25, +1.20]     0.44495    +1.41 %     [+0.11, +2.71]       0.15672
      -67.5     0.01745    +2.81 %     [+1.49, +4.22]     0.44400    +1.19 %     [-0.19, +2.58]       0.17698
      -45.0     0.01753    +3.30 %     [+1.93, +4.68]     0.44050    +0.39 %     [-0.85, +1.61]       0.17745
      -22.5     0.01718    +1.23 %     [+0.09, +2.38]     0.44101    +0.51 %     [-0.60, +1.65]       0.15140
      -11.2     0.01704    +0.38 %     [-0.59, +1.36]     0.43823    -0.12 %     [-1.31, +0.99]       0.12259
       -5.6     0.01695    -0.12 %     [-0.86, +0.66]     0.43827    -0.11 %     [-1.00, +0.79]       0.09179
       -2.8     0.01690    -0.40 %     [-0.92, +0.14]     0.43970    +0.21 %     [-0.46, +0.91]       0.06661
       +0.0     0.01697    +0.00 %     [+0.00, +0.00]     0.43877    +0.00 %     [+0.00, +0.00]       0.00000
       +2.8     0.01699    +0.08 %     [-0.48, +0.64]     0.43884    +0.01 %     [-0.66, +0.73]       0.06632
       +5.6     0.01698    +0.06 %     [-0.71, +0.79]     0.43877    +0.00 %     [-0.86, +0.88]       0.09114
      +11.2     0.01707    +0.57 %     [-0.33, +1.54]     0.43753    -0.28 %     [-1.31, +0.71]       0.12206
      +22.5     0.01725    +1.64 %     [+0.52, +2.82]     0.44288    +0.94 %     [-0.26, +2.14]       0.15331
      +45.0     0.01747    +2.91 %     [+1.56, +4.26]     0.43691    -0.42 %     [-1.65, +0.87]       0.17996
      +67.5     0.01728    +1.82 %     [+0.38, +3.20]     0.44056    +0.41 %     [-0.94, +1.81]       0.18039
      +90.0     0.01703    +0.33 %     [-0.86, +1.55]     0.43962    +0.19 %     [-1.14, +1.49]       0.16119
     +135.0     0.01747    +2.91 %     [+1.41, +4.43]     0.44274    +0.90 %     [-0.53, +2.41]       0.20396
     +180.0     0.01686    -0.64 %     [-1.88, +0.59]     0.44713    +1.90 %     [+0.40, +3.53]       0.17016

    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) | condition E
      deg   test loss =    r(loss)       adj. query CI  log-STFT MS   r(logmse)       adj. query CI   consistency =   (full names: loss = test loss = STFT L1 + 0.01 x decay; log_mse = log-STFT MSE; consistency = consistency = mean|log-spec(k) - log-spec(0)|)
     -135.0     0.01647    +2.02 %     [+0.94, +3.12]     0.42188    -0.10 %     [-1.43, +1.31]       0.13468
      -90.0     0.01620    +0.36 %     [-0.48, +1.17]     0.42395    +0.39 %     [-0.33, +1.09]       0.07682
      -67.5     0.01646    +1.94 %     [+0.98, +2.93]     0.42089    -0.34 %     [-1.49, +0.93]       0.11661
      -45.0     0.01641    +1.69 %     [+0.69, +2.71]     0.42245    +0.03 %     [-1.26, +1.41]       0.12521
      -22.5     0.01631    +1.06 %     [+0.23, +1.88]     0.42223    -0.02 %     [-1.03, +1.00]       0.10329
      -11.2     0.01616    +0.10 %     [-0.55, +0.76]     0.42201    -0.07 %     [-0.74, +0.61]       0.07262
       -5.6     0.01611    -0.19 %     [-0.63, +0.25]     0.42215    -0.04 %     [-0.45, +0.37]       0.04545
       -2.8     0.01612    -0.12 %     [-0.42, +0.17]     0.42208    -0.05 %     [-0.32, +0.21]       0.02744
       +0.0     0.01614    +0.00 %     [+0.00, +0.00]     0.42231    +0.00 %     [+0.00, +0.00]       0.00000
       +2.8     0.01619    +0.33 %     [+0.03, +0.62]     0.42138    -0.22 %     [-0.51, +0.07]       0.02771
       +5.6     0.01625    +0.68 %     [+0.21, +1.14]     0.42006    -0.53 %     [-1.01, -0.07]       0.04605
      +11.2     0.01630    +1.00 %     [+0.34, +1.64]     0.41912    -0.75 %     [-1.45, -0.08]       0.07404
      +22.5     0.01636    +1.35 %     [+0.48, +2.20]     0.42017    -0.51 %     [-1.50, +0.52]       0.10690
      +45.0     0.01649    +2.14 %     [+1.13, +3.11]     0.42431    +0.48 %     [-0.82, +1.83]       0.12743
      +67.5     0.01641    +1.69 %     [+0.72, +2.67]     0.42402    +0.41 %     [-0.74, +1.54]       0.11550
      +90.0     0.01622    +0.48 %     [-0.27, +1.25]     0.42368    +0.33 %     [-0.31, +0.95]       0.07693
     +135.0     0.01642    +1.75 %     [+0.67, +2.85]     0.42113    -0.28 %     [-1.60, +1.09]       0.13195
     +180.0     0.01611    -0.17 %     [-0.99, +0.63]     0.42036    -0.46 %     [-1.08, +0.13]       0.07116

    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) | condition E
      deg   test loss =    r(loss)       adj. query CI  log-STFT MS   r(logmse)       adj. query CI   consistency =   (full names: loss = test loss = STFT L1 + 0.01 x decay; log_mse = log-STFT MSE; consistency = consistency = mean|log-spec(k) - log-spec(0)|)
     -135.0     0.01628    +2.47 %     [+1.56, +3.38]     0.41666    +0.14 %     [-0.86, +1.20]       0.11877
      -90.0     0.01603    +0.88 %     [+0.32, +1.43]     0.41765    +0.38 %     [-0.15, +0.92]       0.06456
      -67.5     0.01629    +2.48 %     [+1.68, +3.30]     0.41476    -0.31 %     [-1.18, +0.57]       0.10233
      -45.0     0.01628    +2.47 %     [+1.57, +3.37]     0.41710    +0.25 %     [-0.75, +1.28]       0.11277
      -22.5     0.01612    +1.41 %     [+0.68, +2.10]     0.41686    +0.19 %     [-0.54, +0.93]       0.09174
      -11.2     0.01592    +0.18 %     [-0.45, +0.77]     0.41634    +0.07 %     [-0.52, +0.67]       0.06750
       -5.6     0.01588    -0.05 %     [-0.47, +0.37]     0.41692    +0.20 %     [-0.15, +0.58]       0.04305
       -2.8     0.01588    -0.07 %     [-0.34, +0.21]     0.41655    +0.12 %     [-0.11, +0.35]       0.02623
       +0.0     0.01589    +0.00 %     [+0.00, +0.00]     0.41607    +0.00 %     [+0.00, +0.00]       0.00000
       +2.8     0.01591    +0.13 %     [-0.15, +0.40]     0.41532    -0.18 %     [-0.42, +0.05]       0.02714
       +5.6     0.01594    +0.30 %     [-0.13, +0.73]     0.41483    -0.30 %     [-0.72, +0.12]       0.04514
      +11.2     0.01600    +0.70 %     [+0.09, +1.30]     0.41464    -0.34 %     [-0.98, +0.30]       0.07011
      +22.5     0.01616    +1.70 %     [+0.96, +2.45]     0.41516    -0.22 %     [-1.05, +0.63]       0.09548
      +45.0     0.01631    +2.61 %     [+1.74, +3.49]     0.41969    +0.87 %     [-0.14, +1.92]       0.11424
      +67.5     0.01611    +1.36 %     [+0.60, +2.13]     0.41898    +0.70 %     [-0.25, +1.61]       0.10196
      +90.0     0.01594    +0.27 %     [-0.26, +0.81]     0.41963    +0.86 %     [+0.26, +1.49]       0.06676
     +135.0     0.01622    +2.06 %     [+1.16, +2.94]     0.41799    +0.46 %     [-0.56, +1.52]       0.11712
     +180.0     0.01586    -0.23 %     [-0.73, +0.29]     0.41542    -0.16 %     [-0.67, +0.38]       0.06047

    released (simple, checkpoints/xRIR_unseen.pth) | condition E
      deg   test loss =    r(loss)       adj. query CI  log-STFT MS   r(logmse)       adj. query CI   consistency =   (full names: loss = test loss = STFT L1 + 0.01 x decay; log_mse = log-STFT MSE; consistency = consistency = mean|log-spec(k) - log-spec(0)|)
     -135.0     0.01746    +2.86 %     [+1.35, +4.40]     0.44915    +2.36 %     [+0.80, +3.95]       0.20811
      -90.0     0.01697    -0.02 %     [-1.25, +1.20]     0.44495    +1.41 %     [+0.11, +2.71]       0.15672
      -67.5     0.01745    +2.81 %     [+1.49, +4.22]     0.44400    +1.19 %     [-0.19, +2.58]       0.17698
      -45.0     0.01753    +3.30 %     [+1.93, +4.68]     0.44050    +0.39 %     [-0.85, +1.61]       0.17745
      -22.5     0.01718    +1.23 %     [+0.09, +2.38]     0.44101    +0.51 %     [-0.60, +1.65]       0.15140
      -11.2     0.01704    +0.38 %     [-0.59, +1.36]     0.43823    -0.12 %     [-1.31, +0.99]       0.12259
       -5.6     0.01695    -0.12 %     [-0.86, +0.66]     0.43827    -0.11 %     [-1.00, +0.79]       0.09179
       -2.8     0.01690    -0.40 %     [-0.92, +0.14]     0.43970    +0.21 %     [-0.46, +0.91]       0.06661
       +0.0     0.01697    +0.00 %     [+0.00, +0.00]     0.43877    +0.00 %     [+0.00, +0.00]       0.00000
       +2.8     0.01699    +0.08 %     [-0.48, +0.64]     0.43884    +0.01 %     [-0.66, +0.73]       0.06632
       +5.6     0.01698    +0.06 %     [-0.71, +0.79]     0.43877    +0.00 %     [-0.86, +0.88]       0.09114
      +11.2     0.01707    +0.57 %     [-0.33, +1.54]     0.43753    -0.28 %     [-1.31, +0.71]       0.12206
      +22.5     0.01725    +1.64 %     [+0.52, +2.82]     0.44288    +0.94 %     [-0.26, +2.14]       0.15331
      +45.0     0.01747    +2.91 %     [+1.56, +4.26]     0.43691    -0.42 %     [-1.65, +0.87]       0.17996
      +67.5     0.01728    +1.82 %     [+0.38, +3.20]     0.44056    +0.41 %     [-0.94, +1.81]       0.18039
      +90.0     0.01703    +0.33 %     [-0.86, +1.55]     0.43962    +0.19 %     [-1.14, +1.49]       0.16119
     +135.0     0.01747    +2.91 %     [+1.41, +4.43]     0.44274    +0.90 %     [-0.53, +2.41]       0.20396
     +180.0     0.01686    -0.64 %     [-1.88, +0.59]     0.44713    +1.90 %     [+0.40, +3.53]       0.17016

## 4. Paired cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth) − control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth) at k = 0, and H2 per angle

k = 0; nominal 95.00 % intervals, unadjusted, descriptive (two single-seed checkpoints). Same references and same Griffin-Lim phases for both models: this supersedes exp_01's epoch-12 comparison.

    metric             [A]        [B]        diff    rel %                query CI                 room CI      n   ([A] = cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth); [B] = control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth))
    EDT error (s)       0.04494    0.04699   -0.00205    -4.4 [  -0.00255,   -0.00154] [  -0.00377,   -0.00013]   6337
    C50 error (dB)       1.2256     1.2351    -0.0095    -0.8 [   -0.0227,    +0.0032] [   -0.0381,    +0.0151]   6337
    T60 error (%)         9.436      9.559     -0.123    -1.3 [    -0.201,     -0.044] [    -0.361,     +0.110]   6337
    log-STFT MSE        0.41607    0.42231   -0.00624    -1.5 [  -0.00823,   -0.00422] [  -0.02185,   +0.00422]   6337
    test loss = STFT L1 + 0.01 x decay   0.015893   0.016142  -0.000249    -1.5 [ -0.000336,  -0.000161] [ -0.000457,  -0.000024]   6337

H2 (the cylindrical backbone degrades less): D_k = r_cyl - r_control, paired on the same resamples; decided cell by cell at the H1-passing cells of the primary model, where a cell passes when D_k is negative and its adjusted upper bound is below zero; the aggregate is "supported" when every such cell passes, "partially supported" when some do, "not supported" when none does and "not evaluable (H1 has no passing cell)" when H1 has no passing cell. Intervals: adjusted query-level 99.72 % and room-cluster 99.72 %. TOST equivalence of r_cyl to zero within +-2% at the adjusted level (query / room) at the patch-aligned angles. Rendered as "equivalent" or "not established" (a failed TOST does not show inequivalence).

    EDT error (s)
      deg      r[A]     r[B]       D_k       adj. query CI          adj. room CI   TOST r[A] (query / room)   ([A] = cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth); [B] = control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth))
      -90.0    +1.98 %    +0.68 %    +1.30 %     [-0.44, +2.94]     [+0.23, +2.55]   not established / not established
      -45.0    +3.07 %    +0.84 %    +2.23 %     [+0.38, +4.17]     [-0.44, +3.06]   not established / not established
      -22.5    +1.54 %    +0.36 %    +1.18 %     [-0.54, +2.88]     [-1.35, +2.06]   not established / not established
       -5.6    -0.73 %    -0.82 %    +0.10 %     [-1.11, +1.31]     [-1.01, +1.19]   –
       +0.0    +0.00 %    +0.00 %    +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   –
       +5.6    +0.23 %    +0.34 %    -0.12 %     [-1.31, +1.06]     [-1.91, +1.00]   –
      +22.5    +1.22 %    +1.02 %    +0.20 %     [-1.53, +2.03]     [-1.78, +1.76]   not established / not established
      +45.0    +2.55 %    +1.34 %    +1.21 %     [-0.66, +3.05]     [-2.88, +2.46]   not established / not established
      +90.0    +2.05 %    +1.25 %    +0.79 %     [-0.86, +2.42]     [-2.15, +3.12]   not established / not established
     +180.0    +0.94 %    -0.06 %    +1.00 %     [-0.67, +2.67]     [-0.97, +4.79]   not established / equivalent

    C50 error (dB)
      deg      r[A]     r[B]       D_k       adj. query CI          adj. room CI   TOST r[A] (query / room)   ([A] = cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth); [B] = control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth))
      -90.0    +1.82 %    +0.94 %    +0.89 %     [-0.86, +2.61]     [+0.13, +1.72]   not established / not established
      -45.0    +5.34 %    +4.55 %    +0.79 %     [-1.13, +2.64]     [-1.85, +1.84]   not established / not established
      -22.5    +3.23 %    +2.58 %    +0.65 %     [-1.01, +2.27]     [-1.70, +2.10]   not established / not established
       -5.6    -0.01 %    -0.03 %    +0.02 %     [-1.18, +1.23]     [-0.99, +1.22]   –
       +0.0    +0.00 %    +0.00 %    +0.00 %     [+0.00, +0.00]     [+0.00, +0.00]   –
       +5.6    +0.49 %    +0.84 %    -0.35 %     [-1.65, +0.94]     [-1.94, +0.49]   –
      +22.5    +3.37 %    +4.12 %    -0.74 %     [-2.61, +1.15]     [-3.06, +1.63]   not established / not established
      +45.0    +5.19 %    +5.09 %    +0.10 %     [-1.82, +2.08]     [-3.56, +1.73]   not established / not established
      +90.0    +2.61 %    +1.95 %    +0.67 %     [-0.99, +2.37]     [-2.04, +3.38]   not established / not established
     +180.0    +0.27 %    -0.27 %    +0.55 %     [-1.21, +2.31]     [-1.26, +5.78]   equivalent / not established

## 5. Delay-flip audit and decomposition

(query, reference) pairs whose integer direct-path delay moves under the rotation -- the numerical noise condition P excludes (producer field delay_flips):

    k                                                           0    4    8   16   32   64   96  128  192  256  320  384  416  448  480  496  504  508
    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth)      0    0    0    2    0    0    4    0    0    0    0    0    2    0    4    4    2    0
    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth)        0    0    0    2    0    0    4    0    0    0    0    0    2    0    4    4    2    0
    released (simple, checkpoints/xRIR_unseen.pth)              0    0    0    2    0    0    4    0    0    0    0    0    2    0    4    4    2    0

Decomposition (cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth), k = 32, 1 batch): receiver-view tokens 2.08e-07, pooled receiver feature 1.67e-02, query-source coordinate embedding 7.28e-01, output log-spectrogram 3.46e-03. Scope: receiver-view tokens and pooling plus the query-source coordinate embedding; reference-coordinate features not decomposed. Relative changes of different representation spaces; not additive.

## 6. Selected evaluation settings (from each run's metrics meta)

    run                                                       backbone     checkpoint                              n    batch  canonical  tf32   gl_seed  elapsed (min)
    control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth)    simple       ckpt/xRIR_simple_8_shot/epoch_12.pth     6337     16  True       False        0         120.8
    cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth)      cylindrical  ckpt/xRIR_cyl_8_shot/epoch_12.pth        6337     16  True       False        0         125.5
    released (simple, checkpoints/xRIR_unseen.pth)            simple       checkpoints/xRIR_unseen.pth              6337     16  True       False        0         119.0
