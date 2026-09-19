# exp_05 param_efficiency — analysis (Planner draft, 2026-09-18)

Status: final, 2026-09-18. Every number below is taken from the generated `param_efficiency_results.md` / `param_efficiency_01_results.html` (rendered by the reviewed generators from the canonical JSON) and is bound by `binding_report_20260918T234858147838Z.json` (sha256 60810502078f4bcc0a287e626d88182a1e33da50796595bb822b80847eec29ef, `check_record.py` exit 0 at main `73e90d4`).

## 1. What was tested

Six arms: SimpleViT and CylindricalViT encoders at three matched capacities — S (2 766 080 / 2 777 984 encoder parameters), M (19 703 296 / 19 750 912; the released configuration, exp_01's pair) and L (43 709 184 / 43 780 608) — identical recipe (12 epochs, effective batch 64, same optimizer and schedule), everything outside the encoder held fixed. H1 asks whether the cylindrical curve dominates the SimpleViT curve at every tested tier (EDT and C50, K = 8); H2 asks whether a smaller cylindrical model reaches a larger SimpleViT's accuracy (Cyl-S vs Base-M, Cyl-M vs Base-L: TOST ±3 % or superiority).

## 2. Results

**H1 (K = 8): partial on EDT and partial on C50.** Relative change cylindrical − SimpleViT with adjusted two-sided intervals: EDT S −1.3 % [−3.0, +0.4], M −4.2 % [−5.4, −2.9], L −2.2 % [−3.6, −0.9]; C50 S +0.6 % [−1.0, +2.3], M −0.6 % [−1.9, +0.7], L −2.5 % [−3.8, −1.3]. Dominance on EDT holds at M and L, not at S; on C50 only at L. Room-cluster intervals include zero except L C50. T60 (descriptive): cylindrical lower at M (9.44 vs 9.58 %) and L (9.37 vs 9.54 %), equal at S (9.64 vs 9.63 %).

**H1 (K = 1): not supported.** With a single reference the cylindrical encoder is worse at every tier: EDT S +4.5 % [+3.6, +5.4], M +2.2 % [+1.6, +2.8], L +0.6 % [−0.03, +1.2]; C50 S +1.2 %, M +2.8 %, L +0.8 %. This is the same single-reference reversal exp_03 found at M, now shown to be tier-independent and largest at S.

**H2 (K = 8).** Cyl-M vs Base-L (2.21× fewer encoder parameters): superior on EDT (−4.4 % [−5.7, −3.1]) and C50 (−3.5 % [−4.7, −2.1]). Cyl-S vs Base-M (7.09× fewer): comparable within ±3 % on EDT (−1.3 %, TOST interval inside the margin) and on C50 (+1.7 % [+0.4, +3.1], TOST inside the margin; note the point estimate is worse). At K = 1 neither Cyl-S vs Base-M cell reaches its target; Cyl-M vs Base-L reaches it only on C50.

**Curve shape.** The SimpleViT curve is flat across 16× in encoder parameters (EDT 47.00 / 47.02 / 47.13 ms; C50 1.252 / 1.238 / 1.275 dB; T60 9.63 / 9.58 / 9.54 %). Capacity is not the lever for this model: the output is a weighted combination of the reference spectrograms, so the encoder shapes the reference weighting, which saturates at the S tier. The cylindrical curve is also nearly flat (46.4 / 45.1 / 46.1 ms), with its largest margin at M.

**Yaw robustness (descriptive, seed 42, K = 8).** Rotating the scene costs every arm 3–6 % on C50 and 1–3 % on EDT at the diagonal headings (k = 32/64/448 of 512) at every tier and for both backbones; exp_03's "no robustness gain for the cylindrical encoder" holds at S and L as it did at M.

## 3. Reading for the paper

At the released size the cylindrical encoder improves EDT by 4 % at fixed parameter count and beats a SimpleViT with 2.2× more encoder parameters on both EDT and C50; a 7× smaller cylindrical encoder matches the released SimpleViT within 3 %. The gain is a property of the positional code, not of capacity: adding capacity to the SimpleViT does nothing, and the cylindrical advantage disappears when the query is conditioned on a single reference (K = 1), where the SimpleViT is 1–5 % better. Claims must be scoped to the tested tiers (S/M/L) and to K = 8.

## 4. Limitations and disclosures

One training seed per arm (training realisations unpaired; the M pair predates this experiment: different day, launcher v1 cadence, unrecorded PYTHONHASHSEED); the S and L runs were co-tenant with other jobs for parts of their wall time (timing only, no effect on the objective); seed SDs are over evaluation seeds. The L tier fit at micro-batch 32 × 2 (44.4 GiB peak). Evaluation of the M pair was redone through exp_05's own entry point (`evaluator_exp04` unused).
