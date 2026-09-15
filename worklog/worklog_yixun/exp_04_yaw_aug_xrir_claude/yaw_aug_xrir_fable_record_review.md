**Reviewer:** Claude Fable 5.1 (Agent subagent, model fable) · **Date:** 2026-09-15

# Record review — exp_04 yaw_aug_xrir (SOP final check of the committed record)

Scope: does the committed record (HEAD `9388c14`; record files last changed in `cab6f42` / `862923e`) tell the truth about the canonical artefacts under `ckpt/yaw_aug/results/` and the bound run evidence? Read first: `worklog/experiment_SOP.md`, plan v4 (§3, §4, §8, §11 A1–A10), notebook entries 2026-09-13 → 2026-09-15, and the record files (`yaw_aug_xrir_results.md`, `yaw_aug_xrir_01_results.html`, `yaw_aug_xrir_analysis.md`, `yaw_aug_xrir_params_set_up.md`, `yaw_aug_xrir_command.md`, `commits_yaw_aug_xrir.md`, `worklog/worklog_yixun/model_comparison.md`, `yaw_aug_xrir_results_assets/approved_digests.json`). Everything was read from the real tree; the only executed scripts (`check_record.py`, both generators) are read-only apart from `--out`, which pointed at the scratch directory. No GPU work (`CUDA_VISIBLE_DEVICES=''`), no process signalled, working tree unchanged outside `worklog/` (this file is the only write).

**Bottom line.** The canonical layer is sound: `check_record.py` exits 0; the binding report covers exactly the 46 completed run directories, the 7 producer JSONs, the certified attempt, the probe receipt and the audit; every pin in `approved_digests.json` equals the digests recorded in the run manifests / attempt and recomputes identically at the reviewed commit, at HEAD and by independent import; both record pages regenerate byte-identically from the seven JSONs (only the git HEAD in the identity line differs, consistently); every verdict string is verbatim; every decision-carrying number in the analysis traces to a JSON value. The defects are all in the prose of the analysis, the params document and the commits file: five quantitative statements are wrong or overstated (convergence maximum, EDT grid range, "only effect whose room interval excludes zero", producer commits, review count), one comparison uses exp_03's single-seed number where the paired five-seed number exists, one claim about the control has no source, the commits file is missing the last 24 commits, and the params "Code state" line is still the pre-launch draft. None changes a verdict or a table.

---

## Findings

Severity: **blocker** = the record misstates a canonical artefact or a decision; **should-fix** = a wrong or unsupported statement that must be corrected before the record is cited; **nit** = clarity.

### Blockers

None.

### Should-fix

**1. Producer commits misreported.** `yaw_aug_xrir_analysis.md:46` — "Producers at `daa7eef`/`cab6f42`". The sidecars record `producer.commit` = `87d3e54` (H1_K8, H1_K1, TABLE_V1), `fa219ef` (H2_K8, TOST_K8, EPOCH9_K8) and `f851be1` (GRID_SEED42); the binding report lists the same. `daa7eef` is the commit whose code the producer closures were pinned from (digests unchanged since, verified below) and `cab6f42` is the commit that added the record files — neither is a producer run. Fix: "producer closures pinned from `daa7eef` (unchanged through HEAD); producer runs at `fa219ef` (H2/TOST/EPOCH9), `87d3e54` (H1/TABLE) and `f851be1` (GRID), as recorded in the sidecars".

**2. Convergence maximum wrong.** `yaw_aug_xrir_analysis.md:38` — "maximum endpoint movement / width ≤ 0.021". Recomputed over all 30 gated intervals of the four confirmatory JSONs (`cells[*].convergence.{decision,superiority}.ratio`): maximum **0.0277** (TOST_K8, EDT, −90°, decision gate), then 0.0256 (TOST_K8, C50, −22.5°), 0.0212 (H2_K8, C50, +22.5°). All 30 pass the 0.10 tolerance, so the conclusion stands; the number does not. Fix: "≤ 0.028".

**3. EDT grid range wrong.** `yaw_aug_xrir_analysis.md:30` — "EDT r_k within ±0.7 % everywhere". `GRID_SEED42.json` EDT cell at −5.625° (k = 504): estimate **−0.845 %**, interval [−1.56, −0.17] %. The other nine are within ±0.7 % (max +0.68 % at −45°). The notebook's 04:48 entry ("+0.1 to +0.7 %") has the same omission. Fix: "|r_k| ≤ 0.85 % everywhere (largest −0.85 % at −5.6°)".

**4. "Only effect whose room interval excludes zero" is false as written.** `yaw_aug_xrir_analysis.md:40`. Room-cluster intervals that exclude zero in the canonical JSONs: H1_K8 C50 [+0.85, +7.74]; H1_K1 C50 [+0.25, +7.25]; **H2_K8 T60 +22.5° [−2.40, −0.37] and +45° [−2.35, −0.22]** (descriptive cells inside H2); **TOST_K8 C50 +45° [+0.07, +6.34] and −45° [+0.19, +6.26]** (decision-driving one-arm cells: the augmented model's own diagonal C50 degradation is real at room level too); TOST_K8 T60 −90° [−2.39, −0.02]. The statement holds only for the decision-driving two-arm cells (H1 EDT/C50, H2 C50/EDT). Fix: "Among the decision-driving two-arm cells, the C50 penalty (K = 8 and K = 1) is the only effect whose 17-room interval excludes zero; the H2 C50/EDT and H1 EDT room intervals all include zero. (Descriptive T60 D_k at +22.5°/+45° and the augmented arm's own C50 r_k at ±45° also have room intervals excluding zero.)"

**5. "Order of magnitude" overstated for EDT.** `yaw_aug_xrir_analysis.md:26`. From TABLE_V1: aug − control EDT = 46.39 − 47.02 = −0.63 ms at K = 8 against seed SDs 0.15/0.19 ms (≈ 3–4×), and 69.52 − 68.28 = +1.25 ms at K = 1 against 0.73/0.61 ms (< 2×); C50 gaps are 0.055 dB vs 0.005–0.006 (≈ 9×) and 0.079 dB vs 0.013–0.015 (≈ 5×). Fix: replace with the ratios, or "well above the seed SDs for C50 and, at K = 8, for EDT; the paired bootstrap carries the inference".

**6. exp_03 comparison uses the single-seed number and mis-states its range.** `yaw_aug_xrir_analysis.md:13` — "from the control's ≈ 4–5 % in exp_03 to ≈ 2–3 %". `ckpt/yaw_rotation/full_stats.json` (`acoustic.control.P.c50`) gives the exp_03 control r_k = +4.12 / +5.09 / +4.55 / **+2.58 %** at +22.5°/+45°/−45°/−22.5° (single seed); exp_04's own five-seed control block (`H2_K8.json` `cells[C50,k].seed_means.control`) gives **3.70 / 5.31 / 4.57 / 2.52 %**, and the augmented arm (`TOST_K8` estimates) 2.68 / 3.05 / 3.07 / 2.03 %. Plan §3 requires any comparison with exp_03's single-seed sweep to be labelled a single-seed diagnostic; here the paired five-seed number exists in the canonical JSON and should be the one quoted. Fix: "from the control's 2.5–5.3 % (five-seed, H2_K8 seed means; exp_03's single seed gave 2.6–5.1 %) to 2.0–3.1 %".

**7. Unsourced claim about the control.** `yaw_aug_xrir_analysis.md:30` — "the C50 deficit grows in the last low-learning-rate epochs, while the control improves there". No exp_04 artefact evaluates the control at epoch 9. The only evidence is exp_01's per-epoch single-seed evaluation (`ckpt/xRIR_simple_8_shot/eval_unseen_epoch09.json` C50 1.2363 dB → `epoch12` 1.2340 dB, −0.19 %, `eval_xRIR_backbone.py`, seed 0, unpaired, outside the canonical set the analysis header claims). Fix: cite it and label it ("exp_01's unpaired single-seed per-epoch evaluation: 1.236 → 1.234 dB"), or drop the clause.

**8. Review count wrong.** `yaw_aug_xrir_analysis.md:46` — "nine Fable 5.1 code reviews (`yaw_aug_xrir_fable_code_*_review.md`)". Fourteen files match that glob (round1, round2, round2_close, round3a, round3a_close, round3b, round3b_close, round4, round5, round7_close, round8, round8_close, branch, final). Fix: "fourteen".

**9. Commits file incomplete.** `commits_yaw_aug_xrir.md:87–127` — the section headed "Planner commits, record phase — 2026-09-15T04:50:31" ends at `6792167` (2026-09-14 morning) and is mostly exp_05 and pre-launch commits. Missing exp_04 commits (from `git log 6792167..862923e`): `265dc83` (training certified), `bb9fa91` (round-8 prompt), Codex round 8 `f41473c` `929ab47` `4827cc1` `4f0abb6`, `424f1f0`, `ecbb96e` (refusals logged), round-8 close-out `51b8c89` `42b4903` `3221624` `c359385`, `7192b6e`, second close-out `82745f6` `60468a0` `7ea196a`, merge `9eacbb4`, `58a8953`, `77ef292`, approval `fa219ef`, producer-run bookkeeping `87d3e54` `f851be1`, record `cab6f42`, and `862923e` itself (add on the next commit). SOP item 12 requires every commit; the round-8 Codex commits and their reviews are not otherwise traceable from this file. Fix: append them with one-line descriptions and re-title the section.

**10. Params document not finalised where it says it is.** `yaw_aug_xrir_params_set_up.md:3` says only the gate-status and code-state lines were updated after the run, but line 40 still reads "round 6 = pre-launch hardening in progress". Final state: launch commit `f19b9b6` certified 2026-09-13 00:53 (rounds 6–7 closed), round 8 (record tooling) merged as `9eacbb4` with the Fable branch review, approval `fa219ef`. Line 37 ("the launch waits for the user's decision") is stale too — keep only if marked historical. Line 13: the aug checkpoint sha256 is now known (`f8e64052…`; epoch-9 `442a0a56…`) and should be written in the Models table instead of "pinned … after training".

**11. §5 omits the GPU-0 co-tenant during the last five evaluations.** `yaw_aug_xrir_analysis.md:41`. Per the notebook (04:48) a foreign FLAC training (`exp_23`, pid 2198556, 3.8 GB) occupied GPU 0 from ≈ 01:26, i.e. during `control_k8_seed46_block` (01:19–02:05), the three re-queued k = 0 runs (02:07–02:41) and the seed-42 grid (02:41–04:39). The evaluation launcher records no GPU-process snapshot (no co-tenant field in any `eval_manifest.json`; `grep -i cotenant tools/exp04_eval_launch.py` is empty), so the notebook is the only record. Evaluations are deterministic (TF32 off, per-query seeded Griffin-Lim, canonical batch 16), so this is not a validity issue, but the deviations list should name it next to the L-tier host sharing.

**12. Epoch-3 slowdown understated.** `yaw_aug_xrir_analysis.md:41` — "≤ 4 min/epoch". `final/history.jsonl`: epoch 3 = 141.47 min against 136.35–136.87 min for the other eleven, i.e. +4.6 to +5.1 min (train phase 140.3 vs 135.4 = +4.9 per the notebook). The S-tier co-tenant effect on the other epochs is ≤ 0.5 min. Fix: "≈ 5 min at epoch 3, ≤ 0.5 min elsewhere; the epoch-1 gate (136.3 vs 145.9 min) was unaffected".

### Nits

**13.** `yaw_aug_xrir_analysis.md:10` — "T60 (descriptive) is negative with bounds below zero at +22.5° and +45°": D_k at −45° is −0.06 pp (upper +1.09) and at −22.5° **+0.18 pp** (upper +1.22); say "≈ 0 at −45°/−22.5°" so "negative" is not read as "at all four". The notebook line 297 ("negative at ±22.5°/+45°") is wrong for −22.5°; the notebook is append-only, so add a correction entry.

**14.** `yaw_aug_xrir_analysis.md:3` — "Every number below is copied from the canonical producer JSON": the provenance numbers (27.43 h, 0.78 %, 0 / 12 800, 0.01599 / 0.01604, epoch minutes, the exp_03 percentages) come from `completion.json`, the probe receipt, `alignment_audit.json`, `history.jsonl` and exp_03's `full_stats.json`. Say "every result number".

**15.** `yaw_aug_xrir_analysis.md:39` — "far larger than exp_01/exp_03's between-run differences": those are between-*model* differences (exp_03 paired k = 0: cyl EDT −4.4 %, C50 −0.8 %, loss −1.5 %), not a training-seed noise estimate; label them, and cite plan §4/§7 (one attempt) rather than "§2" for the single-realisation design.

**16.** `yaw_aug_xrir_params_set_up.md:25` — "(iii) descriptive full grid (exp_03's 18 + 10 + 4 columns)": the grid run is P-only (`conditions P`, `e_acoustic_cols []`, 18 spectral + 10 acoustic columns in `aug_k8_seed42_grid/eval_manifest.json`); drop "+ 4".

**17.** Presentation: neither page marks the "Room-cluster interval (%)" column as the secondary (non-decision) interval; a caption or a line in the protocol table would prevent a reader from treating it as the decision bound. Also the run manifests carry two evaluator digests — the 12-file exp_03 frozen closure `5ba818d8…` in `evaluator_closure` (checked through the metrics meta) and the 14-file `tools/exp04_eval` entry-point closure `0b05245c…` in `source_closures.entrypoint`, which is what the `evaluator` pin means (`tools/paired_compare.py:331` maps `evaluator → entrypoint`); the record only names the latter. A reader who looks up `evaluator_closure.sha256` in a manifest will not find `0b05245c`; one sentence in the params document or the assets README would remove the ambiguity.

---

## Verified

Environment: `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=/home/yixunhu/codespace/xRIR_code`, `XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`, `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`. Scratch outputs under `/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad/record/`.

### 1. Canonical artefacts, binding, pins

- `git rev-parse HEAD` → `9388c14f96d97bc177ba0b6b2342ec9e9f4e3a3b`; `git status --short | grep -v worklog/` → empty (tree clean outside `worklog/`, before and after this review).
- `ls ckpt/yaw_aug/results/` → `{H1_K8,H1_K1,H2_K8,TOST_K8,GRID_SEED42,EPOCH9_K8}.json` + `.json.provenance.json` + `.summary.txt`; `TABLE_V1.json` + sidecar (its second output is `worklog/worklog_yixun/model_comparison.md`, sha256 `c9832bba…` = sidecar digest; no summary by design).
- `python …/check_record.py ckpt/yaw_aug` → **exit 0** (3 m 08 s). It recomputes the latest report `binding_report_20260915T084205518027Z.json` (sha256 `6500c675176572a7…`, `git_HEAD f851be1`) from every run/result/attempt/receipt/audit digest.
- Binding report contents: `inputs.runs` = exactly the 46 non-`_ABORTED_` directories under `ckpt/yaw_aug/eval/` (49 present; the three `_ABORTED_setup_failed` are excluded); `results` = the 7 profiles with producer commits `87d3e54` ×3, `fa219ef` ×3, `f851be1` ×1; `training.path` = `…/attempt_20260913T102051` (= `final` symlink target); `probe_receipt` = `_probe_20260913T101537_probe.json` (`d0d0485a…`); `audit` = `alignment_audit.json` (`cd6f3a29…`).
- Pins vs manifests (script over all 46 `eval_manifest.json`): `source_closures.entrypoint.sha256` = {`0b05245c…`} (14 files: eval_unseen, eval_xRIR_backbone, eval_yaw_rotation, 4 model files, tools/exp04_eval, per_sample_metrics, provenance, reference_manifest, yaw_rotation, dataset, spec_utils); `source_closures.writer.sha256` = {`15f9c984…`} (15 files); `reviewed_commit` = {`f19b9b6…`}; `(confirmatory, allow_dirty_used)` = {(True, False)}; `checkpoint_sha256`: `f8e64052…` ×16 (aug), `442a0a56…` ×5 (aug9), `651e3a37…` ×15 (control), `8ba344ad…` ×10 (cyl). Attempt `train_manifest.json`: `source_closures.launcher.sha256` = `0bff0d4f…` (12 files), `training` = `5b2da250…`, `reviewed_commit f19b9b6`, `mode full`, `allow_dirty False`; `completion.json`: `outputs.epoch_012.pth` = `f8e64052…`, `epoch_009.pth` = `442a0a56…`, `wall_hours 27.4257`, `source_drift_after_spawn []`, `train_manifest_sha256 b2d4d059…`; `cumulative_hours.json` total 27.69 h (one `full` attempt).
- Closure recomputation with `tools.provenance.closure_record` (and `source_closure` for an independent import at HEAD):

| closure | recorded | @ reviewed commit | @ HEAD `9388c14` | independent import @ HEAD | pin |
|---|---|---|---|---|---|
| evaluator (`tools.exp04_eval`, 14 files) | `0b05245c` | `0b05245c` | `0b05245c` | `0b05245c` (same 14 files) | `0b05245c` |
| exp_03 frozen `evaluator_closure` (12 files) | `5ba818d8` | `5ba818d8` | `5ba818d8` | — | (not pinned; checked via metrics meta) |
| writer (15 files) | `15f9c984` | `15f9c984` | `15f9c984` | — | `15f9c984` |
| training launcher (12 files) | `0bff0d4f` | `0bff0d4f` | `9dadbef0` | — | `0bff0d4f` (recorded, not compared, A9; HEAD change = exp_05 launcher work after launch, as the notebook says) |
| producer_paired_compare (8 files) | `1bda0601` | `1bda0601` | `1bda0601` | `1bda0601` | `1bda0601` |
| producer_results_table (9 files) | `6d32a5e2` | `6d32a5e2` | `6d32a5e2` | `6d32a5e2` | `6d32a5e2` |
| producer_descriptive (10 files) | `2f86d7ed` | `2f86d7ed` | `2f86d7ed` | `2f86d7ed` | `2f86d7ed` |

- The three re-queued runs (`aug_k8_seed45_k0`, `aug_k8_seed46_k0`, `aug_k1_seed42_k0`) have manifest `command`s that differ from their sibling seeds only in the manifest path/hash, run label/out-dir and `--gl-seed` — "re-run identically" holds.
- `aug_k8_seed42_grid/eval_manifest.json`: 18 spectral `yaw_cols`, 10 `acoustic_cols` `[0, 8, 32, 64, 128, 256, 384, 448, 480, 504]`, `e_acoustic_cols []`, `conditions P` — the GRID_SEED42 profile's grids.

### 2. Record pages

- Regenerated both pages with the record's generators from the seven JSONs (commands as in `_command.md`, `--out` in scratch). `diff` after normalising the 40-hex HEAD in the identity line: **results.md identical, HTML identical**. The only raw difference is `… at f851be1…` (committed) vs `… at 9388c14…` (now). `f851be1` was HEAD at 04:41 on 2026-09-15 when the pages were generated, one commit before `cab6f42` committed them, and equals the binding report's `git_HEAD` — consistent. Descriptive blocks are titled "Descriptive diagnostics — GRID_SEED42 / EPOCH9_K8" and their figures "Diagnostic …"; T60 rows carry Role "descriptive" and Verdict "—" (18 such rows); TOST verdicts are per cell.
- `model_comparison.md` is the TABLE_V1 sidecar's second output (digest match) and the results.md TABLE_V1 block is `display()` of the same rows (e.g. `47.0 ± 0.2 ms` from `47.0176 ± 0.185805`).

### 3. Number tracing — analysis §1–§3, §5 (+ §6)

| analysis statement | JSON path | value | OK |
|---|---|---|---|
| H1 K8 EDT ρ −1.33 %, upper −0.22 % | `H1_K8.cells[EDT,k=0].estimate / .decision_bound` | −0.013346 / −0.0022303 | yes |
| superior at query level | `.superiority`, `.superiority_interval` | true, [−0.02596, −0.00070] | yes |
| rooms [−3.11, +0.55] | `.room_cluster_interval` | [−0.031144, +0.005521] | yes |
| H1 K1 EDT +1.83 %, upper +2.50 % | `H1_K1.cells[EDT].estimate / .decision_bound` | 0.018292 / 0.024963 | yes |
| C50 K8 +4.47 / +5.53 / [+3.39, +5.53] / [+0.85, +7.74] | `H1_K8.cells[C50]` estimate / decision_bound / companion / room | 0.044673 / 0.055289 / [0.033918, 0.055289] / [0.008473, 0.077366] | yes |
| C50 K1 4.2 % worse, upper +4.62 % | `H1_K1.cells[C50]` | 0.041808 / 0.046223 | yes |
| T60 K8 2.6 % better; K1 unchanged | `H1_K*.cells[T60].estimate` | −0.026459 / +0.000234 | yes |
| verdicts "non-inferior on EDT only" (K8, K1) | `H1_K8.verdict`, `H1_K1.verdict` | verbatim | yes |
| H2 C50 D_k −1.02 / −2.26 / −1.50 / −0.49 | `H2_K8.cells[C50,k=32/64/448/480].estimate` | −0.010227 / −0.022614 / −0.014993 / −0.004880 | yes |
| uppers +0.41 / −0.62 / +0.08 / +0.85 | `.decision_bound` | 0.004141 / −0.006177 / 0.000825 / 0.008541 | yes |
| EDT uppers +0.35 … +1.76 | `H2_K8.cells[EDT,*].decision_bound` | 1.07 / 0.35 / 1.76 / 0.76 % | yes |
| T60 bounds < 0 at +22.5°/+45° | `H2_K8.cells[T60,32/64].decision_bound` | −0.0019 / −0.0028 (−45°: +0.0109; −22.5°: +0.0122, estimate +0.0018) | yes (nit 13) |
| verdict "partially supported" | `H2_K8.verdict` | verbatim | yes |
| TOST equivalent at 90/180/−90 (C50, EDT), EDT also −22.5° | `TOST_K8.cells[*].verdict` | C50 128/256/384; EDT 128/256/384/480 "equivalent"; others "equivalence not established" | yes |
| C50 r_k +2.0 … +3.1 %, uppers +3.6 … +5.0 % | `TOST_K8.cells[C50,32/64/448/480]` | 0.02678/0.03051/0.03067/0.02029; 0.04375/0.04956/0.04930/0.03634 | yes |
| "control's ≈ 4–5 % in exp_03" | `ckpt/yaw_rotation/full_stats.json acoustic.control.P.c50[k=32,64,448,480].r` | 0.0412 / 0.0509 / 0.0455 / **0.0258** | partly (finding 6) |
| §2 table, 18 cells | `TABLE_V1.rows[*].metrics.{T60,C50,EDT}.{mean,sd}` | all 18 equal after rounding (e.g. 9.57524±0.00894945 → 9.575±0.009; 47.0176±0.185805 → 47.02±0.19; 1.96877±0.0153867 → 1.969±0.015) | yes |
| §3 C50 r_k +2.8 / +3.3 / +3.2 / +2.1 | `GRID_SEED42.cells[C50,32/64/448/480].estimate` | 0.02800 / 0.03332 / 0.03245 / 0.02070 | yes |
| §3 EDT within ±0.7 % everywhere | `GRID_SEED42.cells[EDT,*].estimate` | max |r_k| = **0.845 %** at k = 504 | **no** (finding 3) |
| §3 both ≈ 0 at 90/180/−90 | `GRID_SEED42` C50 / EDT at 128/256/384 | −0.40/−0.14/+0.20 %; −0.04/−0.04/+0.21 % | yes |
| §3 EPOCH9 46.89±0.16 / 1.261±0.006 / 9.41±0.01 | `EPOCH9_K8.cells[EDT/C50/T60].mean, .sd` | 46.8876±0.1553 / 1.26065±0.00644 / 9.40738±0.00901 | yes |
| epoch-12 C50 1.294 | `TABLE_V1.rows[YawAugxRIR,8].metrics.C50.mean` | 1.29365 | yes |
| 41 679 updates | 9 × 4631 (`effective_args.train_batches_per_epoch` 9261 / accum 2) | 41 679 | yes |
| §5 convergence ≤ 0.021 | max `convergence.*.ratio` over 30 gated intervals | **0.0277** | **no** (finding 2) |
| §5 no query excluded, 6 337 everywhere | `exclusions.*.total.excluded` (4 JSONs) = 0; `TABLE_V1 …per_seed[*].n_finite` = 6337 (6 rows × 5 metrics × 5 seeds); GRID/EPOCH9 `n_finite` = 6337 | yes |
| §5 rooms include zero for EDT and every H2 cell; C50 only exclusion | `room_cluster_interval` over all cells | see finding 4 | **partly** |
| §5 epoch 3 ≤ 4 min | `final/history.jsonl epoch_minutes` | 141.47 vs 136.35–136.87 (+4.6…+5.1) | **no** (finding 12) |
| §5 best epoch 5, 0.01599 vs 0.01604 | `history.jsonl test_loss` | 0.0159904 (epoch 5, `is_best`) / 0.0160420 (epoch 12) | yes |
| §6 27.43 h; 0.78 %; 0 / 12 800; 46 runs; `f19b9b6`; `fa219ef` | `completion.json wall_hours`; receipt `overhead_fraction`; `alignment_audit.json`; binding report; receipt/train-manifest `reviewed_commit`; `git log` | 27.4257; 0.007810 (ratio 1.007810); pairs 12800 changed 0; 46; yes; yes | yes |
| §6 producers `daa7eef`/`cab6f42`; nine reviews | sidecars; `ls` | `87d3e54`/`fa219ef`/`f851be1`; 14 files | **no** (findings 1, 8) |

Params document final-state lines: 27.43 h (completion 27.4257 ✓), overhead ratio 1.0078 (receipt 1.007810 ✓, `PROBE_NOT_CLEAN false`, `passed true`, reviewed commit `f19b9b6`), epoch 1 136.3 min vs gate 145.9 (history 136.35; 2.431 h = 145.9 ✓), control `args.json` sha256 `13fb7cc0…` (= train manifest `mutable_inputs.control_args` ✓), 9261 micro-batches / 4631 updates (`effective_args` ✓), alignment audit cohort `6720f114…` ✓, exp_03 single-seed table 47.0 ms / 1.235 dB / 9.56 % (`full_stats.json k0 ctrl`: 0.046988 s / 1.23512 dB / 9.5586 % ✓).

### 4. Interpretation labelling (item 3/5 of the brief)

§4 is headed "interpretation, not tested" ✓. §3 is headed "descriptive" ✓; the epoch-9 sentence sits inside it and `EPOCH9_K8.json` has `decision_driving false`, no verdict ✓. GRID_SEED42 `decision_driving false`, `exploratory false`, single seed, labelled on both pages ✓. T60 is called "descriptive" at every mention in §1 ✓. The one unlabelled interpretive clause is finding 7 ("the control improves there"); the one unlabelled single-seed comparison is finding 6.

---

## Verdict

**Record approved with corrections.** The canonical artefacts, the binding, the pins and both generated pages are exactly what the record says they are; every verdict string and every decision-carrying number is correct. Findings 1–12 are prose corrections to the analysis, the params document and the commits file (no artefact needs regenerating, no producer needs re-running); findings 13–17 are nits.

**Blocking findings:** none.
