# Plan — param_efficiency (exp_05): performance–parameter curves for SimpleViT vs CylindricalViT inside xRIR

**Status:** v1, written 2026-09-12 by the exp_04 Planner for execution by another Claude Code agent under `worklog/experiment_SOP.md`. Awaiting the Codex plan review and the user's approval. No code has been written.

## 1. Question

Within one acoustic system (xRIR, K-shot RIR prediction) and one backbone family (a panorama ViT with the token count fixed at 256), does the CylindricalViT encoder (CERPA: per-column gauge alignment, elevation-only positional code, circular relative-position bias) reach lower unseen-room acoustic error than the SimpleViT encoder at every encoder parameter budget (experiment 2 of the user's note), and can a smaller cylindrical encoder match a larger baseline one (experiment 3)?

## 2. Arms: three matched tiers × two backbones

Only the encoder configuration changes; everything else in xRIR (AudioEnc ResNet-18, `lin_proj_0` token pool over 256 tokens, coordinate embeddings, `lin_proj_1/2`, decoder-side weighting) is identical across all six models. `src_proj` (dim → 256) scales with dim; it is counted in the full-system column only, and the encoder column is `source_network` alone. Exact counts (measured with `build_xrir(backbone, num_shot=8, dim=…, depth=…, heads=…, mlp_dim=…)`, `dim_head` = 64 for the cylindrical ViT as in exp_01):

| tier | dim | depth | heads | mlp_dim | SimpleViT encoder / full | CylindricalViT encoder / full |
|---|---|---|---|---|---|---|
| S | 256 | 6 | 4 | 256 | 2.77 M / 15.07 M | 2.78 M / 15.09 M |
| M (= exp_01) | 512 | 12 | 8 | 512 | 19.70 M / 32.08 M | 19.75 M / 32.12 M |
| L | 768 | 12 | 12 | 768 | 43.71 M / 56.15 M | 43.78 M / 56.22 M |

Each pair is matched by construction (the backbones differ only in the positional code, ≤ 0.3 %); the tiers span 16× in encoder parameters. All parameters are trainable in xRIR (nothing frozen), so trainable = full. The M tier is exp_01's existing pair (`ckpt/xRIR_simple_8_shot/epoch_12.pth`, `ckpt/xRIR_cyl_8_shot/epoch_12.pth`) and is **not retrained**; four new runs: S-simple, S-cyl, L-simple, L-cyl.

**Single-delta contract per tier:** the two runs of a tier share the recipe, seed and step budget and differ only in `--backbone`; across tiers only `dim/depth/heads/mlp_dim` change. Same qualification as exp_04 §2: one training seed per run, non-shared data order/reference realizations (unordered-set path listing, worker-local `np.random.choice`), so this is a single intended delta with disabled-path evidence, not bitwise pairing; `PYTHONHASHSEED=0` is pinned for the new runs (unrecorded for exp_01's M runs; disclosed).

## 3. Training recipe (identical to exp_01's control; the only permitted variation is the tier)

`XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms PYTHONHASHSEED=0 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=<gpu> /home/yixunhu/miniconda3/envs/xRIR/bin/python train_xRIR_backbone.py --backbone {simple|cylindrical} --save-dir ckpt/exp05/<tier>_<backbone>/attempt_<ts> --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1 --vit-dim <dim> --vit-depth <depth> --vit-heads <heads> --vit-mlp-dim <mlp_dim>` — 296 334 samples / epoch, 9 261 micro-batches, 4 631 updates / epoch, 55 572 updates total; the epoch-12 checkpoint is confirmatory for every arm (exp_01's M pair at epoch 12 likewise); no `best.pth` or test-selected epoch enters a verdict. No resume for reportable runs (checkpoints do not carry sampler/RNG state; an interrupted attempt is preserved as `_ABORTED_` and restarted).

New trainer flags `--vit-dim/--vit-depth/--vit-heads/--vit-mlp-dim` (defaults = M = current behaviour; `args.json` records them) are the only code change to the trainer. If exp_04's `--yaw-aug` flags have landed by then, they stay at 0 here (no augmentation in this experiment).

Expected time per run at ~1.4 s/iteration for M (28.2 h / 12 epochs): S ≈ 15–18 h, L ≈ 45–55 h (the ViT is run 2 + K = 10 times per sample, so encoder FLOPs dominate); measure in the smoke and gate the launch on a projected ≤ 60 h per run. Four runs ≈ 130 GPU-hours; on two A6000s about 3 days, scheduled around exp_04's training (GPU 0, ≈ 30 h from its launch) — coordinate GPU ownership with the exp_04 session before launching.

## 4. Evaluation protocol (declared; reuse exp_04's tooling when merged, otherwise exp_03's)

- Unseen split, all 6 337 queries, K = 8 (primary) and K = 1 (secondary), `eval_yaw_rotation.py` at k = 0 only (`--yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --decomposition-batches 0`), batch 16 canonical, TF32 off, no test-time augmentation.
- **Five evaluation seeds {42, 43, 44, 45, 46}**: reference manifests `ckpt/yaw_aug/reference_manifest_k{8,1}_seed{s}.json` (built 2026-09-12 by exp_04; hashes in `ckpt/yaw_aug/reference_manifests_index.json`) with `--gl-seed s`; all six arms evaluated under every seed → 6 arms × 2 K × 5 seeds = 60 runs (≈ 10 min at K = 8, ≈ 6 min at K = 1: ≈ 8 h of GPU). The M pair's exp_04 evaluations (same manifests, same seeds) are reused if they exist under the same evaluator closure; otherwise rerun.
- Per-run provenance as in exp_04 (pre-run evaluation manifest, launcher-owned completion sidecar) if that tooling has landed; at minimum, exp_03's `bind_provenance.py`-style sidecars (checkpoint sha256, closure digest, manifest hash, environment).
- Yaw robustness per tier (secondary, K = 8, seed 42 only, condition P at k ∈ {0, 32, 64, 448, 480}) for the four new arms — descriptive, to see whether the exp_03 finding (no robustness gain for cylindrical at M) holds at S and L.

## 5. Pre-registered hypotheses and decision rules

Statistics as exp_03/exp_04: per query, the five per-seed errors are averaged (plain means; a query enters a cell only if finite for all seeds in both arms of the cell); paired per-query differences bootstrapped over queries (20 000 resamples, bootstrap seed 0; query-level primary, this fixed split) with the room-cluster bootstrap (17 rooms) secondary; two-seed convergence gate (max endpoint movement ≤ 10 % of the seed-0 width, else raise `n_boot`). Each hypothesis is a separate family; no omnibus rule.

- **H1 (dominance of the curve; K = 8, EDT and C50 error, epoch 12):** for every tier t ∈ {S, M, L}, δ_t = (mean error_cyl,t − mean error_simple,t) / mean error_simple,t. Family m = 6 (3 tiers × 2 metrics), two-sided adjusted intervals [Q_{0.05/12}, Q_{1−0.05/12}]. "Supported" iff the adjusted upper bound of δ_t is below 0 at all three tiers for at least one metric with no tier showing the opposite sign significantly on that metric; "partially supported" if at some tiers; "not supported" if none. Report the per-tier cells and the sign pattern for both metrics; T60 descriptive. The same table at K = 1 as a separate secondary family (exp_03's K = 1 follow-up showed the reversal at M; the curve tells whether it is tier-dependent).
- **H2 (fixed-target reading; experiment 3):** target fixed now = Baseline-M's five-seed mean EDT and C50 (K = 8). Cylindrical-S "reaches the target" for a metric iff the adjusted upper bound (family m = 2, [Q_{0.0125}, Q_{0.9875}]) of (mean error_cyl,S − mean error_simple,M)/mean error_simple,M is below 0 or the cell is equivalent within ±3 % by TOST at the same level; report the encoder-parameter ratio (2.78 M vs 19.70 M ≈ 7.1×) only if both metrics reach the target. Likewise Cylindrical-M vs Baseline-L (19.75 M vs 43.71 M ≈ 2.2×). No other pairing is tested.
- **Descriptive:** the two curves (x = encoder parameters, log scale; y = error with seed SD and bootstrap intervals), the full-system parameter axis as a second panel, per-epoch test-loss trajectories, the yaw-robustness block per tier, throughput and memory per tier.

## 6. Code (Coder = OpenAI Codex `gpt-6-astra` per the SOP; TDD; Fable 5.1 review per round)

| file | change | tests |
|---|---|---|
| `model/xRIR_cyl.py` | `build_xrir(backbone, num_shot, dim=512, depth=12, heads=8, mlp_dim=512)` already accepts these kwargs — verify; add `count_parameters(model)` → `{"encoder", "full", "trainable", "non_encoder"}` | the three tier counts of §2 reproduced exactly; `dim_head` = 64 kept; token count still 256 (the `lin_proj_0` assertion) |
| `train_xRIR_backbone.py` | flags `--vit-dim --vit-depth --vit-heads --vit-mlp-dim` (defaults = current), passed to `build_xrir`; recorded in `args.json` together with the parameter counts | defaults reproduce the current model bit-for-bit (state-dict shapes and a fixed-seed forward); non-default tiers build and run one step on CPU-sized inputs |
| `tools/exp05_profiles.py` | the six arm definitions (tier, backbone, checkpoint path + sha256 filled after training), the manifest hashes per seed, H1/H2 rules and families, bootstrap seeds | literal, immutable, type-checked |
| `tools/param_curve.py` | consumes the six arms' five-seed k = 0 outputs (exp_04's `paired_compare.py` per cell if available; else a local paired bootstrap through `tools/paired_stats`), writes canonical JSON (per-tier cells, H1/H2 verdicts, curves) + summary with sha256 binding; page generator reads only that JSON | synthetic inputs with known shifts; refusals on missing seeds/mismatched queries/wrong K |
| launcher | exp_04's training launcher (`tools/exp04_launch.sh`) generalised with a `--tier` argument, or a copy `tools/exp05_launch.sh` with the same gates (exact argv golden file, empty attempt dir, resource checks, banner-free here, completion sidecar) | guard tests as exp_04 |

## 7. Validation ladder, parity audit, acceptance criteria

1. static; 2. `pytest tests/`; 3. tiny forward at S and L on GPU 1 (the baseline `apply_delay` hard-codes `.cuda()`, so CPU forwards are impossible — do not "fix" that in this experiment); 4. real-data readback: one real batch through each tier, shapes and loss finite; 5. smoke (3 batches, `--no-save` if available, else a scratch save dir deleted afterwards) at S and L: iteration time and peak memory recorded, projected run time ≤ 60 h; 6. parity: M-tier defaults reproduce the exp_01 model (state-dict key/shape identity with `ckpt/xRIR_simple_8_shot/epoch_12.pth` and a fixed-seed forward equal to the current script); 7. launch. Acceptance criteria per launch (in the notebook before launching): commit SHA, GPU, exact argv, ≥ 1 finite optimizer step, epoch-1 time ≤ 1.05 × the smoke projection, `history.jsonl` per epoch, `epoch_NNN.pth` for every epoch, no resume.

## 8. Deliverables

`_results.md` (tables only, from the canonical JSON), `_analysis.md`, `param_efficiency_01_results.html` (the two curves, the H1/H2 cells, parameter table with encoder/full/trainable columns for all six arms, throughput/memory), `commits_param_efficiency.md`, params/command records, logs, reviews; the parameter columns for the paper's existing table (M tier) come from `count_parameters` and can be reported immediately.

## 9. Cost, risks, decisions

- Cost ≈ 130 GPU-hours training + ≈ 8 h evaluation; ≈ 3 days on two A6000s after exp_04's training frees GPU 0.
- Risks: L may be memory- or time-limited at batch 32 (fallback: `--batch-size 16 --accum-steps 4`, same effective batch, disclosed); S may under-fit within 12 epochs (same budget is the design; the trajectory is reported); one training seed per arm; the M pair predates this experiment (different day, unrecorded `PYTHONHASHSEED`) — disclosed as an operational difference.
- Decisions for the user: (1) tiers S/M/L as above (alternative S+ = dim 384/depth 8: 7.7 M encoder, if a denser curve is wanted at +2 runs); (2) reuse exp_01's M pair rather than retraining it; (3) K = 8 primary, K = 1 secondary; (4) the fixed-target pairings (Cyl-S vs Base-M, Cyl-M vs Base-L).

## 10. Handoff notes for the executing agent

Read `CLAUDE.md`, `worklog/experiment_SOP.md` (roles: Planner = the executing Claude session, Coder = Codex `gpt-6-astra` via `codex exec -s workspace-write`, code Reviewer = a Claude Fable 5.1 subagent, plan Reviewer = Codex), exp_01's and exp_03's records, and exp_04's record (its launcher, evaluation-manifest and producer tooling are designed to be reused; check `commits_yaw_aug_xrir.md` for what has landed). GPU 0 is occupied by exp_04's training for ≈ 30 h after its launch; GPU 1 runs exp_04's evaluations intermittently — agree the schedule with that session before launching. Use the manifests under `ckpt/yaw_aug/` unchanged. Never modify `tools/yaw_rotation.py` (pinned by exp_03's closure). Commit small, log every action in `param_efficiency_worklog.md`.
