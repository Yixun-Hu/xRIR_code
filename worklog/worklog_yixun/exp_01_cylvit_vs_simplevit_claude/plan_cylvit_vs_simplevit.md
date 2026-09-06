# Plan — cylvit_vs_simplevit (RETROSPECTIVE, as executed)

> Written 2026-09-05 from the session record. The experiment ran 2026-09-04/05 before the SOP existed: there was no separate plan document, no reviewer loop, and the user approved the design implicitly by not objecting to the configuration reported at launch. The plan below is what was actually done, in the SOP's format, so later experiments can build on it.

## Question and hypothesis

Does replacing xRIR's SimpleViT geometry encoder with the user's azimuth-equivariant **CylindricalViT** (`model/cylindrical_vit.py`) improve few-shot cross-room RIR prediction on AcousticRooms? **User hypothesis:** yes, CylindricalViT + xRIR outperforms SimpleViT + xRIR.

## Design

1. **Baseline reproduction first** (evaluation integrity): released `checkpoints/xRIR_{unseen,seen}.pth` through the untouched `eval_unseen.py` / `eval_seen.py` on the full published test splits (unseen 6337, seen 6217 queries, K=8). Target: paper Table 1.
2. **Controlled swap**: `xRIR_Cyl(xRIR)` replaces only `source_network` (SimpleViT → CylindricalViT with identical dim/depth/heads/mlp_dim; same `[B, 256, 512]` token layout so `lin_proj_0`, the 256→1 learned pool over tokens, is unchanged). Everything else (shift-and-align, sinusoidal geometry embedding, ResNet-18 audio encoder, reference attention, spectrogram head, loss) is inherited byte-for-byte.
3. **Same-budget control**: because the released checkpoints' training budget is unknown, retrain a SimpleViT control with exactly the flags used for the cylindrical run. Both from scratch, seed 0.
4. **Schedule**: the authors' recipe (AdamW lr 1e-3, wd 1e-4, `ExponentialLR` gamma 0.1) compressed from 200 epochs / decay 50 to **12 epochs / decay 3** (same 4:1 ratio, same LR endpoints 1e-3 → ~1e-7), effective batch 64 as 32 × 2 accumulation (64 OOMs on 48 GB), TF32 on, K=8.
5. **Evaluation**: after every epoch, both models' weights on the full unseen split with identical seeded reference draws → paired per-sample EDT / C50 / T60 / log-STFT MSE / test loss with 95 % bootstrap CIs (`tools/compare_eval.py`). Final judgment from (a) per-sample differences averaged over epochs 5–12 (after LR < 5e-5) and (b) the best-test-loss checkpoint pair (`tools/summarize_epochs.py`). Seen split evaluated once at epoch 9 on request.
6. **Sanity checks before training**: forward-shape self-test of both backbones; equivariance test (scene rotated by one azimuth patch → cylindrical tokens roll exactly, SimpleViT tokens do not); smoke train/eval round trip; the new eval script reproduces the original script's running averages on the released checkpoint.

## Planned code (as implemented)

| File | Change |
|---|---|
| `model/cylindrical_vit.py` | add `from __future__ import annotations` (Python 3.8 env) |
| `model/xRIR_cyl.py` (new) | `xRIR_Cyl`, `BACKBONES`, `build_xrir`; `__main__` forward self-test |
| `train_xRIR_backbone.py` (new) | argparse fork of `train_xRIR_unseen.py`: `--backbone`, workers, accumulation, TF32, intra-epoch `last.pth`, resume, `--max-*-batches`, `history.jsonl` |
| `eval_xRIR_backbone.py` (new) | argparse evaluator reusing `Evaluator`/`griffin_lim`; `--split`, `--max-samples`, `--save-metrics`, `--save-per-sample` |
| `tools/compare_eval.py` (new) | paired bootstrap CIs, `--by-test-rooms` |
| `tools/summarize_epochs.py` (new) | final tables |
| `.gitignore` | `ckpt/` |

Tests: none written (pre-SOP). Verification was by self-test scripts and smoke runs; see `_worklog.md`.

## Infrastructure decisions made during execution

- Data is on a CIFS share; per-sample `os.listdir` + ~20 small reads stalled 32 loader workers. Mirrored the ~24 GB used by the code to local NVMe (`~/data_cache/AcousticRooms`, verified with an rsync dry run) and set `XRIR_DATA_PATH`.
- Per-epoch evaluation pipeline snapshots each run's weights the moment its epoch is logged (a race in the first version lost one snapshot; recovered from `best.pth`).

## Acceptance criteria (as applied)

- Baseline reproduction within run-to-run noise of Table 1.
- Both trainings reach 12 epochs with finite losses and identical configuration.
- Each per-epoch comparison uses all 6337 test samples with identical references (asserted by index equality in `compare_eval.py`).
- Outcome judged on paired CIs, not single-epoch means.
