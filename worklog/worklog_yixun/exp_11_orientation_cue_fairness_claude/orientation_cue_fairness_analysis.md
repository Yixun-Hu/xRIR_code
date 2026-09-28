# Analysis — orientation_cue_fairness (exp_11): does CERPA's HAA advantage survive when the xRIR baselines receive the same loudspeaker-orientation cue?

Status: DRAFT skeleton (2026-09-28 08:2x); §3–§4 are filled only from the canonical `ckpt/exp11/{phase1,phase1b}/stats.json` and the final `ckpt/exp11/stats.json` once produced. No number in this file may come from anywhere else.

## 1. Question and design (plan v3 + amendments A1–A3)

Yixun's request (2026-09-26): fine-tune and evaluate xRIR, xRIR + orientation cue, yaw-augmented xRIR + orientation cue and CylindricalViT + orientation cue under the same HAA protocol; key controlled comparison xRIR + cue vs CylindricalViT + cue. Mechanically the cue has two parts (plan §2.1): the loudspeaker-heading frame (implicit conditioning, a data-level roll `k = 128` in all four rooms) and the two constant azimuth planes of `CylindricalViTOriented` (explicit conditioning, pretrained). For the SimpleViT the planes are not a function-preserving change (Codex plan review, round 1: the first LayerNorm of the patch embedding normalises geometry and azimuth jointly), so "xRIR + orientation cue" is realised literally: a `simple_oriented` backbone (SimpleViT + the same two planes) pretrained from scratch under exp_01's recipe (arm H) and under exp_04's yaw-augmented recipe (arm I), fine-tuned in the heading frame. exp_06's arm D (SimpleViT in the heading frame) is reported as the implicit-conditioning control; the yaw-augmented model in the heading frame is arm G; two zero-initialised azimuth adapters fine-tuned from the exp_01 control and the exp_04 checkpoint (arms J, K, room frame) are the cheap explicit controls, registered as a different design from CERPA's pretraining.

Pre-registered statements (plan §3): Phase 1 N1 (G − D), N1i (interaction (G − E) − (D − A)), N2 (C − G: category, G's non-inferiority at +0.23 dB, C's margin-sized advantage), N3 (G − E: category + equivalence), three screen families; Phase 1b Q1–Q4; Phase 2 P3 = C − H (the only statement that can establish the headline), P3′ (C − I), P4/P4′, P1/P2; R1 historical rows copied from the exp_06/exp_09 canonical JSONs with source hashes; A′ (the released checkpoint) as an external reference row, never paired. Every decision-bearing field is withheld on void/unconverged cells.

## 2. Provenance chain

- **Round 1** (arm G in the extended exp_06 pipeline; the exp_11 summariser phase 1): worktree `exp11-cue` c6233e3 → 5db0c82 → fix cycles → 91fbd8b; Codex code review (request changes) + close reviews 1–2 (approve) at ultra; merged **bb1b59d**, exp_06 approvals re-fill **9f98bbb** (`haa_pipeline_sh` f33d4ffe…, `summarize_haa` 645e74c0…).
- **Round 2** (the exp_11-owned family: `model/{simple_vit_oriented,xRIR_simple_oriented,xRIR_simple_adapter,xrir_exp11_registry}.py`, `tools/exp11_{profiles,recipe,train,finalize,smoke,haa_finetune,haa_eval,pidrecord,pathprobe,lock_holder}.py`, `tools/exp11_{launch,haa_pipeline}.sh`, summariser arms H/I/J/K and phases): c491396 → c8acda2 → 19 fix cycles → **46f0a76**; Codex code review + 19 close reviews at ultra (`orientation_cue_fairness_codex_code_round2*_review.md`). Every scientific part was accepted at close review 2; close reviews 2–19 concerned the pretraining launcher's operational safety (per-arm publication lock → plan A1: kernel flock held by a holder process leased to the launcher's life; crash-safe trainer registration with a `launching` marker, `train.pid`, a permanent `.resolved` tombstone and `--resolve-unregistered`; a single pid-record reader; a single errno-aware path probe → plan A2: a path-free verdict protocol with identity questions by device and inode). Merged **6a299b1**; exp_06 re-fill **6baa337** (`summarize_haa` 3956692a…; record sha256 38fe553a…); exp_11 approvals first fill **6b70843** (eight code keys; reused identities exp_04 2e452117…, exp_06 38fe553a…, legacy receipt 5a124946…; artifacts null until each pretraining is finalised) — both records byte-identical to the reviewer's fill manifest.
- **Frozen**: `tools/exp06_launch.sh` d20ca47a…, `tools/exp06_finalize.py` 9237ed6d…, every pre-existing `model/*.py`, `BACKBONES_EXP06` f9852f85…, the four sim-eval closures; `tests/test_exp09_sim_eval_closures.py` green throughout.
- **Runs**: (to fill: launch commits, GPU, attempt directories, completion digests, artifact pins, summariser HEADs and output hashes per phase.)

## 3. Results (from the canonical JSONs only)

(to fill)

## 4. Reading

(to fill)

## 5. Deviations from the plan and decisions taken

- A1 (kernel flock), A2 (path-free probe protocol), A3 (schedule slip: GPU 1 ≈ Oct 1 10:00, GPU 0 ≈ Oct 3 18:00; sequential on GPU 1).
- The pre-launch/launch automation is marker-gated (`GO_GPU` → Phase-1 chain), as in exp_09.
- Post-merge suite on main (2026-09-28 08:00): 24 failures, none a production defect — nine exp_06 launcher preflight tests hit the real GPU census while a peer job held GPU 1 (pass in isolation), twelve of the peer's exp_10 record-tool tests are timing-sensitive under load (pass in isolation), and three exp_11 tests assumed an unfilled approvals record or a shell that can register an INT trap (a shell started under `nohup`/`setsid` has SIGINT ignored on entry and cannot trap it — in production the launchers are started that way, so TERM is the operative signal and its trap registers). Fixed by a tests-only cycle (fix 20); no approvals key moved.
- Codex close review 19's nonblocking wording items in `tools/exp11_launch.sh` (an old ≈ 60 s example at ≈ 49; the health-check count at ≈ 552) are left as they are: editing the launcher would move the `launch_sh` closure after the fill.

## 6. Open items

(to fill)
