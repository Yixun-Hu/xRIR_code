# Commits — yaw_aug_xrir (exp_04)

- `8cb87d1e0e8cbd85913911fdefd64aa69ced931d` — exp_04 base: plan v4 approved; SOP roles updated; exp_03 K=1 follow-up (Planner)
- `fda92858b789ab33ae5dfdd53d34715d4e1e866e` — exp_04: add strict per-sample batched yaw rotation (Codex round 1, cycle 1) — **superseded and dropped before push** (it edited `tools/yaw_rotation.py`, a member of exp_03's pinned closure; amendment A2 relocated the function)
- `410f205ff3fa340f1eb9f371e8bf226e5a935d11` — exp_04: seeding, offsets and batched rotation in tools/yaw_aug.py (A1 chi-square bound, A2 relocation) (Codex round 1, cycles 1+2)

- `cd7dc994f17b656910ecce7b65633afda10fb806` — exp_04 round 2 cycle 1: trainer yaw integration, no-save, parity and startup guards; 180 changed lines.
- `34d98816a4bc4b79002aa17db78aad7ba94b0eed` — exp_04 round 2 cycle 2: seeded alignment audit CLI, cohort hashing and tests; 183 changed lines.

## Round-2 close-out — 2026-09-12T16:46:56-04:00

- `a619bc86939aebb6b74410a387c602237f7384b7` — loader-derived epoch length and strict counter bound, +45/-18 = 63 lines.
- `da1becc8e95c2bc73c07b753b07dc896287aacbf` — test-only semantic equality, banner, audit and CLI edges, +57/-19 = 76 lines.
- `35f55ad4d5960e99587401cd70aa32f0213ba67c` — strict native counters/enabled contract and exclusive audit args/env, +72/-6 = 78 lines.

All carry the requested Codex trailer; no worklog/ or ckpt/ committed; no push.
