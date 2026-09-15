# Worklog — seen_protocol (exp_07)

## 2026-09-15T09:58:57-04:00 — experiment opened after Yixun's "go for the exp_07" (seen columns of the paper table for the three variants)
- **Goal** — seen-protocol retraining of xRIR / CylindricalViT / YawAug-xRIR (same recipe, `--protocol seen`) and five-seed seen-split evaluation at K = 1, 8; combined LaTeX table.
- **Facts established** — seen train set 296 454 files (9 265 micro-batches), test 6 217 pairs / 131 rooms; 5 124 of the seen-test pairs lie in the unseen protocol's training set and 1 093 in its 17 test rooms (hence no reuse of the exp_04 models).
- **Change** — record folder created; `seen_protocol_yixun_query.md`; `plan_seen_protocol.md` v1. Next: Codex read-only plan review.

## 2026-09-15T10:07:07-04:00 — Codex plan review round 1: request changes (3 blockers on tooling reuse, 6 should-fix) → plan v2; round-2 review launched
- `seen_protocol_codex_plan_review.md`: the frozen evaluator hard-codes the unseen dataset (a seen manifest is rejected: 6 217 ≠ 6 337), `train_data_identity`/launcher recovery assume the unseen inventory, the probe/gates lack M/seen support, the approval/admission layer must be exp_07-specific and role-aware; update count 55 596 (not 55 590), last micro-batch 6, counter domain; estimand to freeze; released-row caveats; measured per-arm times (87 GPU-h); roles per the 2026-09-15 SOP (Opus 5 Coder in a worktree, Codex Reviewer). All applied in v2 (§9 changelog); v1 kept as superseded.

## 2026-09-15T10:13:39-04:00 — Codex plan review round 2: **approve with changes** (5 should-fix, 1 nit; no scientific blocker) → plan v3 (changelog §10); plan approved for implementation (Yixun: "go for the exp_07"); decision on GPU ordering pending (default B)
- `seen_protocol_codex_plan_round2_review.md`. Worktree `/home/yixunhu/codespace/xRIR_code_wt07` (branch `exp07-window`) created for the Opus 5 Coder; round 1 = trainer `--protocol`, protocol-aware training identity, seen-split binding, yaw-aug audit option.
