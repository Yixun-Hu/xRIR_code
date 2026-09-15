# Worklog — seen_protocol (exp_07)

## 2026-09-15T09:58:57-04:00 — experiment opened after Yixun's "go for the exp_07" (seen columns of the paper table for the three variants)
- **Goal** — seen-protocol retraining of xRIR / CylindricalViT / YawAug-xRIR (same recipe, `--protocol seen`) and five-seed seen-split evaluation at K = 1, 8; combined LaTeX table.
- **Facts established** — seen train set 296 454 files (9 265 micro-batches), test 6 217 pairs / 131 rooms; 5 124 of the seen-test pairs lie in the unseen protocol's training set and 1 093 in its 17 test rooms (hence no reuse of the exp_04 models).
- **Change** — record folder created; `seen_protocol_yixun_query.md`; `plan_seen_protocol.md` v1. Next: Codex read-only plan review.
