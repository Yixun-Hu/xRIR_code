# Commits — cylvit_vs_simplevit

No commits yet: the experiment predates the SOP and its code is still uncommitted in the working tree (retrospective record written 2026-09-05). Files belonging to this experiment:

- `model/xRIR_cyl.py` (new) — `xRIR_Cyl` subclass swapping only `source_network`; `build_xrir` factory.
- `model/cylindrical_vit.py` (1 line) — `from __future__ import annotations` so the PEP 604/585 hints import on Python 3.8.
- `train_xRIR_backbone.py` (new) — backbone-selectable fork of `train_xRIR_unseen.py` (argparse, workers, accumulation, resume, bounded flags).
- `eval_xRIR_backbone.py` (new) — backbone-selectable evaluator reusing `eval_unseen.Evaluator`; deterministic order, `--max-samples`, JSON + per-sample dumps.
- `tools/compare_eval.py` (new) — paired bootstrap comparison of two per-sample files (`--by-test-rooms`).
- `tools/summarize_epochs.py` (new) — per-epoch table, epoch-averaged paired differences, best-checkpoint comparison.
- `.gitignore` — `ckpt/` added (and the missing-newline fix on 2026-09-05).

Base-commit handling is decided at the exp_03 plan approval; SHAs will be appended here once committed.
