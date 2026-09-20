# Note for the author of `plot_exp05_efficiency.py`

This file was created (untracked) as `tools/plot_exp05_efficiency.py` on 2026-09-19 at 22:07 by a session other than the exp_07 Planner. Any untracked file outside `worklog/` makes the reviewed experiment launchers refuse (`dirty_outside_worklog requires --allow-dirty`); it blocked nine of exp_07's seen evaluations at 22:08. The Planner moved it here unchanged at 22:2x on 2026-09-19 (see the exp_07 notebook) and committed it under the exp_05 record assets so the main tree is clean.

Caveats for running it from here: its `ROOT = Path(__file__).resolve().parents[1]` assumed `tools/`; from this location use `parents[4]`, or run a copy from a git worktree. It imports `tools.exp05_record`, so keep the repository root on `PYTHONPATH`. Please do all further work on it in a worktree, not in the main checkout, while experiment launches are armed.
