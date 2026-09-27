# exp_10 yaw_pilot — Codex code review, round 8 (close; figure annotation precedence + K1 display label)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt10 --skip-git-repo-check`, stdin closed) · **Reviewed:** `24eb1a6, ec77491, 209bc25` (Claude Opus 5) · **Prompt:** `review_prompts/code_round8_close_prompt.md` · **Raw log:** `yaw_pilot_2026-09-27_08:32:43_codex_code_round8_close.log` · **Date:** 2026-09-27 · **Verdict:** APPROVED; approved for merge; blocking list empty

Reviewed `24eb1a6`, `ec77491`, and `209bc25` after `0c57d45`. **No new findings.**

1. **Prior finding 2 — should-fix, closed.**  
   [exp10_summarize.py:942](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:942) gives unresolved Monte Carlo uncertainty precedence, with `mc?` explained in the footer. Both specified cells render `mc?`. The other 158 nonzero summary cells retain their abbreviations.  
   **Minimal fix:** none remaining.

2. **Prior finding 5 — should-fix, closed.**  
   [exp10_summarize.py:753](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:753), [make_results_md.py:165](/home/yixunhu/codespace/xRIR_code_wt10/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/make_results_md.py:165), and [make_results_html.py:172](/home/yixunhu/codespace/xRIR_code_wt10/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/make_results_html.py:172) display **“trained K = 8, evaluated K = 1”** in figure titles/row labels and both report headers. Machine keys, CSV, HTML IDs, links, and filenames remain unchanged.  
   **Minimal fix:** none remaining.

**Verified**

- All exp_10 suites plus the exp_03 record suite: **384 passed, 1 skipped**. Available real-data CPU smoke tests ran. The sole skip requires two visible GPUs; GPU access was disabled.
- Before/after summarizer JSON, Markdown tables, and CSV are **byte-identical** on a four-arm, five-angle scratch fixture using 10,000 draws, seeds 0/1, and a fixed generation timestamp. Statistics, tables, and writers are unchanged.
- Canonical-data figure comparison: only cylindrical T60 at 270° changes among the 64 standard figure cells. Control `T60_abs` at 45° was verified through a direct panel render because that supplement is outside the standard panels.
- All **10 PNG/PDF assets** are byte-deterministic across fresh renders with `SOURCE_DATE_EPOCH=0`. Labels and footers are visible without clipping.
- Both report renderers passed independent before/after comparisons.
- `py_compile`, `git diff --check`, and all **12 exp_03 source pins** passed.
- Disposable-clone `--no-ff --no-commit` merge onto main **`24715b2bbac01109961762c8ef48ae04e85dc4aa`** succeeded. Exactly the six authorized files change, including the accepted fixture adjustment. Clone deleted; tracked files untouched; review writes confined to `.review-round8/`.

**Round-8 verdict: approved; closed.**  
**Merge verdict: approved for merge.**  
**Exact blocking list: `[]`.**
