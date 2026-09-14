The record generators consume canonical JSON and its adjacent `.provenance.json` only. They never read evaluation arrays or recompute statistics. Use the xRIR interpreter with the repository on `PYTHONPATH`.

Both generators accept `--h1-k8 JSON --h1-k1 JSON --h2-k8 JSON --tost-k8 JSON --table JSON [--diag JSON ...] --out PATH [--draft-ok]`. Use `yaw_aug_xrir_results.md` and `yaw_aug_xrir_01_results.html` as the output filenames when final producer outputs exist. Draft mode permits absent verdicts for tests; it still refuses exploratory or unbound JSON.

TABLE_V1 has no verdict; TOST verdicts are per cell. Family verdict blocks precede the detailed tables, including every TOST cell verdict. Display rounding uses EDT milliseconds (1 decimal), C50 dB (3 decimals), T60 percentages and ratios/bounds (2 decimals). JSON retains full precision. Booleans read yes/no and compound evidence is rendered as labelled text. Tables include protocol, seed exclusions, companion and room intervals, and convergence evidence. SVG coordinate arithmetic only positions producer values.

`--diag` accepts only a descriptive producer profile and never confirmatory JSON. Its canonical JSON, sidecar and summary must stay together.

`bind_provenance.py --runs DIR ... --results JSON ... --attempt FINAL --probe-receipt JSON --audit JSON [--approved JSON] --out REPORT_DIRECTORY` validates completion links and exclusively creates `binding_report_<UTC timestamp>.json` in the existing directory. `check_record.py REPORT_DIRECTORY` verifies the latest timestamped report using its original git HEAD; an invalid latest report fails without fallback. The binder reads run evidence and never modifies run directories. It binds every canonical producer JSON, sidecar and summary to the loaded approval identity and pins, checks result run inputs against bound completion evidence, verifies real directory listings, and requires producer commits to be ancestors of its recorded HEAD.

Operational order:

1. Finish and review code changes, then commit the code. Compute producer closures from that committed code, fill the approved checkpoint/closure digests, and commit the pins in a second reviewed commit before running producers.
2. After training/evaluations complete, run the canonical producers; retain each JSON, summary and sidecar together.
3. Bind the completed run set using `--results` with all five canonical JSONs and any diagnostics; retain the immutable report.
4. Run `check_record.py`, generate Markdown and HTML from the bound JSONs, then run `check_record.py` again before publication.

The living TABLE_V1 `.md` stays digest-bound. Regenerating it with `--force-md` requires a new table JSON/sidecar and a new bind with the current results before verification or publication. Retain earlier reports unchanged; the checker verifies only the latest report.

Run `bash static_checks.sh` from this directory or invoke it by its full path. The CPU regression subset is `python -m pytest tests/test_exp04_record_tools.py tests/test_paired_compare.py tests/test_results_table.py -q -p no:cacheprovider` from the repository root. A restricted installation may need `NUMBA_CACHE_DIR=/tmp/xrir_round8_numba`.

Produce diagnostics with `python tools/exp04_descriptive.py --profile {GRID_SEED42,EPOCH9_K8} --runs DIR [DIR ...] --json PATH --summary PATH`. Approve `closures.producer_descriptive` and (for epoch 9) `checkpoints.aug_epoch9` first; both start null. GRID uses seed 42, 18 spectral and 10 acoustic angles; EPOCH9 uses five seeds at k = 0. Both are P-only descriptions with query bootstrap intervals and no verdict.
