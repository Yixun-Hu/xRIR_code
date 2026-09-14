The record generators consume canonical JSON and its adjacent `.provenance.json` only. They never read evaluation arrays or recompute statistics. Use the xRIR interpreter with the repository on `PYTHONPATH`.

Both generators accept `--h1-k8 JSON --h1-k1 JSON --h2-k8 JSON --tost-k8 JSON --table JSON [--diag JSON ...] --out PATH [--draft-ok]`. Use `yaw_aug_xrir_results.md` and `yaw_aug_xrir_01_results.html` as the output filenames when final producer outputs exist. Draft mode permits absent verdicts for tests; it still refuses exploratory or unbound JSON.

TABLE_V1 has no verdict; TOST verdicts are per cell. H1/H2 aggregate wording appears once above each table. Estimates and intervals retain canonical ratio units; table EDT is already milliseconds in the producer JSON. Tables include protocol, seed exclusions, companion and room intervals, and convergence evidence. SVG coordinate arithmetic only positions producer values.

The optional diagnostic path has an upstream gap: the plan's single-seed descriptive grid and epoch diagnostic have no verdict, and no current producer emits them with the required provenance sidecar. They therefore remain refused. The optional curve renderer supports canonical one-arm cells and labels their actual profile; it does not relabel TOST as the single-seed full grid. A diagnostic producer contract must be settled before that planned grid can be published.

`bind_provenance.py --runs DIR ... --attempt FINAL --probe-receipt JSON --audit JSON [--approved JSON] --out binding_report.json` validates completion links and writes exclusively. `check_record.py binding_report.json` recomputes the evidence using the report's original git HEAD. The binder reads run evidence and never modifies run directories.

Run `bash static_checks.sh` from this directory or invoke it by its full path. The CPU regression subset is `python -m pytest tests/test_exp04_record_tools.py tests/test_paired_compare.py tests/test_results_table.py -q -p no:cacheprovider` from the repository root. A restricted installation may need `NUMBA_CACHE_DIR=/tmp/xrir_round8_numba`.
