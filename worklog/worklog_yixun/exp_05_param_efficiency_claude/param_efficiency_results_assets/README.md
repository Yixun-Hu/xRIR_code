# exp_05 record assets — producers → generators → figures → bind → check

The generators consume the five canonical `tools/param_curve.py` products and their
adjacent `.provenance.json` sidecars, read through `tools/exp05_record.py`. They never
read evaluation arrays and never recompute a statistic. Two columns have no canonical
source and are read from the four certified training attempts instead, every file at the
digest that attempt's own `completion.json` (or the `train_manifest.json` it hash-binds)
records for it: throughput/memory (from the probe receipt the manifest names) and the
per-epoch test-loss trajectory (`history.jsonl`). The parameter table is measured at
generation time with `tools.exp05_params.count_parameters` and refused unless it
reproduces the registered counts of `tools/exp05_profiles.ARMS` exactly.

The six-arm table's **M rows come from exp_05's own M re-evaluations** inside `CURVE_K8`
and `CURVE_K1` — the twenty `ckpt/exp05/eval/M_*` runs through `tools/exp05_eval.py` —
never from exp_04's `TABLE_V1`: the two protocols differ and must not be mixed in a row.

Use the xRIR interpreter with the repository on `PYTHONPATH`:

```bash
conda activate xRIR                     # or /home/yixunhu/miniconda3/envs/xRIR/bin/python
cd /home/yixunhu/codespace/xRIR_code
export PYTHONPATH=$PWD XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
A=worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets
W=worklog/worklog_yixun/exp_05_param_efficiency_claude
R=ckpt/exp05/results
ATTEMPTS="ckpt/exp05/S_simple/final ckpt/exp05/S_cylindrical/final ckpt/exp05/L_simple/final ckpt/exp05/L_cylindrical/final"
JSONS="$R/CURVE_K8.json $R/CURVE_K1.json $R/TARGETS_K8.json $R/TARGETS_K1.json $R/YAW_K8_SEED42.json"
FLAGS="--curve-k8 $R/CURVE_K8.json --curve-k1 $R/CURVE_K1.json --targets-k8 $R/TARGETS_K8.json --targets-k1 $R/TARGETS_K1.json --yaw-k8 $R/YAW_K8_SEED42.json"
```

## 1. Producers (already run on 2026-09-18; repeat only into a fresh directory)

```bash
python tools/param_curve.py --profile CURVE_K8  --runs ckpt/exp05/eval/*_k8_seed*_k0 \
    --json $R/CURVE_K8.json  --summary $R/CURVE_K8.txt
python tools/param_curve.py --profile CURVE_K1  --runs ckpt/exp05/eval/*_k1_seed*_k0 \
    --json $R/CURVE_K1.json  --summary $R/CURVE_K1.txt
python tools/param_curve.py --profile TARGETS_K8 --runs ckpt/exp05/eval/*_k8_seed*_k0 \
    --json $R/TARGETS_K8.json --summary $R/TARGETS_K8.txt
python tools/param_curve.py --profile TARGETS_K1 --runs ckpt/exp05/eval/*_k1_seed*_k0 \
    --json $R/TARGETS_K1.json --summary $R/TARGETS_K1.txt
python tools/param_curve.py --profile YAW_K8_SEED42 --runs ckpt/exp05/eval/*_k8_seed42_yaw \
    --json $R/YAW_K8_SEED42.json --summary $R/YAW_K8_SEED42.txt
```

Each writes the JSON, the text summary and the sidecar together; the writer creates all
three exclusively and rolls back on any failure. Keep the three files together: the
generators refuse a product whose companion summary or sidecar has moved or changed.

## 2. Generators

```bash
python $A/make_results_md.py   $FLAGS --attempt $ATTEMPTS --out $W/param_efficiency_results.md
python $A/make_results_html.py $FLAGS --attempt $ATTEMPTS --out $W/param_efficiency_01_results.html
```

Both refuse an exploratory or deviating product, a product read as the wrong profile or
computed under anything but the registered one, a duplicate canonical input, an attempt
set that is not exactly the four trained arms, and an `--out` that names any file they
read — the five products, their summaries and sidecars, the approval, and each attempt's
`completion.json`, `train_manifest.json`, `args.json`, `history.jsonl` and probe receipt —
by resolved path and by device/inode, so neither a hardlink nor a symlink to an input can
be truncated open.

## 3. Figures (PNG and PDF for the paper)

```bash
mkdir -p $W/param_efficiency_figures
python $A/make_figures.py --curve $R/CURVE_K8.json $R/CURVE_K1.json --outdir $W/param_efficiency_figures
```

Writes `param_curve_<PROFILE>_<METRIC>.{png,pdf}` — four figures, each a two-panel plot
(encoder parameters, then full-system parameters, both logarithmic) of the two backbone
curves with their seed SD and bootstrap intervals. The plot model is the page's
(`make_results_html.series`), so a figure can only show what the page shows. It refuses a
non-curve product, an `--outdir` that holds a canonical input, and — before it draws
anything — every predicted figure file that names an input by path, hardlink or symlink.

## 4. Bind

```bash
mkdir -p ckpt/exp05/reports
python $A/bind_provenance.py --runs ckpt/exp05/eval/* --attempt $ATTEMPTS --results $JSONS \
    --rendered $W/param_efficiency_results.md $W/param_efficiency_01_results.html \
    --figures $W/param_efficiency_figures/* --out ckpt/exp05/reports
```

`--results` must name the five canonical JSONs explicitly (a `*.json` glob would also
match the sidecars). The binder exclusively creates
`binding_report_<UTC timestamp>.json` in the existing directory, reads run directories
and never modifies them, and binds: the four certified attempts with every attempt their
ledgers list and the external logs those name; the M pair's historical exp_01 checkpoints
at the profile's pinned digests; the sixty-six evaluation runs as the registered
arm/K/seed set with their tier metadata, their approved evaluator and writer closures and
their training linkage; the five products with exact per-product input coverage and a
producer closure recomputed from the working tree; the rendered documents (each must cite
every canonical digest and every bound attempt completion digest); the figures; the
approval blob; git HEAD.

**Run it from this checkout.** Every artefact records the absolute paths of the checkout
the producers ran in, and the binder compares them: the approval identity, the producer's
own source files — whose closure it recomputes from the working tree — and each product's
declared inputs. Binding the same record from a git worktree therefore fails on `result
approval identity or pins` even though every digest matches, and pointing `--approved` at
the main checkout does not rescue it: the producer's dependencies are still rooted in the
worktree. Run producers, generators and binder in one checkout.

**Runtime.** `tools/provenance.revalidate` rehashes every input a manifest declares, so a
real bind reads about 1 GiB of AcousticRooms per evaluation run and the 12.2 GiB training
inventory per attempt — about 123 GiB in total, measured at ≈ 8 minutes on the local NVMe
(sha256-bound, one core), and `check_record.py` repeats it. It is idempotent, safe to
repeat, and writes nothing but the new report.

## 5. Check

```bash
python $A/check_record.py ckpt/exp05/reports
```

Recomputes the latest timestamped report from its own recorded inputs, at its own
recorded git HEAD, and refuses any drift. Retain earlier reports unchanged; the checker
verifies only the latest.

## Order and re-runs

Producers → generators → figures → bind → check, and **regenerate the documents after any
producer change**: the binder requires every rendered document to cite every canonical
digest, so a stale page fails the bind rather than being published. Run `check_record.py`
once after binding and again immediately before publication.

## Tests and static checks

```bash
bash $A/static_checks.sh        # py_compile, bash -n, the record tests, both collection orders
python -m pytest tests/test_exp05_record_tools.py -q -p no:cacheprovider
```

The record assets of exp_03, exp_04, exp_05 and exp_07 share file names
(`bind_provenance.py`, `check_record.py`, `make_results_md.py`, `static_checks.sh`), so
every asset is imported by path under an experiment-scoped `sys.modules` key
(`tools/exp05_record.py::load_asset`); `static_checks.sh` runs the four record test files
in both collection orders to keep that property honest. A restricted installation may
need `NUMBA_CACHE_DIR=/tmp/xrir_exp05_numba` and `MPLCONFIGDIR=/tmp/xrir_exp05_mpl`.
