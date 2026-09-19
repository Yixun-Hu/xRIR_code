The exp_07 record generators consume canonical producer JSON and its adjacent
`.provenance.json` only. They never open a run directory, never recompute a statistic and
never read an evaluation array. Use the xRIR interpreter with the repository root on
`PYTHONPATH`, from the repository root.

All four generators take the same inputs:

```
--table    ckpt/exp07/results/TABLE_SEEN_V1.json
--pairs    ckpt/exp07/results/PAIRS_SEEN_V1_{seen_cyl,seen_aug,released_seen}.json
--unseen-table   ckpt/yaw_aug/results/TABLE_V1.json
--unseen-binding ckpt/yaw_aug/binding_report_<UTC timestamp>.json
--out      <document>
```

They refuse: a JSON whose sidecar does not bind its bytes, its profile digest, its inputs,
its approval identity and this producer's approved closure; an exploratory JSON or one
carrying admission deviations; a seen table that is not all eight rows with all five
metrics; a pairing set that is not the three registered pairings, final, descriptive and
thirty cells; a CELL whose statistics have not both converged, that still asks for a larger
`n_boot`, that claims to drive a decision or carries a verdict, or whose pairing, grid,
seed labels, quantiles, statistic set or cohort differ from its parent's; an unseen table
whose digest is not the one exp_04's binding report recorded; a duplicate input; and an
output path that overlaps ANY input -- a canonical JSON, a sidecar, an approval blob, the
exp_04 binding report, or any file those products declare they were built from (a run's
outputs, an arm's `args.json` and manifests, the data inventory) -- in any spelling,
including a hardlink to one (an existing destination is compared with every protected
input by device and inode, not only by resolved path, so an input archived between
admission and publication is refused under both of its names).

`tools/exp07_pairs.py --reconverge` is the registered publication retry: each flagged cell
is recomputed once at the profile's `reconverge_n_boot` (40 000), the first attempt's
count, diagnostic and both intervals are kept inside the cell under `first_attempt` with a
`reconverge_attempts` counter, and a statistic that fails again stays flagged so the
pairing remains non-final and no generator will publish it.

`--evidence gpu_parity=PATH --evidence calibration=PATH` (both required; equivalently
`--evidence gpu_parity=PATH calibration=PATH`, since repeated flags accumulate) bind the
GPU parity receipt and the released-checkpoint calibration. **Both are producer receipts,
not logs**, and the binder refuses a bare log.

`tools/exp07_calibration.py --reviewed-commit SHA --runs DIR ... --json OUT` is the
pre-registered gate of plan section 2. It admits the five released-checkpoint K = 8 seen
evaluations through exactly the checks the table producer applies to that role -- including
the cohort rules `build_table` applies afterwards, `tools.exp07_table.metric_names` (all
five registered metrics, `loss` and `log_mse` included, in every seed's cell) and
`tools.exp07_table.seed_finite_means` over EVERY one of those five columns (a nonempty
finite cohort per seed, per-seed finite counts within the registered tolerance and a
finite seed mean) BEFORE the three historical metrics are selected, so this gate cannot
approve a cohort the publication table will refuse --
derives the five-seed means and sample SDs (ddof 1) of EDT (s), C50 (dB) and T60 (%) from
the runs themselves, applies the registered rule `|mean - historical| <= 3 * sd + 0.02 * |historical|`
against the registered historical values (0.0389 / 1.029 / 7.27, `tools.exp07_profiles.CALIBRATION`)
and writes a canonical JSON plus sidecar binding the run digests, manifest hashes, the
released checkpoint digest, the reviewed commit and its own producer closure. It runs
BEFORE the trainings finish, so the approval file is still all-null: the evaluator and
writer closure pins it needs are computed from the reviewed code at `--reviewed-commit`
(the identity the launcher itself checked at spawn), and the released row needs no training
pin. The binder re-derives the means and SDs from the bound runs and refuses a summary that
does not equal them, so a fabricated finite summary is not evidence. It also binds the
sidecar at its own digest and revalidates its producer-closure records against the
recomputed identity, so a rewritten sidecar changes the report. A failing calibration
still writes its JSON (exit 1) for the investigation the plan requires; the binder refuses it.

`tools/exp07_parity.py --log PATH --reviewed-commit SHA --gpu N` runs the nine registered
deferred GPU node ids under one pytest with a JUnit XML and writes `<log>.receipt.json`
naming each case with its outcome, the pytest exit, the reviewed commit (which must be
HEAD), the CUDA device and the digests of the log and the XML. It owns that evidence: the
log and the XML are created with `O_CREAT|O_EXCL` **before** pytest is spawned (an existing
file at either path is refused and neither is created), the child runs with `PYTEST_ADDOPTS`
cleared and `-o addopts=` so no ambient option can turn it into a help screen or another
set of cases, and the XML must be non-empty and no older than that reservation. It refuses
a run that is missing a case, skipped one, failed one, collected an unregistered one,
exited nonzero, or left an unparseable XML. The binder requires the receipt to cover
exactly those nine ids, all passed, exit 0, a clean checkout, `git_head == reviewed_commit`
and an ancestor of the binding HEAD; it re-hashes the log and the XML **and parses the XML
itself** through the producer's own functions, so a receipt that does not say what pytest's
XML says is refused.

`make_results_md.py` writes `seen_protocol_results.md`, `make_results_html.py` writes
`seen_protocol_01_results.html`, and `make_latex.py` writes `table_seen_unseen.tex`. All
three are byte-stable. EDT is milliseconds in the canonical JSON: the Markdown shows both
seconds and milliseconds, the LaTeX table shows seconds with the SDs in milliseconds in its
footnote. The bold cell per LaTeX column and K is the smallest mean among the three recipe
arms, computed from the canonical means; a tie bolds nothing, and the released checkpoint is
an external reference row that is never part of that comparison. No verdict, significance
marker or superiority claim appears in any rendered cell.

`bind_provenance.py --runs DIR ... --attempt FINAL ... --audit JSON --evidence NAME=PATH
[...] --results JSON ... --rendered FILE ... --unseen-table JSON --unseen-binding JSON
[--approved JSON] --out REPORT_DIRECTORY` validates the completion links and exclusively
creates `binding_report_<UTC timestamp>.json` in that existing directory. It binds the
three certified attempts with their inventory sidecar, probe receipt and hours ledger
(at most one retry per arm) AND every other attempt each ledger lists -- each must end
certified (a completion naming its log, which must be on disk at the recorded digest) or
aborted (an `abort.json` with a reason and the logs it references, which the launcher keeps
OUTSIDE the attempt directory), and every probe attempt that COMPLETED must correspond to
exactly one receipt that records its manifest and completion digests -- so a later change
to any of those files, logs included, changes the report. A probe whose child failed has no
receipt to have (`run_probe` writes it only after `execute_attempt` returns) and is
published through its abort evidence instead. The one abort without a log on disk is the
documented setup failure, required in the exact shape that produces it: reason
`setup_failed`, the `_ABORTED_setup_failed` directory the abort itself named, a null
renamed log beside a named original, no `execution.json` and neither log on disk. That form
is bound with the log recorded absent; any other logless abort is refused. It also binds the released checkpoint at its pinned digest; the
forty seen runs with their `seen_split` bindings, training linkage and agreeing split
identity in both output metas; the seen alignment audit (protocol `seen`, passed, with its
cohort digest and arguments); the required `--evidence` receipts (`gpu_parity`,
`calibration`); the four producer outputs, revalidated through the same canonical checks the
generators apply, each covering EXACTLY its registered run set (the table all forty runs, a
pairing its two arms' twenty) in both its `run_flags` and its `contracts`, each contract
carrying the role, K, evaluation seed and evaluation-manifest digest of the run it is filed
under, and declaring EXACTLY its own dependencies: every artefact its own runs declare (the
reference manifest and the data inventory included), everything the contract of each trained
arm it used reads -- that arm's `args.json`, `train_manifest.json`, `completion.json`, the
`train_inventory.json` sidecar, the `cumulative_hours.json` ledger and the probe receipt --
the approval blob and its own producer closure, and nothing else. An omission, an unknown
input, a differing input, a dependency on a run the product did not use and a file of an arm
it did use that no contract reads (an unused epoch checkpoint, `history.jsonl`, an aborted
attempt's records -- all bound in this report's attempt history, none of them an input) are
each refused; the rendered documents (each must cite every canonical digest); the exp_04 table,
sidecar and binding report; the approval blob and git HEAD.
`check_record.py REPORT_DIRECTORY` recomputes the latest report under its own recorded HEAD;
an invalid latest report fails with no fallback.

Order of operations (the Planner launch sequence; only the steps this directory owns):

0. After merging and before any launch, produce the parity receipt and then the
   calibration, on a free GPU from the merged checkout:

   ```
   PARITY="$REC/gpu_parity_${MERGE}.log"
   "$PY" tools/exp07_parity.py --log "$PARITY" --reviewed-commit "$MERGE" --gpu "$GPU"
   PARITY_RECEIPT="${PARITY%.log}.receipt.json"

   for seed in 42 43 44 45 46; do
     eval07 released_seen simple checkpoints/xRIR_seen.pth 8 "$seed"
   done
   "$PY" tools/exp07_calibration.py --reviewed-commit "$MERGE" \
       --runs ckpt/exp07/eval/released_seen_k8_seed{42..46}_k0 \
       --json ckpt/exp07/results/CALIBRATION_SEEN_V1.json
   ```

   Both artefacts are bound later as
   `--evidence gpu_parity=$PARITY_RECEIPT calibration=ckpt/exp07/results/CALIBRATION_SEEN_V1.json`.
   If the calibration fails, follow the plan's investigation procedure before admitting
   any seen-arm evaluation.

1. Finish and review the code, then commit it. Compute the producer closures from that
   committed code, fill every closure and checkpoint pin together, and commit the approval
   file in a second reviewed commit before running any producer. The schema permits only
   all-null or all-filled pins.
2. Run `tools/exp07_table.py` once and `tools/exp07_pairs.py` once per registered pairing.
   Keep each JSON, its sidecar and its companion (the living Markdown table, the pairing
   summary) together.
3. Generate the Markdown, HTML and LaTeX from those canonical JSONs. Binding requires
   non-empty `--rendered` documents that cite every canonical digest, so the documents
   exist before the first binding: the order is **producers -> generators -> bind ->
   check**, never bind before the documents are written.
4. Run `bind_provenance.py` over the complete run set, every attempt of each arm, all four
   canonical JSONs, the rendered documents and the parity/calibration/audit evidence, then
   run `check_record.py` and keep its exit 0 with the report.

## Archiving evidence after the record is bound

A certified attempt directory may be moved off this disk -- to the NAS, say -- provided a
**directory symlink** is left at the path the record published. Identity here is the
logical path the approval, the manifests, the completions, the products' sidecars and the
binding report all name, never the resolved one, so `check_record.py` and a fresh bind
reproduce the report byte for byte after the move and a regenerated document names the
published path. Digests are unaffected: the bytes are still read, and hashed, through
that name. An attempt's published name is the one its launcher wrote into
`train_manifest.json` as `attempt_path`, so binding `ckpt/exp07/<role>/final` and binding
`ckpt/exp07/<role>/attempt_<ts>full` produce the same record; the approval pins each
checkpoint through `final`, and the binder matches that pin to the attempt's own output
by inode. `final` must still be the symlink the launcher promoted to that attempt: an
inode identifies the checkpoint file, not the directory it was published in, so a
hardlink of it under a repointed `final` -- or under an ordinary directory of that name
-- is refused. Archiving does not disturb this; leave `final` naming the attempt's
published name, which is the directory symlink the move leaves behind.

The same holds for any ancestor: the arm directory (which keeps its `cumulative_hours.json`,
its `_probe_*.json` receipts, its other attempts and its `final` symlink -- CIFS holds no
symlinks, so leave that directory where it is if the archive cannot), an evaluation run
directory, `ckpt/exp07/results` with the products, or `ckpt/exp07` entire. exp_04's table
and binding report live in that experiment's own directory and are not moved by this.
What must keep working is reading through the published name, so leave the symlink in
place and keep each product beside its companion and its sidecar. Archive a directory,
never its individual files: an entry inside an attempt or run directory that resolves
outside it is still refused, and so is any change to the bytes.

Archive only once the record is bound, as that order says. The producers publish through
`tools.exp07_record.write_outputs`, which is inside the calibration receipt's producer
closure and so keeps its reviewed bytes: it compares its `--json` and `--md` destinations
with the admitted inputs by name, so an input archived between a producer's admission and
its publication would be protected under one of its two names only. The generators here
compare by device and inode as well, and the documented order -- producers, generators,
bind, check, and only then archive -- never reaches that gap.

`record_paths.py` holds the two path helpers this record needs -- `logical`, the absolute
name a file is published under, and `identical`, the device-and-inode question "are these
two names one file?" -- and the generators and the binder load it by path. They live here,
not under `tools/`, because every module the producers import is inside a producer closure
that has already been published: `tools.exp07_calibration` imports `tools.exp07_table`,
which imports `tools.exp07_record`, and the calibration receipt in `ckpt/exp07/results`
records that closure's digest. Changing a byte of any of them invalidates that evidence.
The record's assets are in no closure, so they are where this record may still change.

Run `bash static_checks.sh` from this directory or by its full path: it compiles the assets
and the exp_07 producers, checks whitespace, and runs the exp_07 suite plus the record-tool
tests in both collection orders (exp_03, exp_04, exp_05 and exp_07 ship files of the same
name, so each experiment loads its own by path under its own `sys.modules` key).
