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
output path that overlaps ANY input -- a canonical JSON, a sidecar, an approval blob or
the exp_04 binding report -- in any spelling.

`tools/exp07_pairs.py --reconverge` is the registered publication retry: each flagged cell
is recomputed once at the profile's `reconverge_n_boot` (40 000), the first attempt's
count, diagnostic and both intervals are kept inside the cell under `first_attempt` with a
`reconverge_attempts` counter, and a statistic that fails again stays flagged so the
pairing remains non-final and no generator will publish it.

`--evidence gpu_parity=PATH --evidence calibration=PATH` (both required) bind the deferred
GPU parity log and the released-checkpoint calibration. The calibration JSON must state
`role: released_seen`, `protocol: seen`, `num_shot: 8`, `passed: true` and, for each of
EDT (s), C50 (dB) and T60 (%), `mean`, `sd`, `historical` and `passed`; the binder
recomputes the pre-registered rule
`|mean - historical| <= 3 * sd + 0.02 * |historical|` against the plan's historical values
(0.0389 / 1.029 / 7.27) and refuses anything outside it.

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
(at most one retry per arm) AND every other attempt each ledger lists -- aborted full runs
with their `abort.json` and the probe attempts with their receipts -- so a later change to
any of those files changes the report; the released checkpoint at its pinned digest; the
forty seen runs with their `seen_split` bindings and training linkage; the seen alignment
audit (protocol `seen`, passed, with its cohort digest and arguments); the required
`--evidence` artefacts (`gpu_parity`, `calibration`); the four producer outputs, revalidated
through the same canonical checks the generators apply, with their sidecars and companions;
the rendered documents (each must cite every canonical digest); the exp_04 table, sidecar
and binding report; the approval blob and git HEAD.
`check_record.py REPORT_DIRECTORY` recomputes the latest report under its own recorded HEAD;
an invalid latest report fails with no fallback.

Order of operations:

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

Run `bash static_checks.sh` from this directory or by its full path: it compiles the assets
and the exp_07 producers, checks whitespace, and runs the exp_07 suite plus the record-tool
tests in both collection orders (exp_03, exp_04 and exp_07 ship files of the same name, so
each experiment loads its own by path under its own `sys.modules` key).
