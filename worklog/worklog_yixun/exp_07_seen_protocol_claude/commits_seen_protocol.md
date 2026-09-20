# exp_07 seen_protocol — commits


## Complete commit index through record closure (retrospective; cutoff = main `b50202c` on 2026-09-20; `git log f492613..HEAD` over the exp_07 tools/tests/record AND the shared modules the branch touched before amendment A1 and then restored — `train_xRIR_backbone.py`, `tools/provenance.py`, `tools/yaw_aug.py`, `tools/exp04_{launcher,eval,eval_launch}.py`, `tools/exp05_{probe,gates}.py`, `tools/results_table.py`; 192 entries)

```
b50202c 2026-09-20 exp_07: analysis, params set-up, commit index, command record, retained scripts
90642fc 2026-09-20 exp_07: end-game outputs at 0e47341 — seen table, three pairings, md/html/LaTeX seen+unseen table, bound and checked
0e47341 2026-09-20 exp_07: fill reviewed closure and certified checkpoint pins
f5e1cf6 2026-09-20 exp_07: seen_aug evaluations done; all 30 trained-arm evaluations complete
ac4d492 2026-09-20 exp_07: record-review prompt (staged for after the end-game)
e9a0ecf 2026-09-20 exp_07: seen_aug certified; evaluations running
483b37c 2026-09-19 exp_07: seen_cyl seen-split evaluations done (10/10); rows logged
694f4c4 2026-09-19 exp_07 worklog: foreign file relocated; waiters and end-game re-armed
341acdc 2026-09-19 exp_07 worklog: foreign untracked file blocked nine seen_cyl evaluations; queue/chain scripts fixed
b825478 2026-09-19 exp_07: seen_cyl certified; evaluations running
3b7d829 2026-09-19 exp_07 worklog: exp_09 GPU slot answer
c934393 2026-09-19 exp_07 worklog: end-game script armed
27868d4 2026-09-19 exp_07 README: repair the round-12 nit paragraph
2849935 2026-09-19 exp_07: README nit (round-12 review) and merge bookkeeping
a45d277 2026-09-19 Merge exp_07 round 11+12 (relocation-safe record binder, Codex-approved round 12)
22b87a9 2026-09-19 exp_07: Codex round-12 close review (approve; approved for merge) — review and notebook entry written by the reviewer
d7d94e4 2026-09-19 exp_07: round 12 delivered; Codex close-review prompt
68eb435 2026-09-18 exp_07 record: check_record refuses the repointed `final` as well
4ff000c 2026-09-18 exp_07 record: the `final` rule, written down where the Planner archives
d96a516 2026-09-18 exp_07 record: no document is written over evidence that moved
fdac647 2026-09-18 exp_07 record: `final` must be the promotion of the attempt being bound
087dbc5 2026-09-18 exp_07 record: the published calibration keeps its producer's bytes
068d84a 2026-09-18 exp07: merge main (round-11 close review, round-12 prompt) into exp07-window
6449754 2026-09-18 exp_07: Codex round-11 close review (two blockers); round-12 prompt
9e98621 2026-09-18 exp_07: round 11 delivered (relocation-safe binder); Codex close-review prompt
a1c5b9b 2026-09-18 exp07: merge main (exp_05 round 9, exp_06 records, NAS relocation of attempt directories) into exp07-window
5258b31 2026-09-18 exp_07: seen_aug full training launched on GPU 1 (probe 27.3 h); relocation outcomes (exp_05 check passes, exp_04 attempt restored)
b14dc3b 2026-09-18 records: attempt directories relocated to the NAS (swap bug fixed), Codex review notes preserved, seen_aug relaunched
9568591 2026-09-18 exp_07 worklog: GPU 1 released; seen_aug blocked by the disk gate; exp_05 attempts moving to the NAS
dbf42fd 2026-09-18 exp_07 record: the migration the relocation-safe binding permits, written down
6d77a02 2026-09-18 exp_07 record: an archived ancestor may carry the products too
4ae57f6 2026-09-18 exp_07 record: the binding survives a relocated attempt directory
64f3ad7 2026-09-18 exp_07: round-11 prompt (relocation-safe binder, pre-bind); worklog timestamp fixed
d0b9f00 2026-09-18 exp_05: round 8 delivered (relocation-safe bindings); Codex close review launched. exp_07: binder relocation defect noted for a pre-bind round
2e43887 2026-09-18 exp_07 worklog: seen_aug auto-launch paused until exp_06's release ping
d45469c 2026-09-18 exp_07: seen_cyl full training launched on GPU 0 (probe 31.8 h, 30.8 GiB)
2642e4e 2026-09-18 exp_05: record tooling delivered; Codex round-6 review launched. exp_07: known-red runtime-approval test until the pin fill
c6340d9 2026-09-18 exp_07 worklog: exp_06 queues launched; seen_cyl chain re-armed (not before 14:45)
b0bae26 2026-09-18 exp_07: seen_simple seen-split evaluations done (10/10); first seen row
1dc330c 2026-09-18 exp_07 worklog: checkpoint move narrowed to the exp_04 tree until exp_06 summarises; 67 GB free
edf4de4 2026-09-18 exp_07: seen_simple certified; seen_cyl disk-gate abort and cleanup; checkpoint move to NAS; arm chains armed
5de272c 2026-09-17 exp_07 worklog: Codex reasoning-effort drift noted; all exp_07 reviews verified xhigh
17e7688 2026-09-17 exp_07 worklog: second GPU-0 co-tenant 19:22–21:14 (exp_06 test suite) noted for the timing record
d505756 2026-09-17 exp_07 worklog: tonight's GPU-1 arbitration (cylindrical-dinov3 slot to 23:10, exp_06 queue from 23:10)
c013f90 2026-09-17 exp_07 worklog: exp_06 finalised 19:20; GPU 1 to the rot45 sweep until 22:00
c2a5edf 2026-09-17 exp_07 worklog: co-tenant slowdown grew to 35 % on epoch 6; projected +2.5 h
1933c6b 2026-09-17 exp_07 worklog: GPU-0 co-tenant runtime estimate (≈ 9.5 h) and ownership status
47b6229 2026-09-17 exp_07 worklog: GPU-0 co-tenant (cylindrical-dinov3 measurement) slowed seen_simple epoch 5 by 13 %
bb0d1df 2026-09-17 exp_07: seen_simple full training launched on GPU 0 (probe 27.0 h, 30.8 GiB)
3e2c28b 2026-09-16 exp_05/exp_07: GPU-0 chain launched (L_simple certification → L-tier evaluations → seen arms 1-2)
6c8bd8b 2026-09-16 exp_07 worklog: exp_06 epoch-1 passed; agreed two-GPU queue through Sep 19
12ec46b 2026-09-16 exp_07 worklog: GPU-1 co-tenant incident (rot45 sweep, 2 min) and the pencilled GPU queue
5c53ee1 2026-09-16 exp_07 worklog: exp_06 pretraining launched on GPU 1 (6dc0b8e, 12:14); GPU-0 plan
6dc0b8e 2026-09-16 exp_07 worklog: GPU 1 handed to exp_06 at 12:05 after the FLAC job stopped at step 500
bd30c83 2026-09-16 exp_07 worklog: co-tenancy measurement (infeasible, 1.95 vs 1.01 s/it); FLAC exp_23 stop at the step-500 checkpoint by its owner; exp_06 launches on the clear card
5427440 2026-09-16 exp_07 worklog: FLAC exp_23 stop/resume facts; fallback prepared (dry run only); timestamps corrected
7796d6f 2026-09-16 exp_07 worklog: Yixun's directive — exp_06 highest priority; co-tenant measurement first, FLAC release as fallback
515136f 2026-09-16 exp_07 worklog: GPU 1 blocked by FLAC exp_23 chain until ~21:00; ordering decision deferred to that time
3979e08 2026-09-16 exp_07: pre-training gate passed (parity 9/9, audit, released-checkpoint calibration EDT 0.0384 / C50 1.024 / T60 7.27)
4cd1d30 2026-09-16 exp_05: L_cylindrical certified; exp_07: pre-training GPU phase launched on GPU 1
dc65b26 2026-09-16 exp_07 worklog: exp_06 round-3 merge noted (21a2bf6/d854102)
1a1f6ba 2026-09-16 exp_07 worklog: round 10 merged (ee6576f); code phase closed
ee6576f 2026-09-16 Merge exp_07 round 10 (test-only: hermetic parity fixture, Codex-approved)
74aee47 2026-09-16 exp_07: Codex round-10 close review (approve)
146eaa5 2026-09-16 exp_07: round 10 (test-only) delivered; notebook timestamps corrected
cf46275 2026-09-16 exp_07 round 10: the parity fixture pins its own checkout state
e6b70e0 2026-09-16 exp_07: merge bookkeeping (c914311), two environment-dependent parity tests noted, round-10 test-only prompt
c914311 2026-09-16 Merge exp_07 seen protocol (exp07-window f39a441, Codex-approved round 9)
7de6775 2026-09-16 exp_07: Codex round-9 close review — branch APPROVED for merge and GPU work
464b031 2026-09-16 exp_07: round 9 delivered (two blockers); Codex close-review prompt
f39a441 2026-09-15 exp_07 round 9: the exactness of a product's dependency map, stated directly
d95b04f 2026-09-15 exp_07 round 9: each product declares exactly its own dependencies
80f162f 2026-09-15 exp_07 round 9: the calibration gate validates every registered metric
968e46c 2026-09-15 exp_07: Codex round-8 close review (two blockers left); round-9 prompt
90bf4ec 2026-09-15 exp_07: round 8 delivered (blockers 1-4, should-fix 5); Codex close-review prompt
fd6e521 2026-09-15 exp_07 round 8: a certified attempt's log is bound at a recorded digest
da37bb4 2026-09-15 exp_07 round 8: reflow two overlong lines the round-8 edits left
a08c4cb 2026-09-15 exp_07 round 8: the assets README states what round 8 changed
9669054 2026-09-15 exp_07 round 8: the calibration sidecar is bound and its closure revalidated
53d6779 2026-09-15 exp_07 round 8: exact contract coverage and each product's own dependencies
dcc791e 2026-09-15 exp_07 round 8: the calibration gate shares the table's cohort rules
7951b9b 2026-09-15 exp_07 round 8: every attempt's terminal state, and only its real evidence
217ee14 2026-09-15 exp_07 round 8: the binder reads the parity XML, not only the receipt
775b9c9 2026-09-15 exp_07 round 8: the parity producer reserves and owns its own evidence
b86bd42 2026-09-15 exp_07: Codex round-7 close review (request changes, blockers 1-4); round-8 prompt
f35196c 2026-09-15 exp_07: round 7 delivered (blockers 1-4, should-fix 5-9); Codex close-review prompt
d58c2fd 2026-09-15 exp_07 round 7: restate what the binder binds, and drop a dead constant
112db91 2026-09-15 exp_07 round 7: keep the dependency maps out of the binding report
a423358 2026-09-15 exp_07 round 7: document the two new producers and the tightened binding
79c2dfe 2026-09-15 exp_07 round 7: the GPU parity cases run the relocated entry point
88fb3c2 2026-09-15 exp_07 round 7: the binder validates the calibration and the parity receipt
576d593 2026-09-15 exp_07 round 7: every ledger attempt's terminal state and its external logs
c115575 2026-09-15 exp_07 worklog: Sep 16 merge windows agreed with exp_06
5d74f8f 2026-09-15 exp_07 round 7: exact product coverage and no unvalidated dependency
b2eaabe 2026-09-15 exp_07 round 7: a producer for the released-checkpoint calibration
1ed3417 2026-09-15 exp_07 round 7: a GPU parity producer that receipts what actually passed
9b52eff 2026-09-15 exp_07 round 7: hardlink-proof outputs and accumulating --evidence
96edcfe 2026-09-15 exp_07 round 7: the launcher refuses a child that evaluated another split
85b4f9d 2026-09-15 exp_07 round 7: drop the redundant --entry and keep tiny SDs visible
348d47b 2026-09-15 exp_07: Codex round-6 + integrative review (request changes; merge blockers resolved); round-7 prompt
6f14b3b 2026-09-15 exp_07: round 6 delivered (relocation to exp07-only modules); Codex round-6 + integrative review prompt
9c3b458 2026-09-15 exp_07 round 6 half 1: cover the finalize route through the shared recovery
7194dd1 2026-09-15 Merge main into exp07-window (exp_06 launch-gate merge and bookkeeping)
94c4ca3 2026-09-15 exp_07 round 6 half 2: drop an unused import from tools/exp07_record.py
fc80c49 2026-09-15 exp_07 worklog: note exp_06's launch-gate merge (62a4665/e590b43)
d94c739 2026-09-15 exp_07 round 6 half 2: tidy tools/exp07_record.py's imports
16a1981 2026-09-15 exp_07 round 6 half 2: document the new refusals, the retry and the evidence interface
7ffea95 2026-09-15 exp_07 round 6 half 2: the binder validates the products and binds every attempt
16e8e10 2026-09-15 exp_07 round 6 half 2: admission hashes the checkpoint the run actually used
4abf498 2026-09-15 exp_07 round 6 half 2: the registered 40 000-resample reconvergence re-run
4c6e842 2026-09-15 exp_07 round 6 half 2: per-cell admission, output protection, labelled SD footnotes
5e0409a 2026-09-15 exp_07 round 6 half 1: the completion binds the manifest that declares the split
e93edff 2026-09-15 exp_07 round 6 half 1: exp_07's own publication writer; results_table restored
ea3ce0b 2026-09-15 exp_07 round 6 half 1: the split-bound evaluator and its own launcher
5db051f 2026-09-15 exp_07 round 6 half 1: the launcher's run modes, guard, recovery and the probe gate
2074100 2026-09-15 exp_07 round 6 half 1: the seen probe, gates and the launcher's arm decisions
18dacdc 2026-09-15 exp_07 round 6 half 1: restore tools/exp05_{probe,gates}.py to main's bytes
f217d43 2026-09-15 exp_07 round 6 half 1: tools/exp07_provenance.py, protocol-isolated inventories
25b5402 2026-09-15 exp_07 round 6 half 1: restore tools/provenance.py to main's bytes
1dcc5e5 2026-09-15 exp_07 round 6 half 1: tools/exp07_audit.py, the seen alignment audit
9d13581 2026-09-15 exp_07 round 6 half 1: tools/exp07_train.py, the seen-protocol entry point
0a2e518 2026-09-15 exp_07 round 6 half 1: restore the trainer and yaw_aug to main's bytes
453baf9 2026-09-15 Merge main into exp07-window (exp_06 merge + exp_07 record commits)
8cc4fd7 2026-09-15 exp_07 worklog: restore two code spans
3975afe 2026-09-15 exp_07: integrative review (not approved) → amendment A1 (no shared-module edits); round-6 relocation prompt
4d7a6cb 2026-09-15 exp_07: Table-1 reference screenshot moved into the record; exp_06 merge noted
ec7e649 2026-09-15 exp_07: round 5 delivered; Codex round-5 + integrative review prompt
2a27144 2026-09-15 exp_07 round 5 cycle 0: tie the new training literals to the launcher's
cd343b8 2026-09-15 exp_07 round 5 cycle 3: the record assets' static checks and order of operations
671801b 2026-09-15 exp_07 round 5 cycle 3: the record binder and its checker
4f2a16c 2026-09-15 exp_07 round 5 cycle 3: the fixture's sources are a real reviewed commit
8c7e247 2026-09-15 exp_07 round 5 cycle 2: the combined seen/unseen LaTeX table
335e2b7 2026-09-15 exp_07 round 5 cycle 1: the offline seen-protocol HTML page
515eb0d 2026-09-15 exp_07 round 5 cycle 1: the seen, paired and combined Markdown tables
61e1399 2026-09-15 exp_07 round 5 cycle 1: the record generator's admission of canonical JSON
de29cf7 2026-09-15 exp_07 round 5 cycle 0: pin the built identities and the approval lifecycle
c57b9f2 2026-09-15 exp_07 round 5 cycle 0: the exploratory table must say so in the document
53aca52 2026-09-15 exp_07 round 5 cycle 0: all five metrics, and a recursively frozen approval
abcdc9e 2026-09-15 exp_07 round 5 cycle 0: the manifest must describe the run the trainer made
36c37c5 2026-09-15 exp_07 round 5 cycle 0: the timing limits must be the bound receipt's
03b327f 2026-09-15 exp_07 round 5 cycle 0: the training inventory, full-run status and ledger
35e8599 2026-09-15 exp_07 round 5 cycle 0: the fixture's real training evidence
8fb5e07 2026-09-15 exp_07 round 5 cycle 0: recompute the bound training closure digests
32ee878 2026-09-15 exp_07: Codex round-4 review (request changes) folded into round-5 cycle 0
77dd586 2026-09-15 exp_07: seen reference manifests built (10 + index)
daba33d 2026-09-15 exp_07: round 4 delivered; Codex round-4 review prompt
7b3ad6c 2026-09-15 exp_07 round 4 cycle 2: the bound args must state the M capacity tier
5362fa3 2026-09-15 exp_07 round 4 cycle 2: import the Markdown stamp marker where it is used
e9ec070 2026-09-15 exp_07 round 4 cycle 3: the summary, the CLI and publication
831810c 2026-09-15 exp_07 round 4 cycle 3: pairing-level tests over the synthetic run layout
bca1c52 2026-09-15 exp_07 round 4 cycle 3: pairing routing and the cell grid
eff1b2a 2026-09-15 exp_07 round 4 cycle 3: the descriptive paired cell
f87dfbc 2026-09-15 exp_07 round 4 cycle 2: TABLE_V1 aggregation, canonical JSON and the seen table
e7a10a3 2026-09-15 exp_07 round 4 cycle 2: eighteen named admission refusals
9a52940 2026-09-15 exp_07 round 4 cycle 2: role-aware admission over the exp_04 collector
86ec2c9 2026-09-15 exp_07 round 4 cycle 2: the per-run exp_07 contract
8772121 2026-09-15 exp_07 round 4 cycle 2: a synthetic four-role, two-K, five-seed run layout
298859e 2026-09-15 exp_07 round 4 cycle 1: the exp_07 approval loader and its all-null template
83958af 2026-09-15 exp_07 round 4 cycle 1: tie the profile literals to the artefacts they describe
a38afbc 2026-09-15 exp_07 round 4 cycle 1: the frozen TABLE_SEEN_V1 and PAIRS_SEEN_V1 profiles
778b97e 2026-09-15 exp_07 round 4 cycle 0: an independent parity reference and the launcher path
dd0ebe7 2026-09-15 exp_07 round 4 cycle 0: capture the split identity before the manifests are built
5730b81 2026-09-15 exp_07 round 4 cycle 0: accept the canonical split under any path spelling
6b91092 2026-09-15 exp_07 round 4 cycle 0: bind the seen dataset module in the evaluator closure
d992fd5 2026-09-15 exp_07: Codex round-3 review (request changes) folded into round-4 cycle 0
49d4612 2026-09-15 exp_07: round 3 delivered; Codex round-3 review prompt
ab585d0 2026-09-15 exp_07 round 3 cycle 3: --entry exp07 --split seen in the eval launcher
2e7703f 2026-09-15 exp_07 round 3 cycle 2: deferred GPU parity for both splits
23bdddf 2026-09-15 exp_07 round 3 cycle 2: subset, wiring and default-factory coverage
71c5101 2026-09-15 exp_07 round 3 cycle 2: the split-bound evaluation entry point
28a876a 2026-09-15 exp_07 round 3 cycle 1: manifest grid, cwd and CLI coverage
f80a119 2026-09-15 exp_07 round 3 cycle 1: the seen reference-manifest builder
b5cf876 2026-09-15 exp_07 round 3 cycle 0: bind the split the inventory was selected under
9e8cdf7 2026-09-15 exp_07 round 3 cycle 0: one-retry and receipt-completion linkage
bf6027e 2026-09-15 exp_07: Codex round-2 review (request changes) folded into round-3 cycle 0
24085b2 2026-09-15 exp_07: seen inventory pre-built; command file
4f7593c 2026-09-15 exp_07: round 2 delivered; Codex round-2 review prompt
a035d7f 2026-09-15 exp_07 round 2: retire the round-1 seen_split allowlist TODO
e4832f2 2026-09-15 exp_07 round 2: --renew-ceiling help matches the seen arms it now accepts
9a0fbe0 2026-09-15 exp_07 round 2: define train_minimum after the constant it extends
8215702 2026-09-15 exp_07 round 2 cycle 3: seen refusals, launch pass-through and a smoke dry run
57c3a36 2026-09-15 exp_07 round 2 cycle 2: the seen fit-probe, its receipt and the timing gates
6d9578d 2026-09-15 exp_07 round 2 cycle 1c: seen bindings in recovery and the launcher CLI
68fb737 2026-09-15 exp_07 round 2 cycle 1b: per-arm seen controls and protocol-aware build_fields
36e385d 2026-09-15 exp_07 round 2 cycle 1a: seen arms, golden argv and the seen batch count
2fe4395 2026-09-15 exp_07 round 2 cycle 0: normalise legacy inventory cache hits
a498554 2026-09-15 exp_07: Codex round-1 review (approve with changes); round 2 prompt
cbeca19 2026-09-15 exp_07: round 1 delivered; Codex code review briefing and prompt
d7511fb 2026-09-15 exp_07 round 1 cycle 3: --protocol for the yaw-aug alignment audit
e122355 2026-09-15 exp_07 round 1 cycle 2: protocol-aware train_data_identity, seen_split_identity
6ed7bef 2026-09-15 exp_07 round 1 cycle 1: trainer --protocol {unseen,seen}
c65070d 2026-09-15 exp_07: round-1 Coder prompt
```
