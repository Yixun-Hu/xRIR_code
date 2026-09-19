- `2d2a3867b9360964fd5482aaf4fbe426e60c85a1` — exp_05 plan v3 approved and committed for handoff (record folder: query, plan v1–v3, two Codex plan reviews, notebook); also carries exp_04's round-1 review and prompts


## Complete commit index through record closure (retrospective; cutoff = main `ec75704` on 2026-09-18; `git log` over the exp_05 tools, tests, record folder AND the shared launcher `tools/exp04_launcher.py` / `tools/exp04_launch.sh`, whose exp_05 rounds live in the same history — e.g. `0e35e2c`)

```
ec75704 2026-09-18 exp_05: record regenerated at 9b25738 and rebound (deterministic PDFs, consistent revisions); record-review round-2 prompt
9b25738 2026-09-18 exp_05: round 9 merged; regeneration at a fixed revision starts
c5ac3de 2026-09-18 Merge exp_05 round 9 (deterministic figure PDFs, Codex-approved)
d3d5526 2026-09-18 exp_05: Codex round-9 close review (approve)
8437b6e 2026-09-18 exp_05: round 9 delivered (deterministic PDFs); Codex close-review prompt
2d599d6 2026-09-18 exp_05: make the record figures reproduce byte-for-byte in PDF too
5258b31 2026-09-18 exp_07: seen_aug full training launched on GPU 1 (probe 27.3 h); relocation outcomes (exp_05 check passes, exp_04 attempt restored)
b14dc3b 2026-09-18 records: attempt directories relocated to the NAS (swap bug fixed), Codex review notes preserved, seen_aug relaunched
ee5f152 2026-09-18 exp_05: record corrections after Codex record review 1 — analysis to registered wording, errata, SOP disclosure, retrospective command/params/commits records
1f3792d 2026-09-18 exp_05: Codex record review round 1 (not approved; eight corrections); round-9 PDF-determinism prompt
a327ccf 2026-09-18 exp_05: record-review prompt
a6e1f76 2026-09-18 exp_05: record bound and checked (73e90d4) — results tables, page, figures, analysis final, command record
8835f00 2026-09-18 exp_05: record tooling merged (73e90d4); record run launched
73e90d4 2026-09-18 Merge exp_05 record tooling (exp05-record 377cc2c, Codex-approved round 8)
47f3c40 2026-09-18 exp_05: Codex round-8 close review — record tooling APPROVED for merge and record use
d0b9f00 2026-09-18 exp_05: round 8 delivered (relocation-safe bindings); Codex close review launched. exp_07: binder relocation defect noted for a pre-bind round
377cc2c 2026-09-18 exp_05 record: an attempt's name is the one its launcher wrote, not a route to it
0d2bebd 2026-09-18 exp_05 record: an archived ancestor may carry the products too
078548f 2026-09-18 exp_05 record: the document keeps naming a relocated attempt as the report does
82e0f56 2026-09-18 exp_05 record: the binding survives a relocated attempt directory
aee12d1 2026-09-18 exp05: merge main into exp05-record (round 8)
9ff4b3e 2026-09-18 exp_05: Codex round-7 close review (one new blocker: relocation behind directory symlinks); round-8 prompt
8ce27e7 2026-09-18 exp_05 worklog: timestamp corrected
cea8cc5 2026-09-18 exp_05: round 7 delivered; Codex close-review prompt launched
555bff7 2026-09-18 Merge branch 'main' into exp05-record
f3884ec 2026-09-18 exp_05 record: the checkout rule has no worktree escape hatch
e3dda86 2026-09-18 exp_05 record: the page's marks keep their backbone colour
a5671a4 2026-09-18 exp_05 record: the registered profile is admission, not a binder-only check
cf0a22b 2026-09-18 exp_05 record: report the cohort each number was computed on
ede2260 2026-09-18 exp_05 record: admit every run's evaluator and writer closure
d3304b4 2026-09-18 exp_05 record: recompute the producer, and verify its sources
19bbca7 2026-09-18 exp_05 record: every generator protects every file it read
9538479 2026-09-18 exp_05: analysis draft from the canonical producer outputs
45b8d20 2026-09-18 Merge branch 'main' into exp05-record
ff89bbb 2026-09-18 exp_05: Codex round-6 record review (request changes, blockers 1-4); round-7 fix prompt
ecbcc5f 2026-09-18 exp_05: Codex round-6 review killed externally (143); relaunched
2642e4e 2026-09-18 exp_05: record tooling delivered; Codex round-6 review launched. exp_07: known-red runtime-approval test until the pin fill
840e345 2026-09-18 exp_05 record: drop an unused import and a redundant cache mutation
4125407 2026-09-18 exp_05 record: pin every product to the registered profile
abba366 2026-09-18 exp_05 record: name the checkout rule, and assert the yaw headings by row
9fb5eba 2026-09-18 exp_05 record: static checks and the operational README
244d5f3 2026-09-18 exp_05 record: the binder and its recomputation
0c22cc3 2026-09-18 exp_05 record: the offline page and the paper figures
9b956b1 2026-09-18 exp_05 record: the Markdown generator
966dfc8 2026-09-18 exp_05 record: canonical admission shared by the generators and the binder
ad5a556 2026-09-18 exp_05 record: a synthetic world for the record tooling
ddf0d1c 2026-09-18 exp_05: yaw block logged (no cylindrical robustness gain at any tier)
b602a99 2026-09-18 exp_05: confirmatory verdicts logged (H1 K=8 partial on EDT and C50; K=1 not supported; H2 Cyl-M vs Base-L superior, Cyl-S vs Base-M comparable)
0a673e7 2026-09-18 exp_05: round-6 record-tooling prompt; worktree exp05-record
d9e9759 2026-09-18 exp_05: pins filled; producers launched
61f4d1b 2026-09-18 exp_05: fill the approval pins (evaluator a2ffbb38, writer 15f9c984, training 5b2da250, launchers [0bff0d4f @f19b9b6, 9dadbef0 @862923e], producer 10484795; four certified epoch-12 checkpoints)
ddfcb66 2026-09-17 exp_05: S/M evaluation queue auto-starts on GPU 1 tonight in exp_06's gap
71ff90d 2026-09-17 exp_05: L-tier evaluations done (22/22); remaining 44 S/M evaluations queued for GPU 1 after exp_06
076cd40 2026-09-16 exp_05: L_simple certified; L-tier evaluation queue running on GPU 0
3e2c28b 2026-09-16 exp_05/exp_07: GPU-0 chain launched (L_simple certification → L-tier evaluations → seen arms 1-2)
4cd1d30 2026-09-16 exp_05: L_cylindrical certified; exp_07: pre-training GPU phase launched on GPU 1
9388c14 2026-09-15 exp_05: L_simple launched on GPU 0 at 862923e (attempt_20260915T060543)
58a8953 2026-09-14 bookkeeping: merge logged, branch review
9eacbb4 2026-09-14 Merge branch codex-window: exp_04 round 8 (record generators, binder, descriptive producer) and exp_05 rounds 5 (A9 launcher-digest list, receipt causes, template tests)
565a56a 2026-09-14 exp_05: decouple approval tests from committed pin state
27fbb74 2026-09-14 exp_05: round-5 review (approve with changes), fill list
2578ab7 2026-09-14 exp_05: name receipt failures and enforce binding filenames
1f28509 2026-09-14 exp_05: approve launcher sets and pin trainer closure
165fa36 2026-09-14 exp_05: round-4 review, amendment A9 (launcher-digest list), schedule
bb9fa91 2026-09-14 exp_05 round 4 delivered; exp_04 round-8 prompt (result generators)
f7b10b5 2026-09-14 exp_05: bind curve training evidence to checkpoint attempts
34a076a 2026-09-14 exp_05: bind probe evidence to arms and preflight launch budgets
6792167 2026-09-14 exp_05: S_cylindrical certified (12.00 h), L_cylindrical launched
2066830 2026-09-14 exp_05: S_simple certified (11.41 h), S_cylindrical launched
b9ea327 2026-09-13 exp_04 epoch 1 accepted; exp_05 training queue launched on GPU 1 (S_simple attempt_20260913T123905)
3153ad1 2026-09-13 exp_05: fit/timing probes passed for S/L both backbones (L fits at 32x2); command file
ca752ea 2026-09-13 exp_05 bookkeeping: round-3 close-out review, round-4 prompt
8b85119 2026-09-13 exp_04/exp_05 bookkeeping: launch commit f19b9b6 certified; round-2 close-out review
6cfb377 2026-09-13 exp_05 bookkeeping: round-3 close-out delivered
d1f0d61 2026-09-13 fix(exp05): close producer reporting and boundary review findings
1ec0457 2026-09-13 fix(exp05): bind training evidence and complete recipe admission
453b8da 2026-09-13 exp_05 bookkeeping: rounds 2-3 reviews, close-out prompts, amendments A5-A8; exp_04 round-7 review final count
f19b9b6 2026-09-12 fix(exp05): bind probe evidence and audit ceiling renewals
0e35e2c 2026-09-12 fix(launchers): certify completion before honoring tail signals
ba2f281 2026-09-12 fix(exp05): bind golden recipes to tier arms and records
860a334 2026-09-12 test(exp05): cover secondary profiles and provenance boundaries
970bc52 2026-09-12 feat(exp05): report descriptive paired yaw changes by tier
fe0fa94 2026-09-12 fix(exp05): bind tier checks to admitted input bytes
01de56f 2026-09-12 feat(exp05): publish canonical curves with shared provenance writer
a3f81d1 2026-09-12 feat(exp05): fail closed on grouped run admission
c381184 2026-09-12 feat(exp05): validate tier metadata and explicit M evaluator pins
faf3b1c 2026-09-12 test(exp05): bind six-arm synthetic evaluation fixtures
6ac01f2 2026-09-12 feat(exp05): gate six-arm conclusions and target ratios
f2b953a 2026-09-12 feat(exp05): compute registered cohorts and paired intervals
0916864 2026-09-12 feat(exp05): freeze capacity profiles and approval schema
5e8b677 2026-09-12 exp_05 bookkeeping: round 1 delivered and reviewed, goldens, amendments A1-A4, reviewer briefing and prompts
0dba2b3 2026-09-12 fix(exp05): reject stale probes resource drift and invalid metadata types
a19a2a0 2026-09-12 feat(exp05): enforce receipt-derived full-run and epoch-one limits
287e3ed 2026-09-12 feat(exp05): launch single-arm tier probes and write receipts
b74d2ec 2026-09-12 feat(exp05): validate tier receipts and cumulative training budgets
449055c 2026-09-12 feat(exp05): measure tier training validation and scratch-save costs
6a1a209 2026-09-12 feat(exp05): launch tier evaluation with bound training arguments
76c14e9 2026-09-12 feat(exp05): bind tier checkpoints to the shared evaluation loop
5ea3d49 2026-09-12 Preserve exp04 launcher parity and validate exp05 tier commands
7fdff94 2026-09-12 Add frozen exp05 tiers and exact parameter accounting
ed5f2a1 2026-09-12 Protect exp04 completion transactions and restore recovery logs
167722c 2026-09-12 Share signal-safe termination across exp04 launchers
b777f9d 2026-09-12 exp_04: restore recovered attempt names and tighten launch records
e3decd7 2026-09-12 exp_04: harden termination and verify recovery execution evidence
f7596c0 2026-09-12 exp04: check recovery log writers with host-compatible fuser
89521df 2026-09-12 exp04: classify malformed and missing runtime argument records
4772e15 2026-09-12 exp04: preserve recovery charges and require full-run bindings
90a6077 2026-09-12 exp04: move training inventories to digest-bound sidecars
32c3df5 2026-09-12 exp04: retain probe spread and complete artifact inventories
f518224 2026-09-12 exp04: recover preserved attempts with explicit finalize mode
248688b 2026-09-12 exp04: share finalisation and verify process and log closure
5d6d4f9 2026-09-12 exp04: retain recovery evidence and tolerate nonfinite abort state
b48cd7b 2026-09-12 exp04: validate reviewed source blobs and record later drift
8a8e515 2026-09-12 exp04: preserve specific abort diagnostics and owned logs
5d04d0a 2026-09-12 exp04: bind probe admission and count only full training hours
9b47c7a 2026-09-12 exp04: route termination signals through attempt cleanup
bd50f74 2026-09-12 exp04: bind training inputs and enforce launch provenance gates
07f3bc9 2026-09-12 exp04: preserve redirected logs and enforce epoch-one acceptance promptly
d79fa76 2026-09-12 exp04: enforce control parity and account promotion failures once
e8a164d 2026-09-12 exp04: expose smoke probe full and refusal launch modes
67c8819 2026-09-12 exp04: bind probe measurements and exercise launch refusals offline
10ef6ff 2026-09-12 exp04: finalize preserved training attempts after digest revalidation
b7480be 2026-09-12 exp04: police live runtime args and banner through tee
bbf8ad6 2026-09-12 exp04: bind training imports and split identity before spawn
3e5609c 2026-09-12 exp04: guard resources and preserve accounted attempts
38cf05b 2026-09-12 exp04: pin training argv and normalize runtime arguments
8dd5aa6 2026-09-12 exp_05: commits file
2d2a386 2026-09-12 exp_05 plan approved (capacity curve, handoff); exp_04 round-1 review and prompts
```
