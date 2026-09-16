# commits — oriented_cyl (exp_06)

| SHA | branch | description |
|---|---|---|
| (none yet) | | plan v1/v2 and record files are uncommitted under `worklog/` until the peer-agreed merge window (W1 Sep 15 04:30–20:00) |

## Round 1 (Coder: OpenAI Codex gpt-6-astra, branch `exp06-window`, base main `77ef292`) — delivered 2026-09-14 21:16; Fable review pending
| f9300cf | exp06-window | exp06: add heading-conditioned cylindrical encoder |
| ddbebd8 | exp06-window | exp06: add oriented factory preserving pinned routes |
| a960767 | exp06-window | exp06: freeze heading energy windows and column quantization |
| e6f5366 | exp06-window | exp06: require stable two-window heading contrasts |
| 02fdb1b | exp06-window | exp06: preserve HAA draws while rotating geometry |
| dc2d53f | exp06-window | exp06: estimate room headings from hashed training inputs |
| 419ddf6 | exp06-window | exp06: validate heading records and refusal evidence |
| 4066811 | exp06-window | exp06: expose heading estimation and overrides through CLI |
| a950605 | exp06-window | exp06: reject malformed descriptive heading values |

Round 1 CLOSED 2026-09-14 21:32 — Fable review `oriented_cyl_fable_code_round1_review.md`: approve with changes, no blockers; should-fix S1–S3 scheduled as round-2a cycle 0.

## Round 2a (Coder: Claude Opus 5, branch `exp06-window`, base `a950605`) — delivered 2026-09-15 01:00; Codex review pending
| 57b3484 | exp06-window | exp06: bind heading records to git identity and separate refusal exit status |
| 0a1dd8d | exp06-window | exp06: classify every trainer args field and normalize historical runs |
| 8470a8d | exp06-window | exp06: check derived fields, budget completeness and the whole schema |
| ef4c654 | exp06-window | exp06: parser, model construction and recorded arguments for the training entry |
| 5c89ffc | exp06-window | exp06: provenance record and the composed training main |
| 4d4d95b | exp06-window | exp06: finalizer core and the twelve-epoch pretraining contract |
| c4fdf3a | exp06-window | exp06: completion evidence for the HAA fine-tuning and evaluation children |
| 852aebf | exp06-window | exp06: job-level completion and the finalizer command line |
| e38f90b | exp06-window | exp06: bounded in-process smoke runner and the oriented CPU fixture |
| 5bd11f8 | exp06-window | exp06: launch preflight gate inside the finalizer |
| 4f92f54 | exp06-window | exp06: launcher modes smoke, probe, full and finalize |

Round 2a fix cycle (Opus) — delivered 2026-09-15 02:14, Codex close review pending
| 975e999 | exp06-window | exp06: count parameters inside a forked RNG (review nit 8) |
| 8da3354 | exp06-window | exp06: check smoke budgets, entry statuses and the live memory ceiling (review 7) |
| 7914be7 | exp06-window | exp06: name every malformed finalizer input as a refusal (review 9) |
| e9dd0a5 | exp06-window | exp06: bind startup, retained and checkpoint arguments type-strictly (blocker 2) |
| 6d1e870 | exp06-window | exp06: revalidate the closure, the reviewed blobs and the data inventory (blocker 1) |
| 0ce5d13 | exp06-window | exp06: bind the child-exit receipt, the live pid and a stable log (blocker 4a) |
| b822695 | exp06-window | exp06: drain the child pipe to EOF before closing the log (blocker 4b) |
| 3721cc8 | exp06-window | exp06: enforce provenance, heading and artefact contracts on HAA training (blocker 3a) |
| cb9a6f3 | exp06-window | exp06: bind the evaluation child to its checkpoint, heading and side labels (blocker 3b) |
| e9e6e6f | exp06-window | exp06: derive job admission from child roles, hashes and lineage (blocker 3c) |
| 373b39f | exp06-window | exp06: require a valid diagnostic receipt and gate recovery finalization (review 5a) |
| 4c66184 | exp06-window | exp06: resolve one data root through the dataset module and pin it (review 6) |
| 1d5e116 | exp06-window | exp06: share the child lifecycle with smoke, probe and recovery (review 5b) |
| 62655ae | exp06-window | exp06: import the backbone registry directly in the finalizer |
| 2fe25bf | exp06-window | exp06: require each child's role artefacts and refresh the module docstrings |

Round 2a fix cycle 2 (Opus) — delivered 2026-09-15 03:40, Codex close review 2 pending
| 22250f9 | exp06-window | exp06: red tests for a complete training-data identity (blocker 1) |
| e79fc0f | exp06-window | exp06: derive and require the whole training inventory (blocker 1) |
| dddbba3 | exp06-window | exp06: red tests binding the exp06_* fields to the record (blocker 2) |
| 2ac4fee | exp06-window | exp06: bind the exp06_* arguments to the execution record (blocker 2) |
| aead393 | exp06-window | exp06: red tests for malformed nested inputs (review 9) |
| 13da458 | exp06-window | exp06: type every nested provenance container (review 9) |
| 1bcc7e9 | exp06-window | exp06: red tests for a failing sink and the receipt schema (blocker 4) |
| 846e5b4 | exp06-window | exp06: enforce the sink status and the receipt schema (blocker 4) |
| 7ac020b | exp06-window | exp06: red tests for launch ownership and recovery (review 5) |
| d3eeba2 | exp06-window | exp06: launcher ownership, every launch location, real reasons (review 5) |
| 1b54f37 | exp06-window | exp06: red tests for the real HAA history format (blocker 3a) |
| 906b694 | exp06-window | exp06: validate the pipeline's real HAA history (blocker 3a) |
| 6b15002 | exp06-window | exp06: red tests for parsed evaluation metrics (blocker 3b) |
| eaba017 | exp06-window | exp06: parse and cross-check the evaluation metrics (blocker 3b) |
| 4d793c7 | exp06-window | exp06: real job children and a declared job spec (blocker 3c, tests) |
| 84596ca | exp06-window | exp06: red job-admission counterexamples (blocker 3c, tests) |
| 8300de6 | exp06-window | exp06: re-run every child validator inside the job (blocker 3c) |
| 1571715 | exp06-window | exp06: red tests for a child log that was never closed (blocker 3c) |
| 21f7a2c | exp06-window | exp06: re-validate each child's closed log and receipt (blocker 3c) |
| bf51484 | exp06-window | exp06: document the round-2a contracts in the module headers |

Round 2a fix cycle 3 (Opus) — delivered 2026-09-15 04:38, Codex close review 3 pending
| 7c5b74f | exp06-window | exp06: red tests for a live child inside a job (finding 1) |
| 77946c8 | exp06-window | exp06: refuse a live child during job admission (finding 1) |
| 1be0997 | exp06-window | exp06: red tests for unbound child receipts and evidence (finding 2) |
| 2741f37 | exp06-window | exp06: bind every child's receipt and evidence to the re-run (finding 2) |
| d8005a9 | exp06-window | exp06: red tests for unbound per-sample results (finding 3) |
| 961d7e0 | exp06-window | exp06: bind per-sample results to their room and protocol (finding 3) |
| 32db42a | exp06-window | exp06: red tests for malformed metric values (finding 4) |
| d94bf1b | exp06-window | exp06: type and recompute every metric value (finding 4) |
| 4a9f40c | exp06-window | exp06: red CLI tests for nested job-schema failures (finding 5) |
| e7fb365 | exp06-window | exp06: type the whole job schema before it is used (finding 5) |
| a977a68 | exp06-window | exp06: document where a pipeline's launch.pid belongs (finding 1) |

Round 2a fix cycle 4 (Opus) — delivered 2026-09-15 05:09, Codex close review 4 pending
| 3eaec27 | exp06-window | exp06: red job tests for artifact maps that record an extra null (close-3 finding 1) |
| c91e014 | exp06-window | exp06: compare a child's whole artifact mapping, key set first (close-3 finding 1) |
| c4ec284 | exp06-window | exp06: red tests for invalid counters that contradict the observations (close-3 finding 2) |
| 6418e11 | exp06-window | exp06: reconcile the invalid-measurement counters with the observations (close-3 finding 2) |
| 027bd5b | exp06-window | exp06: red CLI test for a JSON integer too wide for a float (close-3 nit 3) |
| f0898a1 | exp06-window | exp06: name the refusal when a JSON integer cannot become a float (close-3 nit 3) |
| 12e06fc | exp06-window | exp06: document how the invalid-measurement counters are reconciled (close-3 finding 2) |

Round 2a CLOSED 2026-09-15 05:19 at `12e06fc` — Codex close review 4: empty blocking list (nit: omitted-T60 NaN-only check → round 2b).

## Round 2b (Coder: Claude Opus 5, branch `exp06-round2b` from `12e06fc`) — delivered 2026-09-15 06:28; Codex review pending
| 9e727af | exp06-round2b | exp06: red regressions for infinity in an omitted room's T60 (close-4 finding 1) |
| f350d51 | exp06-round2b | exp06: an omitted room's T60 must be NaN, never an infinity (close-4 finding 1) |
| ae7ea78 | exp06-round2b | exp06: red tests for the simulated-room evaluation entry point |
| 5f6d49b | exp06-round2b | exp06: simulated-room evaluation entry point on the exp_06 registry |
| c5c81c6 | exp06-round2b | exp06: red tests for the evaluation launcher adapter |
| 65e099f | exp06-round2b | exp06: evaluation launcher adapter with its own bindings and writer closure |
| 483287b | exp06-round2b | exp06: red tests for the HAA fine-tuning wrapper |
| aa61fc4 | exp06-round2b | exp06: HAA fine-tuning wrapper with the heading frame and the child records |
| cc77266 | exp06-round2b | exp06: red tests for the HAA evaluation wrapper's parser, datasets and records |
| d82b126 | exp06-round2b | exp06: HAA evaluation parser, datasets and per-sample meta |
| 6262141 | exp06-round2b | exp06: red tests for per-query Griffin-Lim seeding, side labels and the room summary |
| 855ec9a | exp06-round2b | exp06: the HAA evaluation loop, per-query seeding, side labels and outputs |
| 4957053 | exp06-round2b | exp06: red golden tests for the HAA pipeline queue |
| 140c3c9 | exp06-round2b | exp06: HAA pipeline queue, exclusive children and job-level finalisation |
| 2fce9ef | exp06-round2b | exp06: red end-to-end test of the HAA job contract on written artefacts |
| e5461a6 | exp06-round2b | exp06: let a HAA child record the repository it was launched from |
| 4b30735 | exp06-round2b | exp06: red tests for the finalizer's heading consumer contract (full-train finding 7) |
| d6a4f42 | exp06-round2b | exp06: re-verify every bound heading against its cache and require confirmatory |

Round 2b fix cycle (Opus) — delivered 2026-09-15 07:31, Codex close review pending
| a49e3a3 | exp06-round2b | exp06: red regressions for the validation-only heading binding (finding 4) |
| c1f9829 | exp06-round2b | exp06: revalidate the headings a run validates on, not only those it trains on (finding 4) |
| 93b397f | exp06-round2b | exp06: red tests for the job-spec identity a child is launched under (finding 3) |
| 052bf34 | exp06-round2b | exp06: freeze the job declaration each HAA child was launched under (finding 3) |
| a1a851d | exp06-round2b | exp06: red tests that a failed job preparation launches nothing (finding 1) |
| 497824b | exp06-round2b | exp06: make job preparation a gate the queue cannot launch past (finding 1) |
| 5189b0d | exp06-round2b | exp06: red tests for the post-run check of the finished outputs (finding 2) |
| 3df20b8 | exp06-round2b | exp06: check the finished outputs against the manifest before a run stands (finding 2) |

Training-gate fix cycle (Opus, `exp06-window` 12e06fc..f18383a) — delivered 2026-09-15 07:32, Codex full_train review 2 pending
| 4fb3f7c | exp06-window | exp06: red tests for checkpoint dtype/shape and the probe's recipe flags (nits 8, 9) |
| cf7df40 | exp06-window | exp06: bind checkpoint dtype/shape and give the probe the registered flags (nits 8, 9) |
| ccb06b5 | exp06-window | exp06: red tests for type-strict, bounded operational arguments (finding 4) |
| 4120094 | exp06-window | exp06: validate operational argument types, bounds and cadence (finding 4) |
| 439236f | exp06-window | exp06: red tests for the exit receipt's child pid liveness (finding 3) |
| 56dc945 | exp06-window | exp06: refuse an exit receipt whose child pid is alive or disagrees (finding 3) |
| 3154c3b | exp06-window | exp06: red tests for the approvals module and its null template (finding 1) |
| 04c44c7 | exp06-window | exp06: the approvals schema and its null template, both copies (finding 1) |
| 7abc0c6 | exp06-window | exp06: recompute code digests and admit runs fail-closed (finding 1) |
| 54ad12d | exp06-window | exp06: red tests for the trainer's orchestration and approvals bindings (finding 1) |
| 1e04b8b | exp06-window | exp06: record the launcher, finalizer and approvals identities at spawn (finding 1) |
| 18e8b5c | exp06-window | exp06: red tests for approvals and orchestration at finalisation (finding 1) |
| 7315777 | exp06-window | exp06: verify approvals, code digests and orchestration at finalisation (finding 1) |
| eb0a1c9 | exp06-window | exp06: red tests for the launcher's approval gate (finding 1) |
| 4885e17 | exp06-window | exp06: gate every launch on the approved code digests (finding 1) |
| 265be9c | exp06-window | exp06: red tests for the geometry-input inventory (finding 2) |
| d5ff641 | exp06-window | exp06: inventory and rehash every geometry input of the run (finding 2) |
| 815f58f | exp06-window | exp06: red tests for the diagnostic receipt's identity and budgets (finding 6) |
| 3834d04 | exp06-window | exp06: diagnostics record their runner, budgets, timing and provenance (finding 6) |
| 090ab7f | exp06-window | exp06: red tests for validated diagnostic evidence at finalisation (finding 6) |
| 78523b6 | exp06-window | exp06: require validated diagnostic evidence and its provenance (finding 6) |
| 163f818 | exp06-window | exp06: red tests for a failed diagnostic's status and abort naming (finding 5) |
| f18383a | exp06-window | exp06: a failed diagnostic aborts the launch with the child's status (finding 5) |

Round 2b CLOSED 2026-09-15 07:39 at `3df20b8` — Codex close review: empty blocking list (nits 7–8 → round 3).

Training-gate fix cycle 2 (Opus, `exp06-window` f18383a..a313253) — delivered 2026-09-15 08:43, Codex full_train review 3 pending
| 0f3b79e | exp06-window | exp06: red test for the registered smokes' child dispatch (finding 1) |
| b979f0c | exp06-window | exp06: the launcher tells the exp_06 child what diagnostic it is (finding 1) |
| a74d985 | exp06-window | exp06: red tests for approvals bound to a reviewed commit (finding 2) |
| 5201430 | exp06-window | exp06: preflight binds the approvals to their reviewed commit (finding 2) |
| a83f045 | exp06-window | exp06: red tests for committed approvals at finalisation (finding 2) |
| 79b53f0 | exp06-window | exp06: finalisation reads approvals from their reviewed blob (finding 2) |
| f82c33d | exp06-window | exp06: red tests for complete consumed-data coverage (finding 3) |
| 7868207 | exp06-window | exp06: bind the held-out waveforms and both geometry splits (finding 3) |
| df6a762 | exp06-window | exp06: red tests for diagnostic receipt consistency (finding 4) |
| 58ccc27 | exp06-window | exp06: derive a diagnostic's pass from consistent evidence (finding 4) |
| a313253 | exp06-window | exp06: regression for the exploratory flag reaching the children (finding 1) |

## Round 3a (Coder: Claude Opus 5, branch `exp06-round2b` from `3df20b8`) — delivered 2026-09-15 08:53; Codex review pending
| 4a8189d | exp06-round2b | exp06: red tests that the post-run gate reads the retained completion (nit 7) |
| 626e8d4 | exp06-round2b | exp06: validate the hashes the run published, not the caller's dict (nit 7) |
| 141a166 | exp06-round2b | exp06: red tests that a refused preparation leaves a receipt (nit 8) |
| 1bf2532 | exp06-round2b | exp06: a refused preparation writes its receipt and aborts the job root (nit 8) |
| 2176329 | exp06-round2b | exp06: red tests for the device-preserving probe alignment |
| 8a25004 | exp06-round2b | exp06: device-preserving shift_and_align for the CPU probe |
| 792a16d | exp06-round2b | exp06: red tests for the mirror probe's cohort rule, forward and decision |
| 0c89feb | exp06-round2b | exp06: the mirror probe's cohort rule, recomposed forward and G1 decision |
| d6f19f6 | exp06-round2b | exp06: red tests for the mirror statistics, the gate arms and the bound record |
| ce010d1 | exp06-round2b | exp06: the mirror statistics of one arm, with room-frame side labels |
| c6b41b7 | exp06-round2b | exp06: the legacy reproduction and the full-cohort gate arms |
| abb6d7d | exp06-round2b | exp06: the hash-bound G1 record and its command line |
| 402a42b | exp06-round2b | exp06: red tests for the alpha-aware paired bootstrap |
| 4f4355d | exp06-round2b | exp06: the alpha-aware paired bootstrap and its convergence rule |
| cb3c770 | exp06-round2b | exp06: red tests that an empty batch and an unserialisable record are refused |
| 895f512 | exp06-round2b | exp06: refuse an empty C50 batch and an unserialisable completion record |

Round 3a fix cycle (Opus) — delivered 2026-09-15 09:59, Codex close review pending
| 85c5488 | exp06-round2b | exp06: red tests that G1 binds the bytes it evaluated (findings 1-2, nit 4) |
| f108f12 | exp06-round2b | exp06: capture, load from and re-check G1's input bytes (findings 1-2, nit 4) |
| 8934990 | exp06-round2b | exp06: red tests that a refused preparation never touches an unowned root (finding 3) |
| f62c750 | exp06-round2b | exp06: a refused preparation only ever touches the root it created (finding 3) |
| 7f29877 | exp06-round2b | exp06: cover every gate arm loading from the captured bytes (finding 1) |

Round 3a CLOSED 2026-09-15 10:09 at `7f29877` — Codex close review: empty blocking list (nits 5–6 → round 3b).

## Round 3b (Coder: Claude Opus 5, branch `exp06-round2b` from `7f29877`) — delivered 2026-09-15 11:09; Codex review pending
| 579cf02 | exp06-round2b | exp06: red tests for the probe's last check and its documented exit status |
| 8ae72a5 | exp06-round2b | exp06: check the probe's inputs last and document its refusal status (nits 5-6) |
| b6ba252 | exp06-round2b | exp06: red tests for the approvals schema of 6.4 and its lazy adapter |
| af37615 | exp06-round2b | exp06: the approvals schema of 6.4, behind a lazily imported adapter |
| 126912b | exp06-round2b | exp06: red tests for the 6.4 requirement matrix and the code digests |
| 4b4066b | exp06-round2b | exp06: the per-producer approval matrix and the code-digest helper |
| c2d4c8a | exp06-round2b | exp06: red tests for the summariser's arm table and its legacy hash receipt |
| be03f68 | exp06-round2b | exp06: the summariser's arm table and the reconstructed legacy receipt |
| 776f910 | exp06-round2b | exp06: red tests for the summariser's legacy and new-arm admission branches |
| 06b859b | exp06-round2b | exp06: the summariser's two admission branches |
| 099be39 | exp06-round2b | exp06: red tests for the summariser's cohort policy, verdicts and screen |
| d5396aa | exp06-round2b | exp06: the summariser's cohort policy, verdicts, screen and descriptive cells |
| 94064c2 | exp06-round2b | exp06: red tests for the side split, the approvals gate and the summariser outputs |
| d887e3f | exp06-round2b | exp06: the summariser's side split, approvals gate and published outputs |
| 3ba1872 | exp06-round2b | exp06: red tests for the H3 comparer's fail-closed admission |
| e79a826 | exp06-round2b | exp06: the H3 comparer's fail-closed admission of 6.3's evaluation runs |
| ac79221 | exp06-round2b | exp06: red tests for the H3 statistics, determinism and published outputs |
| 02a01f1 | exp06-round2b | exp06: H3's statistics, verdicts and hash-bound outputs |
| 99ded87 | exp06-round2b | exp06: red tests for the arm's initialisation and the heading JSON identity |
| 622b187 | exp06-round2b | exp06: bind each new arm's initialisation and the heading JSON it read |
| 0ca806c | exp06-round2b | exp06: drop the trailing blank line left by the last test append |

Merge integration (Opus, `exp06-window`): 3a9e86a..786161a tests-only (dead pids, checkout-derived approvals test, strict xfail) and 786161a..a40f81b (A3: job completion without child receipt) — delivered 2026-09-15 11:11; Codex pre-merge review pending
| a2e2550 | exp06-window | exp06: the pipeline fixtures close a child that has really exited |
| d82c31b | exp06-window | exp06: the approvals test derives absence from the checkout, not from a round |
| 786161a | exp06-window | exp06: pin the one merge conflict the fixtures must not hide |
| da5eb93 | exp06-window | exp06: require the A3 job contract of the finalizer (red) |
| c57bca2 | exp06-window | exp06: a job binds its spec, its owner and its children, not a receipt (A3) |
| 6429e1c | exp06-window | exp06: the queue's job finalisation writes no receipt (red) |
| a40f81b | exp06-window | exp06: the pipeline's job finalisation closes no log of its own (A3) |

| 17cdb61 | exp06-window | exp06: a job root must record the owner it binds (red) |
| 5091f6c | exp06-window | exp06: a job binds the owner its root records, always (finding 1) |

Training branch MERGE-READY 2026-09-15 12:17 at `5091f6c` — Codex pre-merge close review: approve for merge.

Round 3b fix cycle (Opus) — delivered 2026-09-15 14:00: merge `90ce3ba` (exp06-window 5091f6c into exp06-round2b), integration `76e8f6d`, fixes …`df7ea24`; Codex close review pending
| daa7eef | exp06-round2b | exp_03: PNG figures of r_k vs yaw (acoustic, spectral, H2 D_k) rendered from the canonical full_stats.json |
| fa219ef | exp06-round2b | exp_04: fill approved_digests.json (second reviewed commit): evaluator 0b05245c, writer 15f9c984, training launcher 0bff0d4f (attempt record), producers at daa7eef, aug epoch-12 f8e64052, epoch-9 442a0a56 |
| 87d3e54 | exp06-round2b | exp_04: H2/TOST/EPOCH9 producer runs logged (H2 partially supported; TOST equivalent at axis-aligned headings only) |
| f851be1 | exp06-round2b | exp_04: H1 (non-inferior on EDT only at K=8 and K=1), TABLE_V1 and the living table |
| cab6f42 | exp06-round2b | exp_04: results.md, results page, record binding report checked; grid diagnostic |
| 862923e | exp06-round2b | exp_04: analysis, params finalised, command file, commits map |
| 4fb3f7c | exp06-round2b | exp06: red tests for checkpoint dtype/shape and the probe's recipe flags (nits 8, 9) |
| cf7df40 | exp06-round2b | exp06: bind checkpoint dtype/shape and give the probe the registered flags (nits 8, 9) |
| ccb06b5 | exp06-round2b | exp06: red tests for type-strict, bounded operational arguments (finding 4) |
| 4120094 | exp06-round2b | exp06: validate operational argument types, bounds and cadence (finding 4) |
| 439236f | exp06-round2b | exp06: red tests for the exit receipt's child pid liveness (finding 3) |
| 9388c14 | exp06-round2b | exp_05: L_simple launched on GPU 0 at 862923e (attempt_20260915T060543) |
| 56dc945 | exp06-round2b | exp06: refuse an exit receipt whose child pid is alive or disagrees (finding 3) |
| 3154c3b | exp06-round2b | exp06: red tests for the approvals module and its null template (finding 1) |
| 04c44c7 | exp06-round2b | exp06: the approvals schema and its null template, both copies (finding 1) |
| 7abc0c6 | exp06-round2b | exp06: recompute code digests and admit runs fail-closed (finding 1) |
| 54ad12d | exp06-round2b | exp06: red tests for the trainer's orchestration and approvals bindings (finding 1) |
| 1e04b8b | exp06-round2b | exp06: record the launcher, finalizer and approvals identities at spawn (finding 1) |
| 18e8b5c | exp06-round2b | exp06: red tests for approvals and orchestration at finalisation (finding 1) |
| f4c42d3 | exp06-round2b | exp_04: record review applied (analysis corrections, params, commits map) — record approved |
| 7315777 | exp06-round2b | exp06: verify approvals, code digests and orchestration at finalisation (finding 1) |
| eb0a1c9 | exp06-round2b | exp06: red tests for the launcher's approval gate (finding 1) |
| 4885e17 | exp06-round2b | exp06: gate every launch on the approved code digests (finding 1) |
| 265be9c | exp06-round2b | exp06: red tests for the geometry-input inventory (finding 2) |
| d5ff641 | exp06-round2b | exp06: inventory and rehash every geometry input of the run (finding 2) |
| 815f58f | exp06-round2b | exp06: red tests for the diagnostic receipt's identity and budgets (finding 6) |
| 3834d04 | exp06-round2b | exp06: diagnostics record their runner, budgets, timing and provenance (finding 6) |
| 090ab7f | exp06-round2b | exp06: red tests for validated diagnostic evidence at finalisation (finding 6) |
| 78523b6 | exp06-round2b | exp06: require validated diagnostic evidence and its provenance (finding 6) |
| 163f818 | exp06-round2b | exp06: red tests for a failed diagnostic's status and abort naming (finding 5) |
| f18383a | exp06-round2b | exp06: a failed diagnostic aborts the launch with the child's status (finding 5) |
| 0f3b79e | exp06-round2b | exp06: red test for the registered smokes' child dispatch (finding 1) |
| b979f0c | exp06-round2b | exp06: the launcher tells the exp_06 child what diagnostic it is (finding 1) |
| a74d985 | exp06-round2b | exp06: red tests for approvals bound to a reviewed commit (finding 2) |
| 5201430 | exp06-round2b | exp06: preflight binds the approvals to their reviewed commit (finding 2) |
| a83f045 | exp06-round2b | exp06: red tests for committed approvals at finalisation (finding 2) |
| 79b53f0 | exp06-round2b | exp06: finalisation reads approvals from their reviewed blob (finding 2) |
| f82c33d | exp06-round2b | exp06: red tests for complete consumed-data coverage (finding 3) |
| 7868207 | exp06-round2b | exp06: bind the held-out waveforms and both geometry splits (finding 3) |
| df6a762 | exp06-round2b | exp06: red tests for diagnostic receipt consistency (finding 4) |
| 58ccc27 | exp06-round2b | exp06: derive a diagnostic's pass from consistent evidence (finding 4) |
| a313253 | exp06-round2b | exp06: regression for the exploratory flag reaching the children (finding 1) |
| a2e2550 | exp06-round2b | exp06: the pipeline fixtures close a child that has really exited |
| d82c31b | exp06-round2b | exp06: the approvals test derives absence from the checkout, not from a round |
| 786161a | exp06-round2b | exp06: pin the one merge conflict the fixtures must not hide |
| da5eb93 | exp06-round2b | exp06: require the A3 job contract of the finalizer (red) |
| c57bca2 | exp06-round2b | exp06: a job binds its spec, its owner and its children, not a receipt (A3) |
| 6429e1c | exp06-round2b | exp06: the queue's job finalisation writes no receipt (red) |
| a40f81b | exp06-round2b | exp06: the pipeline's job finalisation closes no log of its own (A3) |
| 17cdb61 | exp06-round2b | exp06: a job root must record the owner it binds (red) |
| 5091f6c | exp06-round2b | exp06: a job binds the owner its root records, always (finding 1) |
| 76e8f6d | exp06-round2b | exp06: the merged approvals module is the one round 3b reads |
| e5e2620 | exp06-round2b | exp06: refusal tests assert a diagnostic phrase, not a path word (finding 11) |
| 447be92 | exp06-round2b | exp06: red -- a legacy receipt may not omit a registered arm (finding 6) |
| 6e35927 | exp06-round2b | exp06: the legacy receipt enumerates exactly the registered arms (finding 6) |
| 727f0a0 | exp06-round2b | exp06: red -- a void or empty HAA cohort is a reportable cell (finding 8) |
| 578f884 | exp06-round2b | exp06: a void HAA cohort is reported, never raised (finding 8) |
| 9da6d8b | exp06-round2b | exp06: red -- per-role closures, heading identity and the frozen protocol (findings 1-3) |
| 096956f | exp06-round2b | exp06: per-role closures, heading identity and the frozen protocol (findings 1-3) |
| 928592a | exp06-round2b | exp06: the statistics tests read arm dicts, not an admission fixture |
| 09ccbfa | exp06-round2b | exp06: retire the forged new-arm completions (finding 2) |
| d21d884 | exp06-round2b | exp06: red -- admission over children the finalizer really certified (finding 2) |
| 5b8bfdc | exp06-round2b | exp06: new-arm admission re-runs the finalizer's verifiers (findings 1-2) |
| 45eb6bc | exp06-round2b | exp06: red -- the summariser compares actual identities with the approved ones (finding 3) |
| 468263c | exp06-round2b | exp06: the summariser compares actual identities with the approved ones (finding 3) |
| 8e7da03 | exp06-round2b | exp06: red -- bound inputs, final revalidation and staged publication (findings 10, 12) |
| cfa7844 | exp06-round2b | exp06: bind inputs where they are read, revalidate at publication (findings 10, 12) |
| 9fb0538 | exp06-round2b | exp06: the H3 fixture writes real evidence, not a shape |
| d05c063 | exp06-round2b | exp06: red -- H3 requires real completion, source, data and reference evidence (findings 4-5) |
| 949689f | exp06-round2b | exp06: H3 admission requires real evidence and parses the reference (findings 4-5) |
| b1047c9 | exp06-round2b | exp06: red -- explicit exp_04 / exp_05 / exp_06 admission routes (finding 7) |
| 454c843 | exp06-round2b | exp06: explicit exp_04, exp_05 and exp_06 admission routes (finding 7) |
| b148e7b | exp06-round2b | exp06: red -- H3 eligibility, producer identity and safe publication (findings 3, 9, 10, 12) |
| 6c6a156 | exp06-round2b | exp06: H3 eligibility, producer identity and safe publication (findings 3, 9, 10, 12) |
| 92c0dec | exp06-round2b | exp06: red -- the checkpoint identity of every arm is published (finding 5) |
| df7ea24 | exp06-round2b | exp06: publish the checkpoint identity of every arm (finding 5) |

## Merge into main — 2026-09-15 15:04
| eaadb26 | main | merge --no-ff exp06-window (9ccb6ea) |
| 1b49cfc | main | bookkeeping: exp_06 record, exp_02 diagnostics, Codex 2a archive, SOP role change |
| fc1ee2a | main | approvals fill (invalid: extra note key) |
| 53307b6 | main | approvals file conforms to the schema (corrected fill) |

Round 3b fix cycle 2 (Opus) — delivered 2026-09-15 15:42; Codex close review 2 pending
| 2e670da | exp06-round2b | exp06: red -- H3 must enforce the registered experiment (finding 1) |
| 7b19f94 | exp06-round2b | exp06: H3 admits the registered split, inventory and reference (finding 1) |
| 76a3606 | exp06-round2b | exp06: red -- role, registry, class and epoch identities (finding 4) |
| 9acdc0c | exp06-round2b | exp06: H3 binds the registered model, registry and epoch of each role (finding 4) |
| 55e975e | exp06-round2b | exp06: red -- the exp_05 route establishes its args/tier binding (finding 5) |
| 2fae051 | exp06-round2b | exp06: the exp_05 route re-derives its tier from the checkpoint args.json (finding 5) |
| cacfa86 | exp06-round2b | exp06: red -- production approvals are bound to the reviewed commit (finding 7) |
| e391513 | exp06-round2b | exp06: production approvals are the bytes committed at the reviewed HEAD (finding 7) |
| 48f14a0 | exp06-round2b | exp06: red -- the A3 owner is read from the job root launch.pid (finding 6) |
| b2361fd | exp06-round2b | exp06: bind the job owner to the root launch.pid (finding 6) |
| 74fa776 | exp06-round2b | exp06: red -- the registered HAA recipe of plan 6.2 and the sensitivity mode (finding 3) |
| f0d7963 | exp06-round2b | exp06: admit only plan 6.2's recipe, or label the run a sensitivity analysis (finding 3) |
| 747f3a6 | exp06-round2b | exp06: red -- parsed bytes, contradictions and the complete revalidated set (finding 2) |
| e97bcab | exp06-round2b | exp06: bind bytes where they are read and revalidate the complete set (finding 2) |
| f30ac22 | exp06-round2b | exp06: red -- every HAA child artefact is bound and revalidated (finding 2) |
| 9b62055 | exp06-round2b | exp06: carry every verified child artefact into the revalidated set (finding 2) |
| 4617219 | exp06-round2b | exp06: red -- the comparer's own closure is bound and revalidated (finding 2) |
| 07001a0 | exp06-round2b | exp06: bind the comparer's producer closure for revalidation (finding 2) |

| 4dd37ec | exp06-launch-gate | validate the populated approvals record instead of the null template (tests only) — Codex review pending |
| 4ea1b2b | exp06-launch-gate | resolve HEAD lazily so the profiles test module collects outside a git checkout (tests only) — Codex close review pending |

## Launch-gate merge — 2026-09-15 16:38
| 2e1b3ea | exp06-launch-gate | merge main (8cc4fd7) into the branch |
| 62a4665 | main | merge --no-ff exp06-launch-gate (tests-only) |
| e590b43 | main | bookkeeping: launch-gate reviews/fixes, 3b close-2 review, fill verification, params set-up, notebook |

Round 3b fix cycle 3 (Opus) — delivered 2026-09-15 17:47; Codex close review 3 pending
| c0d4753 | exp06-round2b | exp06: red -- the owner pid and its digest come from one read (finding 6) |
| efe66ea | exp06-round2b | exp06: parse and hash one launch.pid snapshot (finding 6) |
| 597589a | exp06-round2b | exp06: red -- the exp_05 tier is read from the bytes its digest identifies (finding 5) |
| 0718c62 | exp06-round2b | exp06: parse and hash one exp_05 args snapshot (finding 5) |
| 9014b8a | exp06-round2b | exp06: red -- a record that changed after the job certified it is refused (finding 2b) |
| ee8d1c9 | exp06-round2b | exp06: bind the parent's digest before delegating, and enforce it after (finding 2b) |
| ed98fe4 | exp06-round2b | exp06: red -- every child-provenance dependency is retained and revalidated (finding 2a) |
| f668747 | exp06-round2b | exp06: retain every validated child-provenance dependency (finding 2a) |
| cf9c38b | exp06-round2b | exp06: red -- primary admission registers the effective validation rooms (finding 3) |
| 93c1a75 | exp06-round2b | exp06: register the effective validation rooms of each stage (finding 3) |
| 6ac832c | exp06-round2b | exp06: regression -- a dependency contradicting an earlier binding is refused (finding 2a) |

Round 3b CLOSED 2026-09-15 18:05 at `6ac832c` — Codex close review 3: empty blocking list (deferred: finalizer `load_job_spec` snapshot gap → separate reviewed round before the evaluation/HAA gate).

## Round-3 merge — 2026-09-16 08:04
| e7555de | exp06-round2b | merge main (1a1f6ba) into the branch |
| 21a2bf6 | main | merge --no-ff exp06-round2b (rounds 3a + 3b) |
| d854102 | main | bookkeeping: round-3 reviews/reports, Sep 16 entries |
| b9f6ebe | main | approvals re-filled at the merge tip (probe_align, all code digests, artifacts.heading) |
Approvals re-fill VERIFIED 2026-09-16 08:14 (Codex, no blockers) — launch binding b9f6ebe.
