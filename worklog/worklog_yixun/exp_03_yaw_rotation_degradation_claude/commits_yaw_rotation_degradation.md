# Commits — yaw_rotation_degradation

- `5bf4640` — base commit (2026-09-06): known-good base (exp_01/exp_02 code + records, exp_03 plan v2). Not exp_03 code.

## Round 1 — yaw rotation tools (Coder: Opus subagent, 2026-09-06)
- `fcac8be` — Add yaw scene-rotation primitives with tests
- `da293e6` — Add oracle and three-view covariance tests for yaw rotation
- `ca1bfea` — Add integer_delays and fixed_alignment for the yaw sweep
- `5fe3f6d` — Add CylindricalViT azimuth token-equivariance test
- `dafe166` — Add full-model yaw periodicity test under a pinned alignment
- `362c76c` — Harden fixed_alignment restore and cover the nested path

### Round 1 fix commits (Codex findings, 2026-09-06)
- `f66b8b8` — Return the model dtype from integer_delays and validate arguments
- `b42ca49` — Install the fixed alignment as a bound method
- `6c56bf0` — Add the fixed-alignment isolation control and real-coordinate parity
- `b44111d` — Close the tie, negative-k and tolerance test gaps

## Round 2 — reference manifest, paired stats, per-sample metrics (Coder: Opus subagent, 2026-09-06)
- `289cd64` — Add the deterministic per-query reference draw
- `076d45d` — Reproduce the dataset's reference candidate set
- `14bc4c8` — Build, hash and persist the reference manifest
- `437adfb` — Pin the manifest of the real unseen test split
- `fb10642` — Specify ManifestDataset before implementing it
- `d2a3463` — Serve the pinned references through ManifestDataset
- `c8df697` — Specify the paired relative-degradation bootstrap
- `ff8dc50` — Add the paired relative-degradation bootstrap
- `c3908ae` — Add the room-cluster resampling unit
- `4ef4582` — Add the Bonferroni level, the verdict rule and TOST
- `1d6d3e6` — Add the cross-model difference in differences
- `61ff4e2` — Split the test loss into per-sample terms
- `5c3193d` — Pin the Griffin-Lim phase per sample
- `80de84f` — Compute the acoustic errors per sample, NaN when invalid
- `e0d4150` — Say the zero-effect TOST case plainly

### Round 2 fix commits (Codex findings, 2026-09-06)
- `b321241` — Enforce the 8000-sample metric window inside acoustic_metrics
- `5107c32` — Pin griffin_lim_seeded to the unrolled eval path
- `1b11ef2` — Report the paired validity split per angle
- `f63a68e` — Make the registered Bonferroni call work
- `04a5421` — Harden the bootstrap arguments and cluster TOST
- `7bea816` — Add discriminating oracles for r_k, the cluster unit and the level
- `14d7cd4` — Canonicalise the manifest and validate it structurally
- `600f093` — Say TypeError in the docstrings and drop the last rtol

## Round 3 — evaluator and summarizer (Coder: Opus subagent, 2026-09-06)
- `2371d49` — Evaluate both yaw conditions from one unrotated batch
- `54cdcfe` — Measure the five spectral and three acoustic metrics per sample
- `69db7f6` — Audit the delay flips and decompose the cylindrical sensitivity
- `a062854` — Make the per-sample metrics batch-size invariant on real queries
- `dba85a5` — Evaluate one batch at every angle, order-independently
- `70e5341` — Validate the angle grids and summarise one angle's samples
- `72a7971` — Sweep one checkpoint over the manifest and write both JSON files
- `8b756d3` — Report the cylindrical decomposition from the first batches
- `7a12a64` — Read a yaw run and report its degradation per angle
- `e42d42e` — Decide H1 from the adjusted lower bound and word the verdict
- `f17b28b` — Compare the two backbones per angle and test patch-aligned equivalence
- `a2caa8d` — Re-evaluate exp_01 at k=0 and diagnose bootstrap convergence
- `a026efa` — Print the eight tables and write the JSON the results page binds to
- `6828c13` — Skip an acoustic metric a run did not measure at every angle

### Round 3 fix commits (Codex findings, 2026-09-06)
- `96735c9` — Pad a short batch to the canonical compute shape
- `6304792` — Record only the real rows of a canonically shaped batch
- `736af14` — Run every batch of the sweep at the canonical shape
- `d752429` — Write strict JSON: an invalid sample is null
- `52d7faf` — Write the outputs atomically and log every ten batches
- `63a7028` — Read the equivalence claim through the cylindrical run's own mask
- `4e4e444` — Decide H2 at the cells where H1 found a degradation
- `75f4779` — Fix the confirmatory family at the pre-registered eighteen tests
- `4c2aaa5` — Gate the confirmatory summary on the pre-registered design
- `ddf432a` — Gate the sweep on the k=0 parity check against exp_01
- `8e28111` — Recheck every decision-driving bound with a second seed
- `8abc647` — Make the summary answerable on its own terms
- `fa836b7` — Drop a leftover unused local
- `62c9107` — Say in the docstring what each mode may claim

### Post-full-review fix round (summarizer only; gate/full validation, band v2/v3) — 2026-09-06
- `be21787` — Name the runs by the checkpoints they used
- `5fbb28a` — Fail the k=0 gate closed on anything it was not given
- `af08a0b` — Drop the superseded k0-gate main test
- `5676f6b` — Pin the pre-registration itself in full mode
- `bfdd9aa` — Require each confirmatory run to say where it came from
- `06fc4bc` — Write the summary as strict JSON
- `b9884c7` — Reconcile each run's metrics file with its per-sample arrays
- `91af25c` — Require the two nuisance runs the amended band is measured on
- `a1db897` — Widen the k=0 band by the phase and TF32 terms it was missing
- `979d4e9` — Measure the reference draw on the model the band judges
- `5febf47` — Let the band account for exp_01's compute shape

### Post-review producer rounds (summarizer JSON schema; numbers unchanged, proven by exploratory diffs) — 2026-09-06, Coder (Opus)
- `7f3532ccb617df56109697f5de788bbb9b5247c2` — exp_03: label the room-cluster intervals as adjusted in summarize_yaw
- `631050da6796f4b21818bffe5b972f590c47d561` — exp_03: print a failed TOST as "not established", not as "no"
- `cfc5e4f8bcc776b9cd396fd0af14578f986df1a6` — exp_03: report how far the data are from H1's +10 % margin
- `13fb53969ef064e6e2c11d8126da2408ffee1084` — exp_03: name and give a unit to every metric the summary emits
- `14134399472491d98d0d24e16b4187153a851da2` — exp_03: serialize the condition glosses, the grid sizes and the run descriptions
- `bfea00db5e0c32eb1d57d9a0288450117aa2f18e` — exp_03: say what the delay-flip integers count
- `4fbcc026072b781beff17551793bb6b8f6cc6ed8` — exp_03: label the k=0 table and the decomposition stages in the JSON
- `9cb512789f9b8ee5be5de791d44ef6c204c7c5fe` — exp_03: serialize H1's decision rule and its verdict wording
- `f7a0c0a26ba6fa0ad22e9a1370af0600673f906d` — exp_03: serialize H2's cell rule, its four-way aggregate and the TOST
- `a9384679804443073b1a5629aecb93d7b3770547` — exp_03: serialize r's definition, the bootstrap sentence and the convergence rule (amended from `9e63387`, pure reordering)

The evaluator and its 11 repo-local imports were never touched after `62c9107b4150e44c4ac410ff4cab359c1e71cc10` (source-closure digest `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`); the sweep ran at HEAD `5febf471318e37e60477ebe72afd48f03879c67e`; the gate decision was taken with the producer at that commit (recomputed byte-identically by the acceptance checker).

### Record commit — 2026-09-07
- `4131f7d9032a56508cfc417066f86fcaec329380` — exp_03: yaw-rotation degradation record (results, analysis, page, provenance tooling, reviews); adds the whole record folder and `tests/test_exp03_record_tools.py`. The unrelated modified `commits_*.md` files of exp_01/exp_02 were left out of this commit on the reviewer's request.
- (bookkeeping commit: this file and the closing worklog entry — SHA recorded in the worklog's closing entry, since a file cannot contain its own commit's hash)
