# Commits — yaw_aug_xrir (exp_04)

- `8cb87d1e0e8cbd85913911fdefd64aa69ced931d` — exp_04 base: plan v4 approved; SOP roles updated; exp_03 K=1 follow-up (Planner)
- `fda92858b789ab33ae5dfdd53d34715d4e1e866e` — exp_04: add strict per-sample batched yaw rotation (Codex round 1, cycle 1) — **superseded and dropped before push** (it edited `tools/yaw_rotation.py`, a member of exp_03's pinned closure; amendment A2 relocated the function)
- `410f205ff3fa340f1eb9f371e8bf226e5a935d11` — exp_04: seeding, offsets and batched rotation in tools/yaw_aug.py (A1 chi-square bound, A2 relocation) (Codex round 1, cycles 1+2)

- `cd7dc994f17b656910ecce7b65633afda10fb806` — exp_04 round 2 cycle 1: trainer yaw integration, no-save, parity and startup guards; 180 changed lines.
- `34d98816a4bc4b79002aa17db78aad7ba94b0eed` — exp_04 round 2 cycle 2: seeded alignment audit CLI, cohort hashing and tests; 183 changed lines.

## Round-2 close-out — 2026-09-12T16:46:56-04:00

- `a619bc86939aebb6b74410a387c602237f7384b7` — loader-derived epoch length and strict counter bound, +45/-18 = 63 lines.
- `da1becc8e95c2bc73c07b753b07dc896287aacbf` — test-only semantic equality, banner, audit and CLI edges, +57/-19 = 76 lines.
- `35f55ad4d5960e99587401cd70aa32f0213ba67c` — strict native counters/enabled contract and exclusive audit args/env, +72/-6 = 78 lines.

All carry the requested Codex trailer; no worklog/ or ckpt/ committed; no push.

## Rounds 3a → 6 (Codex gpt-6-astra) and Planner bookkeeping — 2026-09-12 17:02 → 19:5x

- `ad19454d0140caf5e02a0f35b4c087a90ac59a7c` — exp_04 bookkeeping: round 2 closed (Fable reviews, fix prompt, amendments A3-A5) (Codex, trailer gpt-6-astra)
- `9cd3703789d6de799251a79480e8043d0679c43a` — exp04: fail early on existing audits and tighten trainer fixtures (Codex, trailer gpt-6-astra)
- `d92473783281e4171e7228d7696f433f0f52f780` — exp04: record source provenance and create immutable manifests (Codex, trailer gpt-6-astra)
- `04db141ce1fd1f155a7aac081cbd3ae246ae8dab` — exp04: hash evaluation data and revalidate mutable inputs (Codex, trailer gpt-6-astra)
- `e0633af8b16ccec0abc9c741d92e9832ebb70ede` — exp04: cache training IR inventories with stale-file refusal (Codex, trailer gpt-6-astra)
- `cc1ac1f963ce03dcb1296ad3f02b35947f6c22f9` — exp04: validate immutable evaluation manifests before model loading (Codex, trailer gpt-6-astra)
- `4a9220e77215963589f04a560fce53144ce6df1d` — exp04: add P-only evaluation with exact frozen-path parity (Codex, trailer gpt-6-astra)
- `d639d989b323f5b45a1e6d5fdc2381998529ed4e` — exp04: run manifest-bound sweeps and prove real GPU parity (Codex, trailer gpt-6-astra)
- `e061218b541e734c36297629011b38a8cd597ed4` — exp04: tighten evaluator manifest type and closure validation (Codex, trailer gpt-6-astra)
- `9027f311f765aff72973274bfdfc7a420837d1e4` — exp04: finalize evaluation only after child and log close (Codex, trailer gpt-6-astra)
- `fce00356d8cdbad4bf1b9c38fc352f25dc8512ef` — exp04: launch evaluations with reviewed closures and data identities (Codex, trailer gpt-6-astra)
- `38cf05be8b0898bafb13e6794306e30537b38bd2` — exp04: pin training argv and normalize runtime arguments (Codex, trailer gpt-6-astra)
- `3e5609c2f5d5e1c7accfb2fb0891760513458335` — exp04: guard resources and preserve accounted attempts (Codex, trailer gpt-6-astra)
- `d1550fd0849297574fd4d72a267d4192c83dbb8d` — exp04: expose no-save runtime args and pin counter products (Codex, trailer gpt-6-astra)
- `3e2df320069fe60d23111419c5b73afe5490cb58` — exp04: measure synchronized training throughput after warmup (Codex, trailer gpt-6-astra)
- `bbf8ad67032a4783c5dcf90d547b0caae4be0fd7` — exp04: bind training imports and split identity before spawn (Codex, trailer gpt-6-astra)
- `b7480bec6deba5b106f1437deb98e4618aa45761` — exp04: police live runtime args and banner through tee (Codex, trailer gpt-6-astra)
- `10ef6ff7b70118425862d6c1dac49b4973a7a30d` — exp04: finalize preserved training attempts after digest revalidation (Codex, trailer gpt-6-astra)
- `67c88198b536b8298563c83c249e518b20247722` — exp04: bind probe measurements and exercise launch refusals offline (Codex, trailer gpt-6-astra)
- `e8a164dffd50949ff0c19540742d0f4cc82a936e` — exp04: expose smoke probe full and refusal launch modes (Codex, trailer gpt-6-astra)
- `d79fa760b08b41295d47139c5e930db12d4bf794` — exp04: enforce control parity and account promotion failures once (Codex, trailer gpt-6-astra)
- `07f3bc91c7a8d6a6ce2dcaaef047d73f4578d4f8` — exp04: preserve redirected logs and enforce epoch-one acceptance promptly (Codex, trailer gpt-6-astra)
- `3e34a73caa8ef57accdebabe71126d2c2a7fca66` — exp04: make source closures hermetic and fail closed (Codex, trailer gpt-6-astra)
- `dca9dde7e670d39cdf2509d9a9592687be095e93` — exp04: validate evaluation outputs before finalization (Codex, trailer gpt-6-astra)
- `bd50f745f35ef33f8ed9fa55ce4b2764b1b9058c` — exp04: bind training inputs and enforce launch provenance gates (Codex, trailer gpt-6-astra)
- `eae21dc8ad543809f2cbe93db8a7b8aca3abc5e2` — exp04: define immutable analysis profiles and registered rules (Codex, trailer gpt-6-astra)
- `095c2d1e00462e879d24a8735b0a128fe9a3b904` — exp04: verify pinned checkpoint and reference identities (Codex, trailer gpt-6-astra)
- `a367d06d7924773fe6c6a37098b0376534d930ba` — exp_04 bookkeeping: golden argv files, rounds 3a/3b reviews, amendments A6-A8, Planner benchmark (Codex, trailer gpt-6-astra)
- `cebfe7547670ab53870adfc621303fdb3b9c293e` — exp04: implement exact bounds and all-seed cell masks (Codex, trailer gpt-6-astra)
- `045ec94be74252058985995b390a6fb9b99f4568` — exp04: reuse paired bootstraps with strict decisions and convergence (Codex, trailer gpt-6-astra)
- `0640bf2bf56651da520e76949fe6656215032eab` — exp04: pin the common content-hashed unseen inventory (Codex, trailer gpt-6-astra)
- `47d7c97602a70fa2d9119c412e0ab1f0ef670651` — exp04: add execution-bound synthetic comparison fixtures (Codex, trailer gpt-6-astra)
- `5acae13d6e47f5498eea217fdaf8cb40eb42deae` — exp04: specify CLI admission refusals and source identity checks (Codex, trailer gpt-6-astra)
- `ace93838e4489be8b585b6c3942abb2a25175d79` — exp04: cover profile outputs and fail-closed analysis boundaries (Codex, trailer gpt-6-astra)
- `21df7644f06c47e1de1ac5790d7ffbdf1ea2ee04` — exp04: enforce completion protocol and paired-run admission (Codex, trailer gpt-6-astra)
- `2e9f64c0df349b2800d1205999ea3f2d1a81dbc9` — exp04: produce profile-bound paired analyses and provenance (Codex, trailer gpt-6-astra)
- `b42b559f59e0176ce6ebf19f876ceac848dd5b5e` — exp04: cover authoritative pins and partial-output refusal (Codex, trailer gpt-6-astra)
- `9b47c7aff3c46c9cef22678fad428a2d68fcde84` — exp04: route termination signals through attempt cleanup (Codex, trailer gpt-6-astra)
- `5d04d0aee78402196aae3fff772ed94e947d708d` — exp04: bind probe admission and count only full training hours (Codex, trailer gpt-6-astra)
- `8a8e5156369d0f7b6050363df6c0e10866deed31` — exp04: preserve specific abort diagnostics and owned logs (Codex, trailer gpt-6-astra)
- `b48cd7b44b87ff0caec76f348a67e7a9f03b255d` — exp04: validate reviewed source blobs and record later drift (Codex, trailer gpt-6-astra)
- `5d6d4f9541db63dacede94659c3f5232a743145e` — exp04: retain recovery evidence and tolerate nonfinite abort state (Codex, trailer gpt-6-astra)
- `248688b0415b6a408f68b5d0d63094c69f5c2ce4` — exp04: share finalisation and verify process and log closure (Codex, trailer gpt-6-astra)
- `f5182245687442de68958ff49d15bd4cbe4b3fd6` — exp04: recover preserved attempts with explicit finalize mode (Codex, trailer gpt-6-astra)
- `32c3df56cdbb21b32c1b1d472b23d71d06317ec4` — exp04: retain probe spread and complete artifact inventories (Codex, trailer gpt-6-astra)
- `7306ff63d49f25e9ccb9c6ffbe101a4cc2abfca8` — exp04: isolate generated bytecode in untracked-state fixture (Codex, trailer gpt-6-astra)
- `90a6077b7d0bc642abd09ef7f0dc36c9df9cf191` — exp04: move training inventories to digest-bound sidecars (Codex, trailer gpt-6-astra)
- `4772e15e3ad7b7c58ecd9c078b808a189dfff790` — exp04: preserve recovery charges and require full-run bindings (Codex, trailer gpt-6-astra)
- `89521dfadba62728a580ed5c7d8d66e0d8a267d2` — exp04: classify malformed and missing runtime argument records (Codex, trailer gpt-6-astra)
- `30779cbab931bcc444988935cfaf18c93b00b1f5` — exp04: close evaluation admission and log ownership findings (Codex, trailer gpt-6-astra)
- `f7596c03d3a33fc1a4f7b5b9449f10923c428d13` — exp04: check recovery log writers with host-compatible fuser (Codex, trailer gpt-6-astra)
- `d750597b74975169be2e573690cdc97856de6873` — exp04: load committed runtime approval digests (Codex, trailer gpt-6-astra)
- `fcde6889c8253d451aebdb72a66d7b366eaa9047` — exp04: share grouped admission with runtime approval pins (Codex, trailer gpt-6-astra)
- `310604fb1a03d58bc289074b8c38165bac394cb6` — exp04: enforce confirmatory flags and named exclusions (Codex, trailer gpt-6-astra)
- `e9e2cdd62a36393bf92f70d100a82d6cac8f232a` — exp04: cache dataset hashes within each analysis (Codex, trailer gpt-6-astra)
- `59e62fabee06a4504c59ec9f9eb09bcefe699947` — exp04: aggregate admitted runs into six table rows (Codex, trailer gpt-6-astra)
- `a30d67d8e3e5040c463f520698e1507111cc9177` — exp04: publish canonical table JSON and Markdown exclusively (Codex, trailer gpt-6-astra)
- `e16f2f3990986047a1557f6852b4dd5ed50d0061` — exp04: name pending checkpoint pins and strengthen cache regression (Codex, trailer gpt-6-astra)
- `a2a0b8ba6430db0fc24b5d81d5aabed6a23f9472` — exp04: render estimand and provenance while preserving manual notes (Codex, trailer gpt-6-astra)
- `be5830d8e00fa7f1d74c50f6b3726ad4442e86c9` — exp04: preserve manual edits inside generated table blocks (Codex, trailer gpt-6-astra)
- `37b5f0d676c5911e33afa669d1244f7755ea0855` — exp_04 bookkeeping: rounds 3a/3b close-outs, rounds 4-5 reviews, prompts, amendments A9-A10, rung 4 audit log (Codex, trailer gpt-6-astra)
- `e3decd74b1918fe1f0549d433c11082eec31523c` — exp_04: harden termination and verify recovery execution evidence (Codex, trailer gpt-6-astra)
- `b777f9d1a6b0e3edd6aaa7790dad97b83ea08747` — exp_04: restore recovered attempt names and tighten launch records (Codex, trailer gpt-6-astra)
- `80f31461eef12d004b2dfc5f7b9389cf75e9eadb` — exp_04: fail closed on table edits and require augmented training provenance (Codex, trailer gpt-6-astra)

Round map: 3a = `9cd3703`…`fce0035` (provenance helper, `tools/exp04_eval.py`, evaluation launcher); 3b = `38cf05b`…`07f3bc9` (training launcher/probe); 3a-fix = `3e34a73`…`bd50f74`; 4 = `eae21dc`…`b42b559` (profiles, `paired_compare`); 3b-fix = `9b47c7a`…`f7596c0`; 5 = `d750597`…`be5830d` (A9, `results_table`); 6 = `e3decd7`…`80f3146` (pre-launch hardening). Planner bookkeeping commits: `ad19454`, `a367d06`, `37b5f0d`. Every Codex commit < 200 changed lines with the trailer; exp_03's pinned closure untouched (`git log 62c9107..80f3146 -- <12 files>` empty).
