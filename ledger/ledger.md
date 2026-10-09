# Ledger — lag-ladder

House style, borrowed with provenance {sourceRepo: linear-ceiling, filePath: ledger/ledger.md,
commitSha: 888f745084c63eb52d114acd951dc787db82a71a} (which borrowed it from kv-transfer-replication
docs/ledger.md): hypotheses are pre-registered before any run; verdicts are stated against the rule
as written, before considering which outcome is more interesting to report; entries are numbered,
dated and immutable — an amendment is a new entry, never an edit; status tags are `[VALIDATED]`
(ran, and survived an independent attempt to refute it), `[BASELINE]` (ran; numbers here),
`[STRETCH]` (designed, not run), `[FUTURE]` (not designed), `[SUPERSEDED]`.

From entry 0002 on, every entry records `prior-entries-sha256:` over the entries section above it.
A verdict cell changes only with a machine-readable line `verdict: H-XX = <VERDICT>` in a numbered
entry. Both are recomputed by `ledger_check` in CI, which also refuses any edit to an entry block
already committed at the base revision (the trailing entry included).

## Hypotheses (pre-registered; verdict column is the only cell that ever changes, and only via a numbered entry)

| id | statement | decided by | verdict |
|---|---|---|---|

No hypothesis is registered yet. The first registration is the pilot's (design §9 step 4).

## Entries

### 0001 — 2026-09-22 — Repo created; chassis, pins and the 2026-09-22 rulings on record

**What this repo is.** The measurement repo for *Holdover* — a KV cache held over a weight update.
Writer θ_t, reader θ_{t+k}; every statistic is indexed by the rung k. Design:
`docs/2026-09-19-holdover-design.md` (moved here from linear-ceiling, where it was never committed),
which inherits linear-ceiling's `docs/2026-09-02-e-rl-design.md` at commit `888f745` for the
instrument, controls and dumps.

**Separate repo, by ruling.** linear-ceiling's chassis is COPIED, not shared, so a repo under
active amendment cannot move this one's gates: `hashing`, `rng`, `seal`, `upstream_gate` verbatim;
`config`, `ledger_check`, `lint_scope` adapted — what changed is stated in each module's docstring
and in `UPSTREAM.md`. Every copied module carries `{sourceRepo, filePath, commitSha}` there.

**Inherited definitions, cited not recomputed.** f*(τ) — the oracle selective-recompute fraction —
and τ_K = 0.3186 (= 1 − 0.6814, the archived k = 1 held-out R²) are linear-ceiling ledger entry 0023
(commit `19e91286f680eaacb462733a26d24a233a14fb69`), with the τ ladder {0.3186, 0.10, 0.03} from the
same entry's descriptive addition. This repo does not recalibrate τ; if it ever does, that is a new
entry here naming the new value and the reason.

**Upstreams.** Instrument: `kv-transfer-replication` pinned at
`063f4023fdde67dedbee01a92518ce7f83f6cf5d` (its HEAD today; the Pair revision field is not in it —
design §9 step 2 re-pins). Checkpoint producer: none pinned. `ServiceNow/PipelineRL` main was
observed at `58d393458625ad63ed539f2dcd072c85700c557f` today and is recorded so a later pin can
state what moved; it is not a dependency.

**Rulings of 2026-09-22 (operator; the picked option text is quoted in the design doc §10).**
R6 — Oct 12 (ARR October cycle) dropped; the pilot decides the venue. R7 — primary framing is
prefix-cache persistence: the lag of a prefix cache that is never invalidated is unbounded, so the
long OLMo tail is the engine-relevant result and the 8-shot condition is primary. R8 — OLMo-2-0425-1B-
RLVR1 (13 revisions, stride 200) is the only source until the pilot shows signal; the own run and its
producer are deferred. R9 — this scaffold. Open with recommendations on record: R1–R5 (design §10).

**Scope sentence** (held verbatim in README.md; `lint_scope` refuses a paraphrase anywhere else): The
ladder measures what a KV cache written under one policy checkpoint costs when read under a later
one, as a function of how many optimizer updates apart they are; it does not train on stale
rollouts and does not measure serving latency.

**Status of everything here:** `[FUTURE]` for the pilot, `[STRETCH]` for the design's statistics
(A)–(C). Nothing has run. `config/pilot.toml` carries the design's proposed values with
`registered_by = ""`; the registering entry fixes them.

### 0002 — 2026-09-24 — Ruling: kv-transfer-replication is the instrument for statistic (A) only

prior-entries-sha256: 51bd44723ec5e1e969990966c57d5c889925a3967a0307261383f58e159c7710

**Ruling (operator, 2026-09-24, verbatim):** "kv-transfer-replication stays the instrument for (A) only."

**What it fixes.** Entry 0001 pinned `kv-transfer-replication` as *the* instrument. This entry narrows
that: the upstream is invoked for statistic (A) — the dump writer (`scripts/dump_kv.py`,
`kvt/data.py::dump_kv`) and the per-token scorer (`scripts/score_positions.py --per-token`,
`kvt/pertoken.py`), so that f*(τ_K) is computed by the same bytes that defined it (linear-ceiling
entry 0023) — and for nothing else. Statistic (B)'s log-prob scorer and statistic (C)'s swap scorer
(generation under a supplied cache; design §3) are built in `src/lag_ladder/`, not upstream.

**What it does not change.** τ_K = 0.3186 and the τ ladder (0001); the read-only rule and the
ancestry pin (`UPSTREAM.md` §1); the one upstream change this lane still needs — `kvt/pairs.py::Pair`
gaining a revision / local-path field, plus the bridge items G1, G4, G5 of
`docs/2026-09-24-seed-bridge-pipelinerl-instrument.md` — each of which lands upstream by the
operator's commit and is re-pinned here by a numbered entry.

**Why.** The upstream is a cross-model KV-transfer replication; Holdover uses its dump writer and a
~100-line scorer. Keeping (A) there preserves "same bytes" comparability with Carryover; putting
Holdover-specific generation code there would make an unrelated repository carry async-RL logic
and widen every future re-pin. The alternative — porting the (A) instrument into this repo — was
declined for now; it may be reopened by a new entry if the instrument cannot run on the
operator's local machine (seed G4).

**Consequences recorded elsewhere in this commit set:** design doc §10 (R10) and §11; `UPSTREAM.md`
§1 scope line; `CLAUDE.md` rules; seed §1 precondition marked ruled.

### 0003 — 2026-09-30 — Re-pin: instrument at 9ca6258 (Pair revision/local_path, device(), --dtype, CI)

prior-entries-sha256: b5e0ecf0f2a7201665f4c86aa1eec37fdaeb2ac2ae187672548092424773bdcc

**What it fixes.** `INSTRUMENT_SHA` moves from `063f4023fdde67dedbee01a92518ce7f83f6cf5d` to
`9ca6258c91acf5c43b74026a7cc3649d61165104` — kv-transfer-replication `main` after the operator's
fast-forward of `holdover-instrument` (commits e54c102, 47bdd39, 8b72f19, 9ca6258). This is the
re-pin entry 0002 and `UPSTREAM.md` §1 reserved for "the entry that lands `kvt/pairs.py::Pair`'s
revision / local-path field" (design §9 step 2).

**What the pin now carries, on the (A) paths.** `Pair.with_revision` / `with_local_path` /
`resolve(which)`; `dump_kv` records `revision`, `local_path`, `load_dtype` in meta.json (the nine
prior keys unchanged); `kvt.models.device()` prefers cuda, mps, cpu with `KVT_DEVICE` override;
`--dtype` on the dump (K/V stay float16 on disk); `kvt-dump` / `kvt-score-positions` entry points;
`--probe`; a CI workflow. `scripts/score_positions.py` and `kvt/pertoken.py` are byte-identical to
063f402, so f*(τ_K) is still computed by the bytes linear-ceiling 0023 defined it with.

**Basis.** Upstream suite 165 passed at 8b72f19 and 9ca6258 (CI run 36094690258 green on 8b72f19;
local run 2026-09-27 and 2026-09-30); `git diff --stat 063f402 9ca6258 -- scripts/score_positions.py
kvt/pertoken.py` is empty (re-run 2026-09-30 by the appending session: empty; `origin/main` =
`origin/holdover-instrument` = 9ca6258; `git status --porcelain -- kvt scripts` empty; 9ca6258 is
docs-only over 8b72f19); e54c102 and 47bdd39 are individually broken trees (ImportError DTYPES),
documented in the upstream brief `docs/handoff/2026-09-25-holdover-instrument.md` — the pin is the
tip, never an intermediate.

**What it does not change.** τ_K = 0.3186 and the ladder (0001); R10 scope (0002); `config/pilot.toml`
stays UNREGISTERED with `upstream_sha = "UPSTREAM_SHA_PENDING"` — the registering entry fills it.
Not in this pin: kv-transfer-replication PR #2 (G3 checkpoint manifests, per-side `ModelRef`),
which conflicts with this branch in four files and awaits the operator's design ruling; it will be
a later entry.

**Consequences in this commit set:** `src/lag_ladder/__init__.py::INSTRUMENT_SHA`; `UPSTREAM.md` §1
pinned-commit line and its re-pin sentence; design §6 "fork" sentence amended per the 2026-09-25
checkpoint-config learning (handoff 2026-09-25 named this for the same commit).

### 0004 — 2026-10-08 — Re-pin: instrument at 0d27c68 (G3 checkpoint provenance, per-side ModelRef)

prior-entries-sha256: 88fea86e5df18f2bdeb550ef3ee665d31a40530b8b36a9eb0446ed33a3792c3c

**What it fixes.** `INSTRUMENT_SHA` moves from `9ca6258c91acf5c43b74026a7cc3649d61165104` to
`0d27c6856c7ed6e138bb0540f3874a7e5d37a0fb` — kv-transfer-replication `main` after the operator merged
PR #1 (`50983a2`, Llama 3 pair, 2026-09-30; touches `kvt/pairs.py` only) and PR #2 (`0d27c68`, G3
checkpoint provenance, 2026-10-01 04:00Z). This is the entry 0003 reserved for PR #2 ("it will be a
later entry"). The merge is the design ruling 0003 was waiting on: G3's per-side `ModelRef` replaces
task 1's `Pair.with_revision` / `with_local_path` / `resolve`.

**What the pin now carries, on the (A) paths.** Two of the four changed: `scripts/dump_kv.py` and
`kvt/data.py`. `Pair` gains per-side `source_revision` / `source_local_path` / `target_revision` /
`target_local_path` and `model_ref(which)`; `kvt-dump --revision` / `--local-path` apply to the
`--which` model only and are refused when the pair already pins that side. The dump hashes the
checkpoint before loading (`record_checkpoint`), loads exactly the hashed ref, re-verifies after the
dump (a changed checkpoint renames meta.json to `meta.json.INVALID`), writes `checkpoint_manifest.json`,
and records a `checkpoint` provenance block in meta.json; 0003's `revision` / `local_path` keys are
gone and `model` is the provenance's model_id. `KVDump.load` verifies provenance on every load and
raises `CheckpointMismatch` when the checkpoint is changed, missing or unlocatable; dumps without
provenance load as before. `scripts/score_positions.py` and `kvt/pertoken.py` are byte-identical to
063f402 and 9ca6258, so f*(τ_K) is still computed by the bytes linear-ceiling 0023 defined it with.

**Basis.** Upstream suite 191 passed at 0d27c68 (CI run 36813097382 green 2026-10-01; local run
2026-10-08 in the upstream's own venv); `git diff --stat 063f402 0d27c68 -- scripts/score_positions.py
kvt/pertoken.py` empty (re-run 2026-10-08); `origin/main` = local `main` = 0d27c68 and
`git status --porcelain -- kvt scripts` empty; the gate against the 9ca6258 pin said `REFUSED` on the
live tree 2026-10-08 — the gate working; the answer is this entry, not a wider path tuple.

**What it does not change.** τ_K = 0.3186 and the ladder (0001); R10 scope (0002); `config/pilot.toml`
stays UNREGISTERED with `upstream_sha = "UPSTREAM_SHA_PENDING"` — the registering entry fills it. A
lag-ladder driver that reads dumps must expect `CheckpointMismatch` and the `checkpoint` block in
place of `revision` / `local_path` (design work, not this entry).

**Consequences in this commit set:** `src/lag_ladder/__init__.py::INSTRUMENT_SHA`; `UPSTREAM.md` §1
pinned-commit line and its re-pin sentence; `UPSTREAM.md` §2 "fork" sentence aligned with design §6
(per-step checkpoints are config, learning 2026-09-25 — no fork; a producer pin records the author sha).

### 0005 — 2026-10-09 — Pilot registered: f*(τ_K) on the OLMo-2 RLVR1 ladder, statistic (A) only

prior-entries-sha256: 909907063ec5641b9ac2cdd1e675a99406705a69ef172e5e93222e0c48a25c0f

**Registers** `config/pilot.toml`, now `registered_by = "0005"`: source `allenai/OLMo-2-0425-1B-RLVR1`,
the thirteen `step_*` revisions at stride 200, anchors `step_200` and `step_1200`, held-out sequences
[40, 50) of the instrument's `data/tokens/olmo2-1b-rlvr1_n50_len1024_seed0.npy` (FineWeb-Edu
`sample-10BT`, streaming, shuffle seed 0, fifty sequences of 1,024 tokens, made by the instrument's
`prepare_tokens` on 2026-10-09; sha256 `462a7a743326085462a8fdf0f7a63d79045167e71b596230117066e7ac2eeeeb`;
the instrument ignores `data/`, so the file travels by hash and the driver refuses any other), forward
dtype float32, seed 0. Rule: median over held-out sequences of f*(τ_K = 0.3186) on the K read-out;
HOLDS ≤ 0.15, DEGRADES ≥ 0.50, UNRESOLVED between; τ ladder {0.3186, 0.10, 0.03} descriptive. τ_K and
f* are linear-ceiling 0023 (entry 0001), not recalibrated. Ruled 2026-10-09: dtype, grid, bounds, seal.

**Instrument.** Re-pinned at kv-transfer-replication `1380635da3b80cb5b8e669fab0eee87438abfa96` (`main`
after PR #3: the pair `olmo2-1b-rlvr1`, both sides the same model id, revisions pinned per dump through
`--revision`). The four (A) paths are byte-identical to `0d27c68` (0004): `git diff --stat 0d27c68
1380635 -- scripts/dump_kv.py kvt/data.py scripts/score_positions.py kvt/pertoken.py` is empty.
`INSTRUMENT_SHA`, `UPSTREAM.md` §1 and `pilot.toml.upstream_sha` agree.

**Construction** (`lag_ladder.pilot`). Per anchor a, later revision r and held-out sequence: two
stride-1 single-sequence prefill dumps of the same tokens (the pinned scorer reads one sequence at a
time), scored at every position with the writer's K as the candidate and the reader's own K as the
reference; f*(τ) from the per-token record (`lag_ladder.fstar_record`, which refuses unless the record
hashes, sums to the recorded SSE and reproduces 1 − R²). Per-sequence f*, then the median per lag.

**Controls, run before any rung; the run halts on failure.** Identity: an anchor read by itself must
give f*(τ_K) ≤ 0.0 on every sequence. Scrambled: an anchor's cache for one sequence read against the
same revision's cache for the next held-out sequence must give a median f*(τ_K) ≥ 0.90; below it no
ladder table is produced. Both bounds are stated judgment. Kernel identity on the run machine (Apple
M6, MPS against CPU, float32, 48 tokens, this pin's bytes): OLMo-2-1B `step_200` 1 − R² = 1.7e-9 on K,
per-token centered deviation ≤ 6.6e-9, f* = 0 at every rung; the on-disk float16 arrays differ by at
most one ulp; attention kernel `sdpa_repeat_kv` on both devices. Same on Qwen3-0.6B (1 − R² = 3.0e-9).
The model loads under the instrument: 16 layers, 16 KV heads, d_h 128, peak 5.99 GB at 48 tokens.

**Seal.** `ledger/predictions/olmo2-1b-rlvr1.json`, sidecar sha256 `31f4e5659db321dba1c4c157d29826a9c3056978de933da9320fa37d9a874bde`, written
before any dump: from anchor `step_200`, median f*(τ_K) ≤ 0.15 at lag 200 (HOLDS) and ≥ 0.50 at lag
2400 (DEGRADES).

**Not registered here.** Statistics (B) and (C); the second prompt set (G7: the pilot reads FineWeb-Edu
only); the own run (R8); the precision (G1) and origin (G2) controls; hypotheses, which the verdict entry
states against this rule once `summarize_pilot` has run.

**Consequences in this commit set:** `config/pilot.toml` (`registered_by`, `upstream_sha`,
`tokens_sha256`); `src/lag_ladder/__init__.py::INSTRUMENT_SHA`; `UPSTREAM.md` §1; the sealed prediction
and its sidecar; `tests/test_config.py` (the repo config is now registered).
