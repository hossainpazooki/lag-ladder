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
