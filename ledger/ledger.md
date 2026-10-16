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

### 0006 — 2026-10-09 — Pilot outcome: f*(τ_K) is 0 at every lag to 2400 on OLMo-2 RLVR1; the cache HOLDS [BASELINE]

prior-entries-sha256: 7522fe8ef0b25aac3524b8ccb4c1026bc9cae356d508a2b034575f5421c073ad

**Ran** as registered by 0005: 230 jobs (130 dumps, 230 scores) on the run machine, 2026-10-09
16:28–17:02 UTC, instrument `1380635`, no halt, `run.json` complete; `summary.json` (sha256
`c8290607b615b6f0…`) recomputed from the score records on two machines with identical values on every
field the rule reads (`run.json` sha256 `3ae7d871f9174382…`).

**Controls passed on both anchors.** Identity: f*(τ_K) = 0 on all ten sequences (bound 0.0). Scrambled:
median f*(τ_K) = 0.999 on both anchors (bound 0.90) — the instrument sees the cache.

**Outcome against the rule** (median over held-out sequences [40, 50) of f*(τ_K = 0.3186), K read-out):

| anchor | lags | median f*(τ_K) | f*(0.10), f*(0.03) | band |
|---|---|---|---|---|
| step_200 | 200, 400, …, 2400 (12 rungs) | 0.0 at every rung | 0.0 at every rung | HOLDS at every rung |
| step_1200 | 200, 400, …, 1400 (7 rungs) | 0.0 at every rung | 0.0 at every rung | HOLDS at every rung |

Descriptive, in R²'s units (0023): the whole-cache deviation 1 − R²(K), median over sequences, rises
from 5.5e-5 at lag 200 to 7.3e-4 at lag 2200 from step_200 and flattens from lag ~1200; from step_1200 it
is 4.6e-5 at lag 200 and 3.1e-4 at lag 1400 — about two thirds of the first anchor's at the same lag.
The worst single sequence at any lag is 1.2e-2 (step_200 → step_2400), under the ladder's floor 0.03; the
worst layer is the last (1.2e-3 median at lag 2400) and deviation grows with depth; the largest
single-token deviation observed is 0.70 (one token of 10,240 at lag 2400). f* removes by the mean, and
the mean never reaches 0.03.

**Seal.** The sealed prediction (0005: HOLDS at lag 200, DEGRADES at lag 2400 from step_200) is **met at
lag 200 and falsified at lag 2400**; the outcome is the "holds throughout" shape.

**What the design's rule says** (§9 step 3, ruled 2026-09-22: "if f* ≈ 0 out to lag 2400: short note,
lane closed"): on this source, statistic (A) reads no cost of a stale K cache at any lag the published
ladder offers. Whether the lane closes, or moves to a source whose updates are larger (the RLVR
updates here move K by under a tenth of a percent of its variance), is the operator's ruling, not this
entry's. No hypothesis was registered for the pilot; nothing in the table changes. Status: [BASELINE] —
ran, numbers here, not refuted.

**Not measured.** Statistics (B) and (C); the V read-out as a verdict (recorded in every report, not
read by the rule); the second prompt set (G7); any lag below 200.

### 0007 — 2026-10-09 — Ruling: the ladder's axis is weight distance; the lag ladder is one source of it

prior-entries-sha256: 16ad43d9a4b9afdd8abc795734f7ca2b9e0616bd9fa831239c76baf9f9a7a589

**Ruled 2026-10-09, after 0006** ("re-aim at weight-distance"). The pilot read f*(τ_K) = 0 at every published
lag because the quantity the cache responds to is the distance between the writer's and the reader's
weights, and 2,400 RLVR optimizer steps on this model put almost none on the axis. Lag in optimizer steps
is not the quantity; relative weight distance is, and an RL run's lag is one way of producing it.

**What the lane measures from here** (the scope sentence is unchanged; "how many optimizer updates apart"
is read through the distance those updates produced): f*(τ) as a function of the relative L2 distance
between writer and reader over the **whole parameter vector** (R-D1; the K-projection distance is reported
beside it), with real training directions as labelled points and an isotropic-noise ladder as the
calibrated backbone — design §3's NULL arm, promoted from control to axis.

- **R-D2 — the backbone:** one checkpoint (OLMo-2 RLVR1 `step_200`) perturbed by seeded isotropic Gaussian
  noise at relative norms {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}, each written as a local checkpoint
  and read against the clean checkpoint, ten held-out sequences per rung.
- **R-D3 — the real points:** the Qwen2.5-1.5B family (base → Instruct, base → Math, Math → R1-Distill;
  same architecture, safetensors; Math and R1-Distill carry a different RoPE base, compared in content
  space as the instrument does) and the OLMo-2 1B chain SFT → DPO → RLVR1 (SFT and DPO ship `.bin` weights
  and enter through a converted local checkpoint whose provenance the registering entry records).
- **R-D5 — framing:** the async-RL framing stays. The first result is the pilot's: on a published RL
  ladder the cache never goes stale; the distance curve says why, and where it would.

**Measured today, descriptive, not of record:** relative weight distances on OLMo-2 1B — one RLVR rung
~1e-4, the whole RLVR ladder ~4e-4, SFT → DPO ~9e-4, and the public base checkpoint unrelated to the
post-training lineage (near-zero cosine on every matrix; it is excluded as a writer). These came from an
ad-hoc script over Hub files, not from a summarizer over `results/`, so they are not ledger numbers; the
registering entry's run re-derives every distance from the dumped checkpoints' manifests.

**Open before registration:** the sealed prediction for the backbone (the distance at which median
f*(τ_K) first leaves HOLDS) — R-D4; the rung list and controls for each pair; the conversion step's
provenance rule. Nothing is registered by this entry; `config/pilot.toml` (0005) stands as run.

### 0008 — 2026-10-09 — Weight-distance ladder registered: the isotropic backbone and six real rungs, statistic (A)

prior-entries-sha256: 7239bd4913c034602a9843993cfcf902b18ce2453315f03180c6e77f341e9d92

**Registers** `config/distance.toml`, now `registered_by = "0008"`. Rule: per rung, median over held-out
sequences [40, 50) of f*(τ_K = 0.3186) on the K read-out; HOLDS ≤ 0.15, DEGRADES ≥ 0.50, UNRESOLVED between;
τ ladder {0.3186, 0.10, 0.03} descriptive. The x-axis is the relative L2 distance between writer and reader
over the whole parameter vector, recomputed from the dumped checkpoints by `tools/weight_distance.py` and tied
to the dumps' checkpoint manifests by file hash; the K-projection distance is reported beside it (0007,
R-D1). Forward dtype float32, seed 0.

**Backbone (R-D2).** OLMo-2 RLVR1 `step_200` perturbed by seeded isotropic Gaussian noise at relative norms
{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} — `tools/perturb_checkpoint.py`: one standard-normal draw per
element, one global scale so the whole-vector relative norm equals the target, float32 output, provenance with
the realized norm — each read against the clean checkpoint. Seven rungs; a perturbed checkpoint's weights are
deleted after its last score, its provenance and manifest stay.

**Real rungs (R-D3).** OLMo-2 1B: SFT → DPO, DPO → RLVR1 `step_200`, SFT → RLVR1 `step_2600`; SFT and DPO
ship `pytorch_model.bin` and enter as local safetensors checkpoints written by `tools/convert_bin_checkpoint.py`
from Hub commits 0d85a3d03787 and c4b0485961ab (output sha256 e39d408cae53… and 70a636cfea16…), weights unchanged in their
stored dtype. Qwen2.5-1.5B: base → Instruct, base → Math, Math → R1-Distill (pairs `qwen2.5-1.5b-to-instruct`,
`qwen2.5-1.5b-to-math`, `qwen2.5-math-1.5b-to-r1-distill`; Math and R1-Distill carry `rope_theta` 1e4 against
the base's 1e6 and are compared in content space, as the instrument does). Six rungs. Held-out tokens: OLMo as
0005 (sha256 `462a7a74…`); Qwen `data/tokens/qwen2.5-1.5b-to-instruct_n50_len1024_seed0.npy` (FineWeb-Edu,
`prepare_tokens` seed 0, sha256 01bd7698f741…), shared by the three Qwen pairs.

**Instrument.** Re-pinned at kv-transfer-replication 5d4aa4a98b40f5ab37dd83ebb6b2ee802203994a (`main` after the three Qwen pairs merged); the
four (A) paths are byte-identical to `0d27c68` (0004).

**Controls**, per distinct writer checkpoint, before any rung that uses it; the run halts on failure: identity
f*(τ_K) ≤ 0.0 on every sequence; scrambled median f*(τ_K) ≥ 0.90. Six writers: the two converted OLMo
checkpoints, RLVR1 `step_200`, and the three Qwen writers.

**Seal.** `ledger/predictions/weight-distance.json`, sidecar sha256 `94a315dfddd4d8ad8c71215d88ffc3aacd2e9baf708e196241e6308019b642b8`,
written before any dump (R-D4, ruled 2026-10-09): the backbone's median f*(τ_K) stays within HOLDS through
1e-3, first exceeds 0.15 at 3e-3, and is DEGRADES by 1e-2; a real direction at matched distance reads a
higher median f*(τ_K) than the nearest isotropic rung.

**Not registered here.** Statistics (B) and (C); the own run (R8); any hypothesis in the table. The pilot's
registration (0005) stands as run.

### 0009 — 2026-10-09 — Weight-distance outcome: the backbone HOLDS through 3e-2 and DEGRADES at 1e-1; the sealed knee at 3e-3 is falsified; real directions tie it in f* [BASELINE]

prior-entries-sha256: 555bc9c46403fc5df18d49c25238869f722ec981fb524ef979b54c5dc181a52d

**Ran** as registered by 0008: 250 jobs on the run machine, 2026-10-09 19:32–20:20 UTC, instrument
`5d4aa4a`, no halt, `run.json` complete (sha256 `b398ce827090…`); `summary.json` (sha256 `53f65ab75363…`)
recomputed from the score and distance records on two machines with identical values on every field the rule
reads. Each rung's distance is tied by file hash to the checkpoint manifests of the two dumps it compares.

**Controls passed on all six writers.** Identity: f*(τ_K) = 0 on every sequence (bound 0.0). Scrambled:
median f*(τ_K) = 0.999 on every writer (bound 0.90).

**Outcome against the rule** (median over held-out sequences [40, 50) of f*(τ_K = 0.3186), K read-out; rungs
by relative L2 weight distance over the whole parameter vector, K-projection distance beside it; the median
1 − R²(K) is descriptive, in 0023's units):

| rung | kind | rel. distance, whole | k_proj | median f*(τ_K) | f*(0.10) | f*(0.03) | band | median 1 − R²(K) |
|---|---|---|---|---|---|---|---|---|
| `olmo-dpo-to-rlvr1-step_200` | real | 8.6e-05 | 6.1e-04 | 0.000 | 0.000 | 0.000 | HOLDS | 8.8e-05 |
| `noise-0.0001` | noise | 1.0e-04 | 5.4e-04 | 0.000 | 0.000 | 0.000 | HOLDS | 1.1e-06 |
| `noise-0.0003` | noise | 3.0e-04 | 1.6e-03 | 0.000 | 0.000 | 0.000 | HOLDS | 8.9e-06 |
| `olmo-sft-to-dpo` | real | 8.5e-04 | 4.9e-03 | 0.000 | 0.000 | 0.000 | HOLDS | 1.2e-02 |
| `olmo-sft-to-rlvr1-step_2600` | real | 9.4e-04 | 5.5e-03 | 0.000 | 0.000 | 0.000 | HOLDS | 1.2e-02 |
| `noise-0.001` | noise | 1.0e-03 | 5.4e-03 | 0.000 | 0.000 | 0.000 | HOLDS | 9.7e-05 |
| `noise-0.003` | noise | 3.0e-03 | 1.6e-02 | 0.000 | 0.000 | 0.000 | HOLDS | 8.7e-04 |
| `noise-0.01` | noise | 1.0e-02 | 5.4e-02 | 0.000 | 0.000 | 0.000 | HOLDS | 9.7e-03 |
| `qwen-base-to-instruct` | real | 1.0e-02 | 9.9e-04 | 0.000 | 0.000 | 0.000 | HOLDS | 1.1e-02 |
| `noise-0.03` | noise | 3.0e-02 | 1.6e-01 | 0.000 | 0.000 | 0.998 | HOLDS | 9.1e-02 |
| `noise-0.1` | noise | 1.0e-01 | 5.4e-01 | 0.999 | 0.999 | 1.000 | DEGRADES | 1.7e+00 |
| `qwen-math-to-r1-distill` | real | 1.2e-01 | 1.5e-02 | 0.993 | 0.999 | 1.000 | DEGRADES | 5.1e-01 |
| `qwen-base-to-math` | real | 1.1e+00 | 4.4e-01 | 1.000 | 1.000 | 1.000 | DEGRADES | 5.2e+01 |

**Backbone.** Seven rungs, realized relative norms equal to their targets to three figures. Median f*(τ_K)
= 0 through 3e-2 (HOLDS) and 0.999 at 1e-1 (DEGRADES); no rung reads UNRESOLVED, so at τ_K the knee lies
between 3e-2 and 1e-1 and the backbone resolves it no finer. At τ = 0.03 it lies between 1e-2 and 3e-2
(f*(0.03) = 0.998 at 3e-2). The whole-cache deviation grows as the square of the distance: 1.1e-06 at
1e-4, 9.7e-05 at 1e-3, 9.7e-03 at 1e-2, 1.7e+00 at 1e-1.

**Real directions.** By the rule's statistic, every real rung inside the backbone's HOLDS range ties its
nearest isotropic rung at f*(τ_K) = 0, and the one real rung with an isotropic neighbour in the DEGRADES
range reads below it (Qwen Math → R1-Distill 0.993 at 1.2e-01 against 0.999 at 1e-1). Qwen base → Math
sits at 1.1e+00, ten times past the backbone's last rung and with no neighbour: at this measure Math is
not a fine-tune of the base. Descriptively, in 1 − R²(K) at matched distance, the three OLMo-2 directions
sit 78–130 times above the backbone (SFT → DPO 1.2e-02 against 9.7e-05 at 1e-3), Qwen base → Instruct sits
on it (1.1e-02 against 9.7e-03), and Math → R1-Distill sits at a third of it (5.1e-01 against 1.7e+00).
Distance alone does not set the cache's deviation; the direction does, in either sense.

**Seal.** The sealed prediction (0008: HOLDS through 1e-3, first above 0.15 at 3e-3, DEGRADES by 1e-2; a
real direction at matched distance above the nearest isotropic rung): **met on the first clause (HOLDS
through 1e-3) and falsified on the other three.** The backbone holds more than an order of magnitude past
the sealed knee, and no real direction reads above the backbone by the rule's statistic; Qwen base → Math
reads 1.000 against the last backbone rung's 0.999, but at ten times its distance, which is not a matched
comparison.

**Read with.** f* at τ_K steps from 0 to 0.999 between two adjacent rungs, so the ladder's band is a step
function of distance and the real rungs' ties are ties at zero. The noise checkpoints were written in
float32 and pruned after their last score; their provenance (source commit, seed, realized norm, output
sha256) and the dumps' manifests stay under `results/distance/checkpoints/` and `pairs/`. A first launch
of the run halted before any job was recorded, on a driver fault (the instrument was not told where local
checkpoints live) fixed in `632f914`; the run on record is the second launch.

**Not measured.** Statistics (B) and (C); the V read-out as a verdict; any own run (R8); any hypothesis in
the table. Status: [BASELINE] — ran, numbers here, not refuted.

### 0010 — 2026-10-10 — Ruling: statistic (A) is complete under the inherited instrument; the statement of record, and how to cite it

prior-entries-sha256: da6b6b2a7c8813b693380a5ac13efc0002655e3707339e765e113b59dc1399cd

**Ruled 2026-10-10, after 0009.** The instrument this repository inherited from linear-ceiling (entry 0023:
f*(τ), τ_K = 0.3186, the K read-out, HOLDS ≤ 0.15 / DEGRADES ≥ 0.50; see 0001) has now been run on both
ladders this repository was built for — the lag ladder (0005, 0006) and the weight-distance ladder (0008,
0009) — on the same model, held-out set and dtype, with the same controls. Design §9 step 3's rule ("if f* ≈
0 out to lag 2400: short note, lane closed") has fired, and the distance ladder has located the regime in
which it would not have. No further run of statistic (A) is registered. The linear-ceiling ledger is frozen;
this repository is the record that paper cites for the behaviour of its statistic under a weight change.

**Statement of record** (every number from 0006 and 0009; `summary.json` hashes named there). On OLMo-2
1B, the median over held-out sequences of f*(τ_K) on the K cache is **0 for every writer–reader pair whose
whole-vector relative L2 weight distance is at most 3e-2**: the seven isotropic rungs from 1e-4 to 3e-2;
every published RLVR step from 200 to 2400 apart, from two anchors; every post-training direction measured
on the chain (SFT → DPO at 8.5e-4, DPO → RLVR at 8.6e-5, SFT → RLVR at 9.4e-4); and Qwen2.5-1.5B base →
Instruct at 1.0e-2. It is 0.999 at the isotropic rung 1e-1, and the two Qwen pairs beyond that distance
(Math → R1-Distill at 1.2e-1, base → Math at 1.1) read 0.993 and 1.000. The same holds at τ = 0.10; at
τ = 0.03 the isotropic rung 3e-2 reads 0.998 and everything below it 0. Controls held on every writer:
identity 0, scrambled 0.999.

**What the statistic can and cannot say here.** f*(τ_K) is a step function of distance on this model: it
reads 0 until the whole-cache deviation 1 − R²(K) is of order one and 0.999 one rung later. It therefore
answers "does a cache written under one checkpoint need recomputation under another?" with *no* for every
pair a training run produces, and it does not resolve differences inside that regime — the 78–130-fold
excess deviation of the OLMo directions over isotropic noise at matched distance (0009) is visible in
1 − R² and invisible to f*. Any claim about what the reader *does* with such a cache is outside (A).

**Not ruled here.** Whether the lane continues under statistic (B) or (C) (design §4; never built), or
the own run (R8); the sources and bounds any such continuation would register. The configurations of 0005
and 0008 stand as run; the instrument pin stays `5d4aa4a` (0008).

**Citing.** Cite this repository at the commit that carries this entry; name entries 0006 and 0009 for the
numbers and 0001 for the statistic's provenance. The score records, per-token files and distance records
behind both summaries are retained off-repository with the hashes the entries state.

### 0011 — 2026-10-10 — Ruling: the lane continues under statistic (B), "does the reader notice", defined for this repository; the run moves to a rented GPU

prior-entries-sha256: 77428a8fe762180f8fc2bb2b098cf9293581acaab6ca4dc073dfee87d69a0cb8

**Ruled 2026-10-10, after 0010** (operator; the 09-02 promotion of (B) in linear-ceiling is not binding
here). (A) is a step function of distance (0010). What it cannot read is what the reader does with a cache
that a trained direction has moved a hundredfold more than isotropic noise of the same size (0009). (B)
asks that directly, on the dumps 0008 produced, with no new instrument: the scorer is built here, as 0002
ruled. The full design, with every ruling, is design §13.

**Definition.** For a held-out sequence the first 768 positions are the prefix P and the remaining 256 the
continuation. The reader is run on the continuation twice: over its own cache of P (FRESH) and over the
writer's (STALE), both taken from the float16 dumps of 0008 and injected as the instrument injects them.
Two continuation texts: the corpus text itself, and the reader's own greedy continuation generated under
the STALE cache, the trainer's scenario. Verdict statistic (B1): mean per-token KL(fresh ‖ stale) over the
continuation, median over the held-out sequences, per rung and per text. Descriptive beside it: the
per-token p90 and maximum, the importance-weight dispersion (fresh over stale; ESS over sequences and over
tokens, on text that is a sample of neither policy), and argmax agreement. Bands: HOLDS_B ≤ 1e-3 nats per
token, DEGRADES_B ≥ 1e-1, UNRESOLVED between; ladder {1e-1, 1e-2, 1e-3} descriptive. HOLDS_B is confirmed
or lowered at registration from a float16-floor probe on one reader, run on the run machine and recorded
in the registering entry.

**Controls, per reader, before its rung; the run halts on failure.** The float16 floor (the reader's live
cache against its own float16 dump, both read on the continuation) is the halting control, its limits fixed
from the probe. Identity (the same dump in both arms) is a determinism check against a stated tolerance on
the maximum logit difference. Scrambled (another sequence's prefix) must read DEGRADES_B, or no rung of
that reader is read.

**Sources.** The thirteen rungs of 0008 exactly as dumped, the noise readers regenerated from their seeds
and verified against the recorded checkpoint hashes; Qwen base → Math with the writer's cache re-rotated
to the reader's RoPE base (the one pair whose bases differ; Math → R1-Distill shares its base and injects
as is); one Qwen2.5-1.5B isotropic rung at 1e-2, so the Qwen comparison has a backbone of its own model;
and the pilot's lag ladder (0005) re-dumped at the current pin.

**Seal, in structure; values fixed when the seal is committed, before any score.** (i) On the backbone,
(B1) grows as the square of whole-vector distance over the rungs 1e-3 to 1e-2. (ii) The backbone leaves
HOLDS_B at or below 1e-2. (iii) At the nearest backbone rung on whole-vector distance, each OLMo-2 direction
reads at least 10× the backbone. (iv) Qwen base → Instruct reads within 3× of the Qwen rung. (v) On every
rung above the floor, the on-path reading is at or below the corpus reading.

**Where it runs.** On a rented GPU (RunPod) under linear-ceiling's GPU protocol, float32 as the dumps were
made. The dumps and the converted checkpoints travel from the machine that ran 0008 by a sha-verified
private dataset, and every file is checked against 0008's manifests before any forward. The instrument
stays pinned at `5d4aa4a`.

**Not registered by this entry.** The build (`tools/score_continuation.py`, `src/lag_ladder/reader.py`,
`src/lag_ladder/summarize_reader.py`, `config/reader.toml`), the probe, the seal's values and the
registration belong to the next entry. The configurations of 0005 and 0008 stand as run.

### 0012 — 2026-10-13 — Statistic (B) registered: float16 floor probed, HOLDS_B confirmed at 1e-3; the noise backbone is re-dumped on the run machine

prior-entries-sha256: bd75c57af252f2873490023b1659b3c2f5894ecadaf8ff6c1407dad14b36ac42

**Registers** `config/reader.toml`, now `registered_by = "0012"`, as 0011 and design §13 define statistic (B).
Rule: per rung and text, median over held-out sequences [40, 50) of the mean per-token KL(π_fresh ‖ π_stale) over
the 256-token continuation after a 768-token prefix; HOLDS_B ≤ 1e-3, DEGRADES_B ≥ 1e-1 nats/token, UNRESOLVED
between; ladder {1e-1, 1e-2, 1e-3} descriptive. Two texts: corpus, and on-path generated greedily under the STALE
cache. Forward dtype float32, seed 0. Instrument `5d4aa4a` (0008); the scorer runs under its interpreter and
imports nothing from it. The reader is handed cache rows 0..766 and fed the token at 767 ahead of the continuation,
so all 256 continuation tokens are scored.

**Float16 floor (R-B9, R-B10)**, probed on the run machine (torch 2.14.1+cu130) on reader `step_200`, live float32
cache against its own float16 dump, ten sequences, both texts: median mean-KL 3.9e-8 (corpus), 2.6e-8 (on-path);
maximum per-token KL 1.6e-6; maximum absolute logit difference 1.31e-2. HOLDS_B stays at 1e-3. Halting limits at
1.7× the observed maxima: `floor_max_logit_diff = 2.2e-2`, `floor_max_median_kl = 7e-8`; within 2× of the floor is
"at the floor". Identity tolerance 1e-5 on the maximum logit difference; scrambled must read ≥ 1e-1 or the reader's
rungs are UNEVALUABLE.

**Sources.** The thirteen rungs of 0008, dumps as transferred (dataset revision `59feb035`, manifest sha256
`716585c346ddea40…`, every file verified against it before any forward). Regenerating `noise-0.01` on the run
machine from the same anchor (`model.safetensors` `3dc71849…`), seed and target reproduced the realized norm to
4e-10 but not the bytes (`out_sha256` `6acca7d5…` against 0008's `c711d73f…`; `torch.randn` differs between
arm64 torch 2.14.1 and x86_64 torch 2.14.1+cu130), so 0008's noise dumps cannot be loaded there. The seven noise
rungs are therefore regenerated and dumped on the run machine before their rungs, under the same keys, each tied
to 0008's rung by `realized_rel_norm`; R-B1's hash check is against the run machine's provenance. Added (R-B8):
one isotropic rung of `Qwen/Qwen2.5-1.5B` at whole-vector 1e-2, made the same way. Deferred: R-B4, the pilot's lag
ladder re-dumped at the current pin, to a later entry. Hub readers are pinned at the commits the 0008 dumps
resolved. Qwen base → Math re-rotates the writer's K (θ 1e6 → 1e4) with the instrument's convention.

**Seal.** `ledger/predictions/reader.json`, sidecar sha256 `ba6e3cdce1f0f9cd7d8244e6a42ae5095e50228cfb634af2ed64ccce1b2690e8`,
committed before any score (6bbd9bf): (i) on the backbone over {1e-3, 3e-3, 1e-2} the slope of log median (B1)
on log distance lies in [1.7, 2.3]; (ii) the backbone's median (B1) is ≤ 1e-3 at 1e-2; (iii) at the nearest
backbone rung each OLMo-2 direction reads ≥ 10× the backbone; (iv) Qwen base → Instruct reads within 3× of the
Qwen isotropic rung; (v) on every rung whose corpus (B1) is above 2× the floor, on-path median (B1) ≤ corpus
median (B1).

The summarizer recomputes every statistic from the per-token arrays and refuses on mismatch; its integrity check
bottoms at the driver's run record, as 0009's does. The full-vocabulary KL is recorded at the forward, not
recomputable at home.

**Not registered here.** Statistic (C); the own run (R8); R-B4; any hypothesis in the table. 0005 and 0008 stand as
run.

**Consequences in this commit set:** `config/reader.toml` (`registered_by`, floor limits, pinned readers, the R-B8
rung); the sealed prediction and sidecar (6bbd9bf).

### 0013 — 2026-10-13 — Amendment: the float16 floor halts on median KL only; the run under 0012 halted on a second reader's logit extreme and is restarted

prior-entries-sha256: ef9cb13eb0efc2af55eb83e629c048db7c1e3f2cafe40b323d5b26e4a2c7d7d1

**Amends 0012** (R-B10). `config/reader.toml` is now `registered_by = "0013"`; the sealed prediction
(`ledger/predictions/reader.json`, sidecar `ba6e3cdc…`, 6bbd9bf) is unchanged and still binds. The run was
started under 0012 on the run machine and halted at its 74th job, in the DPO reader's float16-floor control:
one sequence (on-path text) read a maximum absolute logit difference of 5.2e-2 against the registered limit
2.2e-2, at a floor mean-KL of 1.6e-8. The limit had been set at 1.7× one reader's maximum over ten sequences
(`step_200`, 1.31e-2); the maximum over 256 positions and the vocabulary of a float16 rounding difference is a
heavy-tailed extreme that a one-reader probe does not bound across readers, and it carries no information at the
scale of the verdict (the DPO reader's rung reads logit differences of 0.3–0.7 at a mean-KL of 2e-4).

**Ruled.** The float16 floor halts on the median mean-KL alone, limit `floor_max_median_kl = 7e-8` as 0012 fixed it;
the per-reader maximum absolute logit difference is recorded and reported, not a limit. Identity keeps its 1e-5
tolerance; scrambled its 1e-1. `reader.py` and `summarize_reader.py` enforce the amended control. The 74 jobs run
under 0012 are discarded and the run restarted from its first job under this entry; nothing from them enters the
record. What they showed, descriptively: `step_200`'s controls reproduced the probe to the digit (identity 0.0;
scrambled median KL 0.97), and the DPO reader's floor read median mean-KL 3.1e-8 (corpus).

**Also recorded.** The eight isotropic checkpoints regenerated on the run machine (0012): `out_sha256` prefixes
noise-0.0001 `551222c9`, 0.0003 `f9d1b9b6`, 0.001 `0b86f898`, 0.003 `fa3187eb`, 0.01 `6acca7d5`, 0.03 `711d2117`,
0.1 `3ebf590a`, qwen-noise-0.01 `fd23b4a1`; realized relative norms agree with 0008's to better than 1e-9. The
0.01 regeneration reproduced the probe sitting's bytes on a different instance: the draw is fixed by platform and
torch build, not by the machine.

**Not registered here.** Nothing beyond the control amendment; 0012's rule, sources, seal and bands stand.

**Consequences in this commit set:** `config/reader.toml` (`registered_by`, the floor limit block);
`src/lag_ladder/reader.py`, `src/lag_ladder/summarize_reader.py`, `src/lag_ladder/config.py` (the amended
control); `tests/test_reader.py`.

### 0014 — 2026-10-13 — Amendment: the float16-floor limit is band-relative, 1e-4 nats/token; the R-B4 lag rungs registered

prior-entries-sha256: 411ea1d87168993a049ede834cd30710266aa8c42dac82543b42715aeb476b22

**Amends 0013** (R-B10). `config/reader.toml` is now `registered_by = "0014"`; the seal (`ba6e3cdc…`, 6bbd9bf) is
unchanged. The run restarted under 0013 halted at its 310th job in the DeepSeek-R1-Distill-Qwen-1.5B reader's
float16-floor control: median mean-KL 3.6e-5 against the limit 7e-8, with the five readers before it at 1e-8 to
6e-8. The floor is the reader's — its logits are sharper, and float16 rounding of the same cache moves them a
thousandfold more — and a limit set at 1.7× one reader's probe measures that reader, not the band.

**Ruled.** `floor_max_median_kl = 1e-4`, one order of magnitude under HOLDS_B: a reader whose float16 floor reaches
a tenth of the HOLDS band is not read. Per-reader floors are reported; a rung within 2× of its reader's floor is "at
the floor" (§13). The run restarts from its first job under this entry; the 310 jobs under 0013 are discarded.
Descriptively, they read on-path (B1) ≤ corpus (B1) on all five rungs scored, and the Qwen base → Math cell above
the Math reader's scrambled control — the restart's reports are read against that.

**Also registered (R-B4, deferred by 0012).** Five rungs on the RL phase: writer cache at RLVR1 `step_200`, reader
at `step_400`, `step_800`, `step_1200`, `step_1600`, `step_2600` (lags 200–2400 optimizer steps), Hub readers pinned
by commit, reader dumps made on the run machine from the Hub checkpoints. Read against the bands, descriptively; not
in the seal. The Hub's checkpoint stride, 200 steps, is far above a pipelined trainer's operating lag of a few steps,
so a reading here bounds that regime rather than measuring it; the own run (R8), approved in principle 2026-10-13,
is designed and registered by later entries.

**Not registered here.** The own run; statistic (C). 0012's rule, sources, seal and bands stand.

**Consequences in this commit set:** `config/reader.toml` (`registered_by`, the floor limit, five rungs);
`tests/test_reader.py`.

### 0015 — 2026-10-14 — Outcome: statistic (B) on the weight-distance ladder; of the sealed clauses (i) and (iii) HELD, (ii), (iv) and (v) FALSIFIED

prior-entries-sha256: 9c4e6a046d0843bcb427d1d9bf0555086d4de03054b3a291c3ba6f87a4d7e7b1

**Run.** The 1,100 jobs of 0014 (`config/reader.toml`, instrument `5d4aa4a`), on one A100-SXM4-80GB under torch
2.14.1+cu130, completed without a halt. Every reader's three controls passed: float16 floors 1.3e-8 to 1.4e-7
nats/token (limit 1e-4), identity 0 on every reader, scrambled medians 0.47 to 1.9 (limit 0.1). The run machine's
eight regenerations are byte-identical to 0013's. Records: `reader-results-c2.tgz` `828a545b…` (the reports and
`run.json`); `summary.json` `5d761594…`, written by `summarize_reader`, which recomputed every number below from the
per-token arrays and refuses on any mismatch; both retained off-repository.

**Reading** — median (B1), nats/token, corpus / on-path, at whole-vector relative distance x; bands HOLDS_B ≤ 1e-3,
DEGRADES_B ≥ 1e-1. *Isotropic backbone* (OLMo-2 1B RLVR1 step_200): 1e-4 → 6.0e-7 / 4.5e-7; 3e-4 → 4.9e-6 / 3.7e-6;
1e-3 → 5.3e-5 / 4.4e-5; 3e-3 → 4.8e-4 / 3.9e-4; 1e-2 → 5.6e-3 / 4.4e-3; 3e-2 → 6.6e-2 / 5.4e-2; 1e-1 → 4.1 / 4.3.
HOLDS_B through 3e-3, UNRESOLVED at 1e-2 and 3e-2, DEGRADES_B at 1e-1; the slope of log median on log x over the
sealed rungs {1e-3, 3e-3, 1e-2} is 2.02 (corpus) and 2.00 (on-path). *Post-training directions*: SFT → DPO
(x 8.5e-4) 4.1e-2 / 3.9e-2 and SFT → RLVR step_2600 (9.4e-4) 3.8e-2 / 3.5e-2, both UNRESOLVED; DPO → RLVR step_200
(8.6e-5) 2.1e-4 / 1.9e-4, HOLDS_B. *RL phase* (writer step_200, R-B4): lag 200 (x 1.3e-4) 8.4e-5 / 7.4e-5; 600
(2.0e-4) 3.2e-4 / 2.4e-4; 1000 (2.6e-4) 6.5e-4 / 5.0e-4; 1400 (3.0e-4) 6.0e-4 / 5.5e-4; 2400 (4.0e-4) 8.1e-4 /
7.0e-4 — HOLDS_B at every lag, rising with distance, between one and two orders of magnitude above the backbone
rung nearest it. *Qwen2.5-1.5B*: isotropic 1e-2 (R-B8) 3.3e-4 / 2.1e-4, HOLDS_B; base → Instruct (1.0e-2) 3.5e-3 / 2.3e-3,
UNRESOLVED; Math → R1-Distill (1.2e-1) 2.6e-1 / 1.0e-1, DEGRADES_B / UNRESOLVED; base → Math (1.1, cache re-rotated
θ 1e6 → 1e4) 1.26 / 0.31, DEGRADES_B, above the Math reader's scrambled control (0.47) on corpus text. No rung sits
within 2× its reader's floor.

**The seal (6bbd9bf), read by the summarizer against the recomputed statistics.** (i) HELD: slopes 2.02 and 2.00
lie in [1.7, 2.3]. (ii) FALSIFIED: at x = 1e-2 the backbone reads 5.6e-3 / 4.4e-3, above 1e-3; HOLDS_B is left
between 3e-3 and 1e-2. (iii) HELD: at the nearest backbone rung the three OLMo directions read 355× to 875× the
backbone, against the 10× sealed. (iv) FALSIFIED: base → Instruct reads 10.6× / 11.2× the Qwen isotropic rung,
outside 3× either side. (v) FALSIFIED by one rung of nineteen: at noise-0.1 on-path (4.3) exceeds corpus (4.1) — a
reader whose argmax agrees with its own fresh cache on 13% / 10% of tokens, generating under the stale one; the
other eighteen rungs above 2× floor read on-path below corpus by 5–76%.

**What the reading says.** Statistic (B) is graded where (A) was a step: it grows as the square of distance over
three decades of the backbone, and its HOLDS_B edge (3e-3 < x < 1e-2) lies an order of magnitude inside (A)'s
(3e-2 ≤ x < 1e-1, entry 0009). Trained directions are not isotropic to the reader: at matched whole-vector distance
the OLMo-2 post-training steps read two to three orders of magnitude above noise and Qwen's instruct tuning one
order — the excess 0009 saw in 1 − R² and (A) could not read. On the published RL chain a cache 200–2400 optimizer
steps stale stays HOLDS_B at every lag measured; the operating lag of a pipelined trainer (a few steps) lies below
the Hub's stride and is the own run's question (R8).

**Also recorded.** The DeepSeek-R1-Distill reader's float16 floor is 1.4e-7 here against 3.6e-5 in 0014's halted
run — same dumps, same torch build, different host; a reader-and-host property, read against a limit set at the band
(0014), moving no rung. `summarize_reader` had carried each rung's position from a `run.json` field the driver never
wrote, so the slope and backbone matching were empty in every production summary until now; it reads the position
from the distance ladder's record (`results/distance/rungs/<label>/distance.json`, or a local noise checkpoint's
`provenance.json`; the five lag distances and qwen-noise-0.01's provenance were written on the run machine, hashes in
the records above),
resolves run-machine paths by trailing components under the pinned instrument with the recorded sha256 still
enforced, and reads the sealed clauses out itself.

**Not ruled here.** Whether (B) is closed and how it is cited (the analogue of 0010); the own run (R8); statistic
(C). 0012's rule, sources, bands and the configuration of 0014 stand as run.

**Consequences in this commit set:** `src/lag_ladder/summarize_reader.py`, `tests/test_summarize_reader.py`,
`README.md`, `CLAUDE.md`.

### 0016 — 2026-10-14 — Ruling: statistic (B) on the weight-distance ladder is complete; the statement of record, and how to cite it

prior-entries-sha256: ad6d936b07b8ee3c7e58520598352b03643e8c9c03e76e67d95b25f2e4dadf35

**Ruled 2026-10-14, after 0015.** Statistic (B), defined by 0011 and design §13 and registered by 0012–0014, has run
on every rung the registration named — the thirteen rungs of 0008, the Qwen2.5-1.5B isotropic rung, five lags of the
published RLVR1 chain — with the controls on all eighteen readers passed and the seal of 6bbd9bf read out (0015). No
further run of (B) on published checkpoints is registered. The scorer and driver built here remain the registered
instrument for any later rung of the same statistic; the pin stays `5d4aa4a`.

**Statement of record** (every number from 0015; `summary.json` `5d761594…`, reports `828a545b…`). On OLMo-2 1B, the
median over held-out sequences of the mean per-token KL between the reader's next-token distribution over its own
cache and over the writer's (B1), in nats per token, **grows as the square of whole-vector relative weight distance
on the isotropic backbone** — slope 2.02 on corpus text and 2.00 on the reader's own greedy continuation over the
rungs 1e-3 to 1e-2 — from 6.0e-7 at distance 1e-4 to 5.3e-5 at 1e-3, 5.6e-3 at 1e-2 and 4.1 at 1e-1. It is
**HOLDS_B (≤ 1e-3) through 3e-3 and DEGRADES_B (≥ 1e-1) at 1e-1**, the edge of HOLDS_B between 3e-3 and 1e-2, one
decade inside statistic (A)'s (3e-2 to 1e-1, entry 0010). **Trained directions are not isotropic to the reader**: at
matched whole-vector distance the OLMo-2 post-training steps read 355× to 875× the backbone over both texts (on corpus
text SFT → DPO 4.1e-2 at 8.5e-4; SFT → RLVR 3.8e-2 at 9.4e-4; DPO → RLVR 2.1e-4 at 8.6e-5) and Qwen2.5-1.5B base →
Instruct 11× (3.5e-3 against 3.3e-4 at 1e-2). **On the published RL chain** a cache written at step_200 and read 200
to 2400 optimizer steps later reads 8.4e-5 to 8.1e-4 on corpus text and 7.4e-5 to 7.0e-4 on the reader's own
continuation, HOLDS_B at every lag; it rises with distance on the reader's own continuation, while on corpus text lag
1400 reads below lag 1000 (0015's "rising with distance" holds on the former only). The two Qwen pairs beyond distance
1e-1 read 2.6e-1 (Math → R1-Distill) and 1.26 (base → Math, the cache re-rotated across RoPE bases), the latter above
the reader's scrambled control. On eighteen of nineteen rungs the reader's own greedy continuation reads below the
corpus text, by 5–76%; the exception is the isotropic rung at 1e-1, a reader whose argmax agrees with its own cache
on 13% of tokens. Controls on all eighteen readers: float16 floors 1.3e-8 to 1.4e-7, identity 0, scrambled 0.47 to
1.9.

**What the statistic can and cannot say here.** (B) is graded where (A) was a step: it resolves the regime (A) read
as uniformly HOLDS, orders the trained directions above the isotropic backbone, and places the distance past which a
stale cache moves a reader's next-token distribution by more than 1e-3 nats. It is a reader's number, not a trainer's:
the on-path text is a greedy continuation under the stale cache, a sample of neither policy, so (B2) and (B3) are
dispersion diagnostics, not the importance weights a pipelined trainer would see. The smallest lag measured is 200
optimizer steps at distance 1.3e-4; the operating lag of a pipelined trainer, a few steps, lies below the Hub's stride
and is unmeasured here. Two of the five sealed clauses held; the three that fell — the backbone's HOLDS_B reach, Qwen
Instruct within 3× of noise, on-path ≤ corpus without exception — fell toward a more sensitive statistic than sealed.

**Not ruled here.** The own run (R8): a short RL run with the cache dumped every step, lags 1 to 32, on-path text from
the trainer's own rollouts — approved in principle 2026-10-13, to be designed as §14 and registered with its own seal.
Statistic (C). The configurations of 0012–0014 stand as run.

**Citing.** Cite this repository at the commit that carries this entry (tag `statistic-b-closed`); name entry 0015 for
the numbers, 0011 and design §13 for the statistic, and 0010 for statistic (A) beside it. The reports and summary
behind 0015 are retained off-repository with the hashes 0015 states.

**Consequences in this commit set:** `README.md`, `docs/2026-09-19-holdover-design.md` §13, `CLAUDE.md`; the tag.

### 0017 — 2026-10-14 — The own run registered: statistic (B) at lags 1–32 on a GRPO trajectory of Qwen2.5-1.5B (design §14)

prior-entries-sha256: 4a530e8788348ff84de1b9d396832597929fee5d3e35f4bf5688466503bdafbd

**Registers** `config/ownrun.toml`, now `registered_by = "0017"`, as design §14 defines the own run (approved in
principle 2026-10-13, ruled all-as-recommended 2026-10-14, R-C1–R-C8). Rule, bands, controls and scorer are 0012's
as amended by 0013 and 0014: median (B1) per rung and text, HOLDS_B ≤ 1e-3, DEGRADES_B ≥ 1e-1 nats/token; float16
floor on median KL at 1e-4, identity 1e-5, scrambled 0.1, per reader; instrument `5d4aa4a`, float32, seed 0.

**Trajectory.** GRPO on Qwen2.5-1.5B (`8faed761…`), full weight, 32 optimizer steps from the released checkpoint,
one seed: TRL 1.15.0 (`dae7b6ad…`), learning rate 1e-6, no reference-KL term, 16 prompts × 8 completions per step,
temperature 1.0, 256-token completions, fp32 master weights, GSM8K train with an exact-match reward plus a format
term, one fixed system line. Weights kept at steps {0, 1, 2, 4, 8, 16, 32} as local checkpoints with provenance;
every step's whole-vector distance to step 0 is recorded. The held-out and rollout sequences are dumped from every
step on the run machine; nothing is transferred from 0008 but the Qwen token file (`01bd7698…`).

**Cells.** Writer cache at step 0, reader at step g ∈ {1, 2, 4, 8, 16, 32}. Two pairs: the ten held-out Qwen
sequences of 0008 (corpus and greedy on-path texts, as §13), and ten rollout sequences — GSM8K train prompts
{123, 306, 562, 1309, 2014, 2298, 3816, 4754, 6077, 6349}, chosen by seed 0 and excluded from the run's batches,
each with one completion sampled from the step-0 policy at temperature 1.0 on a Mac mini (CPU, torch 2.14.1):
token file `qwen2.5-1.5b-ownrun_rollout_n10_seed0.npy` `7d6dec47…`, spans `….spans.json` `e631fd84…`; prompts
75–117 tokens, completions 146–256 tokens (seven of ten ran to the 256-token cap; none reached the gold answer —
the sample is the policy's, not a solution set). On rollout text the reader teacher-forces the stored completion
under its own prompt cache and under step 0's; each reader's controls run once, on the held-out pair. 12 rungs,
6 readers, 420 jobs.

**Seal** (`ledger/predictions/ownrun.json`, sidecar `49472367…`, pre-fit, committed before any score). (i) Median
(B1) at g = 32 ≤ 1e-3 on corpus and on-path text. (ii) The slope of log median (B1) on log distance over the six
corpus lag rungs lies in [1.5, 2.5], both texts (the summarizer fits it). (iii) At every lag rung whose distance lies
inside 0015's backbone range (1e-4 to 1e-1), (B1) ≥ 10× the OLMo backbone's median at the nearest 0015 rung by log
distance, both texts — read against 0015's `summary.json` (`5d761594…`) in the outcome entry, since this
configuration carries no noise rung. (iv) At every rung above 2× floor, rollout (B1) ≤ on-path (B1) ≤ corpus (B1).
Descriptive, not sealed: (B1) at g = 1 and whether it is at the floor; the per-step distance curve; the per-step
reward mean.

**Halting and reading.** A reader whose floor median exceeds 1e-4 halts the run; a reader whose scrambled median is
under 0.1 is UNEVALUABLE. A step-1 distance under 1e-6 (the policy did not move) stops the sitting before the reader
run, and the outcome entry reads what ran. The trainer is smoke-tested for two steps on the run machine before the
registered run; its outputs are discarded. Rungs are placed by `results/distance/rungs/<label>/distance.json`,
written by the trainer's save hook.

**Not registered here.** The sliding variant (writer at t > 0); a second seed; the 128-step extension (an amendment
if (i) holds at the floor); statistic (C).

**Consequences in this commit set:** `config/ownrun.toml` (`registered_by`, the two rollout hashes, the trainer
pin); `ledger/predictions/ownrun.json` + `.sha256`; `tools/own_run.py` (the instrument pair as an argument; the
interpreter path kept unresolved).

### 0018 — 2026-10-14 — Outcome: the own run (0017); of the sealed clauses (i) and (ii) HELD, (iii) and (iv) FALSIFIED

prior-entries-sha256: 7830e2e87434f1e28d0931c8c024ffdd119423678666499f5d1d41e58d59d4e7

**Run.** The trajectory and the 420 jobs of 0017 (`config/ownrun.toml`, instrument `5d4aa4a`) ran on one
A100-SXM4-80GB under torch 2.14.1+cu130, TRL 1.15.0, without a halt. Step-1 distance 2.29e-5 (the policy moved; the
stop rule did not fire). Every reader's three controls passed: float16 floors 4.3e-8 to 5.3e-8 nats/token on corpus,
2.1e-8 to 3.0e-8 on on-path (limit 1e-4); identity 0; scrambled 0.55 on every reader (limit 0.1). Records: `ownrun-results.tgz`
`440b5055…` (reports, `run.json`, the pod's summary), `ownrun-aux.tgz` `3fd92aee…` (33 provenances, 32 step distances,
12 rung placements, trainer log); kept weights {0, 1, 2, 4, 8, 16, 32} and every step's dumps backed up to the private
dataset under `ownrun/` (manifest `SHA256SUMS`, 20,527 files, 0 missing, 0 LFS-sha mismatches); all off-repository.

**Trajectory.** Whole-vector relative distance to step 0: 2.29e-5 (step 1), 3.52e-5 (2), 5.27e-5 (4), 7.48e-5 (8),
9.55e-5 (16), 1.049e-4 (32) — concave and saturating (d32/d1 = 4.6; a √t law gives 5.7), nearly flat after step 20
(1.00e-4). Exact-match reward mean 0.030 over steps 1–8, 0.148 over 25–32; the format term 0.150 → 0.191 over the same windows; completion
entropy 2.51 → 0.98. The weights almost stop moving while the policy keeps sharpening.

**Reading** — median (B1), nats/token, at distance x; bands HOLDS_B ≤ 1e-3, DEGRADES_B ≥ 1e-1. *Held-out text*
(corpus / on-path): lag 1 (x 2.29e-5) 2.04e-7 / 1.22e-7; 2 (3.52e-5) 5.18e-7 / 2.90e-7; 4 (5.27e-5) 1.44e-6 / 7.16e-7;
8 (7.48e-5) 2.84e-6 / 1.61e-6; 16 (9.55e-5) 4.85e-6 / 2.53e-6; 32 (1.049e-4) 6.65e-6 / 3.54e-6 — HOLDS_B at every
lag, every rung above 2× its reader's floor; slope of log median on log x over the six rungs 2.26 (corpus), 2.19
(on-path). *Rollout text* (the step-0 policy's own GSM8K completions, prompt cache stale, completion teacher-forced):
lag 1 4.68e-3; 2 1.30e-2; 4 2.94e-2; 8 6.38e-2; 16 8.88e-2; 32 8.91e-2 — UNRESOLVED at every lag, four orders of
magnitude above the same reader on held-out text, with the reader's argmax agreeing with its own fresh cache on 98%
(lag 1) to 90% (lag 32) of tokens. The per-position profile is that of a stale prompt, not a misaligned one: the median
per-position KL is largest at the first continuation position (0.033 at lag 1; 2.3 at lag 32) and falls two orders of
magnitude by position 50; maximum |Δlogit| 3.1–4.8 (lag 1) and 9.9–14.1 (lag 32) against 0.01–0.02 and
0.05–0.19 on corpus text.

**The seal (`ownrun`, 49472367…).** (i) HELD: at g = 32, 6.65e-6 / 3.54e-6 ≤ 1e-3. (ii) HELD (read by the
summarizer): slopes 2.26 and 2.19 in [1.5, 2.5]. (iii) FALSIFIED (read by hand against 0015): one rung lies inside
[1e-4, 1e-1], lag 32 at 1.049e-4; the nearest 0015 backbone rung is noise-0.0001 (6.0e-7 / 4.5e-7), ratio 11.1 on
corpus, 7.88 on on-path — under the 10× sealed on on-path, and the clause asks both texts (interpolating the backbone's quadratic to 1.049e-4 gives
10.1 / 7.2; lag 16 at 9.55e-5 sits just outside the range). (iv) FALSIFIED at all six rungs: rollout exceeds corpus
everywhere; on-path ≤ corpus holds at all six (on-path 40–50% below corpus).

**What the reading says.** At the operating lag of a pipelined trainer a cache up to 32 GRPO steps stale is HOLDS_B
on held-out text by two and a half orders of magnitude, and the backbone's quadratic law (0015) holds along a real RL
trajectory at distances 1e-5 to 1e-4, a decade below where 0015 measured it. The excess of RL over isotropic noise at
matched distance (0015: 1–3 orders on OLMo's published chain; here ≈ 10× at the one comparable point) is small on
held-out text — and enormous on the policy's own text: the same step-1 reader reads 2e-7 on web text and 5e-3 on its
own prompts, and by step 32 the prompt cache of step 0 is 0.09 nats/token stale, near DEGRADES_B, while the weights
have moved 1e-4. Where the cache is read matters more than how far the weights moved; the held-out texts of 0012
measure the pretraining manifold, which RL barely touches, and understate staleness on the trainer's own stream by
four orders of magnitude at this scale.

**Also recorded.** `summarize_reader`'s seal readout asked the slope clause for a rollout slope no corpus rung can
carry and printed UNREAD on the pod; it now fits only the texts the sealed rungs carry (no number changed; the pod's
summary is retained). The reward and entropy trajectories are descriptive, not sealed.

**Not ruled here.** Whether (B)'s statement of record (0016) is amended for the rollout text; the sliding writer,
a second seed, the 128-step extension; whether rollout-text controls (floor, scrambled) join the configuration;
statistic (C).

**Consequences in this commit set:** `src/lag_ladder/summarize_reader.py`, `tests/test_summarize_reader.py`,
`README.md`, `CLAUDE.md`.

### 0019 — 2026-10-14 — Ruling: 0016's statement of record is restricted to the held-out texts it was measured on; the trainer's own text is not predicted by weight distance

prior-entries-sha256: 9d45ca8f70ae5028720d200fcb9474f907a4c835d5e582ba258d3395156c4dfb

**Ruled 2026-10-14, after 0018.** The statement of record for statistic (B) (0016) stands on every text it names: the
quadratic law on the isotropic backbone, the HOLDS_B edge between 3e-3 and 1e-2, the excess of trained directions over
noise at matched distance, and the published RL chain's HOLDS_B at lags 200 to 2400 were measured on held-out corpus
text and the reader's own greedy continuation, and 0018 re-read the first of them a decade lower in distance without
disturbing it. 0016 located the gap between a reader's number and a trainer's in the continuation and in the lags below
200. 0018 found it in the prefix: on the policy's own prompts the same readers that read 2e-7 to 7e-6 on held-out text
read 4.7e-3 to 8.9e-2, UNRESOLVED at every lag from 1 to 32, at the same weight distances (2.3e-5 to 1.05e-4).

**Amendment to the statement of record.** The statement of record is read with this clause appended: *every reading
in it is a reading of the held-out texts of 0012; on the text a pipelined trainer serves — its own prompts under the
policy that wrote them — (B1) at this scale is not predicted by whole-vector weight distance, exceeds the held-out
reading by four orders of magnitude, and leaves HOLDS_B at the first optimizer step (0018). The published chain's lag
readings bound a reader of held-out text, not a trainer's staleness.* Nothing else in 0016 is amended; 0015's numbers
and 0016's citation stand.

**What this ruling does not claim.** The rollout magnitudes of 0018 are one model, one seed, ten sequences, 32 steps,
and were read without a floor or scrambled control on rollout text (0017 ran the controls on the held-out pair only).
They support the scope restriction above — the direction and size of the gap need no control to be read — and do not
yet support a statement of record of their own. No band, slope or ratio on rollout text is ruled here.

**Prerequisite registered.** Any later entry that states a rollout-text number as a statement of record first registers
and passes, per reader, a float16-floor and a scrambled control on rollout text under 0014's limits, and places each
rollout rung against its reader's rollout floor. The configuration change is the next registration entry's, not this
one's.

**Citing.** 0016 is cited as before (tag `statistic-b-closed`), read with this entry; the README's citing line names
both. No new tag.

**Not ruled here.** The sliding writer; a second seed; the 128-step extension; whether the own run's rollout reading
becomes a statement of record once its controls exist; statistic (C).

**Consequences in this commit set:** `README.md` (the (B) paragraph's closing sentence and the citing line),
`CLAUDE.md` (the entry list and State).

### 0020 — 2026-10-14 — Amendment: float16-floor and scrambled controls on the rollout text, and the held-out sliding writer, scored from the own run's dumps

prior-entries-sha256: 610118e98ae85df2e25ebbc10e18939db50be16959509e530f459efcfb5154d0

**Registers** `config/ownrun_controls.toml` (`registered_by = "0020"`), a second reader configuration over the own run's
artefacts (0017, 0018): the same instrument (`5d4aa4a`), pins, pairs, bands and control limits as `config/ownrun.toml`,
which stays as run. It answers 0019's prerequisite and rulings R-C9 and R-C10 of this date (design §14, §14.1).

**Controls on the rollout text (0019's prerequisite).** For each reader g ∈ {1, 2, 4, 8, 16, 32}: the float16 floor on
the reader's own cache of the rollout prompts (median KL ≤ 1e-4, as 0014), identity, and the scrambled control on
rollout text (another sequence's prompt cache under the same completion; median KL ≥ 0.1). The driver and summarizer
now plan and judge controls per (reader, pair) when a configuration says `per_pair = true`, on that pair's own texts;
the default is unchanged and the plans of 0012–0018 are as they ran. Each rung is judged against the controls of its
own pair. 0017's six rollout rungs are re-run on the same pod as a descriptive replicate; the record of their reading
stays 0018.

**The sliding writer (held-out text, descriptive).** Fifteen rungs `ownrun-slide-<w>-<r>` with writer at kept step w
and reader at kept step r > w, w, r ∈ {1, 2, 4, 8, 16, 32}, on the held-out pair's corpus and on-path texts, placed by
the measured relative distance ‖θ_r − θ_w‖ / ‖θ_w‖ from the kept weights (`tools/weight_distance.py`, written to
`results/distance/rungs/<label>/distance.json` on the run machine). Rollout sliding is not registered: it needs rollouts
sampled by the step-w policy.

**Seal** (`ledger/predictions/ownrun-controls.json`, sidecar `27af2dc2…`, pre-fit, committed before any score).
(i) Every rollout rung of 0018 reads above 2× its reader's float16 floor on rollout text as measured here (read by hand
against 0018's summary), and the six replicate rungs are not at the floor (read by the summarizer). (ii) The slope of
log median (B1) on log writer–reader distance over the fifteen sliding rungs lies in [1.5, 2.5] on both texts.
Descriptive, not sealed: the sliding rungs against 0018's step-0-writer rungs at matched distance; the replicate against
0018; identity on rollout text.

**Run.** 740 jobs, inference only, on one GPU sitting under the GPU protocol: six kept readers, the held-out and rollout
dumps at the kept steps and step 0 from the private dataset; the results tar, summary and the fifteen distance files
off-repository with hashes. A rollout floor over its limit halts, as 0014; a rollout scrambled under 0.1 makes that
reader's rollout rungs UNEVALUABLE and, in the outcome, withdraws 0018's reading on that rung to descriptive.

**Not registered here.** A second seed and the 128-step extension (R-C11, R-C12: decided after this entry's outcome);
statistic (C) (R-C13: ruled in principle, designed after it); rollout sliding.

**Consequences in this commit set:** `src/lag_ladder/reader.py`, `src/lag_ladder/summarize_reader.py`,
`src/lag_ladder/config.py`, `tests/test_reader.py`, `tests/test_summarize_reader.py`, `config/ownrun_controls.toml`,
`ledger/predictions/ownrun-controls.json` + `.sha256`, `docs/2026-09-19-holdover-design.md` §14 (R-C9–R-C13, §14.1),
`README.md`, `CLAUDE.md`.

### 0021 — 2026-10-15 — Outcome: 0020's controls on the rollout text pass and both sealed clauses HELD; 0019's prerequisite is met

prior-entries-sha256: bee8ac6211baa5d390c566209d0538f01cf126dcb249fff536de2b6337e40c68

**Run.** The 740 jobs of `config/ownrun_controls.toml` (instrument `5d4aa4a`, this repository at `947171a`) ran on one
NVIDIA A40 under torch 2.14.1+cu130 from the own run's dumps and kept weights fetched from the private dataset (4,092
files, every one matching the upload's manifest; `weight_distance.py` on this host reproduces 0018's step_0→step_1
placement to the last digit), without a halt. Records: `controls-results.tgz` `b8306dbd…` (reports, `run.json`, the
pod's summary), `controls-aux.tgz` `7d57b065…` (15 distance files, the distance check, logs); backed up to the private
dataset under `ownrun-controls/` with the fifteen distance files and a manifest (`SHA256SUMS` `142b941c…`, every LFS
digest on the Hub equal to the local one).

**Controls, per (reader, pair), all eleven passed under 0014's limits.** Rollout pair, readers g ∈ {1, 2, 4, 8, 16, 32}:
float16 floor 1.53e-7 to 1.87e-7 nats/token (limit 1e-4; about three times the held-out corpus floor), identity 0,
scrambled 0.115 to 0.126 (limit 0.1). Held-out pair, readers {2, 4, 8, 16, 32}: floors 4.6e-8 to 5.8e-8 on corpus,
2.2e-8 to 2.9e-8 on on-path, identity 0, scrambled 0.552 — the same range as 0018's A100 readings, on an A40.

**Seal (`ownrun-controls`, 27af2dc2…).** (i) HELD: every rollout rung of 0018 reads 2.5e4× (lag 1) to 5.8e5× (lag 16) its
reader's float16 floor on rollout text; the six replicate rungs are not at the floor and reproduce 0018's medians to within
5e-5 relative at every lag (4.679e-3 … 8.914e-2). (ii) HELD: the slope of log median (B1) on log writer–reader distance over
the fifteen sliding rungs is 1.96 on corpus and 1.97 on on-path, in [1.5, 2.5].

**The sliding writer (descriptive).** Writer at kept step w, reader at kept step r, distance 1.63e-5 (1→2) to 9.42e-5 (1→32):
median (B1) 1.41e-7 to 5.28e-6 on corpus, 7.3e-8 to 2.85e-6 on on-path, HOLDS_B at every rung and every rung at least 2.46×
its reader's floor (1→2 on corpus). Against the quadratic fitted to 0018's six step-0-writer rungs, the sliding rungs read
0.93× to 1.85× the prediction at their distance (median 1.1 on both texts); the excess is largest at the one-doubling gaps
(16→32 1.85 / 1.65, 2→4 1.54 / 1.54, 1→2 1.50 / 1.31, 4→8 1.35 / 1.22) and vanishes above x ≈ 5e-5 (0.93 to 1.12 there).
The law off step 0 is the step-0 law within a factor of two.

**Stale against scrambled on the policy's own text (descriptive).** The rollout scrambled control — another rollout
sequence's prompt cache under the same completion — reads 0.12, not the 0.55 of held-out text: every rollout prompt shares
the system line, the GSM8K format and a 75–117-token length, so a wrong prompt is a near neighbour. 0018's rollout median
over the same reader's scrambled median is 0.037 at lag 1, 0.105 at 2, 0.24 at 4, 0.54 at 8, 0.77 at 16 and 0.74 at 32: by
sixteen steps the step-0 prompt cache moves the reader's next-token distribution on its own prompts three quarters as far
as the cache of a different prompt does. The control passed its limit by 15–26%; 0014 set that limit on held-out text.

**What this entry settles.** 0019's prerequisite is met: 0018's rollout numbers have per-reader floor and scrambled controls
on rollout text, passed, and a cross-GPU replicate. Whether they now enter (B)'s statement of record, and in what words, is a
ruling, not a reading, and is not taken here. The sliding reading answers R-C10: the quadratic law is not a step-0 artefact.

**Not ruled here.** The rollout numbers as a statement of record (0019's gate is open); a rollout-specific scrambled control
(a non-GSM8K prompt of matched length) and its limit; the second seed (R-C11); the 128-step extension (R-C12); §15 for
statistic (C) (R-C13); rollout sliding.

**Consequences in this commit set:** `README.md`, `CLAUDE.md`.

### 0022 — 2026-10-15 — Amendment: a far scrambled control on the rollout text, scored from the own run's dumps; rulings after 0021

prior-entries-sha256: e906341d42bef3ecec694e7b3c887fd69534a0d05283a07faafd1a039d7a3b50

**Rulings (2026-10-15, after 0021; design §14 R-C14–R-C18).** The amendment of (B)'s statement of record with the
rollout reading is not taken: one seed, and the scrambled control 0021 passed on rollout text is a near neighbour —
another GSM8K prompt's cache, sharing the system line, the format and the length — that cleared 0014's limit by 15–26%, so
the 0.77 reading of stale over scrambled at lag 16 has a denominator that carries the format. A far control is registered
here (R-C15); the second seed waits for the paper's sentence (R-C16); the 128-step extension is closed (R-C17); statistic
(C) is designed in §15, to be built and registered after this entry's outcome (R-C18).

**Registers** `config/ownrun_far.toml` (`registered_by = "0022"`), a third reader configuration over the own run's
artefacts: the same instrument (`5d4aa4a`), pins, pairs, bands and limits as `config/ownrun_controls.toml`, which stays
as run (0021), with `per_pair = true` and the rollout pair naming the held-out pair as its `far_pair`.

**The far control.** For each reader g ∈ {1, 2, 4, 8, 16, 32} on the rollout pair, beside the float16 floor, identity and
scrambled of 0020: the reader's own float16 dump of held-out sequence 40 + i, of which the scorer reads the first
prefix_len_i − 1 rows as it does for every arm — a web prefix of exactly the rollout prompt's token length, carrying no
information about the prompt — under rollout sequence i's stored completion. Median over the ten sequences of the mean
per-token KL; limit `far_min = 0.1`; a unit under it is UNEVALUABLE, as scrambled. Both controls must pass. The held-out
dumps at the kept steps are on the private dataset; nothing is re-dumped; the scorer is unchanged. The driver plans a
`far` job per sequence after scrambled when a pair names a far pair; the summarizer judges it and reads, per rung and
text, the rung's median over its unit's scrambled and far medians (`over_scrambled`, `over_far`) and, per unit,
`far_over_scrambled`. The default plans of 0012–0021 are unchanged. 0017's six rollout rungs run again as a second
descriptive replicate; the record stays 0018.

**Seal** (`ledger/predictions/ownrun-far.json`, sidecar `324bd425…`, pre-fit, committed before any score). (i) On every
(reader, rollout pair) unit the far median is at least 2× the same unit's scrambled median: an information-free prefix
is at least twice as far as another GSM8K prompt's. (ii) On every replicate rollout rung the median (B1) is at most 0.5×
its unit's far median (read by the summarizer), and the same for 0018's six rollout medians against these far medians
(read by hand, as 0020's clause (i)): the step-0 cache stays nearer than halfway to an information-free one at every lag
to 32. Falsified (i): the format is most of the distance and 0021's 0.77 stands as read. Falsified (ii): by sixteen steps
the stale cache is as bad as no prompt. Descriptive, not sealed: the far medians; stale over scrambled per lag; the
replicate against 0018 and 0021; near over far per reader.

**Run.** 300 jobs, inference only, on one GPU sitting under the GPU protocol: six kept readers, the rollout dumps at
step 0 and the kept steps, the held-out dumps at the kept steps, from the private dataset; the results tar and summary
off-repository with hashes. Halting and reading as 0020.

**Not registered here.** A chat-formatted non-GSM8K prompt as an intermediate control (format without content); the
second seed; statistic (C) (§15: designed, unregistered, unbuilt); rollout sliding.

**Consequences in this commit set:** `src/lag_ladder/config.py`, `src/lag_ladder/reader.py`,
`src/lag_ladder/summarize_reader.py`, `tests/test_reader.py`, `tests/test_summarize_reader.py`,
`config/ownrun_far.toml`, `ledger/predictions/ownrun-far.json` + `.sha256`, `docs/2026-09-19-holdover-design.md`
(§14 R-C14–R-C18, §14.2, §15), `README.md`, `CLAUDE.md`.

### 0023 — 2026-10-15 — Outcome: the far control on the rollout text reads 3.6× to 5.8× the near one; both sealed clauses HELD

prior-entries-sha256: 541b34d371b6a19a554f9facbcb63e498606758ff654089f0e601f4b16972671

**Run.** The 300 jobs of `config/ownrun_far.toml` (instrument `5d4aa4a`, this repository at `0f61a32`) ran on one NVIDIA A40
under torch 2.14.1+cu130 from the own run's dumps and kept weights fetched from the private dataset (4,085 files, every one
matching the upload's manifest), without a halt; no unit UNEVALUABLE. Records: `far-results.tgz` `8b7046a9…` (reports,
`run.json`, the pod's summary), `far-aux.tgz` `09caa4a8…` (logs); backed up to the private dataset under `ownrun-far/` with a
manifest (`SHA256SUMS` `1e859393…`, every LFS digest on the Hub equal to the local one).

**Controls, per (reader, rollout pair), all six passed.** Float16 floor 1.53e-7 to 1.85e-7 nats/token, identity 0, scrambled
0.115 to 0.126 — each equal to 0021's reading of the same cell to four digits on another host. Far — the reader's own cache of
a web prefix of the prompt's length, under the same completion — 0.450 (g = 1), 0.468, 0.514, 0.612, 0.669 (g = 16), 0.611
(g = 32), against the limit of 0.1.

**Seal (`ownrun-far`, 324bd425…).** (i) HELD: far over scrambled 3.57, 3.77, 4.22, 5.14, 5.80, 5.09 at g = 1 … 32, every one
above 2. (ii) HELD: 0018's rollout medians over the same reader's far median 0.010, 0.028, 0.057, 0.104, 0.133, 0.146,
every one below 0.5 — read by hand against 0018 and by the summarizer on the six replicate rungs, which reproduce 0018 to
within 5e-5 relative (the largest gap 4.95e-5, at g = 4) and 0021's replicate to within 5e-6.

**What the three controls say together (descriptive).** On the policy's own prompts a cache that carries no information
about the prompt moves the reader 0.45 to 0.67 nats/token; a cache of a different GSM8K prompt moves it 0.12, so the format
and the system line account for three quarters (g = 1) to five sixths (g = 16) of the information-free distance and the
question's content for the rest. The step-0 cache at lag 16 to 32 reads three quarters of the content term (0021) and one
seventh of the information-free distance. The far distance grows with the reader's step to g = 16 (0.45 to 0.67) and eases
at 32 (0.61) while the near one stays within 0.115 to 0.126: later policies are more certain of their own completions and
lose more when the prompt is gone.

**What this entry settles.** 0021's stale-over-scrambled reading has its anchor: 0.77 at lag 16 is three quarters of a
near-neighbour's distance and one seventh of an information-free one. The control objection of R-C14 is answered; the
one-seed caveat stands.

**Not ruled here.** The rollout numbers as a statement of record (R-C14: now only the seed count in the way); the second seed
(R-C16); statistic (C)'s registration (§15, R-C18); a chat-formatted non-GSM8K prompt as an intermediate control.

**Consequences in this commit set:** `README.md`, `CLAUDE.md`.

### 0024 — 2026-10-15 — Ruling: (B)'s statement of record gains the trainer's own text, on one seed

prior-entries-sha256: 2e1a257cec0784a9897571d419a2393611512b247eb4ea097489d28bb426da1c

**Ruled 2026-10-15, after 0023.** 0019 restricted 0016's statement of record to the held-out texts it was measured on and
set a prerequisite for any rollout-text number to enter it: per-reader float16-floor and scrambled controls on rollout
text, passed under 0014's limits. 0021 met it; 0022 and 0023 added a far control after the near one proved a near
neighbour, and a replicate on a second host. R-C14 (design §14) declined the amendment on 2026-10-15 for two reasons; one
is answered and the other — one seed — is a caveat the sentence below carries. The alternative, a second trainer sitting
first, was weighed and declined: the direction, the four orders of magnitude, the loss of HOLDS_B at the first step and
the two control ratios are not readings a second seed reverses.

**Amendment to the statement of record.** The statement of record for statistic (B) (0016, read with 0019) is read with
this clause appended after 0019's: *on the trainer's own text — its own prompts under the policy that wrote them — on one
seed of one 1.5B model (Qwen2.5-1.5B, GRPO on GSM8K, 32 optimizer steps at learning rate 1e-6; 0017, 0018), (B1) leaves
HOLDS_B at the first optimizer step (4.7e-3 nats/token at distance 2.3e-5) and reads 8.9e-2 at lags 16 and 32 (distances
9.5e-5 and 1.05e-4), UNRESOLVED at every lag from 1 to 32 and four orders of magnitude above the same readers' reading of
held-out text (2.0e-7 to 6.7e-6, HOLDS_B throughout). The readers' float16 floors on that text are 1.5e-7 to 1.9e-7, their
cache of a different GSM8K prompt reads 0.12 and their cache of a web prefix of the prompt's length 0.45 to 0.67 (0021,
0023): by sixteen steps the step-0 cache moves the reader three quarters as far as a different prompt's cache and one
seventh as far as an information-free one. The number is a reader's, on text the trainer serves; it is not an importance
weight, an advantage or a task outcome.* Nothing else in 0016 or 0019 is amended; 0019's clause — that the trainer's own
text is not predicted by whole-vector weight distance and that the published chain's lag readings bound a reader of
held-out text — stands as written.

**What this ruling does not claim.** A law on rollout text (no slope is sealed or read there); anything beyond one
seed, one model, one task, 32 steps, or a cache older than 32 steps; anything a trainer computes (the importance-weight
dispersions of 0018 are descriptive); anything about task output (statistic (C), §15, unregistered).

**Citing.** 0016 is cited as before (tag `statistic-b-closed`), read with 0019 and this entry; name 0018 for the rollout
numbers, 0021 and 0023 for their controls, 0015 for the ladder's. No new tag.

**Not ruled here.** The second seed (R-C16, deferred; it would strengthen this clause, not change it); statistic (C)'s
registration (R-C18, §15); rollout sliding.

**Consequences in this commit set:** `README.md` (the (B) paragraph's closing sentences and the citing line),
`CLAUDE.md` (the entry list and State).

### 0025 — 2026-10-15 — Statistic (C) registered: the greedy answer-flip rate under the step-0 cache, lags 1–32 (design §15)

prior-entries-sha256: 0c63f4278f58aaa119d622f712f9dcfe5c3ce04ffe2448e43fbd175402ac7436

**Registers** `config/ownrun_task.toml`, now `registered_by = "0025"`, as design §15 defines (C) after two 64-prompt
probes on the run machine (2026-10-15, pre-registration, outside `results/`; records off-repository, results tars
`a007049e…` and `5bd9e076…`). Rulings R-C19–R-C23 taken as the design records them. Instrument `5d4aa4a`, float32, seed 0.

**The statistic.** Reader at kept step g ∈ {1, 2, 4, 8, 16, 32}; writer step 0. Per prompt of GSM8K test (all 1,319;
prompts file `918c7e67…`) four greedy generations of ≤ 256 new tokens under an injected prompt cache — FRESH, STALE,
NEAR, FAR — built live and rounded through float16. The answer is the run's reward extraction (the number after the
last `####`, else the last number). **Verdict-bearing: the STALE answer-flip rate against FRESH per g**, with a seeded
bootstrap interval over prompts. The accuracy difference STALE − FRESH, with its paired bootstrap and §4's bands, is
descriptive: both probes read the step-32 policy at 4 of 64 correct, 48 of 64 at the cap, 40 of those a repeated
`#### N` line (the same at a 1,024-token cap on 17 prompts, where every capped completion was the 256-token text's
exact prefix), so a one-point accuracy band has no power on this baseline and the cap is not what limits it.

**Controls and limits.** Per reader, halting: float16 floor (FRESH in float32 against FRESH through float16) flip rate
≤ 0.01 — both probes read 0 of 64, which bounds the rate at 4.6% (rule of three); 1% is a margin ten times under the
seal's 10%, not a measured ceiling; identity (FRESH twice) zero flips. Positive, per reader, the cell UNEVALUABLE
without them: NEAR and FAR each flip ≥ 0.5 of the answers (probes at g = 32: 58 and 64 of 64). Once per sitting, before
any cell: dump equivalence — the live step-0 cache of 0017's ten rollout prompts against the step-0 dump, relative gap
≤ 2⁻¹⁰ per element (one float16 ulp; the probes read exactly one ulp, 2⁻⁷ then 2⁻⁶ absolute on elements of different
magnitude, so the limit is relative), and 0018's rollout (B1) at step_32 reproduced within 1e-3 relative (probes:
6.4e-6, 4.0e-6).

**Seal** (`ledger/predictions/ownrun-task.json`, sidecar `3da7a50d…`, pre-fit). (i) At every g NEAR and FAR each change
the answer on at least half the prompts. (ii) At g = 1 the STALE flip rate is ≥ 10%: the greedy output is not
reproducible from the first optimizer step. (iii) At g = 32 the STALE flip rate is ≥ 10%. (iv) Growth: the g = 1 flip
interval lies entirely below the g = 32 interval. Read, by pre-commitment, as reproducibility — "the greedy output
changes under a stale cache" — not as task degradation; the accuracy claim is the descriptive Δ's. **(ii) and (iv)
are blind**: no probe read g = 1. **(i) and (iii) at g = 32 are confirmations**: both probes read 55 of 64 STALE flips
there. Descriptive, not sealed: Δ per g; FRESH accuracy; the first-`####` answer beside the registered one; cap and
no-answer rates per arm; first divergence; the rank correlation of flips with (B1) on the fresh path; g ∈ {2, …, 16}.

**What the probes also fixed.** The 256-token pass reproduced the first probe's completions on 255 of 256 across two
torch builds (2.14.1+cu130, 2.11.0+cu128); the one difference is a STALE completion that diverges at token 0 on both
builds and lands on a different wrong answer — the flip count is unchanged, so a replicate's one-in-64 STALE hash
mismatch is expected jitter, not a defect. Records are `task-record-v2` and keep the completion text.

**Run.** One A40 sitting under the GPU protocol at the 256-token cap: six readers × 1,319 prompts × (four arms + two
control arms), inference only; the torch build, CUDA flag and host driver recorded in the sitting's runbook; one log per
stage and attempt. Halting as above; the outcome entry reads what ran.

**Not registered here.** Sampled generation (the trainer samples at T = 1.0; every (C) sentence says *greedy*); the
first-`####` answer as verdict; a cap above 256; the second seed (R-C16).

**Consequences in this commit set:** `tools/task_swap.py`, `src/lag_ladder/{config,task,summarize_task}.py`,
`tests/{test_task_swap,test_task,test_summarize_task,test_config,fake_task_swap}.py`, `config/ownrun_task.toml`,
`ledger/predictions/ownrun-task.json` + `.sha256`, `docs/2026-09-19-holdover-design.md` (§15), `README.md`, `CLAUDE.md`.
