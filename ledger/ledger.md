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
