# lag-ladder — repo brief

Read `README.md` for what this is. `docs/2026-09-19-holdover-design.md` is the authority on scope and
carries the rulings (§10), the build order (§9), the re-aim (§12), statistic (B) (§13) and the own run (§14). `ledger/ledger.md` is the record and
is append-only by numbered entry; entry 0001 carries the chassis provenance, the inherited definitions (f\*,
τ_K) and the first rulings, entry 0002 the ruling that the instrument serves statistic (A) only, entries
0003 and 0004 the instrument re-pins (checkpoint selection; per-side `ModelRef` with verified provenance),
entry 0005 the pilot's registration, entry 0006 its outcome (f\* = 0 at every lag to 2400: HOLDS),
entry 0007 the ruling that weight distance is the axis, entry 0008 the weight-distance ladder's registration,
entry 0009 its outcome, entry 0010 the closure of statistic (A) with the statement of record, entry 0011
the ruling that the lane continues under statistic (B) as design §13 defines it, entries 0012–0014 (B)'s
registration and amendments, entry 0015 its outcome, entry 0016 its closure on the ladder with the statement of record,
entry 0017 the own run's registration, entry 0018 its outcome, entry 0019 the ruling that restricts 0016's statement of
record to the held-out texts it names, entry 0020 the registration of rollout-text controls and the sliding writer,
entry 0021 their outcome, entry 0022 the registration of a far scrambled control on rollout text, entry 0023 its
outcome, and entry 0024 the ruling that adds the trainer's own text to (B)'s statement of record, on one seed.

## Rules
- `../kv-transfer-replication` is **read-only** and pinned (`UPSTREAM.md`). Never write there, never
  `import kvt`, never copy its code; `tests/test_imports.py` refuses it. Borrowed facts carry
  `{sourceRepo, filePath, commitSha}`. It is the instrument for **statistic (A) only** (ledger 0002);
  the (B) and (C) scorers are built here.
- The chassis is **copied from linear-ceiling, not shared**. Do not import from it or symlink to it;
  a change there is not a change here until an entry says so.
- Every driver calls `lag_ladder.upstream_gate.check_upstream` on the four (A) paths before it runs.
  A red gate is answered by a numbered re-pin entry, never by widening the path tuple.
- Never write a number into the ledger or a doc that was not recomputed from `results/` by a
  summarizer. A fact from outside the repo carries a primary source and a retrieval date, or the word
  "unverified".
- Never edit a hypothesis after its experiment starts; never edit a committed ledger entry or a
  sealed prediction; append.
- Ledger entries are short: what was registered, ruled or measured, its sources, its status.
- Seeds and thresholds live in `config/*.toml`. Randomness only via `lag_ladder.rng.make_rng`.
- A config with `registered_by = ""` is UNREGISTERED; no driver may run while it is empty. A
  `*_PENDING` pin is refused by name.
- Every GPU run follows linear-ceiling's `docs/gpu-experiment-protocol.md` (R1–R12).
- Build what the next step calls. A module nothing imports does not belong here yet.
- Working documents (handoffs, learnings, seeds, designs in progress) stay out of the repository.

## Commands
```
.venv/Scripts/python.exe -m pytest -q                 # suite (synthetic, offline)
.venv/Scripts/python.exe -m lag_ladder.seal verify
.venv/Scripts/python.exe -m lag_ladder.lint_scope
.venv/Scripts/python.exe -m lag_ladder.ledger_check   # --against <rev> in CI
.venv/Scripts/python.exe -m lag_ladder.pilot plan     # the dump and score jobs; decides nothing
.venv/Scripts/python.exe -m lag_ladder.pilot check    # the gate: registration, pin, token hash, seal
.venv/Scripts/python.exe -m lag_ladder.pilot run      # refuses until the gate passes; resumable, --limit N
.venv/Scripts/python.exe -m lag_ladder.summarize_pilot   # recomputes every f* from disk; refuses on mismatch
.venv/Scripts/python.exe -m lag_ladder.distance plan|check|run   # the weight-distance ladder (config/distance.toml)
.venv/Scripts/python.exe -m lag_ladder.summarize_distance        # rungs by measured distance; refuses on mismatch
.venv/Scripts/python.exe -m lag_ladder.reader plan|check|run|probe [--config config/ownrun.toml]   # statistic (B)
.venv/Scripts/python.exe -m lag_ladder.summarize_reader [--config config/ownrun.toml]
.venv/Scripts/python.exe -m lag_ladder.task plan|check|run [--config config/ownrun_task.toml]   # statistic (C), §15
.venv/Scripts/python.exe -m lag_ladder.summarize_task [--config config/ownrun_task.toml]
```
On macOS/Linux the interpreter is `.venv/bin/python`. The project pins Python 3.12.

The upstream gate, as every driver will call it:
```
.venv/Scripts/python.exe -c "from pathlib import Path; from lag_ladder import INSTRUMENT_SHA; from lag_ladder.upstream_gate import check_upstream; check_upstream(Path('../kv-transfer-replication'), INSTRUMENT_SHA, ('scripts/dump_kv.py','kvt/data.py','scripts/score_positions.py','kvt/pertoken.py'), who='check'); print('gate ok')"
```

## Layout
`src/lag_ladder/` — `hashing` · `rng` · `config` (seal + pilot loaders) · `seal` · `upstream_gate` ·
`ledger_check` · `lint_scope` · `pertoken` (centered deviation, f\*, the band; copied from linear-ceiling 0023) ·
`fstar_record` (one scored job read back: hash, sums, the exact bridge to 1 − R²) · `pilot` (the (A)-only
driver: identity and scrambled controls, then the ladder, by subprocess into the pinned instrument) ·
`summarize_pilot` (recomputes the ladder from disk) · `distance` (the weight-distance driver: rungs of writer
and reader refs, Hub revision or local checkpoint; controls per writer; the noise backbone written and pruned
per rung) · `summarize_distance` (rungs by measured distance, tied to the dumps' manifests) · `reader` (the
(B) driver: plan|check|run|probe; controls per reader, then the rungs, by subprocess into the pinned
instrument; refuses while `reader.toml` is unregistered) · `summarize_reader` (recomputes B1–B4 from the
reports, withholds failed-control rungs, places rungs by the distance ladder's `distance.json`, reads the sealed
clauses out; per-rung texts and ragged continuations for §14) · `task` (the (C) driver, §15: plan|check|run; the
dump-equivalence control first, then one `cells` job per reader with the floor and identity arms; halts on the
equivalence, the floor or the identity) · `summarize_task` (recomputes Δ accuracy with a seeded paired bootstrap, the
(C) bands, flip / no-answer / cap rates, the (B)→(C) rank correlation; withholds a lag whose positive controls do not
fire; reads the sealed clauses out). `tools/` — `perturb_checkpoint.py`,
`convert_bin_checkpoint.py`, `weight_distance.py`, `score_continuation.py` (the (B) scorer: injects a stale vs fresh
cache, scores the continuation; `--text rollout` teacher-forces a stored continuation cut at `--length`): run with
the instrument's interpreter, import nothing from it. `own_run.py` (§14: the GRPO trajectory with its save hook;
run with the trainer's interpreter, imports nothing from here or the instrument). `task_swap.py` (§15, the (C) tool:
`equivalence` | `cells` | `prepare`; builds prompt caches live, rounds them through float16 and injects them through
`score_continuation.py`'s functions; greedy generation per arm; records per prompt; run with the instrument's
interpreter). Built 2026-10-15, UNREGISTERED: `config/ownrun_task.toml` carries a `GSM8K_TEST_PENDING` pin and a
placeholder floor limit until the probe and the registering entry.
`config/` — `seal.toml`, `pilot.toml`, `distance.toml`, `reader.toml`, `ownrun.toml` (0017, as run), `ownrun_controls.toml` (0020; `per_pair` controls), `ownrun_far.toml` (0022; `far_pair` + `far_min`), `ownrun_task.toml` (§15, UNREGISTERED). `ledger/` — `ledger.md`, `predictions/`. `results/fstar/` and
`mappers/` — seal artifact roots, gitignored past their placeholders. `docs/` — the design doc.

## State
The pilot ran (ledger 0005, 0006): on OLMo-2 1B RLVR1, f\*(τ_K) = 0 at every lag from 200 to 2400 on
both anchors, every τ; the sealed DEGRADES at lag 2400 was falsified. Ruled (0007): the axis is weight
distance. The weight-distance ladder ran (0008, 0009) on the instrument pinned at kv-transfer-replication
`5d4aa4a`: the isotropic backbone on OLMo-2 1B HOLDS through a relative distance of 3e-2 and DEGRADES at
1e-1 (the sealed knee at 3e-3 falsified); the six real directions tie the backbone at f\* = 0 wherever
both are measured and scatter around it by two orders of magnitude in whole-cache deviation. **Statistic
(A) is closed (0010)**: the statement of record is in that entry and the README; linear-ceiling's paper
cites this repository for it. Nothing is registered beyond 0010. **Statistic (B) is ruled (0011)** as
design §13 defines it: per-token KL of the reader over the writer's cache against its own, on the 0008 dumps,
to run on RunPod. **Statistic (B) is registered (0012)**: the float16 floor was probed on the run machine
(median mean-KL 3.9e-8, HOLDS_B confirmed at 1e-3), the seal is committed, `config/reader.toml` is pinned, and
the noise backbone is re-dumped on the run machine because 0008's noise checkpoints do not regenerate
byte-identically there. 0013 and 0014 amended the float16-floor control after the first run halted twice on other readers'
floors (KL only; limit band-relative at 1e-4); 0014 also registers the five R-B4 lag rungs. **The (B) run completed
(0015)**: all controls passed; (B1) grows as x² on the backbone (slope 2.0) and leaves HOLDS_B between 3e-3 and 1e-2;
OLMo directions read 355–875× the backbone at matched distance, Qwen Instruct 11×; RL-phase lags 200–2400 all HOLDS_B;
seal clauses (i), (iii) HELD, (ii), (iv), (v) FALSIFIED. **(B) on the ladder is closed (0016)**, statement of record in
that entry and the README; the scorer stays the registered instrument for later rungs. **The own run (§14) is complete
(0017, 0018)**: `config/ownrun.toml` registered by 0017 with the rollout pins and the seal; one A100 sitting ran the
32-step GRPO trajectory and the 420 jobs without a halt; held-out text HOLDS_B at every lag (lag 32: 6.65e-6 corpus,
3.54e-6 on-path; slopes 2.26 / 2.19), rollout text UNRESOLVED at every lag (4.68e-3 to 8.91e-2); clauses (i), (ii)
HELD, (iii), (iv) FALSIFIED. Records and dumps are off-repository under the hashes 0018 states. **0019 restricts 0016's
statement of record** to the held-out texts it names (the trainer's own text is not predicted by weight distance) and
registers a prerequisite: no rollout-text number becomes a statement of record before per-reader float16-floor and
scrambled controls on rollout text pass under 0014's limits. **0020 registers those controls** (`config/ownrun_controls.toml`,
`per_pair = true`: controls per (reader, pair) on the pair's own text; seal `ownrun-controls`) together with fifteen
descriptive sliding rungs on held-out text (writer and reader at kept steps, placed by writer–reader distance) and a
replicate of the six rollout rungs — 740 inference-only jobs from the Hub's dumps and kept weights. **0021 is its
outcome**: one A40 sitting, no halt; all eleven (reader, pair) control units pass (rollout floors 1.53e-7 to 1.87e-7,
rollout scrambled 0.115 to 0.126 against the 0.1 limit); clause (i) HELD (0018's rollout rungs 2.5e4× to 5.8e5× their
floors, replicate within 5e-5), clause (ii) HELD (sliding slopes 1.96 / 1.97); the sliding rungs read 0.93× to 1.85× the
step-0 law. 0019's prerequisite is met; the rollout reading's entry into (B)'s statement of record is a ruling not yet
taken. Records and the Hub backup (`ownrun-controls/`) under the hashes 0021 states. Rulings R-C14–R-C18 (design §14,
2026-10-15): the statement-of-record amendment not taken (one seed; the near scrambled control passed by 15–26%); **0022
registers a far control** (`config/ownrun_far.toml`, §14.2: the rollout pair names the held-out pair as `far_pair`; each
reader's rollout controls gain a FAR cell — its own dump of held-out sequence 40+i cut to the prompt's length under the
same completion, limit `far_min` 0.1; seal `ownrun-far`: far ≥ 2× scrambled per unit, median (B1) ≤ 0.5× far per lag; 300
inference-only jobs). **0023 is its outcome**: one A40 sitting, no halt; far 0.450 to 0.669 against scrambled 0.115 to
0.126 (far/scrambled 3.57 to 5.80, clause (i) HELD); 0018's rollout medians over far 0.010 to 0.146 (clause (ii) HELD);
the replicate matches 0018 to 5e-5 on a second host. Records and the Hub backup (`ownrun-far/`) under the hashes 0023
states. **0024 amends (B)'s statement of record** with the rollout reading, on one seed (the clause names the seed count,
the controls and the two ratios; 0019's restriction stands). Open: R-C16 second seed (would strengthen 0024, not change
it), R-C18 the build and registration of statistic (C) from §15 (whole-prompt cache swap, GSM8K test, greedy, FRESH/STALE/NEAR/FAR arms,
Δ accuracy with a paired bootstrap, flip rate descriptive); 128-step closed.
