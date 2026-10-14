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
registration and amendments, entry 0015 its outcome, and entry 0016 its closure on the ladder with the statement of record.

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
clauses out; per-rung texts and ragged continuations for §14). `tools/` — `perturb_checkpoint.py`,
`convert_bin_checkpoint.py`, `weight_distance.py`, `score_continuation.py` (the (B) scorer: injects a stale vs fresh
cache, scores the continuation; `--text rollout` teacher-forces a stored continuation cut at `--length`): run with
the instrument's interpreter, import nothing from it. `own_run.py` (§14: the GRPO trajectory with its save hook;
run with the trainer's interpreter, imports nothing from here or the instrument). (C)'s scorer does not exist.
`config/` — `seal.toml`, `pilot.toml`, `distance.toml`, `reader.toml`, `ownrun.toml` (UNREGISTERED). `ledger/` — `ledger.md`, `predictions/`. `results/fstar/` and
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
that entry and the README; the scorer stays the registered instrument for later rungs. **The own run (§14) is designed
and built, UNREGISTERED**: `tools/own_run.py`, the reader's per-pair heldout/spans and the `rollout` text,
`config/ownrun.toml` with `ROLLOUT_*_PENDING` pins. Next: sample the rollouts on the run machine (hashes), the registration
entry 0017 with the seal, the sitting, the outcome entry. (C) is not ruled.
