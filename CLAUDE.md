# lag-ladder — repo brief

Read `README.md` for what this is. `docs/2026-09-19-holdover-design.md` is the authority on scope and
carries the rulings (§10), the build order (§9) and the re-aim (§12). `ledger/ledger.md` is the record and
is append-only by numbered entry; entry 0001 carries the chassis provenance, the inherited definitions (f\*,
τ_K) and the first rulings, entry 0002 the ruling that the instrument serves statistic (A) only, entries
0003 and 0004 the instrument re-pins (checkpoint selection; per-side `ModelRef` with verified provenance),
entry 0005 the pilot's registration, entry 0006 its outcome (f\* = 0 at every lag to 2400: HOLDS), and
entry 0007 the ruling that weight distance is the axis.

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
per rung) · `summarize_distance` (rungs by measured distance, tied to the dumps' manifests). `tools/` —
`perturb_checkpoint.py`, `convert_bin_checkpoint.py`, `weight_distance.py`: run with the instrument's
interpreter, import nothing from it. The (B) and (C) scorers do not exist yet (design §9 step 5).
`config/` — `seal.toml`, `pilot.toml`, `distance.toml`. `ledger/` — `ledger.md`, `predictions/`. `results/fstar/` and
`mappers/` — seal artifact roots, gitignored past their placeholders. `docs/` — the design doc.

## State
The pilot ran (ledger 0005, 0006): on OLMo-2 1B RLVR1, f\*(τ_K) = 0 at every lag from 200 to 2400 on
both anchors, every τ; the sealed DEGRADES at lag 2400 was falsified. Ruled (0007): the axis is weight
distance. The instrument is pinned at kv-transfer-replication `1380635` (the OLMo pair); the three Qwen2.5
pairs are a branch upstream awaiting merge and a re-pin. The weight-distance driver, summarizer, tools and
`config/distance.toml` (seven noise rungs, six real rungs, four pairs) are built and tested against the
stand-in; the config is unregistered, so `distance run` refuses. Next: the Qwen pairs merged and pinned,
the SFT/DPO conversions and the Qwen token file made on the run machine, the registering entry with the
backbone's sealed knee (R-D4), then the run.
