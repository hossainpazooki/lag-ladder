# lag-ladder — repo brief

Read `README.md` for what this is; `docs/2026-09-19-holdover-design.md` is the authority on scope and
carries the rulings (§10) and the build order (§9). `ledger/ledger.md` is append-only by numbered entry.
The paper's working title is *Holdover*; the repo is named for the instrument so a reviewer cannot walk
from the title to it.

## Rules that override nothing global but must never be broken here
- `../kv-transfer-replication` is **read-only** and pinned (`UPSTREAM.md`). Never write there, never
  `import kvt`, never copy its code. Borrowed facts carry `{sourceRepo, filePath, commitSha}`. It is the
  instrument for **statistic (A) only** (ledger 0002); the (B) and (C) scorers are built here.
- The chassis is **copied from linear-ceiling, not shared**. Do not import from it or symlink to it;
  a change there is not a change here until an entry says so.
- Never write a number into the ledger that was not recomputed from `results/` by a summarizer.
- Never edit a hypothesis after its experiment starts; never edit a sealed prediction.
- Seeds and thresholds live in `config/*.toml`. Randomness only via `lag_ladder.rng.make_rng`.
- `config/pilot.toml` is UNREGISTERED (`registered_by = ""`) until a numbered entry fixes it; no driver
  may run while it is empty.
- Every GPU run follows linear-ceiling's `docs/gpu-experiment-protocol.md` (R1–R12) and gets a dated
  runbook under `docs/`.

## Commands
```
.venv/Scripts/python.exe -m pytest -q                 # suite (synthetic, offline)
.venv/Scripts/python.exe -m lag_ladder.seal verify
.venv/Scripts/python.exe -m lag_ladder.lint_scope
.venv/Scripts/python.exe -m lag_ladder.ledger_check   # --against <rev> in CI
```
On Linux/web the interpreter is `.venv/bin/python`.

## Layout
`src/lag_ladder/` — `hashing` · `rng` · `config` (seal + pilot loaders) · `seal` · `upstream_gate` ·
`ledger_check` · `lint_scope`. No experiment driver or summarizer exists yet (design §9 steps 2–5).
`config/` — `seal.toml`, `pilot.toml`. `ledger/` — `ledger.md`, `predictions/`. `results/fstar/` and
`mappers/` — seal artifact roots, gitignored past their placeholders. `docs/` — the design doc.

## State (2026-09-22)
Scaffolded, nothing run, no hypothesis registered. Next: design §9 step 1 (prior-art search under the
prefix-persistence framing) and step 2 (upstream `Pair` revision field), then the pilot.
