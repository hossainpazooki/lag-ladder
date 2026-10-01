# Handoff — ledger 0003 landed: the instrument is pinned at `9ca6258`; the next session's checkout question is closed

2026-10-01. Newest commit this brief describes: lag-ladder `6377036` (`main`, pushed, 0 ahead / 0 behind
at 03:33Z); kv-transfer-replication `main` = `9ca6258c91acf5c43b74026a7cc3649d61165104` (the
`holdover-instrument` fast-forward, verified `origin/main` = `origin/holdover-instrument`). The 2026-09-25
brief still describes everything else; this one records one thing — the re-pin — and what it unblocks.
Executed from `~/dev/briefs/2026-09-30-seed-lag-ladder-0003-repin-instrument.md`, every §2 precondition
re-verified before acting.

## Current state

| item | tag | re-verify |
|---|---|---|
| **Ledger 0003** — instrument re-pinned from `063f402` to `9ca6258` (Pair `revision`/`local_path`, `device()` with mps + `KVT_DEVICE`, `--dtype`, `--probe`, console entry points, CI); three entries chained | built, committed `6377036` | `.venv/Scripts/python.exe -m lag_ladder.ledger_check` → `ledger ok (blocks unchanged vs HEAD)`; `grep -c '^### ' ledger/ledger.md` → `3` |
| `INSTRUMENT_SHA` / `UPSTREAM.md` §1 agree on `9ca6258`; `063f402` no longer in `UPSTREAM.md` prose | built | `pytest -q` → `86 passed` (`test_imports` asserts the agreement); `grep -c 9ca6258 UPSTREAM.md src/lag_ladder/__init__.py` |
| **Upstream gate passes against the real checkout on the four (A) paths** | built | `.venv/Scripts/python.exe -c "from pathlib import Path; from lag_ladder import INSTRUMENT_SHA; from lag_ladder.upstream_gate import check_upstream; check_upstream(Path('../kv-transfer-replication'), INSTRUMENT_SHA, ('scripts/dump_kv.py','kvt/data.py','scripts/score_positions.py','kvt/pertoken.py'), who='pickup'); print('gate ok')"` |
| The two scorer paths are byte-identical to the old pin, so f\*(τ_K) is computed by the bytes linear-ceiling 0023 defined it with | built | `git -C ../kv-transfer-replication diff --stat 063f402 9ca6258 -- scripts/score_positions.py kvt/pertoken.py` → empty |
| Design §6 "fork" sentence replaced (per-step checkpoints are config) | built | `grep -n "no fork is needed" docs/2026-09-19-holdover-design.md` |
| Design §11 still says "through a fork whose only changes are additive config" | **stale, one sentence** — outside the seed's scope | `grep -n "fork whose only changes" docs/2026-09-19-holdover-design.md` → one hit |
| `config/pilot.toml` still `registered_by = ""`, `upstream_sha = "UPSTREAM_SHA_PENDING"` | planned (the registering entry fills both) | `grep -n -E "registered_by|upstream_sha" config/pilot.toml` |
| kv-transfer-replication PR #2 (G3 checkpoint manifests, per-side `ModelRef`) | planned; conflicts with the merged branch; needs the operator's design ruling; becomes a later entry | `gh pr view 2 --repo hossainpazooki/kv-transfer-replication --json state,mergeable` |
| Everything in the 09-25 brief's planned list (Mac mini pre-flight, §9 step 1 search, bridge controls, pilot driver/summarizer, (B)/(C) scorers) | planned, unchanged | that brief |

## Locked decisions

All of 09-22 and 09-25 stand. New:

| decision | reason | where |
|---|---|---|
| Pin = the branch **tip** `9ca6258`, never an intermediate commit | `e54c102` and `47bdd39` are individually broken trees (learning 2026-09-25); `9ca6258` is docs-only over the green `8b72f19`, and it is what `main` points at | ledger 0003 |
| Pin only after the fast-forward landed on `main`, not the unmerged branch | entry 0002: upstream changes "land upstream by the operator's commit" and are re-pinned after | seed §2; ledger 0003 |
| PR #2 is **not** in this pin and waits for a ruling | it conflicts with the merged branch in four files and competes with task 1's `Pair` design; a re-pin is one entry, and R12 wants the sha that actually ran on record | ledger 0003 "Not in this pin" |
| A seed's precomputed chain value is never pasted; re-derive on the written file | the seed's own value was wrong (learning 2026-10-01) | — |

## Reuse map

- The **gate one-liner** in Current state is the pre-run check every driver should call through
  `lag_ladder.upstream_gate.check_upstream` with its own `who=`; the four-path tuple is the (A) surface
  entry 0002 names.
- `kvt-dump --revision step_N --dtype bfloat16 --probe` / `kvt-score-positions` are now on the pinned
  bytes (no `PYTHONPATH`); `KVT_DEVICE=cpu|mps|cuda` forces the device for the kernel-identity control.
- Appending a ledger entry: write the block, then `chain_hash(text, heading.start(), entries_start)` on the
  file as written (two learnings now say so); `ledger_check --against HEAD` accepts an uncommitted trailing
  entry and refuses any touched committed block.
- The seed itself (`~/dev/briefs/2026-09-30-seed-lag-ladder-0003-repin-instrument.md`) is the template for
  the next re-pin entry (0004 for PR #2 or the (C) scorer re-pin): facts table with re-verify lines, the
  entry text, the three companion files, the gate.

## Invariants

The thirteen of 09-22 and the amendments of 09-25 hold. Specific to this state:

- **The pin is `9ca6258` on all three surfaces** (`__init__.py`, `UPSTREAM.md`, ledger 0003). Changing one
  without the others is a `test_imports` red, and changing the ledger one is a refused block edit.
- **`pilot.toml.upstream_sha` is still the placeholder.** The pilot driver must refuse on it (`check_upstream`
  refuses a non-40-hex sha by name). Do not copy `9ca6258` into it outside the registering entry.
- **Any upstream change after `9ca6258` to `scripts/dump_kv.py`, `kvt/data.py`, `scripts/score_positions.py`
  or `kvt/pertoken.py` turns the gate red** — including merging PR #2. That is the gate working; the answer
  is entry 0004, not a wider path tuple.
- Git history is the operator's; the seed-driven session only outputs commit blocks (this one was committed
  as `6377036` within minutes, as usual).

## Open / next

Unchanged from the 09-25 brief, now unblocked: **(1) Mac mini pre-flight** — `KVT_DEVICE=mps` vs `cpu`
kernel-identity on a 48-token toy against the E9 bound, written into a dated runbook with the mini's memory
and torch version; **(2) design §9 step 1** — the prefix-persistence prior-art search, carried three times;
**(3)** bridge controls registration, pilot driver/summarizer, pilot registration (R1/R2 still owed).

Small, do with the next design edit: §11's "fork" sentence. Needs a ruling: PR #2's G3 design vs task 1's
`Pair` fields (kv-transfer-replication's own brief has the comparison).
