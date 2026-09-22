# lag-ladder

The measurement repo for **Holdover**. An asynchronous RL engine pushes new weights to its inference
workers while rollouts are in flight; if it leaves their KV cache alone, every token generated after the
sync is conditioned on state an older policy produced. Frameworks disagree by assertion — some reset on
every sync, some keep going and say it does not matter — and the only published number is one divergence
curve out to lag 32. Here the same quantity is put on a ladder from one update to thousands, per token
and per task, with the tolerance calibrated in [linear-ceiling](https://github.com/hossainpazooki/linear-ceiling).

The scope sentence, held verbatim: The ladder measures what a KV cache written under one policy
checkpoint costs when read under a later one, as a function of how many optimizer updates apart they
are; it does not train on stale rollouts and does not measure serving latency.

## How the record stays auditable

- `ledger/ledger.md` — numbered, dated, immutable entries; hypotheses registered before any run;
  every entry hashes the ones above it, and CI refuses an edit to a committed entry.
- `ledger/predictions/` — sealed pre-run predictions (`python -m lag_ladder.seal`).
- `config/*.toml` — every seed and threshold; nothing numeric lives in code.
- `UPSTREAM.md` — the pinned instrument, the (unpinned) checkpoint producer, and where each copied
  module came from.
- Numbers reach the ledger only through a fail-closed summarizer that recomputes them from `results/`.

## Setup

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Linux/web: .venv/bin/python
.venv/Scripts/python.exe -m pytest -q
```

## Docs

- `docs/2026-09-19-holdover-design.md` — the design, its rulings and the build order.
- Inherits `linear-ceiling/docs/2026-09-02-e-rl-design.md` for the instrument, controls and dumps,
  and `linear-ceiling/docs/gpu-experiment-protocol.md` for every GPU run.
