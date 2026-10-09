# lag-ladder

What happens to a language model's KV cache when the weights change underneath it?

Asynchronous RL training runs generation and training as two loops. The trainer keeps producing new
weights; the inference workers keep generating. When a new checkpoint arrives, a worker already holds a
KV cache, the attention state for every token it has processed so far, and that cache was computed by
the old weights. The engine either recomputes it under the new weights, paying GPU time at every sync,
or keeps it and carries on. Systems in both camps state their choice. One divergence curve, 32 updates
deep on one model, is the only published measurement, and nobody reports where "it does not matter"
stops being true, or reports it in task units.

## What is measured, and where it is read

The scope sentence, held verbatim: The ladder measures what a KV cache written under one policy
checkpoint costs when read under a later one, as a function of how many optimizer updates apart they
are; it does not train on stale rollouts and does not measure serving latency.

The independent variable is *k*, the number of optimizer updates between the weights that wrote the
cache and the weights that read it. The same quantities are read at every rung.

| rung | *k* | what it is |
|---|---|---|
| short | 1 … 8 | an in-flight rollout crossing one sync |
| middle | 12, 32 | where prior work stops |
| long | 200 … 2400 | a prefix cache that is never invalidated |

The long end matters because of the prefix cache: a shared system prompt, tool schema or few-shot
block stays cached across every sync unless something clears it, so its age is unbounded.

| question | statistic | how it is read |
|---|---|---|
| (A) How much of the cache is stale? | f\*(τ), the fraction of tokens an oracle would have to recompute to bring the cache inside a tolerance τ | cache dumps only; no generation |
| (B) Does the trainer notice? | the stale-to-fresh importance ratio and its effective sample size | teacher-forced log-probs |
| (C) Does the task notice? | paired exact-match accuracy, stale minus fresh, with a bootstrap interval | generation under each cache |

τ is inherited from [linear-ceiling](https://github.com/hossainpazooki/linear-ceiling), where the same
statistic was calibrated on a cache carried across a change of *context* (the paper *Carryover*). Here
the context is held fixed and only the weights move.

## How a rung is measured

For each prompt, the old weights generate part of an answer and the new weights finish it, reading one
of four caches for the shared prefix. The tokens are identical across arms, so any difference is the
cache.

- **FRESH**: the cache rebuilt by the new weights.
- **STALE**: the cache as the old weights wrote it. FRESH against STALE is the question.
- **NULL**: the old cache plus random noise of the same size as the update. It asks whether any weight
  change of that size would do the same, or whether RL updates are special.
- **SCRAMBLED**: the cache of a different prompt. It must fail. If it does not, the instrument cannot
  see the cache and no result is read.

Statistic (A) needs no generation. It is read from cache dumps alone by the pinned instrument,
`kv-transfer-replication`, which this repository invokes and never imports.

## Where the ladder runs

- **OLMo-2-0425-1B-RLVR1**: thirteen published checkpoints, 200 updates apart, out to 2,400. This is
  the long end and the pilot: statistic (A) alone, dumps only, decides whether the rest is run.
- **An own RL run**, one checkpoint per optimizer step, for the short end. Deferred until the pilot
  shows signal.

## Status

The pilot has run: on a published RL ladder of thirteen checkpoints 200 updates apart, the K cache written
by the first checkpoint and read by any later one needs no token recomputed at any tolerance on the ladder,
because 2,400 updates of that run move the weights by a few hundredths of a percent. The ledger records the
ruling that followed: the axis is the distance between the writer's and the reader's weights, and the
optimizer lag is one way of producing it. A second ladder, over weight distance, is built and unregistered.
No hypothesis is registered. The ledger holds the founding record (chassis
provenance, inherited definitions, the first rulings), the ruling that the instrument serves
statistic (A) only, and two re-pins of the instrument as it gained checkpoint selection and verified
checkpoint provenance. The pilot's driver and summarizer exist: two controls (the anchor read by itself,
and one sequence's cache read against another's) run before any rung, and every number is recomputed
from disk. The pilot's configuration is unregistered; the driver refuses to run until a numbered entry
registers it. `ledger/ledger.md` is the record of what has been registered
and ruled, and it says what has not.

## How the record stays auditable

- `ledger/ledger.md`: numbered, dated, immutable entries; hypotheses registered before any run; every
  entry hashes the ones above it, and CI refuses an edit to a committed entry.
- `ledger/predictions/`: sealed pre-run predictions (`python -m lag_ladder.seal`).
- `config/*.toml`: every seed and threshold; nothing numeric lives in code.
- `UPSTREAM.md`: the pinned instrument, the checkpoint producer (observed, not pinned), and where each
  copied module came from. A gate checks the instrument's invoked paths against the pin before any run.
- `docs/2026-09-19-holdover-design.md`: the design, its rulings and the build order.
- Numbers reach the ledger only through a fail-closed summarizer that recomputes them from `results/`.

## Setup

```
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # Windows: .venv/Scripts/python.exe
.venv/bin/python -m pytest -q
```

## License

Apache-2.0; see `LICENSE`. The chassis modules are copied from the same author's `linear-ceiling`
(`UPSTREAM.md`); no third-party material is included. See `NOTICE`.
