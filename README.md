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
the context is held fixed and only the weights move. Only (A) has been built and run; (B) and (C) are
designed and unbuilt (ledger 0010).

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

## Where the ladder ran

- **OLMo-2-0425-1B-RLVR1**, thirteen published checkpoints 200 updates apart, out to 2,400: the lag
  ladder (ledger 0005, 0006).
- **Weight distance directly** (ledger 0007, 0008, 0009), after the lag ladder showed that optimizer lag
  on a published run puts almost no weight distance on the axis: an isotropic-noise backbone on the same
  model at seven relative norms from 1e-4 to 1e-1, and six real training directions as labelled points
  (OLMo-2 1B SFT → DPO → RLVR; Qwen2.5-1.5B base → Instruct, base → Math, Math → R1-Distill).
- **An own RL run** for the short end: not run.

## Result and status

**Statistic (A) is complete (ledger 0010).** On OLMo-2 1B, the median f\*(τ_K) on the K cache is 0 for
every writer–reader pair whose whole-vector relative weight distance is at most 3e-2: every published RLVR
step from 200 to 2,400 updates apart, every post-training direction measured on the chain (SFT → DPO,
DPO → RLVR, SFT → RLVR, all below 1e-3), Qwen2.5-1.5B base → Instruct at 1e-2, and isotropic noise up to
3e-2. It is 0.999 at isotropic noise of 1e-1. A cache written under one checkpoint of a training run and
read under another needs no token recomputed at this tolerance; the sealed prediction that the backbone
would degrade by 1e-2 was falsified (entry 0009), as was the pilot's sealed prediction that the cache
degrades 2,400 updates out (entry 0006). The statistic is a step function of distance here: it does not
resolve the hundredfold difference in whole-cache deviation between a trained direction and isotropic
noise of the same size, which the entries record descriptively.

Nothing further is registered. Statistics (B) and (C) and the own run are not ruled on. The ledger holds
the founding record (chassis provenance, inherited definitions, the first rulings), the ruling that the
instrument serves statistic (A) only, two re-pins of the instrument, the two registrations with their
sealed predictions, the two outcomes, and the closure. Every number in it was recomputed from `results/`
by a summarizer that refuses on any mismatch, and the score records behind both outcomes are retained
off-repository with the hashes the entries state.

**Citing.** Cite this repository at the commit that carries ledger entry 0010; name entries 0006 and 0009
for the numbers and 0001 for the statistic's provenance.

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
