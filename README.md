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
| (B) Does the reader notice? | the per-token KL between the reader's next-token distribution over its own cache and over the writer's, on the corpus continuation and on the reader's own greedy continuation | teacher-forced log-probs; one greedy generation per sequence |
| (C) Does the task notice? | the fraction of prompts whose greedy final answer changes under the stale cache, with a bootstrap interval; the paired accuracy difference beside it, descriptive | generation under each cache |

τ is inherited from [linear-ceiling](https://github.com/hossainpazooki/linear-ceiling), where the same
statistic was calibrated on a cache carried across a change of *context* (the paper *Carryover*). Here
the context is held fixed and only the weights move. (A) has been run and closed (ledger 0010). (B), defined for
this repository (ledger 0011; design §13) and registered with its seal committed (0012), has run and is closed on
the ladder (0015, 0016): the reader's KL grows as the square of weight distance on the isotropic backbone and
leaves HOLDS_B between 3e-3 and 1e-2, an order of magnitude inside (A)'s edge; trained directions read one to
three orders of magnitude above noise at matched distance. The own run (0017, 0018) read (B) along a 32-step GRPO
trajectory of Qwen2.5-1.5B: HOLDS_B at every lag on held-out text, four orders of magnitude higher on the policy's
own prompts. (C) ran (0025 registered, 0026 restated the dump-equivalence limit, 0027 read): the step-0 cache changes the greedy
answer on 71% of prompts at g = 1 and 87% at g = 32; all four sealed clauses held. A second trainer seed of the own run
ran (0028 registered, 0029 read): all five agreement clauses against the first seed held — distances within 1.3%,
held-out medians 0.62–0.91 of seed 1's, rollout medians 0.80–1.05, slopes 2.13 / 2.05, every control passed. A fourth
text is registered on seed 1's readers (0030, not yet run): fifty windows of the MLPerf Agentic Inference trajectories, the
same fifty linear-ceiling reads under its mapper, sealed to land with held-out text rather than with the trainer's prompts.
A second model family ran under the own run's recipe (0031 registered, 0032 read; design §14.5): Llama-3.2-3B with prompts
rendered plain because the base ships no chat template. Held-out text HOLDS_B at every lag (slopes 2.34 / 2.51, the on-path slope
0.006 past the sealed 2.5), the policy's own text UNRESOLVED from the first step at 2e3× to 6e3× held-out text (Qwen: 1e4× to
5e4×); every control passed; the excess over isotropic noise at matched distance 3.5× to 19× against the 10× sealed on every rung.
A third family is registered (0034, not yet run; instrument re-pinned pair-only at de6ce9e by 0033): SmolLM3-3B-Base under the same
recipe and plain prompts, its held-out windows byte-identical to Llama's because SmolLM3 ships the Llama 3 tokenizer.

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

**Statistic (B) on the weight-distance ladder is complete (ledger 0015, 0016).** On OLMo-2 1B, the median
per-token KL between the reader's next-token distribution over its own cache and over the writer's grows as
the square of whole-vector weight distance on the isotropic backbone (slope 2.0 over 1e-3 to 1e-2), from
6.0e-7 nats/token at 1e-4 to 5.6e-3 at 1e-2 and 4.1 at 1e-1: HOLDS_B (≤ 1e-3) through 3e-3, DEGRADES_B
(≥ 1e-1) at 1e-1, the edge one decade inside (A)'s. Trained directions are not isotropic to the reader: at
matched distance the OLMo-2 post-training steps read 355–875× the backbone and Qwen2.5-1.5B base → Instruct
11×. On the published RL chain a cache 200 to 2,400 updates stale reads 8.4e-5 to 8.1e-4 on corpus text, HOLDS_B
at every lag. Of the five sealed clauses, the quadratic backbone and the OLMo excess held; the backbone's HOLDS_B
reach, Qwen Instruct's nearness to noise, and on-path ≤ corpus without exception were falsified, each toward
a more sensitive statistic than sealed. (B) is a reader's number, not a trainer's; the operating lag of a
pipelined trainer (a few steps) is below the published chain's stride and was the own run's question. Entry 0019
restricts the statement to the held-out texts it names: on the trainer's own text (0018) the reader's KL is not
predicted by weight distance. Entry 0024 adds that text to the record, on one seed: on the trainer's own prompts (B1)
leaves HOLDS_B at the first optimizer step and by sixteen steps reads three quarters of a different prompt's cache and
one seventh of an information-free one, with the floor, near and far controls of 0021 and 0023 behind it. (C) is not
ruled on.

**The own run is complete (ledger 0017, 0018).** Qwen2.5-1.5B trained by GRPO for 32 steps (lr 1e-6, no
reference-KL term, GSM8K); the reader at step g ∈ {1, 2, 4, 8, 16, 32} reads the step-0 cache. The weights moved
2.3e-5 (step 1) to 1.05e-4 (step 32), concave and nearly flat after step 20, while reward rose fivefold. On
held-out text (B1) is HOLDS_B at every lag, 2.0e-7 to 6.7e-6 nats/token on corpus, and grows as the square of
distance (slopes 2.26 and 2.19), so the backbone's law holds a decade below where 0015 measured it. On the
policy's own prompts the same readers read 4.7e-3 (lag 1) to 8.9e-2 (lag 32): UNRESOLVED at every lag, four
orders of magnitude above held-out text, with the stale-prompt profile (KL largest at the first continuation
token, falling a hundredfold by position 50). Of the four sealed clauses, HOLDS_B at the operating lag and the
quadratic slope held; the 10× excess over isotropic noise (11× on corpus, 7.9× on on-path at the one in-range rung)
and rollout ≤ on-path ≤ corpus were falsified. Entry 0019 restricts (B)'s statement of record to the held-out
texts and makes rollout-text controls the prerequisite for any rollout number to become a statement of record; entry
0020 registers those controls and the sliding writer on held-out text; entry 0021 reads them from the run's dumps: the
controls pass per reader on rollout text (floor 1.5e-7 to 1.9e-7, scrambled 0.12), the rollout numbers stand 2.5e4× to
5.8e5× their floors and replicate across GPUs to 5e-5, and with the writer moved off step 0 the held-out law keeps its
slope (1.96 / 1.97) within a factor of two of the step-0 fit. 0019's prerequisite is met; whether the rollout reading
enters the statement of record is a ruling not yet taken: the scrambled control on rollout text is a near neighbour
(another GSM8K prompt, 0.12 against held-out text's 0.55), so entry 0022 registers a far control — a web prefix of the
prompt's length, carrying no information about it — and entry 0023 reads it: 0.45 to 0.67, 3.6 to 5.8 times the near
control; the step-0 cache at lag 16 to 32 moves the reader one seventh as far as an information-free cache and three
quarters as far as a different GSM8K prompt's. Both sealed clauses held; one seed remains the caveat.

The ledger holds the founding record (chassis provenance, inherited definitions, the first rulings), the
ruling that the instrument serves statistic (A) only, two re-pins of the instrument, the registrations with
their sealed predictions, the outcomes, and the two closures. Every number in it was recomputed from
`results/` by a summarizer that refuses on any mismatch, and the score records behind the outcomes are
retained off-repository with the hashes the entries state.

**Citing.** For statistic (A), cite this repository at the commit that carries ledger entry 0010 (tag
`statistic-a-closed`); name entries 0006 and 0009 for the numbers and 0001 for the statistic's provenance.
For statistic (B), cite the commit that carries entry 0016 (tag `statistic-b-closed`), read with 0019 and 0024; name
0015 for the ladder's numbers and 0011 for the statistic. For the own run, name 0018 for the numbers, 0017 for the
registration and 0021 and 0023 for the controls on rollout text.

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
