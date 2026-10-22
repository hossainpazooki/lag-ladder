# Holdover — a KV cache held over a weight update (async RL on the weights axis)

**Working title (2026-09-23):** *Holdover: The Shelf Life of a KV Cache Across Policy Updates* (pairs
with *Carryover: The Reuse Margin of a Stale KV Cache* — a cache carried across a change of context; a
holdover is one held over a weight update; "Weight Syncs" for an MLSys audience, "Policy Updates" for
ARR/ICML). Fallback if the tail shows nothing: *Holdover: Stale KV Caches in Asynchronous RL, from One
Update to Thousands*. **Repo:** `hossainpazooki/lag-ladder`, package `lag_ladder`
— named for the instrument, not the claim, so it survives a reframing and does not lead a reviewer from
the title to the repo. Both locked 2026-09-20; the repo was scaffolded 2026-09-22 (§11) and this document
lives there.

**Date:** 2026-09-19 · **Status:** design, PROPOSED, unregistered, unnumbered, nothing run. **Status
2026-10-10:** statistic (A) complete and closed under the inherited instrument (ledger 0010); §12 carries the
outcome. **Status 2026-10-10, later:** statistic (B) ruled and defined for this repository (ledger 0011,
§13), unbuilt and unregistered; the run moves to RunPod; (C) and the own run not ruled. **Status 2026-10-15:** the
own run (§14) is complete and its rollout-text controls passed (0017–0021); §14.2 registers a far control (0022); §15
designs statistic (C) on the own run, unregistered and unbuilt. **Rulings
2026-09-22 (§10):** Oct 12 dropped, pilot first; prefix-cache persistence is the primary framing; OLMo is the
only source until the pilot shows signal; the repo is scaffolded today. Inherits
`docs/2026-09-02-e-rl-design.md` (the 09-02 design) for the instrument, lag ladders, controls, dumps and
upstream mechanics; this document states only what changes for an ARR paper. Where the two disagree, this
one governs the ARR paper and the 09-02 design governs nothing new. No figure here is a ledger figure.

Verified 2026-09-19: ARR October 2026 cycle — submission Oct 12, reviews due Nov 16, meta-reviews Dec 17,
commitment NAACL 2027 / COLING 2027 Dec 23 (`aclrollingreview.org/dates`). NOT verified: submission hour
and timezone, page limit, anonymity and concurrent-submission rules (not on that page).

## 1. The framing, with the asymmetry flipped

A cached KV goes stale when it is written under one condition and read under another. Two conditions can
shift: **context** (new position, new prefix — the handoff) and **weights** (written by θ_t, read by
θ_{t+k} — an async RL rollout that survives a weight update).

The two-axis framing failed mentor review for Carryover because the second axis was a design. Here the
asymmetry is reversed and that objection does not apply: the **weights axis is this paper's result**; the
context axis is **prior work, cited in the third person** (the non-archival workshop paper), used for one
thing only — the instrument and its τ arrive already exercised, so the weights axis is read on a yardstick
that has a measured point on it. One paragraph in §2 of the paper, no figure, no re-reporting of its
numbers beyond the τ values. If the brief's abstract table is reused: τ = 0.03 is **a tenth** of τ_K
(0.3186 / 10), not a thirtieth.

**Primary framing (ruled 2026-09-22): prefix-cache persistence.** The 09-19 draft read the lag axis as
the in-flight generation — a rollout that straddles one sync, lag 1–8 — and called the 200–2400 tail "no
engine runs there". That is true of in-flight generations and false of the *prefix* cache: a shared
system prompt, tool schema or few-shot block sits in the engine's prefix cache across every sync unless
something invalidates it, so its lag is unbounded (this is exactly SkyRL #2246's mechanism). Under that
reading the long ladder is the engine-relevant result, the 8-shot condition is primary, and the OLMo tail
is the main event rather than a descriptive appendix. The in-flight reading survives as the short end of
the same ladder. Prior-art search for this framing, run 2026-10-09 from the engines' source (shas in §2):
the premise holds as a **shipped default** in NeMo RL's async path (`recompute_kv_cache_after_weight_updates:
false`, prefix caching on by default on Ampere+) and in slime with `--flush-cache-interval <= 0`, as an opt-in
in SkyRL (`use_cache_salt: false`), and as a bug class (SkyRL #2246, open; vllm-omni #6739, cache resets were
no-op stubs until 2026-09). It is **bounded** where the cache is version-salted — prime-rl, SkyRL's default,
PSRL's `multi_version_kv`, TRL (a fresh adapter name per sync), open-instruct (a salt per request) — and
absent where every sync resets (verl, OpenRLHF, ROLL, AReaL, SGLang's `flush_cache=True`). PipelineRL's own
config disables vLLM prefix caching, so its stale cache is in-flight only; the SkyRL mechanism is not
PipelineRL's. vLLM's API makes the choice one boolean, `pause_generation(clear_cache=…)`, and its doc names
the False branch "stale KV cache" with no cost attached. Nobody has measured either side: the version salt
re-prefills every shared prefix after every sync, the keep-default serves prefixes of unbounded age, and the
ladder's (A)/(C) at lag k is the number both decisions are made without.

Topic sentence *(proposed under the ruling)*: "An async RL engine that does not invalidate its prefix
cache at a weight sync serves every later rollout from KV an older policy wrote. We measure, per token
and per task, what that costs as a function of how many updates old the cache is — from one update to
thousands."

## 2. What is new against PipelineRL (the sentence the paper needs)

Prior art, re-read from the primary PDFs on 2026-09-20 (rev 2 of this section; rev 1 called PipelineRL
"one operating lag", which is wrong — corrected below):

- **PipelineRL §5.1 (arXiv 2509.19128)** is the closest prior measurement and it *has a lag axis*.
  Qwen 2.5 base 7B, 222 optimizer steps, starting checkpoints 0 / 100 / 190, g_max = 32, L = 2048; the
  behavior policy swaps to the next checkpoint every L/g_max tokens. Figure 7 plots KL(μ ‖ π_{C+g})
  against lag g = 0…32 for three curves: Conventional RL, PipelineRL (stale cache), and "PipelineRL with
  KV cache recomputed". Text: "using stale KV-cache for mixed policy sequences introduces only slightly
  higher divergence compared to recomputing the cache." The stale-vs-recomputed gap is visible and grows
  with g in panel (a) *(read off the plot by eye, not a reported number)*. The conclusion calls it "the
  stale KV-cache risk" and says recompute "will still lower the throughput."
- **Magistral (arXiv 2506.10910), "Asynchronous generations":** one sentence, no experiment, figure, lag
  or model size attached — "For performance, we find that recomputing the key-value cache is not
  necessary, potentially due to off-policy corrections inherent to the loss function."
- **Nemotron 3 Super (arXiv 2604.12374) §3.2.4 and the Async RL Infrastructure paragraph:** "We do not
  recompute the KV cache after updating the model weights on the inference workers", with inference
  workers held to "at most one step behind the latest model version." It reports a design choice at
  lag ≤ 1; it does **not** report testing recomputation. A widely read survey post (Huang, 2026-05-31)
  says both Magistral and Nemotron "tested recomputing KV caches after weight syncs and found no
  benefit"; for Nemotron the report does not support that, and a reviewer may still believe it.
- vLLM ships the switch (`clear_cache`) with no guidance; AReaL recomputes; Laminar calls recompute a cost
  and staleness a risk, and measures neither (09-02 design §1, quotes verified there).

Concurrent and adjacent work, search run 2026-09-20 (18 sources) and re-run 2026-10-09 for the prefix-persistence
framing from the engines' source — vLLM, PipelineRL `58d3934`, prime-rl `88faa6d`, SkyRL `b896332`, NeMo RL
`c989568`, slime `c933787`, PSRL `0eeead8`, verl `fc72e2f`, OpenRLHF `dc2a7ad`, ROLL `581046a`, TRL `f4526e1`,
open-instruct `1182625`, AReaL `298412a`, SGLang docs — plus KVCOMM, C2C and the papers below (**coverage gap
remaining:** Nemotron 3 Super, Magistral, MiniMax, DeepSeek, GLM and LlamaRL were not re-read for a
prefix-cache policy; Kimi's flush policy was not found in `checkpoint-engine`). Marked *[checked]*
where I read the primary source myself, *[relayed]* where the wording came through a summarizer:

- **No in-RL stale-vs-recomputed experiment found beyond PipelineRL Fig. 7.** The systems split by
  assertion: reset on sync (Laguna 2605.27605, AsyncOPD 2606.24143, StaleFlow 2601.12784, RhymeRL
  2508.18588 — recompute discussed as a throughput cost only); avoid it by single-version trajectories
  (DORA 2604.26256); keep the stale cache "without hurting accuracy" (Olmo 3, 2512.13961 — its ablation
  table is throughput only). *[relayed]*
- **SkyRL issue #2246** (opened 2026-09-19, open, 0 comments) *[checked via gh api]*: stale prefix-cache
  blocks after a non-colocated weight sync, `max_staleness_steps=1`, |rollout − train logprob| mean 0.87 /
  max 28.6, grad_norm 8 → 136 on a 3-step smoke run. An unreproduced bug report, no model named, and the
  magnitude at lag ≤ 1 is implausibly large for a cache-only effect — motivation at most, not evidence.
- **arXiv 2609.17109** (Rajput, 2026-09-15) *[abstract checked]*: Qwen3-1.7B, base-written prefill KV read
  by a GSM8K LoRA, *paired* GSM8K EM, and a sweep of the boundary at which the reader takes over:
  Δ = −4.6 EM at a 160-token budget, −3.0 at 320, −0.8 under a second seed, "only the first excluding
  zero"; "partial recomputation provided no demonstrated advantage." Outside RL, one weight gap, no lag
  axis, no null control — but **paired GSM8K accuracy for a cache written under other weights is no
  longer new as a read-out**, and its boundary sweep is the same move as §3's s-ladder. Cite and
  differentiate. Its effect sizes (≤ 5 points, unstable across seeds, at a LoRA-sized gap) are also the
  best available prior for how small (C) will be.
- **KVShareArena 2609.10266** (already in the Carryover bibliography): "damage follows weight distance
  rather than lineage" from a two-point k-projection distance — a qualitative precursor of (iv), n = 2,
  uncalibrated. **LRAgent 2602.01053**: value-cache cosine across LoRA agents, descriptive, never
  regressed on task loss. **DroidSpeak 2411.02820**: owns layer-wise selective recompute under a
  writer/reader weight mismatch. *[relayed]*

Net: nothing found pre-empts (i), (ii)'s tail, the NULL arm, or (iv)'s calibration; (iii) must be worded
as *in-RL, fixed-policy, across a lag ladder*, not as a new kind of read-out.

Added 2026-10-09 *[checked]*: **Olmo 3 (2512.13961) §4.4.3** follows PipelineRL — "continue generating,
without invalidating the KV cache … up to 4x faster with the same resources, without hurting accuracy" —
and its Table 23 has no accuracy column; **Motif 3 (2608.09119) §5.2.2** "retain the existing KV cache rather
than recomputing it after each update", no measurement. **CacheReforge (2609.30884)**: stale KV under continual
LoRA updates at serving time, a per-layer sensitivity/drift statistic deciding reuse vs bounded recompute,
KL only (Qwen2.5-1.5B/7B, 16K QA) — the adapter-time precedent of (iv), without RL, task units, a lag axis or
a null arm. **"Aborted but Not Forgotten" (2608.15939)**: a "same-token/different-cache audit" — identical
decision-step tokens, cached prefix stale or rebuilt — on the *context* axis (same weights, a discarded
branch), seven families 3.8B–36B; the construction of (i), one axis over. **UniRL #94** (Tencent) measured
keeping the radix cache across syncs as flat on single-turn DAPO because "cross-sync KV is stale; the real
benefit is multi-turn / shared-prefix" — the throughput side of the trade, with the cost side unmeasured.
**vllm-omni #6739** saw stale prefix KV after updates as a logprob-divergence jump and doubled generation
length, unquantified. ThunderSyncRL (2610.05935), VenusRL (2610.03286) and CacheRL (2606.14179) do not touch
the axis.

So the field is split by assertion — keep the stale cache (PipelineRL, Magistral, Nemotron, Olmo 3) or
reset it (AReaL, Laguna, AsyncOPD, StaleFlow), or design around it (DORA) — and the only measurement
under either position is one KL figure to lag 32 on one 7B run. Nobody reports where "it does not
matter" stops being true, and nobody reports it in task units. That disagreement is the paper's opening.

Novelty *(proposed, to be attacked before any GPU spend)*:
**(i)** a paired, same-token contrast at a single clean cache age, **on the weights axis**. PipelineRL's two
curves are two separately *sampled* distributions each compared to on-policy by sequence KL, and the cache
inside one sequence is a mixture of ages 1…g; here the prefix tokens and reader weights are identical, the
cache is written by exactly θ_t, and the difference is read per token and per prompt. The construction
itself has a precedent on the context axis (2608.15939's same-token/different-cache audit, same weights);
claim the axis and the clean age, not the contrast.
**(ii)** range, *not* the existence of a ladder. Within lags ≤ 32 a ladder is prior art (PipelineRL
Fig. 7); what is new is the tail beyond it (OLMo, 200–2400), a second model family, and the NULL arm —
no prior report asks whether the effect is about RL updates or about any weight delta of that size.
**(iii)** fixed-policy paired task accuracy as the read-out. Prior read-outs are a sampling-distribution KL
(PipelineRL) and an undocumented training outcome (Magistral). Do not claim "first task-level evidence"
without that qualifier.
**(iv)** whether a cheap tensor-space statistic (f*, no generation needed) predicts the task-level loss —
i.e. τ calibrated in task units. Untouched in RL and in task units; CacheReforge (2609.30884) is the
adapter-time precedent of a statistic that decides recompute, read against KL. (iv) is the contribution an NLP reviewer
can use, and it is also the gap Carryover's reviewers will raise; it is also the item with nothing to
predict if (C) is flat, so it cannot carry the paper alone.

Consequence for §6 *(proposed)*: add own-run lags 12 and 32 — 32 so the ladder overlaps PipelineRL's
whole axis and (B)/(C) can be laid beside their KL, 12 because it is the lag at which the survey post
reports current IS corrections failing. Lag 32 from anchor step 8 fits inside the 40-step run.

If (i)–(iv) cannot be defended against PipelineRL §5.1 **and Figure 7** in one paragraph, stop here.

## 3. Construction: the mid-generation swap

For a task prompt P and lag k: θ_t generates greedily to completion (length T_t); the swap position s is
a fraction of **T_t** (the 09-19 text said "of the FRESH generation's length", which is circular — FRESH
depends on s). The "in-flight" prefix is y_<s with the KV θ_t wrote. The engine updates to θ_{t+k}.
θ_{t+k} completes the generation under one of:

| arm | cache read by θ_{t+k} | production analogue |
|---|---|---|
| FRESH | KV(P, y_<s) recomputed under θ_{t+k} | vLLM `clear_cache=True`; AReaL |
| STALE | KV(P, y_<s) as θ_t wrote it | vLLM `clear_cache=False`; PipelineRL |
| NULL | KV from θ_t perturbed by an isotropic delta of the pair's relative weight norm (09-02 §5) | — (the control that says an effect is about RL steps, not any delta of that size) |
| MAPPED *(appendix only; 3 lags)* | per-head linear map K_t → K_{t+k} | the linear-ceiling thread |

The prefix tokens are identical across arms, so any difference is the cache. **Every arm's KV is built by
the same teacher-forced prefill over (P, y_<s)** — STALE prefills under θ_t, FRESH under θ_{t+k} — never by
replaying θ_t's decode loop: a decode-written cache and a prefill-written cache are not bit-identical
even under the same weights (different kernels, shapes and reduction order, certainly in bf16), and the
k = 0 identity control would halt the run on numerics rather than on a real defect. This departs from
production, where the stale KV really was written during decode; one Limitations sentence. At k = 0 all
arms are bit-identical to FRESH; the run halts otherwise (09-02 §5 identity control, extended to the task
score) — **run on the GPU the grid runs on, not only on CPU** (the 09-19 Sep 26 gate said CPU; a CPU pass
does not transfer).

Decoding is greedy in every arm; production rollouts are sampled. One Limitations sentence.

Swap position s: 50 % of T_t for the main grid; {25, 50, 75} % at three lags only — 25 % is the
sensitive end (at 50 % most of a GSM8K chain of thought is already fixed in tokens identical across arms),
so if the grid budget forces a choice, the s-ladder outranks a lag. Prefix length: **8-shot primary**
(ruled 2026-09-22: under prefix-cache persistence the stale *prompt* KV is the engine's common case),
zero-shot as the contrast.

## 4. Three statistics, one per question

**(A) How much of the cache is stale — f*(τ_K), τ ladder {0.3186, 0.10, 0.03}.** Unchanged from 0023 and
the 09-02 design; τ not recalibrated. Needs dumps, no generation.

**(B) Does the trainer notice — stale-vs-fresh importance ratio, ESS / N.** As inherited from 09-02 §2(B).
**Superseded 2026-10-10 by §13** (ledger 0011): (B) is "does the reader notice", per-token KL as the verdict,
the importance-weight dispersion descriptive.

**(C) Does the task notice — NEW, verdict-bearing for this paper.** Per (task, lag, arm): final-answer
exact-match accuracy; the paired difference STALE − FRESH with a seeded paired bootstrap (2,000 reps) —
**the bootstrap interval carries the verdict**; McNemar on the discordant pairs is reported beside it, not
verdict-bearing; first-divergence position and continuation exact-match as descriptives (09-02 §5's
behavioral control, promoted by this document — a named amendment when registered).

**Positive control (added 2026-09-22; the 09-19 draft had none).** NULL controls for the *size* of the
delta, not for the instrument's sensitivity: with the swap at 50 % the final answer is largely decided in
tokens shared by every arm, so HOLDS_C is cheap by construction and a HOLDS from an instrument never shown
to fire is vacuous. Add one arm that must degrade — **SCRAMBLED**: the prefix KV taken from a different
prompt of the same length (same reader weights, same tokens y_<s) — and require DEGRADES_C on it at every
lag before any HOLDS_C cell is read. If SCRAMBLED holds, (C) cannot see the cache and the cell is
UNEVALUABLE, not HOLDS.

Multiplicity: the "crossing lag" is a maximum over cells, and no correction is claimed; each cell's
interval is reported as is and the crossing is stated as the first cell whose interval excludes the
HOLDS bound, with the count of cells named.

Bounds for (C) *(proposed; stated judgment, to be fixed in the registration entry)*:
HOLDS_C — paired accuracy drop ≤ 1.0 point and its 95 % interval below 2.0; DEGRADES_C — drop ≥ 5.0
points with the interval above 2.0; UNRESOLVED between. Per Nanda's threshold advice and the ledger's
habit, no claim rests on .01 < p < .05. *Two cautions for R2:* the band leaves 1–5 points UNRESOLVED,
which is where a practically material 3-point drop lands; and "n = 1,319 resolves ~1.5 points" holds only
near 7 % discordance (half-width ≈ 1.96·√(d/n)) — at 10 % it is 1.7, and the second task at n = 1,000 is
worse. The best available prior for the effect size is arXiv 2609.17109 (§2): ≤ 5 points and unstable
across seeds at a LoRA-sized gap. Expect UNRESOLVED at engine lags.

**The paper's actual claim is the relation between them:** at which lag each of (A), (B), (C) leaves
HOLDS, per source, and whether (A)'s crossing precedes (C)'s. Rank correlation of per-prompt f* with
per-prompt task flips is descriptive.

## 5. Tasks (the NLP task quality)

- **GSM8K test (1,319 problems)** — primary. It is the own run's RL reward and in OLMo-2 RLVR1's training
  mix, so the policy moves on it; verifiable final answer; a 0.6B–1B model sits mid-range, away from floor
  and ceiling *(to be probed at k = 0 before registration)*. Paired n = 1,319 resolves ~1.5 points.
- **One task the RL run never rewarded** — to separate "the cache is stale" from "the policy moved on this
  task". *(proposed)* IFEval for OLMo (in its mix; then it is a second in-distribution task) **or** a
  reading-comprehension set with long prompts (DROP or SQuAD v2 dev, 1,000 seeded examples), which also
  supplies the long-prefix condition naturally. Pick one at registration; two tasks total, not more.
  *Recommendation (2026-09-20, unruled):* neither option fits — IFEval is in OLMo's mix (fails the bullet's
  own purpose) and is not exact-match; DROP/SQuAD answers are ~5 tokens, so a mid-generation swap
  degenerates and the condition collapses into the 8-shot case. An unrewarded chain-of-thought task with a
  verifiable answer (ARC-Challenge, or a BBH subset) fits; check its absence from RLVR1's mix first.

## 6. Sources (ruled 2026-09-22: OLMo only until the pilot shows signal)

- **OLMo-2-0425-1B-RLVR1 — primary, and the only source this week:** 13 revisions, stride 200, lags
  200–2400 (base `OLMo-2-0425-1B-DPO`; `main` is not a lag point — 09-02 design pick-up). Under the
  prefix-persistence framing (§1) this tail is engine-relevant, not "decides nothing". The upstream
  per-side `Pair` revision pin it needs landed (ledger 0003; G3 `ModelRef` shape at 0004).
- **Own run — deferred.** Producer options, in order of preference once the pilot shows signal: the
  author's `ServiceNow/PipelineRL` unmodified (real in-flight updates; 4×H100 minimum documented, Linux,
  conda + flash-attn; per-step checkpoints are config — `save_checkpoint_steps` / `also_save_steps`,
  checked 2026-09-25 — so no fork is needed; not funded at this writing), else a
  single-GPU GRPO loop stated as a limitation. If run: Qwen3-0.6B, GRPO on GSM8K train, full-weight, a
  checkpoint every optimizer step to **step 40**; lags 1–8, 10, 12, 20 and **32** from anchor step 8 (= step
  40; the 09-19 text listed lag 40 from step 8, which needs step 48 — dropped; 12 and 32 added per §2). Add
  steps 80 and 160 if cheap: 40 steps of a 0.6B may never leave HOLDS.

## 7. What the paper can say in each outcome

- **STALE holds through engine lags on (A), (B), (C):** "`clear_cache=False` costs nothing measurable at
  lags ≤ 8 on these tasks; the recompute AReaL pays buys nothing here" + the tail shows where that ends.
  A null with a boundary; publishable only with the OLMo tail and the NULL arm.
- **(A) or (B) leaves HOLDS before (C):** the tensor statistic is conservative — useful as a cheap alarm.
- **(C) leaves HOLDS before (A):** τ_K is too loose in task units. This would also bear on Carryover and
  must be reported there; it is the most valuable outcome and the least comfortable.
- **NULL matches STALE everywhere:** the effect is about delta size, not RL; the paper is a short one.

## 8. Out of scope, stated

Whether *training on* stale-cache rollouts changes the final policy (ruled 2026-09-02: not an
RL-algorithms paper); any serving-latency number (that is the MLSys reading); a selective-recompute scheme
that achieves f* (the `[STRETCH]` of 0023). Each is a Limitations sentence; ARR requires that section.

## 9. Build order (ruled 2026-09-22: pilot first, no deadline of record)

Oct 12 is dropped (R6). The ARR October kill table of the 09-19 draft is retired; the next venue is decided
on the pilot's result — ARR December cycle or ICML if (C) has something to say, MLSys if only (A)/(B) do,
a short note if nothing does.

Inherited blockers (09-02 §7–8): upstream `Pair` revision / local-path field — landed (ledger 0003/0004);
no scorer exists for teacher-forced log-probs or generation under a supplied cache; (B)'s bounds blank.

| step | what | needs | decides |
|---|---|---|---|
| 0 | `lag-ladder` repo scaffolded (§11), design doc moved there, ledger 0001 | — | — |
| 1 | prior-art search under the prefix-persistence framing (§1) + the Q6 framework list the 09-20 search missed — **done 2026-10-09**: §2 survives; §1 premise holds for named defaults, bounded by version salts elsewhere | web | whether §2 survives the reframe |
| 2 | upstream `Pair` gains `revision` / local-path; pinned by ledger entry — **done** (0003; G3 per-side `ModelRef` at 0004) | kv-transfer-replication commit | first dump |
| 3 | **pilot:** (A) alone — f*(τ_K), τ ladder — on the 13 OLMo revisions, anchors step_200 and step_1200; dumps only, no generation. Driver + summarizer built 2026-10-09 (`lag_ladder.pilot`, `summarize_pilot`); registered by ledger 0005; **RAN 2026-10-09 (ledger 0006): f*(τ_K) = 0 at every lag to 2400 on both anchors, HOLDS throughout; 1 − R²(K) ≤ 7e-4 median, ≤ 1.2e-2 worst sequence** | one GPU-day at most; A100 class for 1B fp32 dumps | if f* ≈ 0 out to lag 2400: short note, lane closed. If f* crosses τ_K in the tail: there is a paper, and (iv) has something to predict |
| 3b | **weight-distance ladder** (§12; ledger 0007–0009): isotropic backbone 1e-4..1e-1 + six real directions — **RAN 2026-10-09 (ledger 0009): f*(τ_K) = 0 through 3e-2, 0.999 at 1e-1; sealed knee 3e-3 falsified.** Step 3's rule fired; **statistic (A) closed, ledger 0010 (2026-10-10)** | the Mac mini, ~1 h | the statement of record for (A) |
| 4 | registration entry: statistics, bounds (R2), tasks (R1), lags, seeds, SCRAMBLED positive control | pilot result | — |
| 5 | swap scorer for (C) upstream; k = 0 identity on the run GPU; (B) + (C) on OLMo | GPU | the verdict cells |
| 6 | own-run producer chosen (§6); own-run grid | funding | the short end of the ladder |

Compute for the pilot: 13 revisions × 2 anchors' partners ≤ 13 dumps of the held-out set at
131,072 B/token (OLMo 1B, fp16 K+V; 09-02 design) — sized at registration, not here. The full grid's
09-19 estimate (≈ 44k generations) counted one prefix condition and one source; with two prefix conditions
and OLMo as primary it is ≈ 1,319 × 12 × 4 arms × 2 ≈ 127k generations before the s-ladder and the
second task — still a few GPU-days on a ≤ 1B model, but not the number the draft gave.

## 10. Rulings

Ruled 2026-09-22 (option text quoted as picked):

- **R6 — "Drop Oct 12; pilot first."** §9 rewritten accordingly. Not a ruling on the venue.
- **R7 — framing: "Prefix-cache persistence."** §1, §3 (8-shot primary), §6 (OLMo tail engine-relevant).
- **R8 — checkpoint producer: "OLMo only for now."** §6; the own run and its producer wait on the pilot.
- **R9 — scope today: "Doc + scaffold lag-ladder."** §11; the design doc moves to the new repo.
- **Names (2026-09-20):** paper *Holdover*, repo `lag-ladder`, package `lag_ladder`.
- **R10 — instrument scope (2026-09-24): "kv-transfer-replication stays the instrument for (A) only."**
  Ledger entry 0002. (B)'s log-prob scorer and (C)'s swap scorer are built in `src/lag_ladder/`; the
  upstream's `Pair` revision field and G3 landed (ledger 0003/0004); bridge items G1/G4/G5 still owed
  (seed 2026-09-24).

Open, with the recommendation on record (unruled; a recommendation is not a ruling):

- **R1.** Second task — recommend an unrewarded chain-of-thought task with a verifiable answer
  (ARC-Challenge or a BBH subset), not IFEval or DROP/SQuAD (§5). Decide at registration (§9 step 4).
- **R2.** (C)'s bounds (§4, with the two cautions) and (B)'s blanks (09-02 §8). Decide at registration.
- **R3.** MAPPED arm — recommend **drop** from the paper body; appendix at three lags only if a page is free.
- **R4.** Vocabulary — recommend **lift** the ban on "recompute floor" once (A) is measured, not before;
  "stale fraction" until then.
- **R5.** Anonymity — mechanical: `lcfm_anon`-pattern copy at submission; the repo name `lag-ladder` does
  not appear in the title, and Carryover is cited in the third person. No ruling needed unless the venue
  is not double-blind.

## 11. Repo shape (locked 2026-09-20; scaffolded 2026-09-22)

- A separate repo, `lag-ladder`, with linear-ceiling's chassis **copied, not shared**: ledger + checker,
  sealed predictions, scope linter, `config/*.toml` seeds and thresholds, `rng`, `hashing`, the
  upstream ancestry gate, fail-closed summarizers, one CI job. Each copied module gets a provenance row
  `{sourceRepo: linear-ceiling, filePath, commitSha}`. GPU protocol linked, not copied.
- Ledger starts at 0001; 0001 cites linear-ceiling entry 0023 for f* and τ_K by `{repo, entry, sha}`.
- Two upstreams in `UPSTREAM.md`: the **instrument** `kv-transfer-replication` (pinned by numbered entry,
  0004 at this writing; per-side `Pair` revision pins landed; **statistic (A) only, by R10** — the (B) and
  (C) scorers live here) and the
  **checkpoint producer** — the author's
  `ServiceNow/PipelineRL` (Apache-2.0) unmodified — per-step checkpoints are config (`save_checkpoint_steps`
  / `also_save_steps`, checked 2026-09-25), so no fork (§6); the gate records the author sha and asserts
  trainer paths unchanged from it. Checked 2026-09-20: the
  author repo's code search has no `recompute` / `kv_cache` / `reset_prefix_cache` hit (default branch
  only), so Fig. 7's measurement code is not published; smallest documented config is 4 H100s.

## 12. Re-aim (ruled 2026-10-09, ledger 0007): weight distance is the axis

The pilot (0005/0006) read f*(τ_K) = 0 at every published lag to 2400 on OLMo-2 1B RLVR1: the cache responds
to the distance between writer and reader weights, and 2,400 RLVR steps put ~4e-4 of relative L2 distance on
the axis (descriptive, re-derived under the next registration). Consequences, by ruling:

- **Axis (R-D1):** relative L2 distance over the whole parameter vector; K-projection distance reported beside it.
- **Backbone (R-D2):** §3's NULL arm promoted — isotropic Gaussian noise added to RLVR1 `step_200` at relative
  norms {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}, each a local checkpoint read against the clean one; ten
  held-out sequences per rung; seeded.
- **Real points (R-D3):** Qwen2.5-1.5B base → Instruct, base → Math, Math → R1-Distill (content-space comparison;
  Math and R1-Distill use a different RoPE base) and the OLMo-2 1B chain SFT → DPO → RLVR1 (SFT, DPO converted
  from `.bin` to local safetensors checkpoints with provenance). The public OLMo-2 1B base is off the
  post-training lineage (near-zero weight cosine) and is excluded.
- **Framing (R-D5):** async RL stays the framing; the pilot's null is the first result, the distance curve
  explains it and locates the knee.
- **Open (R-D4):** the sealed prediction for the backbone's knee, at registration.

Build order from here: the perturbation and conversion tools (run with the instrument's interpreter; no
`kvt` import), a rung-based config and driver generalised from the pilot's (writer ref → reader ref, revision
or local path), the Qwen pairs upstream, then registration, seal, run.

**Outcome (ledger 0009, 2026-10-09) and closure (ledger 0010, 2026-10-10).** The ladder ran as registered: the
backbone reads median f*(τ_K) = 0 through 3e-2 and 0.999 at 1e-1 (knee bracketed, not located; at τ = 0.03 it
lies in (1e-2, 3e-2]); every real direction with whole-vector distance ≤ 1e-2 reads 0; Qwen Math → R1-Distill
(1.2e-1) and base → Math (1.1, not a small update of the base at this measure) read 0.993 and 1.000. The
sealed knee at 3e-3 was falsified, as was "real above backbone" under the rule's statistic; descriptively the
OLMo directions carry 78–130× the backbone's K-cache deviation at matched distance, Qwen Instruct sits on it,
Math → R1 below. Statistic (A) under the inherited instrument is thereby complete: a step function of
distance that answers *no* for every pair a training run produces. Entry 0010 states the citable result
and rules nothing about (B), (C) or the own run.

## 13. Statistic (B), "does the reader notice" (ruled 2026-10-10, ledger 0011)

**Why.** (A) is complete and a step function of distance (§12, 0010). The (A) run surfaced one thing (A)
cannot read: at matched distance the trained OLMo-2 directions move the K cache by two orders of magnitude
more than isotropic noise, and Qwen Instruct does not (0009, descriptive). (B) asks what the reader does
with such a cache. It is continuous where (A) is a step, needs no new instrument, and runs on the dumps 0008
produced. The 09-02 definition of (B) in §4 (importance ratio and ESS as verdict) is superseded by this
section; the operator ruled that linear-ceiling's 09-02 promotion does not bind this repository.

**Construction.** A dump at the pin holds K and V for every layer at every position of a 1,024-token
held-out sequence, from one forward over the whole sequence, stored float16, K with the writer's RoPE
applied. Attention is causal, so the first 768 positions are exactly the cache of the prefix P; the
remaining 256 tokens are the continuation. Two arms, both injected into the reader as the instrument's
`build_cache` / `forward_with_cache` do (re-implemented in `tools/`, importing nothing from `kvt`): FRESH,
the reader's own dump of P; STALE, the writer's. For the one pair whose RoPE base differs (Qwen base →
Math) the writer's K is stripped of its rotation and re-rotated for the reader, a borrowed fact from
`kvt/rope.py` at the pin. Two continuation texts, every statistic computed per text: *corpus*, the
sequence's own tokens teacher-forced; *on-path*, 256 tokens the reader generates greedily under the STALE
cache (the trainer's scenario), stored with their hash and then teacher-forced under both arms.

**Statistics.** (B1), verdict-bearing: mean per-token KL(π_fresh ‖ π_stale) over the continuation,
median over the held-out sequences, per rung and text; the direction follows linear-ceiling 0049. Beside
it, descriptive: the per-token p90 and maximum (a mean-then-median hides single-token spikes); (B2) the
per-sequence log importance weight Σ(log π_fresh − log π_stale) and ESS/N over sequences; (B3) the same at
token level; (B4) argmax agreement. (B2) and (B3) are importance-weight dispersion on text that is a sample
of neither policy, not a trainer's number.

**Bands.** HOLDS_B: median (B1) ≤ 1e-3 nats/token. DEGRADES_B: ≥ 1e-1. UNRESOLVED between. Ladder {1e-1,
1e-2, 1e-3} descriptive. Stated judgment; no published per-token-KL threshold was searched for. A
pre-registration probe measures the float16 floor on one reader on the run machine, is recorded in the
registering entry, and HOLDS_B is confirmed or lowered from it before the seal is committed.

**Controls, per reader, before its rung; halt on failure.** *Float16 floor* (halting): the reader's live
float32 cache of P against its own float16 dump of P, both read on the continuation; limits on the maximum
logit difference and the median KL fixed from the probe; a rung within 2× of the floor is "at the floor".
*Identity*: the same dump in both arms; a determinism check on a tolerance of 1e-5 in the maximum logit
difference, not a loader check. *Scrambled*: another sequence's prefix with this sequence's continuation
must read DEGRADES_B, or the reader's rungs are UNEVALUABLE. *Null*: the isotropic backbone.

**Sources.** The thirteen rungs of 0008 as dumped (six writers, thirteen readers); the noise readers are
regenerated from their seeds by `tools/perturb_checkpoint.py` and verified against the `out_sha256` in
their provenance before use, so a changed torch draw halts rather than drifts. Added: one Qwen2.5-1.5B
isotropic rung at whole-vector 1e-2 (a backbone of the Qwen reader's own model for clause (iv)) and the
pilot's lag ladder (0005) re-dumped at the current pin.

**Seal (structure here; values in the sealed file, committed before any score).** (i) Backbone (B1) grows
as the square of whole-vector distance over {1e-3, 3e-3, 1e-2}: least squares on log medians, slope in
[1.7, 2.3]; the rungs 3e-2 and 1e-1 are reported, not sealed, since at 3e-2 the cache deviation is already
of order 0.1 and f*(0.03) reads 0.998 (0009). (ii) The backbone leaves HOLDS_B at or below 1e-2. (iii) At
the nearest backbone rung on whole-vector distance, each OLMo-2 direction reads ≥ 10× the backbone's median
(B1): KL and 1 − R² are both quadratic in the same cache perturbation to first order, and 0009's deviation
ratios leave an eightfold margin. (iv) Qwen base → Instruct within 3× of the Qwen rung. (v) On every rung
whose corpus (B1) is above 2× the floor, on-path median (B1) ≤ corpus median (B1).

**Where it runs.** RunPod, under linear-ceiling's `docs/gpu-experiment-protocol.md` (R1–R12) on the shape
of its 2026-10-06 RunPod runbook; float32. The dumps and the SFT/DPO conversions travel from the Mac mini
by a private Hugging Face dataset in the R8 layout and are verified against 0008's manifests on the pod
before any forward; the Mac keeps its copy until then. The probe runs on the pod. The instrument is a clone
at `5d4aa4a` with `KVT_CHECKPOINT_ROOT` set before any score.

**Build (what 0012 calls).** `tools/score_continuation.py` (reader spec, fresh and stale dump dirs, token
file and index, `--prefix-len`, RoPE specs, `--live`, `--text corpus|on-path`, `--generate-under`; writes
`report.json` with every input's sha256); `src/lag_ladder/reader.py` (plan / check / run / probe on
`config/reader.toml`, the distance driver's gate order, controls per reader, resumable `run.json`);
`src/lag_ladder/summarize_reader.py` (recomputes everything from the reports, refuses on mismatch, withholds
a rung whose reader's controls failed, ties rungs to the dumps' manifests); fake-instrument tests;
`tests/test_imports.py` widened to `tools/`.

**Rulings (2026-10-10, all as recommended).**

| ruling | ruled |
|---|---|
| R-B1 | noise rungs keep (A)'s direction: reader = the perturbed checkpoint, regenerated and sha-verified |
| R-B2 | HOLDS_B ≤ 1e-3, DEGRADES_B ≥ 1e-1 nats/token; HOLDS_B confirmed or lowered from the floor probe at registration |
| R-B3 | Qwen base → Math included, re-rotated, labelled |
| R-B4 | the pilot's lag ladder re-dumped at the current pin for a (B) reading |
| R-B5 | seal clauses (i)–(v) as above; matched distance = nearest backbone rung on whole-vector distance |
| R-B6 | prefix 768 / continuation 256 |
| R-B7 | both texts, corpus and on-path, in this registration ("greedy variant is worth the compute at this stage of experimentation") |
| R-B7a | on-path text generated under the STALE cache |
| R-B8 | one Qwen2.5-1.5B isotropic rung at whole-vector 1e-2 |
| R-B9 | pre-registration float16-floor probe on `step_200`, on the run machine, recorded in the registering entry |
| R-B10 | float16 floor is the halting control, limits from the probe; identity on a tolerance, not "any nonzero" |
| R-B11 | dumps transferred to the pod (private dataset, sha-verified), not re-dumped; the run moves to RunPod |

**What (B) cannot say.** Nothing about task accuracy ((C), unbuilt); nothing about a cache older than
1,024 tokens or a prefix shared across prompts; nothing beyond 1.5B-scale models; nothing a trainer would
compute. A (B1) at the float16 floor is "at the floor", not zero.

**Outcome (ledger 0015, 2026-10-14) and closure (ledger 0016, 2026-10-14).** Registered by 0012 (floor probed,
HOLDS_B confirmed at 1e-3, noise rungs re-dumped on the run machine), amended by 0013 and 0014 (floor halts on
median KL only, limit band-relative at 1e-4; five R-B4 lag rungs), the ladder ran as registered on 19 rungs
and 18 readers with every control passed. Median (B1) grows as x² on the backbone (slope 2.02 / 2.00 over
1e-3..1e-2), HOLDS_B through 3e-3, UNRESOLVED at 1e-2 and 3e-2, DEGRADES_B at 1e-1; the OLMo directions read
355–875× the backbone at matched distance, Qwen Instruct 11×; the RL lags 200–2400 read 8.4e-5–8.1e-4 on corpus
text, HOLDS_B throughout. Seal: (i) and (iii) HELD; (ii), (iv), (v) FALSIFIED — the backbone leaves HOLDS_B before
1e-2, Qwen's instruct tuning is an order of magnitude above noise, and on-path exceeds corpus on the destroyed
reader at 1e-1. 0016 closes (B) on the weight-distance ladder with the statement of record and leaves the scorer
as the registered instrument for the own run (§14); it rules nothing about (C).

## 14. The own run: statistic (B) at lags 1–32 (approved in principle 2026-10-13; ruled all-as-recommended 2026-10-14; registration pending)

**Why.** 0015/0016 measured (B) on the published RLVR1 chain at lags 200–2400 optimizer steps: HOLDS_B at every
lag, 8.4e-5 to 8.1e-4 on corpus text, one to two orders of magnitude above isotropic noise at matched distance.
The lag a pipelined trainer runs at is 1 to 32 optimizer steps (PipelineRL §5.1, Fig. 7, `max_lag`), below the
Hub's stride by a factor of 6 to 200; 0015 bounds that regime from above and does not measure it. The only way to
put the operating lag on the axis is a trajectory of our own with a checkpoint every optimizer step. Steps are not
comparable across runs — an RLVR1 step moved the weights by ≈ 6e-7 relative (`olmo-rlvr1-lag-200`), an Adam step
at learning rate 1e-6 on Qwen2.5-1.5B is expected near 2.4e-5 (lr over the weights' RMS 4.2e-2) — so the reading
is made on the distance axis, as 0007 ruled, with the step count beside it; the published chain's lag rungs are
the long end of the same curve.

**Construction.** One short RL trajectory: GRPO on Qwen2.5-1.5B, full-weight, 32 optimizer steps from the
released checkpoint (step 0), one seed. After every optimizer step the just-saved checkpoint dumps the ten
held-out sequences and the ten rollout sequences through the pinned instrument (`dump_kv --local-path`, float32
forward, float16 dumps, as every dump here), its whole-vector distance to step 0 is written with
`tools/weight_distance.py` under the rung labels, and its weights are deleted unless the step is in the keep set
{0, 1, 2, 4, 8, 16, 32}. Kept weights are the readers and give the distance axis directly. Cells, §13's: writer
cache at step 0, reader at step g ∈ {1, 2, 4, 8, 16, 32} — six lag rungs, log-spaced, PipelineRL's axis — with
§13's controls per reader (run once, on the held-out pair) and both of §13's texts. A third text, *rollout*, is
the trainer's own scenario, which §13 could only imitate: for each of ten GSM8K training prompts (seeded, excluded
from the run's batches) the step-0 policy samples one completion at the trainer's temperature; prompt+completion
is one row of a padded token file with a per-sequence `prefix_len`/`length` sidecar (spans); the reader at step g
teacher-forces the completion under its own cache of the prompt and under step 0's. Rollouts are sampled before
registration, written to the two paths `config/ownrun.toml` names under the instrument checkout's gitignored
`data/tokens/` (where every held-out token file lives; the checkout's code stays untouched), and their hashes
pinned by the registering entry, like the floor probe of 0012. A sliding variant
(writer at t, reader at t+g) is not registered; the dumps exist at every step, so it can be scored later from the
same sitting's artefacts if an entry calls for it.

**Statistics, bands, controls.** §13's, unchanged: (B1) verdict-bearing, (B2)–(B4) descriptive; HOLDS_B ≤ 1e-3,
DEGRADES_B ≥ 1e-1; float16 floor (median KL, limit 1e-4 as 0014 set it), identity, scrambled, per reader, halting
as before. On *rollout* text (B2) is, for the first time, the dispersion of a quantity with the trainer's shape —
the log importance weight of a sampled completion between the fresh and the stale cache — though still not the
trainer's number (one sample per prompt, no advantage, no clipping); the entry says so. A rollout continuation
has its own length per sequence; token-level statistics pool the tokens, sequence-level ones take each row.

**Trajectory (R-C1, R-C2, R-C3).** TRL `GRPOTrainer`, pinned by commit at registration; transformers and torch
as the (B) sitting had them. Policy Qwen2.5-1.5B (base; `8faed761…`, 0012's pin). Task: GSM8K train, reward exact
match on the final number plus a 0.2 format term; prompts in the chat template with one fixed system line. Per
optimizer step 16 prompts × 8 completions, max 256 completion tokens, temperature 1.0, learning rate 1e-6, no
reference-KL term (β = 0: the movement is the policy gradient's alone and the distance axis is not shaped by a
pull back to step 0), fp32 master weights, seed 0. Expected distance at g = 32: 1.4e-4 if steps are uncorrelated,
up to 7.7e-4 if aligned — the band where the published chain read 8.4e-5 to 8.1e-4 and the backbone 5e-6 to
5e-5. The prediction below is a prediction.

**Seal (R-C5; structure here, values in `ledger/predictions/ownrun.json`, committed before any score).** (i)
HOLDS_B at the operating lag: median (B1) at g = 32 ≤ 1e-3 on corpus and greedy on-path text. (ii) Growth: the
slope of log median (B1) on log distance over the six rungs lies in [1.5, 2.5] on both texts. (iii) The RL
direction's excess reproduces: at every rung whose distance falls inside the backbone's measured range, (B1) ≥
10× the backbone's at the nearest rung (0015 read 66× to 875× for every trained direction). (iv) Rollout ≤ greedy
on-path ≤ corpus at every rung above 2× floor. Descriptive, not sealed: (B1) at g = 1 and whether it is at the
floor.

**Where it runs (R-C7).** One RunPod sitting, secure A100-SXM4-80GB, under the GPU protocol and the Part-C
runbook: training ≈ 1 h (no vLLM), 330 dumps ≈ 30 min, distances ≈ 10 min, the reader run ≈ 2 h (6 readers × 40
control jobs + 12 rungs × 10 sequence-texts ≈ 420 jobs). Budget ≈ $7, watchdog with both ceilings set. Artefacts:
the results tar, the dumps' manifests, seven kept checkpoints with provenance (sha256 of the safetensors), 32
distance files; dumps and kept weights go to the private dataset as 0008's did. Nothing from the trainer's logs
enters the ledger except the per-step reward mean, descriptively, so a reader can see the policy moved.

**Build (what the registration calls; built 2026-10-14).** `tools/own_run.py` (`sample-rollouts`, `train` with
the save hook: dump → distance to step 0 for every step under `results/distance/ownrun/`, copied to each rung label
the config gives that step → delete; the labels are read from `config/ownrun.toml`'s `local:step_<g>` readers; run
with the trainer's interpreter, imports nothing from `lag_ladder` or `kvt`); the loader refuses a reader whose first
rung is a rollout rung, so controls always run on the held-out pair; `lag_ladder.reader`: a pair's own `heldout`, a `spans` sidecar giving `--prefix-len`/`--length` per
sequence, a per-rung `texts`; `tools/score_continuation.py`: `--text rollout` (teacher-forced like corpus) and
`--length`; `summarize_reader`: rollout verified against the token file like corpus, per-rung texts, ragged
continuations; `config/ownrun.toml`, UNREGISTERED, with `ROLLOUT_*_PENDING` pins until the rollouts are sampled.
The seal file is written at registration.

**What §14 cannot say.** Nothing about in-flight updates as an engine performs them: the trainer is synchronous
and the stale cache is modelled, as in §13, by dumping the writer's cache and reading it under the reader. Nothing
beyond one seed, one 1.5B model, one task, 32 steps; a result at the floor means "a 1.5B GRPO step moves the
weights too little for the reader to notice at 1e-3", not "lag is harmless". Nothing a trainer would compute.

**Decision rules.** If (i) holds and g = 32 is within 2× the floor, a second registered sitting extends the
trajectory to 128 steps at a larger learning rate (an amendment, not a new design). If (i) falls, the paper's
sentence is the distance at which HOLDS_B is left, and the sliding variant is scored from the same dumps to show
it is not a step-0 artefact. If (iii) falls, the RL direction's excess over noise is a property of the long-trained
chain, not of a step.

**Rulings (2026-10-14, all as recommended).**

| | ruling |
|---|---|
| R-C1 | model: Qwen2.5-1.5B base (`8faed761…`, 0012's pin); not binding |
| R-C2 | trainer: TRL GRPO; lr 1e-6, β = 0, 16 × 8 per step, T = 1.0, fp32 master, seed 0 |
| R-C3 | 32 steps; keep {0, 1, 2, 4, 8, 16, 32}; dump every step |
| R-C4 | texts: corpus + greedy on-path (§13) + rollout; rollout registered, not deferred |
| R-C5 | seal clauses (i)–(iv); (B1) at g = 1 descriptive |
| R-C6 | rollout prompts: ten GSM8K train prompts excluded from the batches, seeded; one sample each at T = 1.0, ≥ 64 tokens or resampled |
| R-C7 | dumps and kept weights to the private dataset; reports tar and summary off-repository with hashes |
| R-C8 | the PipelineRL Fig. 7 comparison is stated descriptively on the lag axis; no claim about their run |

**Rulings (2026-10-14, after 0018 and 0019; R-C9–R-C13).** Neither of the decision rules above fired: (i) held and
lag 32 read far above the floor, so neither the 128-step extension nor the sliding variant is called by them. What the
rules did not anticipate is the rollout text (0018, 0019). The rulings answer to that.

| | ruling |
|---|---|
| R-C9 | rollout-text controls: registered now (0020) as a second reader configuration over the own run's dumps — float16 floor and scrambled per reader on the rollout pair under 0014's limits; no retraining; 0019's prerequisite |
| R-C10 | the sliding writer: held-out text only, descriptive, in the same sitting — writer and reader at kept steps, placed by the measured writer–reader distance; rollout sliding deferred (it needs rollouts sampled by the step-w policy) |
| R-C11 | a second seed: deferred until 0020's outcome is read |
| R-C12 | the 128-step extension: kept open, decided after 0020's outcome; any run would be at a larger learning rate (distance saturates at 1e-4 by step 20) |
| R-C13 | statistic (C): ruled in principle — 0018 gives it something to predict on rollout text; §15 is designed after 0020's outcome; the scorer stays unbuilt until registered |

**Rulings (2026-10-15, after 0021; R-C14–R-C18).** 0021 met 0019's prerequisite. What it did not settle: one seed, and a
scrambled control on rollout text that passed its limit by 15–26% because another GSM8K prompt's cache is a near
neighbour — the 0.77 stale-over-scrambled reading at lag 16 has a denominator that carries the format and the system line.

| | ruling |
|---|---|
| R-C14 | the statement-of-record amendment for the rollout reading: **not taken** — too soon; the direction and size of the gap are on record through 0019, the magnitudes and the 0.77 ratio wait for R-C15 and a second seed |
| R-C15 | a far scrambled control on rollout text: **registered now (0022, §14.2)** — a held-out web prefix of the rollout prompt's length under the same completion, an information-free cache; the near control stays and both must pass |
| R-C16 | a second seed: deferred until the paper's sentence is needed (a trainer sitting; the only thing that addresses one-seed). **Ruled 2026-10-20: a second trainer seed; its (B) half before the submission (§14.3, registered by 0028); (C) on the second seed deferred past the reviewer round** |
| R-C17 | the 128-step extension: **closed** (nothing in 0021 reopened its trigger) |
| R-C18 | statistic (C): **§15 designed now**, built and registered after 0022's outcome; the scorer stays unbuilt until registered |

### 14.1 Controls on the rollout text and the sliding writer (0020)

`config/ownrun_controls.toml`: the same instrument, pins, pairs, bands and limits as `ownrun.toml`, which stays as run,
with `per_pair = true` — every reader's controls run per pair its rungs use, on that pair's own texts (floor on the
pair's texts; identity and scrambled on its first text, `rollout` for the rollout pair). Rungs: fifteen sliding rungs
`ownrun-slide-<w>-<r>` (w < r, both in {1, 2, 4, 8, 16, 32}) on the held-out pair, corpus and on-path, placed by
`tools/weight_distance.py` on the kept weights; 0017's six rollout rungs re-run as a descriptive replicate (the record
stays 0018). Seal `ownrun-controls` (pre-fit): (i) every rollout rung of 0018 reads above 2× its reader's rollout
floor; (ii) the slope of log median (B1) on log writer–reader distance over the fifteen sliding rungs lies in
[1.5, 2.5] on both texts. 740 jobs, inference only, from the private dataset's dumps and kept weights.

### 14.2 The far scrambled control on the rollout text (0022)

`config/ownrun_far.toml`: the same instrument, pins, pairs, bands and limits as `ownrun_controls.toml` (which stays as run,
0021), `per_pair = true`, and the rollout pair names the held-out pair as its `far_pair`. For each reader g ∈ {1, 2, 4, 8,
16, 32} the controls on rollout text are the float16 floor, identity, scrambled (0020: another rollout sequence's prompt
cache under the same completion, `scrambled_min` 0.1) and **far**: the reader's own float16 dump of held-out sequence
40 + i, of which the scorer reads the first prefix_len_i − 1 rows exactly as for every arm — a non-GSM8K prefix of the
rollout prompt's token length, carrying no information about the prompt — under rollout sequence i's stored completion;
median over the ten sequences, limit `far_min` 0.1; under it the unit is UNEVALUABLE, as scrambled. The held-out dumps at
the kept steps already exist on the private dataset (1,024 tokens, so the cut is always possible); nothing is re-dumped and
the scorer is unchanged (it checks `T ≥ rows` on the stale dump). 0017's six rollout rungs run again as a second
descriptive replicate; the record stays 0018. The summarizer gains, per rung and text, the rung's median over its unit's
scrambled and far medians (`over_scrambled`, `over_far`), and per unit `far_over_scrambled`. 300 jobs, inference only.

Seal `ownrun-far` (pre-fit): (i) far ≥ 2× scrambled on every (reader, rollout pair) unit — the information-free cache is
at least twice as far as the near neighbour; (ii) median (B1) over far ≤ 0.5 on every replicate rung (summarizer) and on
0018's six rollout medians (by hand) — the step-0 cache stays nearer than halfway to an information-free one at every lag
to 32. Falsified (i): the format is most of the distance and 0021's 0.77 stands as read. Falsified (ii): by sixteen steps the
stale cache is as bad as no prompt. Not registered: a chat-formatted non-GSM8K prompt (same system line, other task) as an
intermediate control — a later registration if far reads large and near small.

### 14.3 The second trainer seed (R-C16 as ruled 2026-10-20; 0028)

0024's statement of record for the trainer's own text carries the qualifier "on one seed". The second seed is a second
trajectory under 0017's recipe — the same `tools/own_run.py` at the same commit with the same arguments, the trainer seed
1 in place of 0 — with per-step dumps, kept steps {0, 1, 2, 4, 8, 16, 32} and the (B) reader run; not a second bootstrap
seed on seed 1's records, which moves interval edges and says nothing about the trajectory. `config/ownrun_seed2.toml`:
the same instrument, pins, pairs, bands and limits as `ownrun.toml`, which stays as run; `per_pair = true` so 0017's and
0020's controls run in the one sitting; the rollout pair names the held-out pair as `far_pair` with 0022's `far_min`, as
evaluability insurance rather than as a replication of 0023: `evaluable_under_far = true` pre-commits that a reader whose
rollout scrambled misses 0.1 but whose far clears it is EVALUABLE under the far control (seed 1 cleared scrambled by
15–26%), the driver running its rungs and the summarizer judging them, with the unit marked `evaluable_under = "far"`;
a reader that misses both is UNEVALUABLE as before. The rollout rows are 0017's: the step-0 policy's samples under the
same step-0 checkpoint, so the same text is read under both trajectories and the comparison is on the trajectory alone.
Twelve rungs `ownrun-seed2-lag-<g>` and `ownrun-seed2-rollout-lag-<g>`; 660 jobs.

The clauses are agreement clauses against seed 1's readings (0018), whose medians the seal carries, not 0017's bands
re-tested. Seal `ownrun-seed2` (pre-fit): (i) held-out text HOLDS_B at every lag on both texts, above 2× floor, and the
slope of log median (B1) on log distance over the six held-out lag rungs in [1.5, 2.5], both texts; (ii) at every lag the
rollout median (B1) within a factor of three of seed 1's at the same lag, and at g = 1 at least 1e-3 — HOLDS_B is lost at
the first optimizer step on this seed too, kept as its own half because it is the paper's sentence; (iii) at every lag the
rollout median at least 1e3× the same reader's median on corpus and on on-path text (seed 1: 1.3e4× at the weakest point);
(iv) every (reader, pair) unit passes its controls under 0014's limits, no unit UNEVALUABLE. The summarizer reads all four
(`i-holds`, `i` for the slope, `ii`, `iii`, `iv`). Descriptive: the per-step distance curve against seed 1's thirty-two
values — expected to agree closely and uninformative, since Adam's first step has a magnitude nearly independent of the
batch (seed 1's two-prompt smoke read a step-1 distance within 1.1% of the registered run's) while the seed acts on the
update's direction, which the rollout KL sees; the reward and entropy trajectories; rollout over scrambled per lag.

Halting: the step-0 weights the trainer saves must hash to seed 1's `3779c0b9…` before the first optimizer step — the
same checkpoint under the same libraries writes the same bytes, so a different digest is an environment drift, not a
seed, and the sitting stops before it bills the trajectory; a step-1 distance under 1e-6 stops the sitting before the
reader run (0017); a floor over 1e-4 halts the reader run (0014). Not registered on this seed: the sliding writer, (C).

## 15. Statistic (C), "does the task notice", on the own run (designed 2026-10-15, revised after review the same day and after two probes; built 2026-10-15; registered by 0025)

**Why.** (B) at the operating lag on the trainer's own prompts: the reader notices from the first optimizer step (0018),
with per-reader controls passed (0021). (C) asks whether the task output changes — the number a practitioner weighs
against the throughput a persistent prefix cache buys. §4 defined (C) for a mid-generation swap on a published chain; this
section redefines it for the own run, where the swap is the whole prompt cache at the prompt/completion boundary — the
pipelined trainer's case — and §3's mid-generation construction is not used.

**Prior from (B).** On rollout text, teacher-forced, the per-token argmax agreement between the fresh and the step-0 cache
reads 0.980 at lag 1 and 0.898 at lag 32 (0018, descriptive), with the KL concentrated at the first continuation tokens.
Under greedy generation of ~200 tokens a 2% per-token flip means nearly every completion diverges somewhere, so
continuation exact match is ≈ 0 at every lag and carries nothing; what can move is the **final answer**: whether a
diverged path lands on the same number. arXiv 2609.17109 (§2) is the only prior for the size — ≤ 5 points, unstable
across seeds, at a LoRA-sized gap. Expect UNRESOLVED on accuracy and a readable flip rate.

**Construction.** Prompts: GSM8K test, all 1,319, in the run's chat template with its system line; gold = the `####`
number. Reader at kept step g ∈ {1, 2, 4, 8, 16, 32}. Per (prompt, g) four arms, each a greedy generation of ≤ 256 new
tokens from the same prompt under an injected prompt cache: FRESH (the reader's own cache), STALE (step 0's — the
writer's), NEAR (another test prompt's cache under the reader's weights: the next prompt in ascending token-length
order, so the source is at least as long and is cut to this prompt's length; the longest takes the shortest's, padded —
the one pairing that is not a true cut), FAR (a held-out web sequence's cache under the reader's weights, cut to this
prompt's length). Caches are built live — one float32 forward of the arm's weights over the
prompt, rounded through float16 (the dump path's precision) — and injected as `tools/score_continuation.py` injects a dump;
nothing is written to disk at this scale (1,319 × 6 × 2 dumps would be ≈ 90 GB). The final answer is the run's reward
extraction: the number after the last `####` marker, else the last number in the completion; exact match against gold.
The number after the *first* `####` is kept beside it, descriptive: the probes found greedy completions that restate
`#### N` to the cap.

**Statistics (as revised 2026-10-15 after the probes; R-C19).** (C1), verdict-bearing: per g, the **answer-flip rate** —
the fraction of the 1,319 prompts whose final answer differs between STALE and FRESH, right or wrong — with a seeded
bootstrap over prompts (2,000 reps) carrying the interval. (C2), descriptive: Δ = acc(STALE) − acc(FRESH) in points with
its seeded paired bootstrap and §4's bands (the first draft's verdict statistic, demoted because the probes read the
step-32 policy at 6% under greedy decoding at the cap — 48 of 64 completions capped, 40 of them a repeated `#### N`
line, the same at a 1,024-token cap — so a one-point band is a sixth of the baseline and carries nothing; the cap is
not the limit, the policy's greedy loops are); the per-arm no-answer rate (no extractable number, the cap reached) —
a stale cache can change the truncation rate itself, so it is recorded per arm, not folded into "wrong"; the
first-divergence token position; and the (B)→(C) link: per test prompt, (B1) is computed by the same tool on the FRESH
arm's greedy completion teacher-forced under the FRESH and the STALE cache (one extra forward; §13's on-path text with
the fresh path as the continuation — a new text for (B1), named so, descriptive), and the per-prompt flip is rank-correlated
with it. (B)'s rollout (B1) of 0018 exists for ten training prompts only; nothing in (C) is correlated with it. McNemar
on the discordant pairs beside the interval, not verdict-bearing. Multiplicity as §4: six cells, no correction claimed,
the count named.

**Bands (§4's, the sign fixed; descriptive under R-C22).** HOLDS_C: |Δ| ≤ 1.0 point with the interval inside ±2.0.
DEGRADES_C: Δ ≤ −5.0 with the interval below −2.0. UNRESOLVED between. The interval's half-width is ≈ 1.96·√(d/n) for
discordance d: 1.5 to 1.7 points at 7–10%, so HOLDS_C needs discordance well under 7%. The bands were drafted (§4) with
no baseline in mind; the probes read FRESH at 4 of 64 (6.25%), so a one-point band is a sixth of the baseline. R-C22
(2026-10-15): the bands stay as written and are read as descriptives; the verdict is the flip rate's. No band on flips
is defined — the seal names thresholds (10%, 50%) and a growth clause instead.

**Controls, per g, before the cells; the cell is UNEVALUABLE without them.** Positive: NEAR and FAR must each flip at
least half the answers against FRESH (`control_min_flip_rate` 0.5; the instrument fires on a wrong cache; §4's SCRAMBLED,
in two strengths, as §14.2; the probes read 0.906 and 1.000 at g = 32 — the first draft asked for DEGRADES_C, which at
a 6% baseline NEAR and FAR reach only at n = 1,319, not at the probe's 64). NEAR is a stronger control here
than in (B): under another prompt's cache the model answers the other question, and accuracy against this prompt's gold
collapses whatever the format shared — 0021's near-neighbour worry does not transfer, though the probe confirms NEAR
degrades before the full run spends its generations on it. Floor: FRESH built through float16 against FRESH in float32
— the answer-flip rate between them, halting above 1% (13 of 1,319 per reader; both probes read 0 of 64, which bounds
the rate at 4.6% by the rule of three, so 1% is a margin ten times under the seal's 10%, not a measured ceiling; a hard
zero would be six chances per sitting to halt on one prompt). Identity: FRESH twice — zero flips, a determinism check.
**Dump equivalence, once per sitting, before any cell** — the control that ties this tool to the one every (B) number
came through: for the ten rollout prompts of 0017 the tool builds the step-0 cache live from the step-0 weights and
compares it layer by layer with the step-0 dump on the private dataset, and reproduces 0018's rollout (B1) at one lag
(g = 32, 8.914e-2) by teacher-forcing the stored completions under the live caches. R-C21 as restated by 0026: the gap
limit is **absolute, 2⁻⁵** — one float16 ulp in the top binade the caches occupy — with the (B1) reproduction within 1e-3
relative as the material check; the per-element relative gap and the share of elements that differ after the rounding
are descriptives in the record. 0025 had registered the limit as one ulp *relative to the element* (2⁻¹⁰, float16 minimum
normal as the floor) on the probes' reading of exactly one ulp at the max-abs element (2⁻⁷ at magnitude 8–16, 2⁻⁶ at
16–32 on another torch build); the sitting of 2026-10-16 halted on it at 0.33 with the same 2⁻⁷ absolute and the same
(B1) to seven digits. The discrepancy is float32 forward-to-forward noise (prompt-only forward against the dump's
full-row forward; the live path is deterministic), of order 1e-5 absolute whatever the element's size: on a V element
of magnitude 1e-4 that is hundreds of its own ulps, so no live cache meets a per-element relative limit; large elements
flip one float16 ulp or none. The absolute limit is what the noise model supports, with 4× margin over 2⁻⁷ and 2× over
2⁻⁶. A rounding point, a layout or a scaling factor wrong in the re-implemented injection shows here, not in the verdict
cells.

**Seal (R-C23; `ledger/predictions/ownrun-task.json`, 0025).** (i) At every g NEAR and FAR each change the answer on
at least half the prompts. (ii) At g = 1 the STALE flip rate is ≥ 10%: the greedy output is not reproducible from the
first optimizer step. (iii) At g = 32 the STALE flip rate is ≥ 10%. (iv) Growth: the g = 1 flip interval lies entirely
below the g = 32 interval. Read, by pre-commitment, as "the greedy output changes under a stale cache", not "the task
degrades": a flip with Δ ≈ 0 is path divergence that lands on a different, equally (in)accurate answer, and a
practitioner weighs reproducibility and accuracy separately; the accuracy claim is the descriptive Δ's. The asymmetry is
stated: both probes read g = 32 only, so (i) and (iii) at g = 32 are confirmations (55 of 64 STALE flips, twice) and
**(ii) and (iv) are the blind predictions** — (B1) at g = 1 is an order of magnitude under g = 32 (0018), and whether a
cache one step stale still turns the greedy answer on a tenth of the prompts is what the sitting tests. The first draft's
(ii), a location clause on Δ at g = 1, is withdrawn with the statistic: at a 6% baseline the interval's location says
nothing. Descriptive, not sealed: Δ at every g; FRESH accuracy; the first-`####` answer beside the registered one; the
flip rate at g ∈ {2, …, 16} and its growth against (B1)'s; the no-answer and cap rates per arm; NEAR against FAR.

**Where it runs; cost (estimate, not a record).** One A40 sitting under the GPU protocol: 1,319 × 6 × 4 ≈ 31.7k greedy
generations of ≤ 256 tokens plus the cache forwards, batched; 3–5 h, $2–3; the probe first (it also sizes the TTL). Step 0
and the kept weights from the private dataset; the step-0 dump of the rollout prompts for the equivalence control.
Records: per (g, arm) a JSONL (`task-record-v2`) of {index, prompt_len, answer (None when no number is extractable),
the first-`####` answer, reached_cap, gold, correct, completion sha256 and the decoded completion text, first divergence
vs FRESH, (B1) on the fresh path}, the summary, hashes in the outcome entry. The text is kept because the first probe
stored only the hash and its three confounds (cap, loops, extraction) could not be separated offline; the second probe
separated them.

**Build (after 0022's outcome, R-C18; built 2026-10-15; registered by 0025).** `tools/task_swap.py` (the instrument's
interpreter; imports nothing from `kvt` or `lag_ladder`; re-implements the injection as `score_continuation.py` does),
`lag_ladder/task.py` (gate, plan, run, probe, record, halting), `lag_ladder/summarize_task.py` (flip and accuracy
bootstraps, bands, seal readout), `config/ownrun_task.toml`; TDD with a fake. Two probes (2026-10-15, 64 prompts at
reader step_32, the second storing text and running 17 prompts at a 1,024-token cap as well) fixed the values. Rulings
taken by 0025: R-C19 greedy decoding, with the **flip rate as the verdict statistic** and accuracy descriptive (T = 1.0
sampling not registered); R-C20 the full test set, n = 1,319, cap 256; R-C21 the four arms and the dump-equivalence
control (one relative ulp, restated by 0026 to an absolute 2⁻⁵ after the sitting halted on it); R-C22 the accuracy bands kept as descriptives; R-C23 the seal above, (ii) and (iv) blind.

**What §15 cannot say.** Nothing about sampled generation: the trainer samples at T = 1.0, sampling adds its own floor
of flips against which stale-cache flips may be small, and greedy divergence is a different regime — every sentence (C)
produces is about the *greedy* task output, and the outcome entry's wording carries the word; nothing about a swap
inside a generation; nothing beyond one seed, one 1.5B model, one task, 32
steps; nothing about a cache older than 32 steps; nothing a reward model would compute; nothing about *accuracy* beyond
the descriptive Δ — under greedy decoding at a 256-token cap the step-32 policy answers 6% of GSM8K test and loops on
most of the rest, so this section measures whether the answer changes, not whether it worsens.

## 16. Hybrid state: carryover decomposed by state kind (Kimi Linear 48B-A3B) (seeded 2026-10-22; R-K1–R-K8 PROPOSED, none taken; probe pending; nothing registered)

**Why.** Every rung so far reads one state kind: K and V at every layer of a dense or GQA transformer, so a stale cache
is stale everywhere at once and nothing in (B) can say *which* part of the carried state the reader notices. A hybrid
model separates the kinds. Kimi Linear (`moonshotai/Kimi-Linear-48B-A3B-Base` and `-Instruct`, arXiv 2510.26692;
27 layers, 7 multi-head latent attention (MLA) layers at one-based `[4, 8, 12, 16, 20, 24, 27]`, 20 Kimi Delta
Attention (KDA) layers; 48B total, 3B active, 256 experts, bf16 on the Hub) keeps three state kinds in one cache, and a
serving engine that holds "the prefix cache" across a weight sync holds all three. The carryover definition (§13) extends
verbatim — the change in the reader's next-token distribution from consuming state written under a different
computation — and the question this section adds is where the injury sits: in the seven attention layers' K/V, in the
twenty recurrent layers' states, or in both. It is the same ladder (the text fixed, the model varied) with a new
species of reader, not a linear-ceiling cell (R-K6). The MLPerf-named Kimi K3 (2.8T, multi-node) is out of reach; Kimi
Linear is the same recipe at one-node scale with a Base and an Instruct checkpoint released.

**The cache (verified from `modeling_kimi.py` and flash-linear-attention `main` on 2026-10-22; digests in the private
seed brief).** `KimiDynamicCache` is its own class with four lists, one slot per layer. MLA layers: materialized
per-head K `[B, 32, T, 192]` (128 + 64 dims, none rotated; the 64 come from one shared head expanded across the 32, so a
third of every K row is identical across heads — nothing for the arms, a smaller dump if wanted) and V `[B, 32, T, 128]`,
bf16, concatenated along the sequence
— not the latent; 20 KB per token per layer, ≈ 143 MB per 1,024-token sequence over the 7 layers. KDA layers: the
delta-rule recurrent state `[B, 32, 128, 128]` in **float32** (the chunk kernel allocates it in float32, the fused
kernel asserts it) — 2 MiB per layer, 40 MiB per sequence over 20 layers, independent of prefix length — plus three
short-convolution windows (q, k, v) of `[B, 4096, 4]` in the activation dtype bf16, whose last three inputs carry into
the next forward. Three facts shape the design. (1) *Kernel path:* `mode = 'fused_recurrent' if q_len <= 64 else
'chunk'`, so the 768-token prefill and the 256-token continuation both run the chunk kernel; the floor is one 1,024-token
chunk pass against a 768-token chunk pass followed by a 256-token chunk pass from its float32 state and conv windows —
the same kernel at a different tiling plus the state hand-off, not chunk-vs-fused. (2) *Length and positions:*
`get_seq_length` reads the first MLA layer's key cache and the model derives `cache_position` from it, so **every arm
carries an MLA cache of the full prefix**; a KDA-only stale arm injects the reader's own K/V, never an empty one.
(3) *No positions in the cache:* `mla_use_nope` is true and the MLA forward asserts it — no rotary embedding is applied
to Q or K anywhere in the modeling file (the 64 "rope" dims are plain extra key dims), so positions enter only through
the causal mask and the recurrence's order; a stale K/V is injected as is, and the driver asserts the two configs equal
on every cache-shaping field (the rope fields among them, since a difference would mean another family) and otherwise
refuses. (The seed brief of the morning said K carries RoPE; corrected from `modeling_kimi.py` 378 the same day.) A `state_v_first` layout flag exists in FLA (`[N, HV, V, K]`); K = V = 128 hides a
transposition, so the dump records the flag read on the machine.

**The pinned instrument does not apply (R-K1).** `kvt` builds a `transformers.DynamicCache` from arrays, expects K and
V of one width at every layer, registers an `sdpa` attention, derives `kv_shape` from `num_key_value_heads × head_dim`
(32 × 72 here; the cache is 32 × 192 / 128) and runs float32 (192 GB for this model). Kimi Linear forces
`flash_attention_2` at construction and 20 of its 27 layers have no K/V. Under 0002 ("the instrument serves (A) only;
the (B) and (C) scorers are built here") this is a second dump-and-inject driver in this repository, not a new pair in
`kvt/pairs.py`; `kv-transfer-replication` stays read-only. Statistic (A), f*(τ_K) on a K read-out, has no meaning for a
recurrent state and stays closed (0010).

**Construction.** Rungs (R-K4, R-K8): the isotropic-noise backbone on Base at the rungs bf16 can carry — target
relative norms 1e-2, 3e-2, 1e-1, each placed on the x-axis by the distance measured on the weights as the reader loads
them (below) — written by `tools/perturb_checkpoint.py`, dumped and pruned per rung as 0008 did; then Base → Instruct,
the one real direction available (the analogue of 0008's `qwen-base-to-instruct`). Instruct → Base is run on the same dumps with the reader swapped and reported descriptively,
not as a second sealed rung. No own run: a GRPO trajectory on a 48B MoE is outside this programme's compute; the
per-step dump hook is what a partner would get. Texts: §13's held-out FineWeb windows re-tokenized under Kimi's
163,840-vocabulary tokenizer (a new token file, prepared after the probe asserts the Base and Instruct tokenizer files
byte-identical), P = 768, C = 256, held-out rows as §13; the MLPerf `agentic` windows if that text has been registered
by then. Dtype (R-K2): bf16 weights and activations as the Hub serves them, the recurrent state float32 as the kernels
keep it. Per rung and reader, four arms over the same prefix, each a 256-token teacher-forced continuation under an
injected cache:

| arm | MLA K/V | KDA recurrent + conv states |
|---|---|---|
| FRESH | reader's own | reader's own |
| FULL-STALE | writer's | writer's |
| MLA-STALE | writer's | reader's own |
| KDA-STALE | reader's own | writer's |

**R-K8 (proposed) — the backbone under bf16.** `tools/perturb_checkpoint.py` writes, today, one float32
`model.safetensors` with `torch_dtype` float32 in the copied config, because bf16's mantissa step (2⁻⁸ ≈ 3.9e-3
relative per element) cannot carry a 1e-4 relative perturbation; `config/distance.toml` registers the seven rungs
1e-4 … 1e-1 on that float32 path, and the OLMo and Qwen readers of record loaded float32. Under R-K2 this reader loads
bf16, so a float32 checkpoint at 1e-4, 3e-4 or 1e-3 rounds back to the anchor at load, element by element, and those
rungs would measure the floor, not the noise. Three options. (a) *Restrict* the backbone to the rungs bf16 carries,
1e-2, 3e-2, 1e-1 (per-element steps 2.5 to 25 mantissa units), and name the x-axis gap under 1e-2 as this family's
limit. (b) *Place by measured distance:* write float32 at any target, then compute the rung's x-coordinate with
`tools/weight_distance.py` on the two checkpoints **as the reader loads them** (both cast to bf16), so a rung sits
where it really is and a rung that collapses onto the anchor is dropped by rule (measured distance under half its
target). (c) *Float32 anchors:* load the reader in float32 — 192 GB of weights, two H200s or four A100s, with FLA's
Triton kernels unverified in float32 — a different design and a different cost. **Recommend (a) with (b)'s
placement**: targets 1e-2, 3e-2, 1e-1 only, each rung's distance measured on the bf16-loaded pair and recorded as its
x-coordinate; 3e-3 attempted once at the probe and kept only if its measured distance is within 2× of target,
otherwise the family's backbone starts at 1e-2 and the section says so. Two consequences the probe must read before
the run: the tool holds every tensor and its noise in host memory (≈ 480 GB for 48B parameters: the bf16 source, a
float32 noise draw and the float32 output) and writes 192 GB per rung, so it needs a streaming rewrite for this model
(tensor by tensor, per-tensor seeds, the same rule) and the volume plan is two 96 GB checkpoints plus one 192 GB
perturbed anchor plus dumps, ≈ 500 GB, not 350 GB; and whether a float32 write of a bf16 MoE checkpoint reloads
under `KimiLinearForCausalLM` with the expert layout intact — the probe perturbs one rung at 1e-2, reloads it in
bf16, measures its distance against the anchor as loaded, and runs one FRESH forward, or the backbone is out.

**Statistic (R-K3).** (B1) as registered — median over held-out sequences of the mean per-token KL(π_fresh ‖ π_stale)
over the continuation, bands HOLDS_B ≤ 1e-3 / DEGRADES_B ≥ 1e-1 — read on each of the three stale arms against FRESH.
The arms are worded on the **state the reader consumes**, not on the layers whose weights changed: a writer whose MLA
weights moved writes different hidden states into its KDA layers, so its recurrent and conv states differ too, and
KDA-STALE reads above zero with no KDA weight moved (the offline fake reproduces this; only a change confined to the
last layer kind gives a clean split). So the decomposition says where the injury *enters* the reader, which is the
serving question, and FULL-STALE is not an identity in the two partial arms: the ratio FULL / (MLA + KDA) is reported
per rung as a descriptive, never sealed.

**Controls, per reader, halting (R-K2, R-K5).** *bf16 floor:* on the reader's own weights and its own cache, one pass
over P + C against the cached FRESH continuation — median (B1), limit 1e-4 as 0014 set it; a reading above the limit
makes HOLDS_B UNEVALUABLE for this family and ends the design, written up as such. *Dump equivalence:* the FRESH cache
written through the dump layout, read back and injected, against the live FRESH cache — an absolute limit per 0026's
lesson, its value fixed at the probe. *Identity:* FULL-STALE from the reader's own dump, max logit difference on a
tolerance (R-B10), the float32 state round trip the only expected source. *Scrambled:* the next held-out sequence's
prefix under the reader's weights in every state kind, ≥ 0.1 (DEGRADES_B), or the reader is UNEVALUABLE. The probe
also asserts the shapes above on the machine, per layer, and halts on any mismatch: the dump layout would be wrong.

**Seal (to be committed at registration; values blind until then).** (i) Floor, dump equivalence, identity and
scrambled pass on every reader. (ii) At every rung FULL-STALE ≥ max(MLA-STALE, KDA-STALE): consuming more stale state
never reads less — a sanity clause, falsifiable. (iii) **Direction, picked blind (R-K3's recommendation,
proposed 2026-10-22): at Base → Instruct, median (B1) under KDA-STALE ≥ under MLA-STALE** — 20 of 27 layers, and the recurrent
state integrates the whole prefix while K/V is a per-token store, each row partly recoverable by attention over fresh
tokens. The
probe's two-sequence arm readings are taken after this direction is on record, so they size the bands without
informing the clause. Descriptive, not sealed: the ratio FULL / (MLA + KDA) per rung; the backbone's slope of log median
(B1) on log relative norm per arm against §13's x² (0015); Instruct → Base on all arms; the bf16 floor against the
float16 floors of record; the (B1) on the `agentic` text if run.

**Where it runs; cost (estimates of 2026-10-22; nothing measured).** One H200 NVL 143 GB (RunPod secure, $3.79/h on
2026-10-22; re-read on the day), ≈ 500 GB volume (two 96 GB checkpoints, one 192 GB float32 perturbed anchor, dumps;
R-K8); two A100-80GB
with `device_map` the fallback. Probe (R-K5, the 0015 pattern, outside `results/`, halting order fixed): install
(torch for the host driver, `fla-core` pinned, flash-attn — the model forces it and the wheel is the named risk; no wheel
in 20 min stops the probe, since falling back to sdpa changes the MLA kernel and is a design change) → load → tokenizer
byte-identity → shape asserts → floor and dump equivalence → identity → one inject per arm on two sequences →
timing (prefill, cached continuation, dump write) → the backbone check of R-K8 (one float32 rung at 1e-2, reloaded in
bf16, its measured distance and one FRESH forward, or the backbone is out). Spend (R-K7): probe cap $15, run cap $40,
campaign cap $55; the watchdog line written into the runbook with the day's price. Probe 2–3 h ≈ $10; run 6–8 h, $25–30 (4 arms ×
50 sequences × 8 rungs ≈ 2,000 cached forwards plus controls, per-rung perturb-write-dump-prune, two Hub downloads).
Records: one `hybrid-dump-v1` directory per (model, sequence, prefix) — `meta.json` with per-file sha256,
`mla_layerNN.npz` K/V, `kda_layerNN.npz` recurrent + conv_q/k/v — and reader reports keeping the reader-report keys plus
`arm`, `sources` and `state_v_first`, so the per-rung reading can reuse `summarize_reader` with an arm axis; whether it
does, or gets its own summarizer, is decided after the probe shows the report shape.

**Build.** *Built (c770838, 2026-10-22, tests first):* `src/lag_ladder/hybrid_state.py` — the dump layout and its
validation (refuses a moved sha256, a non-float32 recurrent state, a shape that disagrees with `meta.json`, a dump with
no MLA layer), the four arms and their sources, `compose`/`fill` into a cache-shaped object, `kl_per_token`; numpy only,
so the suite stays offline. `tests/fake_hybrid.py` — a two-layer stand-in (softmax attention with dk ≠ dv, then a gated
delta rule over a width-4 causal conv with a float32 state) whose cache mirrors `KimiDynamicCache` field for field;
`tests/test_hybrid_state.py`, 14 tests, including the clean split under a last-kind-only change and the upstream leak
under an MLA-only change. *Planned:* `tools/score_hybrid.py` (torch + fla, the pod's interpreter, imports nothing from
`kvt` or `lag_ladder`, as `score_continuation.py`'s rule) — dump a `KimiDynamicCache` through the layout, fill one from a
composed arm, teacher-force the continuation, write the report; the probe scripts; the token file; the registration
entry with the seal and a config under `config/`; the streaming rewrite of `perturb_checkpoint.py` if R-K8 keeps the
backbone. Rulings PROPOSED 2026-10-22, none taken; the recommendations are: R-K1 a second driver here; R-K2 bf16, floor
band-relative at 1e-4; R-K3 three arms on consumed state, (ii)–(iii) as above, additivity descriptive; R-K4 backbone
then Base → Instruct, Instruct → Base descriptive, no own run; R-K5 probe first, outside `results/`, allowed to end the
design; R-K6 this section; R-K7 the caps above; R-K8 the bf16-carried rungs placed by measured distance. Nothing below
the probe is built until they are taken.

**What §16 cannot say.** Nothing about a trained direction on this family — Base → Instruct is one post-training
step of unknown composition, and the backbone is noise; nothing about which *layers* carry the injury, only which
*state kind* the reader consumes it through; nothing about decoding (the fused kernel is never on the path (B) reads);
nothing about float16 or float32 for this model (bf16 is the registered dtype and its floor is its own); nothing
beyond one hybrid model at one scale and one prefix length; nothing until the probe's floor is read — a floor above
1e-4 closes this section as UNEVALUABLE at HOLDS_B, and that is a result the record keeps.
