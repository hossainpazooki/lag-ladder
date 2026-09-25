# Holdover — a KV cache held over a weight update (async RL on the weights axis)

**Working title (2026-09-23):** *Holdover: The Shelf Life of a KV Cache Across Policy Updates* (pairs
with *Carryover: The Reuse Margin of a Stale KV Cache* — a cache carried across a change of context; a
holdover is one held over a weight update; "Weight Syncs" for an MLSys audience, "Policy Updates" for
ARR/ICML). Fallback if the tail shows nothing: *Holdover: Stale KV Caches in Asynchronous RL, from One
Update to Thousands*. **Repo:** `hossainpazooki/lag-ladder`, package `lag_ladder`
— named for the instrument, not the claim, so it survives a reframing and does not lead a reviewer from
the title to the repo. Both locked 2026-09-20; the repo was scaffolded 2026-09-22 (§11) and this document
lives there.

**Date:** 2026-09-19 · **Status:** design, PROPOSED, unregistered, unnumbered, nothing run. **Rulings
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
the same ladder. Prior-art search for this framing specifically: **not yet run** (the 09-20 search asked
the in-flight question); it is the first item of §9.

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

Concurrent and adjacent work, search run 2026-09-20 (18 sources; **coverage gap:** SGLang, slime, prime-rl,
verl, OpenRLHF, NeMo-RL, LlamaRL, ROLL, Kimi, MiniMax, GLM, DeepSeek, Composer, KVCOMM, C2C and the vLLM /
open-instruct / TRL / AReaL issue trackers were not reached; re-run before submission). Marked *[checked]*
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

So the field is split by assertion — keep the stale cache (PipelineRL, Magistral, Nemotron, Olmo 3) or
reset it (AReaL, Laguna, AsyncOPD, StaleFlow), or design around it (DORA) — and the only measurement
under either position is one KL figure to lag 32 on one 7B run. Nobody reports where "it does not
matter" stops being true, and nobody reports it in task units. That disagreement is the paper's opening.

Novelty *(proposed, to be attacked before any GPU spend)*:
**(i)** a paired, same-token contrast at a single clean cache age. PipelineRL's two curves are two
separately *sampled* distributions each compared to on-policy by sequence KL, and the cache inside one
sequence is a mixture of ages 1…g; here the prefix tokens and reader weights are identical, the cache is
written by exactly θ_t, and the difference is read per token and per prompt.
**(ii)** range, *not* the existence of a ladder. Within lags ≤ 32 a ladder is prior art (PipelineRL
Fig. 7); what is new is the tail beyond it (OLMo, 200–2400), a second model family, and the NULL arm —
no prior report asks whether the effect is about RL updates or about any weight delta of that size.
**(iii)** fixed-policy paired task accuracy as the read-out. Prior read-outs are a sampling-distribution KL
(PipelineRL) and an undocumented training outcome (Magistral). Do not claim "first task-level evidence"
without that qualifier.
**(iv)** whether a cheap tensor-space statistic (f*, no generation needed) predicts the task-level loss —
i.e. τ calibrated in task units. Untouched by any of the above. (iv) is the contribution an NLP reviewer
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

**(B) Does the trainer notice — stale-vs-fresh importance ratio, ESS / N.** Unchanged from 09-02 §2(B).
Bounds remain "the operator's stated judgment".

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

Bounds for (C) *(proposed; operator's stated judgment, to be fixed in the registration entry)*:
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
  prefix-persistence framing (§1) this tail is engine-relevant, not "decides nothing". Needs the upstream
  `Pair` revision field before the first dump.
- **Own run — deferred.** Producer options, in order of preference once the pilot shows signal: a fork of
  the author's `ServiceNow/PipelineRL` (real in-flight updates; 4×H100 minimum documented, Linux, conda +
  flash-attn; per-step checkpoint saving unverified; not fundable on the 09-18 RunPod balance), else a
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

Whether *training on* stale-cache rollouts changes the final policy (the operator's 09-02 ruling: not an
RL-algorithms paper); any serving-latency number (that is the MLSys reading); a selective-recompute scheme
that achieves f* (the `[STRETCH]` of 0023). Each is a Limitations sentence; ARR requires that section.

## 9. Build order (ruled 2026-09-22: pilot first, no deadline of record)

Oct 12 is dropped (R6). The ARR October kill table of the 09-19 draft is retired; the next venue is decided
on the pilot's result — ARR December cycle or ICML if (C) has something to say, MLSys if only (A)/(B) do,
a short note if nothing does.

Inherited blockers (09-02 §7–8): upstream `Pair` has no revision or local-path field; no scorer exists for
teacher-forced log-probs or generation under a supplied cache; (B)'s bounds blank.

| step | what | needs | decides |
|---|---|---|---|
| 0 | `lag-ladder` repo scaffolded (§11), design doc moved there, ledger 0001 | — | — |
| 1 | prior-art search under the prefix-persistence framing (§1) + the Q6 framework list the 09-20 search missed | web | whether §2 survives the reframe |
| 2 | upstream `Pair` gains `revision` / local-path; pinned by ledger entry | kv-transfer-replication commit | first dump |
| 3 | **pilot:** (A) alone — f*(τ_K), τ ladder — on the 13 OLMo revisions, anchors step_200 and step_1200; dumps only, no generation | one GPU-day at most; A100 class for 1B fp32 dumps | if f* ≈ 0 out to lag 2400: short note, lane closed. If f* crosses τ_K in the tail: there is a paper, and (iv) has something to predict |
| 4 | registration entry: statistics, bounds (R2), tasks (R1), lags, seeds, SCRAMBLED positive control | pilot result | — |
| 5 | swap scorer for (C) upstream; k = 0 identity on the run GPU; (B) + (C) on OLMo | GPU | the verdict cells |
| 6 | own-run producer chosen (§6); own-run grid | funding | the short end of the ladder |

Compute for the pilot: 13 revisions × 2 anchors' partners ≤ 13 dumps of the held-out set at
131,072 B/token (OLMo 1B, fp16 K+V; 09-02 design) — sized at registration, not here. The full grid's
09-19 estimate (≈ 44k generations) counted one prefix condition and one source; with two prefix conditions
and OLMo as primary it is ≈ 1,319 × 12 × 4 arms × 2 ≈ 127k generations before the s-ladder and the
second task — still a few GPU-days on a ≤ 1B model, but not the number the draft gave.

## 10. Rulings

Ruled 2026-09-22 (operator, via the four-question pick; option text quoted as picked):

- **R6 — "Drop Oct 12; pilot first."** §9 rewritten accordingly. Not a ruling on the venue.
- **R7 — framing: "Prefix-cache persistence."** §1, §3 (8-shot primary), §6 (OLMo tail engine-relevant).
- **R8 — checkpoint producer: "OLMo only for now."** §6; the own run and its producer wait on the pilot.
- **R9 — scope today: "Doc + scaffold lag-ladder."** §11; the design doc moves to the new repo.
- **Names (2026-09-20):** paper *Holdover*, repo `lag-ladder`, package `lag_ladder`.
- **R10 — instrument scope (2026-09-24): "kv-transfer-replication stays the instrument for (A) only."**
  Ledger entry 0002. (B)'s log-prob scorer and (C)'s swap scorer are built in `src/lag_ladder/`; the
  upstream still needs the `Pair` revision field and the bridge items G1/G4/G5 (seed 2026-09-24).

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

## 11. Repo shape (locked 2026-09-20; scaffolded 2026-09-22, uncommitted — see the repo's own CLAUDE.md)

- A separate repo, `lag-ladder`, with linear-ceiling's chassis **copied, not shared**: ledger + checker,
  sealed predictions, scope linter, `config/*.toml` seeds and thresholds, `rng`, `hashing`, the
  upstream ancestry gate, fail-closed summarizers, one CI job. Each copied module gets a provenance row
  `{sourceRepo: linear-ceiling, filePath, commitSha}`. GPU protocol linked, not copied.
- Ledger starts at 0001; 0001 cites linear-ceiling entry 0023 for f* and τ_K by `{repo, entry, sha}`.
- Two upstreams in `UPSTREAM.md`: the **instrument** `kv-transfer-replication` (pinned as now; needs
  the `Pair` revision field; **statistic (A) only, by R10** — the (B) and (C) scorers live here) and the
  **checkpoint producer** — the author's
  `ServiceNow/PipelineRL` (Apache-2.0) through a fork whose only changes are additive config; the gate
  records both shas and asserts trainer paths unchanged from the author sha. Checked 2026-09-20: the
  author repo's code search has no `recompute` / `kv_cache` / `reset_prefix_cache` hit (default branch
  only), so Fig. 7's measurement code is not published; smallest documented config is 4 H100s.
