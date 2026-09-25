# Seed — bridging the quality and robustness gap between PipelineRL and kv-transfer-replication

<!-- Paste this file as the opening message of a fresh session in ~/dev/lag-ladder. Self-contained
     except for the three governing docs it links. Scoped to the bridge: it does not run the pilot,
     register a hypothesis, or write the (C) swap scorer. Do not let the session expand past §6. -->

## 1. Role and blast radius

You are closing the gap between two repositories that Holdover depends on and that were never built
to meet:

- **The producer** — `ServiceNow/PipelineRL` (Apache-2.0, author implementation of in-flight weight
  updates; `main` observed at `58d393458625ad63ed539f2dcd072c85700c557f` on 2026-09-22). It is the
  engine whose behaviour the paper measures: multi-GPU, vLLM in bf16, NCCL weight broadcast after
  every optimizer step, checkpoints on a step interval.
- **The instrument** — `hossainpazooki/kv-transfer-replication` pinned at
  `063f4023fdde67dedbee01a92518ce7f83f6cf5d`. A CPU-scale replication of a cross-model KV-transfer
  paper: float32 models, custom SDPA, fp16 dumps, a per-token scorer, no CI, cuda-or-cpu only.

The design (`docs/2026-09-19-holdover-design.md`) reads the weights axis "on a yardstick that has a
measured point on it" — the instrument's f\*(τ_K). That sentence is only true if the instrument
measures the quantity the engine actually has. Today it does not, on at least five counts (§3). Your
job is to make each count either closed by evidence or stated as a limitation with its size measured.

Blast radius: **`lag-ladder` and a branch of `kv-transfer-replication`.** Never `import kvt`; never
edit PipelineRL (a fork with additive config only, if at all — §3 G3); never write git history
(output commit commands). Every upstream change lands as a commit the operator makes, then a re-pin
here by a numbered ledger entry (`UPSTREAM.md` §1).

Precondition the operator has **not** ruled on (2026-09-24): whether kv-transfer-replication stays the
instrument for statistic (A) only (recommended) or is ported into `lag_ladder/`. This seed assumes
**(A) only**; if the ruling is "port", §3 G4–G6 become `lag_ladder/` work and the pin clause of ledger
0001 is superseded by a new entry first.

## 2. What is established (re-verify before building; nothing below is to be trusted from this file)

| fact | basis (captured 2026-09-24, both repos read at the shas above) | re-verify (read-only) |
|---|---|---|
| PipelineRL **keeps the stale cache** across a weight update | `pipelinerl/vllm1.py:160` — `await self.engine.pause_generation(mode="keep", clear_cache=False)` | `gh api repos/ServiceNow/PipelineRL/contents/pipelinerl/vllm1.py --jq .content \| base64 -d \| grep -n clear_cache` |
| The engine's cache is **bf16, written by vLLM during decode** | `conf/base.yaml:61` `dtype: bfloat16` under `vllm_config`; weights arrive per tensor via `model_update_group.broadcast` + `load_weights` (`vllm1.py:110–126`) | same file, `grep -n dtype conf/base.yaml` |
| Trainer master weights are **fp32** (FSDP param/reduce/buffer) | `conf/base.yaml:99–101` | `grep -n _dtype conf/base.yaml` |
| Checkpoints are HF `save_pretrained` safetensors on a **step interval that is config, not code** | `conf/finetune/base.yaml:69` `save_checkpoint_steps: 100`, `:71` `keep_intermediate_checkpoints: True`, `:85` `also_save_steps: []`, `:86` `use_safetensors: true`; `finetune/checkpoints.py:174,197–199` | `gh api …/conf/finetune/base.yaml … \| grep -n -E "save_checkpoint_steps\|also_save_steps\|use_safetensors"` |
| Weight update requested after each optimizer step; `max_lag: null` by default | `finetune_loop.py:205 send_weight_update`; `conf/base.yaml:106` | `grep -n max_lag conf/base.yaml` |
| The instrument loads **float32** with a custom `sdpa_repeat_kv` attention and picks **cuda or cpu only** | `kvt/models.py:6–8` `device()`; `:68` `{"dtype": torch.float32, "attn_implementation": ATTN_IMPLEMENTATION}` | `sed -n '1,12p;66,72p' ../kv-transfer-replication/kvt/models.py` |
| Dumps are a **prefill** (`model(input_ids, use_cache=True, logits_to_keep=1)`) saved as **fp16** K/V at a stride, with a RoPE-spec halt check | `kvt/data.py:33–66` | `sed -n '33,66p' ../kv-transfer-replication/kvt/data.py` |
| f\* is computed from float64 per-token squares that sum exactly to the recorded SSE | `kvt/pertoken.py::per_token_squares`, `moments` | `sed -n '1,30p' ../kv-transfer-replication/kvt/pertoken.py` |
| `Pair` has **no revision or local-path field** | `kvt/pairs.py` — `Pair(name, source, target)` | `grep -n "class Pair" -A4 ../kv-transfer-replication/kvt/pairs.py` |
| The instrument has **no CI workflow**; 22 test files | `ls ../kv-transfer-replication/.github/workflows` → nothing; `ls tests \| wc -l` → 22 | same |
| f32 + GQA + `enable_gqa` fell to the math kernel and OOMed on the E9 box (51.5 GiB at T = 29k); `sdpa_repeat_kv` fixed it at 16.5 GiB | linear-ceiling `docs/2026-09-02-e9-gpu-runbook.md`, amended-on-the-box section | read that section |

## 3. The gap, enumerated (each: what breaks if unbridged → the bridge → the evidence that closes it)

**G1 — Precision.** The engine writes and reads the cache in bf16; the instrument dumps fp32 prefill
cast to fp16 and scores in float64. If the bf16-vs-fp32 deviation of the *same* weights on the *same*
tokens is not far below the τ ladder's floor (0.03), a "stale" reading includes a precision effect the
engine does not distinguish from the weights effect. → Add a `--dtype {float32,bfloat16}` load option
upstream (default unchanged) and a **precision control**: dump one checkpoint both ways, score with
`score_positions`, report per-token deviation as a fraction of the K norm. → Closed when the
control's median and 99th percentile are stated in a ledger entry and sit below 0.03; otherwise the
paper reads at bf16 and says so.

**G2 — Cache origin.** The engine's stale KV was written during **decode** under paged attention;
the instrument's is written by a **prefill**. Design §3 already rules that every arm is built by the
same teacher-forced prefill (so the k = 0 identity control cannot halt on kernel numerics) and
declares the departure a limitation. Its size is unmeasured. → **Origin control**: for one checkpoint
and a few hundred generated tokens, KV from HF `generate` (decode-written, `past_key_values`) vs KV
from a prefill of the identical tokens, same weights, same dtype. → Closed when the deviation is
stated beside G1's and the limitation sentence carries a number. Extracting vLLM's paged KV is out of
scope; HF decode is the proxy and the seed says so.

**G3 — Checkpoint interface.** The pilot reads OLMo revisions from the Hub; the own run (deferred,
ruling R8) will read PipelineRL checkpoints from disk. Both need `Pair` to name a **revision or a
local path** and the dump's `meta.json` to record which. PipelineRL needs nothing beyond
`save_checkpoint_steps: 1` (+ `also_save_steps` for sparse lags) — **config, not a patch**; a fork is
unnecessary unless a step-numbered directory layout has to be fixed. → Add `revision: str | None` and
`local_path: Path | None` to `Pair`; `load_model` passes `revision=` through; a **manifest** (sha256
of every safetensors shard per checkpoint, written before any dump; the summarizer refuses on
mismatch) is the join between producer and instrument. → Closed when a dump of two OLMo revisions
from a fresh checkout records both revisions in `meta.json` and the two checkpoints' shard hashes
differ in the manifest.

**G4 — Device.** The instrument's `device()` returns cuda or cpu. The operator now runs locally on an
Apple-silicon Mac mini (MPS). → `device()` gains `mps` when available; then the **kernel-identity
control** on MPS: a 48-token dump + `score_positions` on the pinned models must match the CPU path
within float32 rounding (the E9 runbook's bound: max |ΔK| 9.2e-4 on a scale of 423). `sdpa_repeat_kv`
in float32 on Metal has never run. → Closed when the MPS-vs-CPU deviation is recorded; if Metal
falls back to the math kernel, the memory budget (§4) is recomputed and stated.

**G5 — Engineering robustness.** No CI upstream; a subprocess invocation that depends on
`PYTHONPATH=$PWD`; a f32 memory profile that once OOMed a 40 GB slice. → A CI workflow upstream
(pytest, cpu torch, the 22 files); a console entry point or `python -m kvt.scripts.…` so the
PYTHONPATH trap disappears; a `--memory-probe` that reports the attention kernel actually selected
and the peak allocation on a short sequence before any long one (protocol R2, "budget the forward").
→ Closed when the upstream CI is green on the branch and `lag_ladder.upstream_gate` can name the
paths it pins.

**G6 — Statistic bridge (the quality gap a reviewer will name).** PipelineRL §5.1 reports a
**sequence KL** between a mixed-policy behaviour distribution and on-policy sampling; the instrument
reports a **per-token tensor deviation**. Neither is wrong; they are not comparable, and the paper
needs to put its number beside theirs. → Reproduce Fig. 7's construction on the same checkpoints the
ladder uses (swap the behaviour policy every L/g_max tokens, KL to on-policy, stale vs recomputed) as
statistic (B)'s companion — not verdict-bearing, one figure. → Closed when both curves exist for one
source at three lags and the design's §4 says which one each claim rests on.

**G7 — Prompt distribution.** The instrument's held-out set is FineWeb-Edu (comparability with
Carryover's τ). The engine's cache is over math prompts and sampled generations. → (A) is read on
both: FineWeb-Edu for the τ-comparable number, the RL run's own prompt distribution (GSM8K for the
own run; OLMo's RLVR mix is not public in prompt form — state that) for the engine-relevant one. →
Closed when the pilot config names both sets and the summarizer reports both.

## 4. Sizing the bridge work (derived from the facts in §2; re-derive, do not trust)

All controls are short-sequence (≤ 1,024 tokens) on a 1B model: minutes of compute, single-digit GB
of memory in fp32, well under a Mac mini with ≥ 24 GB unified memory or any 24 GB card. G6 is the
only item with real compute — Fig. 7's construction at three lags on OLMo 1B is a few hundred
generations. Nothing here needs the pilot's disk budget.

## 5. Definition of done (build nothing beyond this)

1. A branch of kv-transfer-replication carrying G1 (`--dtype`), G3 (`Pair.revision` / `local_path`,
   manifest writer), G4 (`mps` in `device()`), G5 (CI + entry point + memory probe). Tests for each
   (the suite is synthetic and offline; keep it so). Commit commands output, not run.
2. In `lag-ladder`: a ledger entry **proposal** (not appended — the operator appends) that re-pins the
   instrument at the branch's merge commit and registers the four controls (G1 precision, G2 origin,
   G4 kernel identity, G7 both prompt sets) with their bounds as *operator's stated judgment*; the
   design doc §3/§4 amended to cite the controls; `config/pilot.toml` gains `dtype`, `prompt_sets`,
   `manifest` keys with the loader validating them.
3. A bridge note `docs/2026-09-24-bridge-controls.md`: one table, G1–G7, each row *closed by* /
   *stated as limitation* / *open*, with the measured number where one exists. Every number in it
   recomputed from a file under `results/` by a script that refuses on a missing input.
4. `pytest` green in both repos; `lint_scope`, `ledger_check`, `seal verify` green here.

## 6. Explicitly out of scope

The pilot itself; registering H-P1; the (C) swap scorer (generation under a supplied cache — that is
`lag_ladder/` work after this seed); extracting vLLM's paged KV; running PipelineRL; any change to
τ_K; any change to the f\* definition; the Algoverse box (the operator runs locally now).

## 7. Invariants (from `CLAUDE.md`, `UPSTREAM.md`, linear-ceiling's protocol; violated ones are refusals)

- Upstream is read-only from this repo's point of view: branch there, commit commands for the
  operator, re-pin here by entry. Never `import kvt`; never copy its code into `lag_ladder/` without
  a provenance row.
- No number into any doc that a script did not recompute from a raw artifact; a refusal is a finding.
- Registered before requested: no GPU/MPS run of a *control* needs registration (controls are
  instrument checks, protocol R1's exemption for identity/null controls), but their **bounds** do
  before they decide anything.
- Seeds and thresholds in `config/*.toml`; randomness via `lag_ladder.rng.make_rng` only.
- The scope sentence in README stays verbatim and unparaphrased (`lint_scope`).
- Python 3.12 for both venvs; on the Mac, torch from PyPI (MPS wheels are the default build) — the
  cu128 note in the handoff brief is Algoverse-specific and does not apply.
- Git history is the operator's. Output commit blocks grouped by repo, one commit per concern.

## 8. Working style

Read §2's re-verify column and run it before anything else; a fact that fails re-verification is a
finding to report, not a reason to patch silently. Work G3 → G4 → G1 → G2 → G5 → G7 → G6 (the order
in which each unblocks the next on a Mac). Refute every "closed" claim once before writing it into
the bridge note: rerun the control from a clean shell, diff against the recorded number. End with a
handoff brief (`/rigor:handoff`), which is also where the §2 facts become learnings entries.
