# lag-ladder — repo brief

Read `README.md` for what this is. `docs/2026-09-19-holdover-design.md` is the authority on scope and
carries the rulings (§10), the build order (§9), the re-aim (§12), statistic (B) (§13), the own run (§14; §14.5 the own run
on two more families, Llama-3.2-3B then SmolLM3-3B-Base, R-F1–R-F10 ruled 2026-10-23; Llama registered by 0031, read by 0032), statistic (C)
(§15) and the hybrid-state seed on Kimi Linear (§16; R-K1–R-K8 taken 2026-10-22; the probe ran the same day and read a bf16 floor of ≈ 1e-3 against the 1e-4 limit — HOLDS_B UNEVALUABLE under R-K2; the float32 reference of 2026-10-23 showed the floor is bf16 arithmetic ambiguity across computation orders, and a floor-relative registration is drafted for after the families; nothing registered). `ledger/ledger.md` is the record and
is append-only by numbered entry; entry 0001 carries the chassis provenance, the inherited definitions (f\*,
τ_K) and the first rulings, entry 0002 the ruling that the instrument serves statistic (A) only, entries
0003 and 0004 the instrument re-pins (checkpoint selection; per-side `ModelRef` with verified provenance),
entry 0005 the pilot's registration, entry 0006 its outcome (f\* = 0 at every lag to 2400: HOLDS),
entry 0007 the ruling that weight distance is the axis, entry 0008 the weight-distance ladder's registration,
entry 0009 its outcome, entry 0010 the closure of statistic (A) with the statement of record, entry 0011
the ruling that the lane continues under statistic (B) as design §13 defines it, entries 0012–0014 (B)'s
registration and amendments, entry 0015 its outcome, entry 0016 its closure on the ladder with the statement of record,
entry 0017 the own run's registration, entry 0018 its outcome, entry 0019 the ruling that restricts 0016's statement of
record to the held-out texts it names, entry 0020 the registration of rollout-text controls and the sliding writer,
entry 0021 their outcome, entry 0022 the registration of a far scrambled control on rollout text, entry 0023 its
outcome, entry 0024 the ruling that adds the trainer's own text to (B)'s statement of record, on one seed, and entry
0025 the registration of statistic (C) on the own run with its seal, entry 0026 the restatement of its
dump-equivalence limit (absolute 2⁻⁵) after the sitting halted on the relative one, entry 0027 the outcome of the
(C) sitting, entry 0028 the registration of the own run's second trainer seed (§14.3), entry 0029 its outcome
(all five clauses held), entry 0030 the registration of the fourth text `agentic` (§14.4), entry 0031 the registration of
the own run on Llama-3.2-3B (§14.5), entry 0032 its outcome ((i-holds), (ii), (iii) HELD; (i) and (iv) FALSIFIED), entry 0033
the pair-only re-pin at de6ce9e for the SmolLM3 family (`INSTRUMENT_SHA` stays 5d4aa4a), entry 0034 the registration of the own run
on SmolLM3-3B-Base, entry 0035 the registration of the Instruct-initialised own run (Qwen2.5-1.5B-Instruct), both halves, and
entry 0036 the outcome of the agentic text (all four clauses HELD).

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
.venv/Scripts/python.exe -m lag_ladder.reader plan|check|run|probe [--config config/ownrun.toml]   # statistic (B)
.venv/Scripts/python.exe -m lag_ladder.summarize_reader [--config config/ownrun.toml]
.venv/Scripts/python.exe -m lag_ladder.task plan|check|run|probe [--config config/ownrun_task.toml]   # statistic (C), §15; probe: --out OUTSIDE results/, --n 64 [--max-new N overrides the config's cap]
.venv/Scripts/python.exe -m lag_ladder.summarize_task [--config config/ownrun_task.toml]
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
per rung) · `summarize_distance` (rungs by measured distance, tied to the dumps' manifests) · `reader` (the
(B) driver: plan|check|run|probe; controls per reader, then the rungs, by subprocess into the pinned
instrument; refuses while `reader.toml` is unregistered) · `summarize_reader` (recomputes B1–B4 from the
reports, withholds failed-control rungs, places rungs by the distance ladder's `distance.json`, reads the sealed
clauses out; per-rung texts and ragged continuations for §14; from §14.5 also `ii_rollout_lost_at_lag_1` (`ii-lost`) and the
machine-readable `iii_rl_excess_over_0015_backbone` (`iii-excess`: the sealed rungs inside a distance range against the
nearest noise rung of another summary, pinned by sha256; 0017's prose-only form is left alone)) · `task` (the (C) driver, §15: plan|check|run|probe; the
dump-equivalence control first, then one `cells` job per reader with the floor and identity arms; halts on the
equivalence, the floor or the identity) · `summarize_task` (recomputes Δ accuracy with a seeded paired bootstrap, the
(C) bands, flip / no-answer / cap rates, the (B)→(C) rank correlation; withholds a lag whose positive controls do not
fire; reads the sealed clauses out). `tools/` — `perturb_checkpoint.py`,
`convert_bin_checkpoint.py`, `weight_distance.py`, `score_continuation.py` (the (B) scorer: injects a stale vs fresh
cache, scores the continuation; `--text rollout` teacher-forces a stored continuation cut at `--length`; `--text agentic`
scores a stored continuation exactly as `corpus`): run with the instrument's interpreter, import nothing from it.
`score_hybrid.py` (§16, PROPOSED, nothing registered: the hybrid-state scorer for Kimi Linear — `shapes` | `dump` | `score`;
the `hybrid-dump-v1` layout that `lag_ladder.hybrid_state` reads, arms FRESH / FULL-STALE / MLA-STALE / KDA-STALE on the
consumed state plus ONE-PASS as the bf16 floor; torch, transformers and fla imported inside the functions so the layout and
report test offline; `--dtype float32 --attn sdpa [--device-map auto]` for the float32 reference probe and `gaps`, the torch-free
split-point readout over ONE-PASS reports; run with the pod's interpreter, imports nothing from here or the instrument).
`agentic_tokens.py` (the 2026-10-22 plan, cell 1b: the MLPerf Agentic Inference windows as a held-out token file, built
from linear-ceiling's rendered export under the instrument pair's tokenizer by the 0016 §4 window rule restated there, seed 8,
rows suite by suite — the SAME fifty windows as linear-ceiling's `data/e8m` file, which `--assert-equal` checks id for id;
writes the .npy and a manifest sidecar; needs `tokenizers`, imports only `lag_ladder.rng`). `own_run.py` (§14: the GRPO trajectory with its save hook;
`--prompt-format {chat,plain}` on both commands from §14.5 — `chat` fails loudly on a template-less tokenizer, `plain` renders the registered
string and keeps the tokenizer's special tokens as TRL 1.15.0 does for string prompts; the format, prompt token counts, peak CUDA memory
and seconds per step go to `run_config.json` / the spans sidecar; run with the trainer's interpreter, imports nothing from here or the instrument). `task_swap.py` (§15, the (C) tool:
`equivalence` | `cells` | `prepare`; builds prompt caches live, rounds them through float16 and injects them through
`score_continuation.py`'s functions; greedy generation per arm; one `task-record-v2` per prompt and arm, keeping the
decoded completion text and the first-`####` answer beside the registered last-`####` answer; run with the instrument's
interpreter). Built 2026-10-15, registered by 0025 after two probes: `config/ownrun_task.toml` pins the GSM8K test
prompts file, floor 1%, positive controls at 0.5 flips, dump equivalence at an absolute 2⁻⁵ (0026; the relative
gap and the share of differing elements descriptive).
`config/` — `seal.toml`, `pilot.toml`, `distance.toml`, `reader.toml`, `ownrun.toml` (0017, as run), `ownrun_controls.toml` (0020; `per_pair` controls), `ownrun_far.toml` (0022; `far_pair` + `far_min`), `ownrun_task.toml` (0025; §15, the (C) arms and controls), `ownrun_seed2.toml` (0028; §14.3, the second trainer seed: `per_pair`, `far_pair`, `evaluable_under_far`), `ownrun_agentic.toml` (0030, read by 0036; the fourth text `agentic` on the seed-1 readers: six per-rung rungs on the pair `qwen2.5-1.5b-mlperf`, every row held out, the token file linear-ceiling's `data/e8m` draw id for id), `ownrun_smollm3.toml` (0034; §14.5 on the third family, `upstream_sha` de6ce9e, pair `smollm3-3b-base` + rollout pair `smollm3-3b-base-ownrun-rollout`, labels `ownrun-smollm3-*`), `ownrun_instruct.toml` (0035; (B) on Qwen2.5-1.5B-Instruct@989aa79, chat prompts, pair `qwen2.5-1.5b-to-instruct` with the Instruct side dumped as `target`, rollout pair `qwen2.5-1.5b-instruct-ownrun-rollout`, labels `ownrun-instruct-*`), `ownrun_instruct_task.toml` (0035; (C) on that run: cap 768, n 1,319, `equivalence_expect_b1` a pointer at the run's own (B) summary, seal `ownrun-instruct-task` with clauses (v) and (vi) on accuracy), `ownrun_llama.toml` (0031; §14.5: 0028's layout on the pair `llama3.2-3b-to-llama3.1-8b` plus the rollout pair `llama3.2-3b-ownrun-rollout`, hashes pinned; a `[trainer]` table the pod script and the entry read — `prompt_format = "plain"`, the model revision, the pins). `ledger/` — `ledger.md`, `predictions/`. `results/fstar/` and
`mappers/` — seal artifact roots, gitignored past their placeholders. `docs/` — the design doc; `docs/provenance/` — verbatim outside records with their hashes (the Instruct Step-1 baseline probe of 2026-10-21, run by a co-author; not evidence, the readings it informed are 0035's rulings).

## State
The pilot ran (ledger 0005, 0006): on OLMo-2 1B RLVR1, f\*(τ_K) = 0 at every lag from 200 to 2400 on
both anchors, every τ; the sealed DEGRADES at lag 2400 was falsified. Ruled (0007): the axis is weight
distance. The weight-distance ladder ran (0008, 0009) on the instrument pinned at kv-transfer-replication
`5d4aa4a`: the isotropic backbone on OLMo-2 1B HOLDS through a relative distance of 3e-2 and DEGRADES at
1e-1 (the sealed knee at 3e-3 falsified); the six real directions tie the backbone at f\* = 0 wherever
both are measured and scatter around it by two orders of magnitude in whole-cache deviation. **Statistic
(A) is closed (0010)**: the statement of record is in that entry and the README; linear-ceiling's paper
cites this repository for it. Nothing is registered beyond 0010. **Statistic (B) is ruled (0011)** as
design §13 defines it: per-token KL of the reader over the writer's cache against its own, on the 0008 dumps,
to run on RunPod. **Statistic (B) is registered (0012)**: the float16 floor was probed on the run machine
(median mean-KL 3.9e-8, HOLDS_B confirmed at 1e-3), the seal is committed, `config/reader.toml` is pinned, and
the noise backbone is re-dumped on the run machine because 0008's noise checkpoints do not regenerate
byte-identically there. 0013 and 0014 amended the float16-floor control after the first run halted twice on other readers'
floors (KL only; limit band-relative at 1e-4); 0014 also registers the five R-B4 lag rungs. **The (B) run completed
(0015)**: all controls passed; (B1) grows as x² on the backbone (slope 2.0) and leaves HOLDS_B between 3e-3 and 1e-2;
OLMo directions read 355–875× the backbone at matched distance, Qwen Instruct 11×; RL-phase lags 200–2400 all HOLDS_B;
seal clauses (i), (iii) HELD, (ii), (iv), (v) FALSIFIED. **(B) on the ladder is closed (0016)**, statement of record in
that entry and the README; the scorer stays the registered instrument for later rungs. **The own run (§14) is complete
(0017, 0018)**: `config/ownrun.toml` registered by 0017 with the rollout pins and the seal; one A100 sitting ran the
32-step GRPO trajectory and the 420 jobs without a halt; held-out text HOLDS_B at every lag (lag 32: 6.65e-6 corpus,
3.54e-6 on-path; slopes 2.26 / 2.19), rollout text UNRESOLVED at every lag (4.68e-3 to 8.91e-2); clauses (i), (ii)
HELD, (iii), (iv) FALSIFIED. Records and dumps are off-repository under the hashes 0018 states. **0019 restricts 0016's
statement of record** to the held-out texts it names (the trainer's own text is not predicted by weight distance) and
registers a prerequisite: no rollout-text number becomes a statement of record before per-reader float16-floor and
scrambled controls on rollout text pass under 0014's limits. **0020 registers those controls** (`config/ownrun_controls.toml`,
`per_pair = true`: controls per (reader, pair) on the pair's own text; seal `ownrun-controls`) together with fifteen
descriptive sliding rungs on held-out text (writer and reader at kept steps, placed by writer–reader distance) and a
replicate of the six rollout rungs — 740 inference-only jobs from the Hub's dumps and kept weights. **0021 is its
outcome**: one A40 sitting, no halt; all eleven (reader, pair) control units pass (rollout floors 1.53e-7 to 1.87e-7,
rollout scrambled 0.115 to 0.126 against the 0.1 limit); clause (i) HELD (0018's rollout rungs 2.5e4× to 5.8e5× their
floors, replicate within 5e-5), clause (ii) HELD (sliding slopes 1.96 / 1.97); the sliding rungs read 0.93× to 1.85× the
step-0 law. 0019's prerequisite is met; the rollout reading's entry into (B)'s statement of record is a ruling not yet
taken. Records and the Hub backup (`ownrun-controls/`) under the hashes 0021 states. Rulings R-C14–R-C18 (design §14,
2026-10-15): the statement-of-record amendment not taken (one seed; the near scrambled control passed by 15–26%); **0022
registers a far control** (`config/ownrun_far.toml`, §14.2: the rollout pair names the held-out pair as `far_pair`; each
reader's rollout controls gain a FAR cell — its own dump of held-out sequence 40+i cut to the prompt's length under the
same completion, limit `far_min` 0.1; seal `ownrun-far`: far ≥ 2× scrambled per unit, median (B1) ≤ 0.5× far per lag; 300
inference-only jobs). **0023 is its outcome**: one A40 sitting, no halt; far 0.450 to 0.669 against scrambled 0.115 to
0.126 (far/scrambled 3.57 to 5.80, clause (i) HELD); 0018's rollout medians over far 0.010 to 0.146 (clause (ii) HELD);
the replicate matches 0018 to 5e-5 on a second host. Records and the Hub backup (`ownrun-far/`) under the hashes 0023
states. **0024 amends (B)'s statement of record** with the rollout reading, on one seed (the clause names the seed count,
the controls and the two ratios; 0019's restriction stands). **0025 registers statistic (C)** (`config/ownrun_task.toml`;
§15 as revised after two 64-prompt probes at reader step_32: whole-prompt cache swap, GSM8K test n = 1,319, greedy at
a 256-token cap, FRESH/STALE/NEAR/FAR arms; **verdict on the STALE answer-flip rate** per lag with a bootstrap
interval, Δ accuracy descriptive since the probes read FRESH at 4 of 64 with 48 capped in `#### N` loops; floor 1%,
positive controls NEAR/FAR ≥ 0.5 flips, dump equivalence at an absolute 2⁻⁵ with (B1) within 1e-3 (0026 restated
R-C21 after the 2026-10-16 sitting halted at 0.33 relative with 2⁻⁷ absolute: float32 forward noise on near-zero
elements, not rounding); seal
`ownrun-task`: (i) controls fire at every g, (ii) STALE flips ≥ 10% at g = 1 [blind], (iii) ≥ 10% at g = 32 [probes read
55/64], (iv) the g = 1 interval below the g = 32 interval [blind]). **0027 reads the (C) sitting** (one A40, 57.8 h,
torch 2.14.1+cu130): all four clauses HELD — STALE flips 0.710 [0.686, 0.735] at g = 1, 0.869 [0.851, 0.886] at g = 32,
peak 0.895 at g = 4 (not monotone); controls fire at all six readers; floor 2–7 of 1,319, identity 0; FRESH accuracy
4.5–5.7% at g ≤ 16 and 12.3% at g = 32, Δ descriptive (DEGRADES_C only at g = 32). Hub `ownrun-task/`. **0028 registers
the second trainer seed** (R-C16 ruled 2026-10-20; `config/ownrun_seed2.toml`, seal `ownrun-seed2`): 0017's recipe with
trainer seed 1, the same rollout rows, controls per (reader, pair) with the far control as evaluability insurance
(`evaluable_under_far`: a scrambled miss carried by a passing far is EVALUABLE, pre-committed), 660 jobs on one A100
sitting; agreement clauses against 0018's medians, which the seal carries — (i) held-out HOLDS_B + slope, (ii) rollout
within 3× of seed 1 per lag and ≥ 1e-3 at g = 1, (iii) rollout ≥ 1e3× held-out, (iv) all control units pass; the
summarizer reads all four. Halting: the saved step-0 weights must hash to seed 1's. **0029 reads it**: step-0 digest equal
to seed 1's, no halt, all twelve control units passed (none carried by far), all five clauses HELD — distances within 1.3%
of seed 1's, held-out medians 0.62–0.91 of 0018's at equal lag, rollout medians 0.80–1.05, slopes 2.13 / 2.05, rollout
over held-out 2.0e4×–5.1e4×. Hub `ownrun-seed2/` (20,527 files). 128-step closed; (C) on seed 2 deferred. **0030 registers
the fourth text `agentic`** on seed 1's readers (`config/ownrun_agentic.toml`, seal `ownrun-agentic`). **0036 reads it**: one A40
sitting at the registering commit `ebe5ee4`, no halt, every control passed, HOLDS_B at every lag (medians 3.61e-6 to 1.76e-4,
14.6×–26.5× seed 1's corpus, 1/506–1/1723 of its rollout, the two suites within 1.4×), lands with held-out text by 1.28–2.07
decades, slope 2.48 read by `3add994`'s summarizer (the fit over per-rung texts landed after the sitting, disclosed); all four
clauses HELD. Hub `ownrun-agentic/` (10,864 files). **0031 registers
the own run on Llama-3.2-3B** (§14.5; `config/ownrun_llama.toml`, seal `ownrun-llama`, plain prompt format, the same ten rollout
prompts sampled from this family's step 0); the trainer's `--prompt-format` and the `ii-lost` / `iii-excess` readouts landed ahead.
**0032 reads it**: one A100 sitting at `54c4d48` (two fixes past the registering commit, disclosed), no halt, step 0 equal to the
snapshot by value, step 1 at 4.71e-5 against the ≈ 5.0e-5 RMS prediction, step 32 at 2.11e-4; all twelve control units passed;
held-out HOLDS_B at every lag, slopes 2.3404 / 2.5063 (on-path 0.0063 past the sealed 2.5: (i) FALSIFIED); rollout 1.19e-3 at
g = 1 to 6.14e-2 at g = 32, 1.9e3×–5.8e3× held-out ((ii) HELD); (iii) HELD; the excess over 0015's backbone 3.5×–19× on four
in-range rungs, one of four over 10× ((iv) FALSIFIED). Hub `ownrun-llama3.2-3b/` (20,537 files). **0033 re-pins pair-only at de6ce9e**
(PR #4, `smollm3-3b-base`; `INSTRUMENT_SHA` stays) and **0034 registers the own run on SmolLM3-3B-Base** (`config/ownrun_smollm3.toml`,
seal `ownrun-smollm3`; the same fifty held-out windows as Llama's, disclosed; step size predicted 1.95e-5 so clause (iv) is expected
UNREAD; not yet run). **0035 registers the Instruct-initialised own run** (`config/ownrun_instruct.toml` + `config/ownrun_instruct_task.toml`,
seals `ownrun-instruct` / `ownrun-instruct-task`; Qwen2.5-1.5B-Instruct@989aa79 under 0017's recipe with chat prompts, the Instruct side
dumped as `target`, 0017's fifty held-out windows, disclosed; (C) at cap 768 with (v) FRESH accuracy ≥ 50 points at every reader and
(vi) STALE below FRESH at g = 32, g = 1 accuracy descriptive; the equivalence (B1) read from the run's own (B) summary and recorded with
its digest; to run on external compute, both halves; not yet run).
