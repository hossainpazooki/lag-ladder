# Handoff — session close 2026-09-25: R10 ruled, the bridge seed written, the instrument branch half-committed

2026-09-25. Newest commit this brief describes: lag-ladder `04169c6abbb7fb01e1b0818192e8ef11442b6330`
(`main`, 0 ahead / 0 behind at 04:27Z); kv-transfer-replication branch `holdover-instrument` at
`47bdd39` (pushed) **with uncommitted work on top** (see Current state); linear-ceiling `3902a16`
(unchanged by this session since `39b13b4`). Supersedes nothing: the 2026-09-22 brief still describes
the scaffold; this one describes what moved since. Two facts changed the plan since that brief and
are not in it: **the operator runs experiments on a local Apple-silicon Mac mini now, not the Algoverse
box**, and **kv-transfer-replication is the instrument for statistic (A) only** (ledger 0002).

## Current state

| item | tag | re-verify |
|---|---|---|
| Ledger entries 0001 (scaffold, pins, rulings R6–R9) and **0002 (R10: instrument for (A) only)**, chained | built | `cd ~/dev/lag-ladder && .venv/Scripts/python.exe -m lag_ladder.ledger_check` → `ledger ok (blocks unchanged vs HEAD)`; `grep -c '^### ' ledger/ledger.md` → `2` |
| Design doc: working title, R7–R10 in §10, §6 rewritten (OLMo only), §9 pilot-first build order, three review defects folded into §3/§4 | built | `grep -n "R10 — instrument scope" docs/2026-09-19-holdover-design.md`; header line 3 carries "The Shelf Life of a KV Cache Across Policy Updates" |
| Visual README for coauthors (three mermaid diagrams, four-arm table); scope sentence verbatim once | built; **mermaid rendering on GitHub not checked** | `.venv/Scripts/python.exe -m lag_ladder.lint_scope` → `scope ok`; open the repo on GitHub and look at the three diagrams |
| Bridge seed `docs/2026-09-24-seed-bridge-pipelinerl-instrument.md` — G1–G7 with what-breaks / bridge / closes-when; precondition marked ruled | built | file exists; §1 says "**ruled 2026-09-24** (ledger entry 0002" |
| Learnings: 4 entries 2026-09-22 + 3 entries 2026-09-25, all with read-only re-verify lines | built | `docs/learnings/LEARNINGS.md` → 7 rows; run any entry's `re-verify:` |
| Suite, seal, scope, ledger gates green at HEAD | built | `pytest -q` → `86 passed`; `seal verify` → `OK (no sealed predictions yet)` |
| **kv-transfer-replication `holdover-instrument`**: Task 4 (CI workflow, `kvt-dump` / `kvt-score-positions` entry points, `--probe`) committed `e54c102`; Task 2 (`device()` cuda→mps→cpu, `KVT_DEVICE` override) committed `47bdd39`; both pushed — **and the pushed tip is RED on its own** (`1 failed` on the `--probe` test: the committed script uses code that is still uncommitted; verified in a throwaway worktree, learning 2026-09-25; that repo's own brief `docs/handoff/2026-09-25-holdover-instrument.md` says the same) | built as a working tree, **not green as pushed, not merged, not re-pinned** | `git -C ~/dev/kv-transfer-replication log --oneline -3 holdover-instrument`; `cat ../kv-transfer-replication/.github/workflows/ci.yml` |
| kv-transfer-replication Tasks 1 and 3 (`Pair.revision` / `local_path`, `resolve`, `--revision` / `--local-path`, `--dtype`, `meta.json` keys `revision` / `local_path` / `load_dtype`) | **in-progress: in the working tree, uncommitted** — 98 insertions over `kvt/pairs.py`, `kvt/models.py`, `kvt/data.py`, three test files; the suite on that tree is `165 passed` | `git -C ~/dev/kv-transfer-replication status --short` → six ` M` lines; `git -C … diff --stat` |
| The low-effort prompt that produced the branch | consumed and deleted (never committed) | `git -C ~/dev/lag-ladder log --all --oneline -- docs/2026-09-24-prompt-kvt-branch.md` → nothing |
| Re-pin of the instrument at the merged branch (ledger 0003) | planned; blocked on the merge above | — |
| Prior-art search under the prefix-persistence framing + the framework list (design §9 step 1) | planned; **still not done** — carried over from the 09-22 brief | — |
| Bridge controls G1 precision / G2 origin / G4 kernel identity / G7 both prompt sets — registration and runs | planned | seed §3 |
| Pilot driver + fail-closed summarizer; pilot registration; (B)/(C) scorers in `lag_ladder/` | planned | no `src/lag_ladder/pilot*.py`; `config/pilot.toml` `registered_by = ""` |
| Mac mini specifics (unified memory size, torch/MPS version, whether `sdpa_repeat_kv` in fp32 runs on Metal) | **unknown** | — |

## Locked decisions

Carried from the 09-22 brief (names; R6 no Oct 12; R7 prefix-cache persistence; R8 OLMo only; R9 scaffold;
chassis copied not shared; τ inherited not recalibrated; same-prefill for all arms; SCRAMBLED control;
GPU protocol linked) — unchanged. New since:

| decision | reason | where |
|---|---|---|
| **R10 — kv-transfer-replication is the instrument for statistic (A) only**; (B)'s log-prob scorer and (C)'s swap scorer are `src/lag_ladder/` code | keeps "same bytes" comparability for f\*/τ_K with Carryover while not making a cross-model-transfer replication carry async-RL generation code; the port alternative is reopenable by entry if the instrument cannot run on the Mac | ledger 0002; design §10 R10; `UPSTREAM.md` §1; `CLAUDE.md` |
| Working title *Holdover: The Shelf Life of a KV Cache Across Policy Updates* ("Weight Syncs" for MLSys); fallback *Holdover: Stale KV Caches in Asynchronous RL, from One Update to Thousands* | parallels Carryover's subtitle; commits to no result | design header; README line 3 |
| Experiments run locally on the Mac mini; the Algoverse path is a fallback, not the plan | operator bought the machine for this; removes the grant dependency and the ≈50 GB shared-disk constraint | this brief; the 09-22 brief's box invariants 8–11 are now conditional |
| The upstream branch is the operator's to merge; lag-ladder re-pins by a numbered entry, never by editing `__init__.py` alone | `tests/test_imports.py` asserts `UPSTREAM.md` ⇔ `__init__.py` pins; the entry is what makes the pin auditable | `UPSTREAM.md` §1 |
| No fork of PipelineRL is needed for per-step checkpoints | `save_checkpoint_steps` / `also_save_steps` are config (learning 2026-09-25) | design §6 still says "fork"; amend at the next design edit |

## Reuse map

Everything in the 09-22 brief's reuse map still applies (linear-ceiling `e9.py` / `summarize_e9.py` as the
driver/summarizer shapes; `hf_backup.sh`; the E9 runbooks; `config/e9.toml`). New:

- **`kv-transfer-replication@holdover-instrument`** — `kvt-dump --revision <step_N> --dtype bfloat16 --probe`
  is the pilot's dump command once merged; `--probe` prints `device / dtype / attn / peak_bytes` before
  the full dump and records them under `meta.json["probe"]` (the seed's G5 memory probe). `KVT_DEVICE`
  forces a device for the kernel-identity control (seed G4).
- **`Pair.resolve(which)` / `with_revision` / `with_local_path`** (uncommitted, see above) — how the pilot
  names an OLMo revision or an own-run checkpoint directory without touching `PAIRS`.
- **Seed §2's fact table** — every producer/instrument fact with a read-only re-verify line; start there,
  not from memory.
- **Learnings 2026-09-25** — the engine's actual cache behaviour and dtypes, with `gh api` one-liners.
- **`ledger_check.chain_hash`** — compute a new entry's chain *after* writing it, over the file as it stands,
  up to the new heading (learning 2026-09-25); the gate refuses anything else.

## Invariants

All thirteen from the 09-22 brief hold, with these amendments:

- Invariants 8–11 there (websocket cells, MIG slice, cu128 wheel, shared-login release, box disk) apply
  **only if** the Algoverse fallback is used. On the Mac: torch from PyPI (MPS is in the default build);
  the kernel-identity control (seed G4) runs **before** any dump is trusted; disk is local and the
  pull-verify-delete loop is not needed, but R6/R8 (nothing exists in one place; HF backup as transport)
  still do.
- **R10:** nothing Holdover-specific beyond (A)'s dump-and-score lands upstream. A PR to the instrument
  that adds generation, log-prob scoring or a swap arm is a refusal.
- **Re-pin discipline:** merging `holdover-instrument` changes the instrument's HEAD; `lag_ladder.upstream_gate`
  pins by *ancestry + invoked paths unchanged*, so the current pin `063f402` keeps passing for paths the
  branch did not touch — and **fails for `kvt/data.py` / `kvt/models.py` / `kvt/pairs.py`, which it did**.
  Any driver that invokes those must record the post-merge sha via ledger 0003 first.
- **Uncommitted upstream work is invisible to the gate and to CI.** Tasks 1 and 3 exist only in a working
  tree on one machine until committed; a `git checkout` there discards them.
- The scope lint flags ordinary restatements (learning 2026-09-22); the design doc and the one
  evidence-quoting learning are exempt by exact filename, nothing else.

## Open / next

**First: finish the instrument branch.** In `~/dev/kv-transfer-replication`, commit Tasks 1 and 3 as a
**forward commit** (the suite on that tree is 165 passed; the two pushed commits are broken on their own
and pushed history is not rewritten — the repair is the next commit, which makes the tip green; CI on
the branch stays red until it lands), push, merge
`holdover-instrument` into `main`, then in lag-ladder append **ledger 0003** re-pinning the instrument at
the merge sha (also `UPSTREAM.md` §1 and `src/lag_ladder/__init__.py::INSTRUMENT_SHA`; `test_imports`
enforces agreement) and amend design §6's "fork" sentence per the checkpoint-config learning. Blocker:
none — it is the operator's commit.

**Second: the Mac mini pre-flight**, which is seed G4 before anything else: `KVT_DEVICE=mps kvt-dump …
--probe` on a 48-token toy vs `KVT_DEVICE=cpu`, `score_positions` between the two, deviation recorded
against the E9 bound (max |ΔK| 9.2e-4 on a scale of 423). Needs the mini's memory size and torch version
written into the runbook it starts. If Metal falls back to the math kernel, recompute the pilot budget.

**Third: design §9 step 1** — the prefix-persistence prior-art search and the framework list the 09-20
search never reached. Twice carried over; it gates whether §2's novelty paragraph survives the reframe.

**Then:** register the bridge controls (seed §5 item 2), build the pilot driver/summarizer on the
`e9.py` shape, register the pilot (needs R1 second task and R2 bounds — still owed by the operator), run.

**Not for the next session:** the (C) swap scorer, the own run, anything on the Algoverse box.
