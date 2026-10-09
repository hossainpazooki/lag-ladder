# Upstreams (read-only) and chassis provenance

Three external trees, three roles. `src/lag_ladder/__init__.py` is the single source of truth for
every sha below; `tests/test_imports.py` asserts this file names exactly those shas.

## 1. Instrument for statistic (A) only — `kv-transfer-replication` (invoked, never imported; ledger 0002)

- Repo: https://github.com/hossainpazooki/kv-transfer-replication
- Pinned commit: `5d4aa4a98b40f5ab37dd83ebb6b2ee802203994a` — `main` after the three Qwen2.5-1.5B
  weight-distance pairs merged; ledger 0008, 2026-10-09. Before it, 1380635 (ledger 0005: the `olmo2-1b-rlvr1`
  pair) and 0d27c68 (ledger 0004: PR #1 Llama 3 pair and PR #2 G3 checkpoint provenance), to which the four (A)
  paths are byte-identical: per-side `ModelRef` pins on `Pair`
  (`source_revision` / `target_revision` / `*_local_path`), checkpoint manifest hashed before load and
  verified on every `KVDump.load`, `device()` with mps, `--dtype`, `--probe`, console entry points, CI.
  Ledger 0003's `Pair.with_revision` interface (pin 9ca6258) is replaced. The prior pins (linear-ceiling's
  entry-0036 pin, "RoPE spec from the model's rotary embedding"; 9ca6258) are named in ledger
  0001/0003/0004; `scripts/score_positions.py` and `kvt/pertoken.py` are byte-identical to both. Later
  re-pins come by numbered entry (the swap scorer for (C)); each new sha is recorded here and in
  `config/*.toml` after it lands upstream. Scope (ruled 2026-09-24, entry 0002): the dump writer and the
  per-token scorer for statistic (A). Statistic (B)'s log-prob scorer and (C)'s swap scorer are
  `src/lag_ladder/` code.
- Local path (used by `config/seal.toml` as `${upstream}`): `../kv-transfer-replication`
- Rule: nothing in this repo writes into the upstream tree, imports `kvt`, or copies its code. Dumping,
  fitting and scoring are invoked there, by subprocess, in the upstream's own environment. Experiment
  gates check their OWN recorded pin by ancestry + invoked-paths-unchanged (`lag_ladder.upstream_gate`).

## 2. Checkpoint producer — NOT PINNED (design §6, ruled 2026-09-22)

- The pilot uses published checkpoints (`allenai/OLMo-2-0425-1B-RLVR1`, 13 `step_*` revisions at stride
  200) and needs no trainer.
- Preferred producer once the pilot shows signal: the author implementation
  https://github.com/ServiceNow/PipelineRL (Apache-2.0), unmodified — a checkpoint every optimizer
  step is config (`save_checkpoint_steps` / `also_save_steps`, checked 2026-09-25), so no fork. `main`
  observed at
  `58d393458625ad63ed539f2dcd072c85700c557f` on 2026-09-22 — recorded so a later pin can state what
  moved. Checked that day: the repo's code search has no `recompute` / `kv_cache` /
  `reset_prefix_cache` hit (default branch only), so Figure 7's measurement code is not published;
  the smallest documented configuration is 4×H100.
- When pinned: record the author sha (and a fork sha only if a fork ever becomes necessary); the gate
  asserts the trainer paths are unchanged from the author sha.

## 3. Chassis — copied from `linear-ceiling` with provenance

- Repo: https://github.com/hossainpazooki/linear-ceiling, read at
  `888f745084c63eb52d114acd951dc787db82a71a` (`src/` clean at that HEAD on 2026-09-22).
- `src/lag_ladder/pertoken.py` is copied from `src/linear_ceiling/e9_pertoken.py` at the same commit
  (`centered_delta`, `token_mean`, `layer_mean`, `F_STAR_REL_TOL`, `f_star`, `band_outcome` verbatim; the seam,
  block, bootstrap and null-pairing helpers dropped), added 2026-10-09 for the pilot driver.
- Copied, not shared, so a repo under active amendment cannot move this one's gates. The GPU protocol
  (`docs/gpu-experiment-protocol.md`, rules R1–R12) is LINKED, not copied: it governs every GPU run here.

| what | filePath @ 888f745 | here | change |
|---|---|---|---|
| canonical JSON bytes + sha256 helpers | `src/linear_ceiling/hashing.py` | `src/lag_ladder/hashing.py` | verbatim (package name) |
| the one seeded generator | `src/linear_ceiling/rng.py` | `src/lag_ladder/rng.py` | verbatim |
| pre-artifact seal (write / require / verify) | `src/linear_ceiling/seal.py` | `src/lag_ladder/seal.py` | verbatim; a "pair" is writer → reader checkpoint |
| upstream pin by ancestry + paths | `src/linear_ceiling/upstream_gate.py` | `src/lag_ladder/upstream_gate.py` | verbatim |
| TOML → frozen config | `src/linear_ceiling/config.py` | `src/lag_ladder/config.py` | seal loader verbatim; E0/E7/E8/E9 loaders dropped; `load_pilot_config` new |
| ledger lint (structure, chain, block diff, cell provenance) | `src/linear_ceiling/ledger_check.py` | `src/lag_ladder/ledger_check.py` | manifest check and frozen map dropped; chain REQUIRED from 0002 |
| scope-sentence lint | `src/linear_ceiling/lint_scope.py` | `src/lag_ladder/lint_scope.py` | mechanism verbatim; sentence and exemptions are this repo's |
| CI job | `.github/workflows/ci.yml` | `.github/workflows/ci.yml` | torch install dropped |
| tests for the above | `tests/{conftest,test_hashing,test_rng,test_seal,test_upstream_gate}.py` | same names | verbatim (package name); `tiny_snapshot` dropped |
| tests, adapted | `tests/{test_config,test_ledger_check,test_lint_scope,test_imports}.py` | same names | re-anchored on this repo's sentence, pins and checks |
| ledger house style | `ledger/ledger.md` header | `ledger/ledger.md` | chain from 0002, `verdict:` lines from 0001 |
| f*(τ), τ_K = 0.3186, τ ladder | `ledger/ledger.md` entry 0023 @ `19e91286` | ledger entry 0001 | cited, not recomputed |
