# Handoff — Holdover / lag-ladder: running the pilot and the grid on the Algoverse box with linear-ceiling's workflow

2026-09-22. Newest commit this brief describes: lag-ladder `e79d3ad4789467cec39ffb5ca733a8b07a0c0173`
(`main`, pushed, 0 ahead / 0 behind at 23:09Z); linear-ceiling `39b13b4` (pointer stub for the moved
design doc). Pick-up measures drift from those two. Written for the session that will drive this repo
on the Algoverse JupyterHub GPU grant; that session runs `/rigor:pickup` against this brief and
re-verifies every claim below rather than trusting it.

Three documents govern everything here and this brief does not restate them:

1. `docs/2026-09-19-holdover-design.md` — the design; §9 is the build order, §10 the rulings, §11 the repo shape.
2. `../linear-ceiling/docs/gpu-experiment-protocol.md` — rules R1–R12 for every GPU run, and "The box,
   concretely (Algoverse TLJH, JupyterHub only)" for the box. **Linked, not copied**, by ruling.
3. `../linear-ceiling/docs/2026-09-02-e-rl-design.md` — the instrument, controls, dump sizing (§4), τ ladder (§5).

## Current state

| item | tag | re-verify |
|---|---|---|
| Repo scaffold: chassis copied from linear-ceiling `888f745`, package `lag_ladder`, CI | built | `cd ~/dev/lag-ladder && .venv/Scripts/python.exe -m pytest -q` → `86 passed`; `git log --oneline -1` → `e79d3ad` |
| Ledger entry 0001 (rulings, pins, inherited definitions); no hypothesis registered | built | `.venv/Scripts/python.exe -m lag_ladder.ledger_check` → `ledger ok (blocks unchanged vs HEAD)`; `grep -c '^### ' ledger/ledger.md` → `1` |
| Seal, scope lint, upstream gate present and green | built | `python -m lag_ladder.seal verify` → `OK (no sealed predictions yet)`; `python -m lag_ladder.lint_scope` → `scope ok` |
| Instrument pin `063f402` = kv-transfer-replication HEAD | built | `git -C ../kv-transfer-replication rev-parse HEAD` → `063f4023fdde67dedbee01a92518ce7f83f6cf5d`; `grep -c 063f4023 UPSTREAM.md src/lag_ladder/__init__.py` |
| `config/pilot.toml` with the design's proposed values, `registered_by = ""` | built, **unregistered** | `grep registered_by config/pilot.toml` → `registered_by = ""` |
| Design doc §2 prior art re-read from primary PDFs (PipelineRL Fig 7 has a lag axis 0–32; Nemotron never says it tested recompute) | built (learnings 2026-09-22) | `docs/learnings/LEARNINGS.md` re-verify lines |
| Concurrent-work search (18 sources; framework list NOT covered) | built, partial | design §2 "Concurrent and adjacent work"; coverage gap listed there |
| Prior-art search under the **prefix-cache persistence** framing (design §9 step 1) | planned | — |
| Upstream `kvt/pairs.py::Pair` revision / local-path field (design §9 step 2) | planned; **blocks the first dump** | `grep -n "revision" ../kv-transfer-replication/kvt/pairs.py` → no hit today |
| Pilot driver (dump 13 OLMo revisions, score f*(τ_K) per pair) + fail-closed summarizer | planned | no `src/lag_ladder/pilot*.py` exists |
| Swap scorer for statistic (C) (generation under a supplied cache) | planned | — |
| Registration entry (statistics, bounds, tasks, lags, seeds, SCRAMBLED control) | planned; needs rulings R1, R2 | — |
| Own-run checkpoint producer | deferred by ruling R8 | `UPSTREAM.md` §2 |

Compute sizing (derived in the 2026-09-22 session from the 09-02 design §4 per-token cost and the E9
runbooks; not yet in the design doc): pilot = ≤ 12 GB GPU peak, minutes of GPU compute, **disk-bound**
(6.7 GB per checkpoint at stride 1 → 87 GB for 13; 17 GB for the 10 held-out docs; 22 GB at stride 4),
≈ 40 GB of snapshot downloads, hours of CPU scoring, 2–4 h wall on any 24 GB card. Grid (C) ≈ 330k
generations ≈ 100M tokens; 7–14 h on an A100 **assuming** 2–4k tok/s with an injected cache in HF
`generate` — the one unmeasured number.

## Locked decisions

| decision | reason | where |
|---|---|---|
| Separate repo; chassis **copied**, not shared | a repo under active amendment (linear-ceiling) must not move this one's gates | design §11; ledger 0001 |
| Names: paper *Holdover*, repo `lag-ladder` | repo named for the instrument so it survives a reframe and a reviewer cannot walk from title to repo | design header |
| R6 — Oct 12 / ARR October **dropped**; the pilot decides the venue | 23 days with no code was judged tighter than the ICLR window already refused; likely null at engine lags | design §9, §10 |
| R7 — primary framing **prefix-cache persistence** | an un-invalidated prefix cache has unbounded lag, so the 200–2400 OLMo tail is engine-relevant and the 8-shot condition is primary | design §1 |
| R8 — **OLMo-2-0425-1B-RLVR1 only** until the pilot shows signal | no trainer needed; PipelineRL's smallest documented config is 4×H100 and is not fundable now | design §6; `UPSTREAM.md` §2 |
| Pilot = statistic (A) only, dumps and no generation | cheapest experiment that can close the lane (f* ≈ 0 to lag 2400 ⇒ short note) or open it | design §9 step 3 |
| f*, τ_K = 0.3186, τ ladder {0.3186, 0.10, 0.03} **inherited from linear-ceiling 0023, not recalibrated** | comparability with Carryover; recalibration would be a new entry | ledger 0001 |
| Every arm's KV built by the **same teacher-forced prefill**, never a replayed decode | decode-written and prefill-written caches are not bit-identical; the k = 0 identity control would halt on numerics | design §3 |
| SCRAMBLED positive control required before any HOLDS_C is read | NULL controls delta size, not instrument sensitivity | design §4 |
| GPU protocol R1–R12 is linked, not copied | one text to amend | design §11 |
| Names in the repo stay off the paper's title (`lcfm_anon` pattern at submission) | double-blind | design §10 R5 |

Open rulings (recommendations recorded, not rulings): R1 second task, R2 (C)/(B) bounds, R3 MAPPED arm,
R4 vocabulary — design §10.

## Reuse map

**In this repo** — `src/lag_ladder/`: `upstream_gate.check_upstream(upstream, sha, paths, who=)` for every
gate; `seal` for pre-run predictions (a "pair" is writer→reader checkpoint); `config.load_pilot_config`;
`hashing.sha256_file_bytes` / `hash_json_obj` for manifests and fingerprints; `rng.make_rng` — the only
generator allowed (tests grep for it).

**In linear-ceiling (read, adapt with a provenance row, never import):**
- `src/linear_ceiling/e9.py` — the driver shape to copy for the pilot: `--check` gate, `--align-only`,
  per-unit checkpointing, `--resume`, kept-subset fingerprints in `report.json`.
- `src/linear_ceiling/summarize_e9.py` — the fail-closed summarizer shape (recompute from raw, refuse on
  any seam, controls checked before figures, `calibrate_tau`).
- `src/linear_ceiling/e8.py::dump_agent` / `e9.py::score_pairs` — how the upstream `dump_kv.py` and
  `score_positions.py --per-token` are invoked by subprocess with `PYTHONPATH=$PWD` on the box.
- `tools/jupyterhub/jh.py`, `pull.py` — the only way onto the Algoverse box; `tools/jupyterhub/README.md`.
- `tools/hf_backup.sh` + `tools/hf_verify_backup.py` — R8 backup; `--check` first, `--num-workers 1` on Windows.
- `docs/2026-09-02-e9-gpu-runbook.md` and `docs/2026-09-10-e9l-gpu-runbook.md` — runbook shape, the
  amended-on-the-box section is the trap list in narrative form.
- `config/e9.toml` — config shape (rule / controls / gate sections) the pilot config should grow toward.

**In kv-transfer-replication (invoke only):** `scripts/dump_kv.py`, `scripts/score_positions.py --per-token`,
`kvt/pertoken.py` (f* is computed from its per-token squares), `kvt/pairs.py::Pair` (needs the revision field).

**Memory notes (the driving session should load these):** `jupyterhub-box-driving` (the trap list),
`shared-gpu-login-release-check` (R7 step 0: list before deleting), `relaunch-redirect-destroys-halt-logs`,
`hf-upload-large-folder-windows-stall`, `bash-tool-large-heredocs-fail`.

## Invariants

1. **Registered before requested (R1).** No dump, no score, no GPU request while `config/pilot.toml` has
   `registered_by = ""`. The registering entry fixes seed, τ, ladder, anchors, revisions; after it, none
   of those change without a new entry. Breaks: the verdict is un-pre-registered and the ledger's house
   style is void.
2. **The ledger is append-only and chained.** `ledger_check` refuses an edit to any committed entry block
   and requires `prior-entries-sha256:` from 0002 on. A wrong entry is superseded, never edited.
3. **No number enters the ledger except from a fail-closed summarizer reading `results/`.** Nothing under
   `results/` is ever committed (`tests/test_imports.py` enforces the placeholder set).
4. **Upstream is read-only, invoked by subprocess, pinned by ancestry + invoked-paths-unchanged.** A re-pin
   is a ledger entry plus `UPSTREAM.md` plus `__init__.py` (`test_imports` asserts the three agree).
   Never `import kvt`. The instrument change this lane needs (the `Pair` revision field) is a commit
   **upstream**, by the operator, then a re-pin here.
5. **Scope sentence held verbatim in README and nowhere paraphrased** (`lint_scope`, threshold 0.45 on
   distinctive-stem overlap). It flagged two of my own sentences on 2026-09-22; cite it, do not restate it.
6. **Seeds and thresholds live in `config/*.toml`; randomness only via `make_rng`.**
7. **Every arm's cache is built by the same prefill path; k = 0 identity control runs on the run GPU** before
   any cell is read (design §3). Breaks: a numerics halt or, worse, a false STALE≠FRESH.
8. **On the box: R4–R7.** Launch detached with `setsid nohup … < /dev/null &` on a line of its own; never a
   `pkill -f` pattern that matches its issuer; rotate the log before every relaunch; every remote command
   ≤ ~40 s or the websocket drops; pull → verify by sha → delete; **before any release, `ls -la ~` and
   `ps -u $(whoami)` and abort on anything foreign** (the box login is shared; 2026-09-09 incident).
9. **Nothing may exist only on the box** (R6); the verified home mirror is backed up to an HF dataset
   (R8, public is fine by the 2026-09-11 ruling) and the backup is transport, not evidence.
10. **Disk is the pilot's binding constraint** (≈ 50 GB shared-disk policy on the Algoverse box; 87 GB of
    stride-1 dumps for 13 revisions). The pilot must dump the held-out subset or stride 4, and pull-delete
    per checkpoint. Breaks: the box fills mid-run and the driver's own checkpointing is the only recovery.
11. **Box specifics that cost hours last time:** torch from the `whl/cu128` index (driver is CUDA 12.8; PyPI
    torch is cu130 and `cuda.is_available()` is False); the grant is an H100 MIG `3g.40gb` slice via
    `CUDA_VISIBLE_DEVICES`; upstream scripts need `PYTHONPATH=$PWD` on the box; uploads one stream, 32 MB
    parts; the server can be SIGTERMed without explanation — checkpoint per unit.
12. **Local dev is Python 3.12** (`pyproject` pins `<3.13`; the machine's default is 3.14 — build the venv
    from `py -3.12` or linear-ceiling's interpreter).
13. **Git history is the operator's.** Output commit commands; never commit or push. Rigor's git-guard also
    matches remote git in Bash text — remote git steps go in an uploaded script.

## Open / next

**First thing:** design §9 step 1 — the prior-art search under the prefix-persistence framing plus the
framework list the 09-20 search never reached (SGLang, slime, prime-rl, verl, OpenRLHF, NeMo-RL, LlamaRL,
ROLL, Kimi, MiniMax, GLM, DeepSeek, Composer; vLLM / open-instruct / TRL / AReaL issue trackers). If
someone has already measured a persistent prefix cache across syncs, §2 changes before any code is written.
No blocker.

**Second:** step 2 — the `Pair` revision / local-path field in `kv-transfer-replication` (operator commits
upstream; this repo re-pins by entry 0002). **Blocker for everything after it.**

**Third:** the pilot driver + summarizer, modelled on `e9.py` / `summarize_e9.py`, then the registration
entry, then the GPU request. Before the request: a runbook under `docs/` in the E9 shape, sized for the
disk constraint (invariant 10), with the k = 0 identity control on the first pair.

**Rulings still owed by the operator before registration:** R1 (second task), R2 (bounds). R3/R4 can wait
for the paper.

**Model note for the driving session:** Claude Opus 5.5's default effort is `medium`; run this work at
`high` or above — the E9 days were lost to under-checked assumptions (attention kernel, cu130 wheel), not
to compute.
