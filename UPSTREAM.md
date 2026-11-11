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
  `config/*.toml` after it lands upstream. Pair-only re-pin for one family (ledger 0033, 2026-10-24): `main` at
  de6ce9e (`de6ce9e944d9…`; the full sha is in ledger 0033 and the config) after PR #4 (`kvt/pairs.py` only, the `smollm3-3b-base` identity pair); the four
  (A) paths are byte-identical to 5d4aa4a's blobs, `INSTRUMENT_SHA` stays 5d4aa4a, and only `config/ownrun_smollm3.toml` pins
  de6ce9e. Scope (ruled 2026-09-24, entry 0002): the dump writer and the
  per-token scorer for statistic (A). Statistic (B)'s log-prob scorer and (C)'s swap scorer are
  `src/lag_ladder/` code.
- Local path (used by `config/seal.toml` as `${upstream}`): `../kv-transfer-replication`
- Rule: nothing in this repo writes into the upstream tree, imports `kvt`, or copies its code. Dumping,
  fitting and scoring are invoked there, by subprocess, in the upstream's own environment. Experiment
  gates check their OWN recorded pin by ancestry + invoked-paths-unchanged (`lag_ladder.upstream_gate`).

## 2. Checkpoint producer — the pilot's producer is unpinned (design §6, ruled 2026-09-22); PipelineRL is pinned for the agentic arm (design 2026-11-09, subsection below)

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
- Pinned for the agentic arm on 2026-11-09 (subsection below): the author sha is recorded; a fork sha only if a fork
  ever becomes necessary; the byte-pinned inputs and the fidelity gate assert the trainer-side inputs are unchanged from
  the author sha. The pilot's published checkpoints remain unpinned.

### PipelineRL — pinned for the agentic arm (design 2026-11-09, reduced scope)

- Pin: https://github.com/ServiceNow/PipelineRL at `58d393458625ad63ed539f2dcd072c85700c557f` (`main`, last pushed
  2026-08-05, Apache-2.0), the sha observed on 2026-09-22 above. Unmodified: every run changes it through
  Hydra overrides only (`config/pipelinerl_tir_overrides.txt`, `config/pipelinerl_tir_sample_overrides.txt`). PipelineRL is
  never imported, vendored or executed from lag-ladder; the pod scripts run it in its own environment and lag-ladder
  reads its stream files and checkpoints. The recipe lives in the two override files; `[trainer]` carries the pin, file
  names, model, and the step count and kept steps the driver reads.
- Facts the driver `tools/task_swap_tir.py` and the stream tool `tools/pipelinerl_stream.py` depend on, read raw at the
  pin (`{sourceRepo: ServiceNow/PipelineRL, commitSha: 58d3934…}`):

| Fact | filePath (at the pin) |
|---|---|
| TIR system prompt (a YAML `\|` block at :8-28, 20 lines, so the text ends with one newline; sha256 `7ee8c06e…c12b41`), `task_template: "{task}"`, `agent_max_loops: 3`, `sandbox_endpoint: ${oc.env:SANDBOX_ENDPOINT,…}` (an environment variable, loopback default), `sandbox_timeout: 10.0`; `_self_` sets `finetune.rl.policy_loss: gspo`, `seq_parallel: 8`, `gradient_accumulation_passes: 1024`, `vllm_config.vllm_kwargs.max_model_len: 32000`, `llm.parameters.max_tokens: 16000`, `finetune.seq_length: 32000` (`:85`, overridden to 8192), `finetune.rl.overlong_filtering: true` (`:90`, pinned explicitly by the override file) | `conf/tir.yaml`:8-30, 55-56, 65-79, 84-90 |
| Defaults: `seed: 42`, `llm.parameters` (`max_tokens: 8192`, `temperature: 1.0`), `enable-prefix-caching: false`, `eval_every_n_versions: 78000`, `attempts: ${finetune.attempts}`, `debug.mode: ""`, `streams_from`, `wandb.use_wandb: true` | `conf/base.yaml`:7, 46-55, 66, 89, 107, 110-111, 124 |
| Finetune defaults: `train_batch_size: 1`, lr 1e-6, cosine, 50 warmup steps, 1024 passes, `max_train_steps`, `seq_length: 12000`, `save_checkpoint_steps: 100`, `keep_intermediate_checkpoints`, `also_save_steps: []`, `use_safetensors: true`, `seq_parallel`, `attempts: 8`, `rl.policy_loss: reinforce`; `grpo.yaml`: `attempts: 8`, `rl.policy_loss: ppo`. Samples per optimizer step = `gradient_accumulation_passes × train_batch_size` over all lead trainers (`finetune_loop.py:629-637`; the passes must divide by the trainer count, :340-342), so 128 passes × 1 = 128 samples = 16 prompts × 8 attempts whatever the GPU split | `conf/finetune/base.yaml`:31, 37, 41, 43, 45, 53, 63, 69, 71, 85, 86, 88, 96, 101; `conf/finetune/grpo.yaml`; `pipelinerl/finetune_loop.py`:340-342, 629-637 |
| Tool schema (`run_python_code`, `MathAnswer`), blocked-code patterns, sandbox output formatting (`[stderr]`, `[execution timed out]`, `[no output]`, `[execution error: …]`), `_parse_tool_arguments` fallbacks, tool messages `Answer submitted: …` / `Unknown tool: …`, the turn loop (messages = system + `task_template.format(task=problem["task"])`, the prompt rendered each turn with the tools, `min_generation_tokens = 256`, break on no tool calls, break after `MathAnswer`), `gold = problem["answer"]`, and `text.finished = ctx.submitted_final_answer` on every turn's TrainingText | `pipelinerl/domains/tir/rollouts.py`:25-37, 40-66, 69-116, 119-149, 152-188, 253-365 |
| Training text: `labels` = generated token ids, masked with `MASKED_TOKEN_ID = -100` over the prompt; `input_ids = prompt_token_ids + labels`; `prompt_tokens` = vLLM's prompt count; `finished = finish_reason != "length"` (overridden by the TIR rollout, above); vLLM calls use `logprobs: 1`, `include_stop_str_in_output: True`, `skip_special_tokens: False`; the chat kwargs carry `tools` | `pipelinerl/async_llm.py`:103-105, 215-346; `pipelinerl/rollouts.py`:13 (`TrainingText`) |
| Hermes parser: `<tool_call>(.*?)</tool_call>\|<tool_call>(.*)` (DOTALL), ids `call_<i>`, `arguments = json.dumps(args, ensure_ascii=False)`, malformed JSON skipped, `content` = the whole model output when no final-response wrapper; the `<tool_calls>` aggregate and bare-JSON paths tried first | `pipelinerl/rl_tool_parser_plugin.py`:18-40, 57-67, 75-201 |
| Actor: `random.seed(cfg.seed)`; train loop `random_iter` (with replacement), test loop never starts under `debug.mode`; `attempts = cfg.attempts if is_training else 1`; metadata `model_version`, `rollout_index`, `step_index`; one stream line per group = the list of every turn's TrainingText; topics `actor` / `actor_test`; the actor model is `finetune/current` if it exists, else `model_path` | `pipelinerl/actor.py`:186, 212-223, 298-299, 312, 345, 489, 786, 796-799, 814-818, 865 |
| Custom datasets: a name with no loader resolves to `datasets/custom/<name>.jsonl` (cwd or repo root), rows keep their keys and get `source_dataset`, `source_id`, `dataset`; `id = row index` | `pipelinerl/domains/math/load_datasets.py`:204-272, 558-562 |
| GSM8K loaders: `task = question`, `answer = answer.split("####")[1]`, ids in row order | `pipelinerl/domains/math/load_datasets.py`:75-83, 446-458 |
| Checkpoints: `output_dir/finetune/current` and `intermediate/<step>`; `time_to_save = (step % save_checkpoint_steps == 0 or step in also_save_steps) and not time_to_stop`, so the final step is written to `current` only; intermediates kept only under `keep_intermediate_checkpoints`; safetensors | `pipelinerl/finetune_loop.py`:353-354, 604-605, 855-858, 1004-1019, 1039-1053; `conf/finetune/base.yaml`:71 |
| `debug.mode` disables weight updates and `debug.mode=actor` launches actor + environment + actor_llm only; the actor's `streams_from` and the finetune job are not started | `pipelinerl/launch.py`:194-198, 230-231, 673-677, 686-687 |
| Streams: file backend `<output_dir>/streams/<topic>/<instance>/<partition>/<shard_id>.jsonl`, one JSON line per write | `pipelinerl/streams.py`:52-59, 238-243, 249-277 |
| Chat template at the model revision: with `tools`, the system turn carries the tool schema inside `<tools>…</tools>`; an assistant turn with `tool_calls` is `content` then `<tool_call>\n{"name": …, "arguments": <arguments \| tojson>}\n</tool_call>` per call (a second JSON encoding of the arguments string); a `tool` message is rendered inside a `user` turn as `<tool_response>\n…\n</tool_response>`; eos `<\|im_end\|>`, pad `<\|endoftext\|>`, no bos | `Qwen/Qwen2.5-1.5B-Instruct@989aa798…` (the full revision is in `config/pipelinerl_tir.toml`; this file carries the three pins only, `tests/test_imports.py`), `tokenizer_config.json` (`chat_template`) |
| Stack: torch ~2.10.0, vllm 0.18.1, transformers ~4.57, deepspeed ~0.18, Python ≥ 3.12, extra `tir = ["sandbox-fusion>=0.3.7"]`; flash-attn 2.8.3 | `pyproject.toml`; `README.md` |

- The SandboxFusion REST shape the driver's client reproduces (`POST <endpoint>/run_code` with `{code, language,
  run_timeout}`; response `{status, message, run_result: {status, stdout, stderr, return_code}}`) is from the
  `sandbox-fusion` 0.3.7 SDK (`src/sandbox_fusion/client.py:85-94`, `models.py:28-40, 92-116`), read, not installed.

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
