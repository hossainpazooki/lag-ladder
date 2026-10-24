# Diagnostic, NOT the registered probe: Step 1 without the repetition penalty

> **Not evidence under linear-ceiling; see [../README.md](../README.md) for the header, status and terminology.** This is the runner's original note, redacted. Here "registered probe" means the runbook's Step 1 probe run unedited (lag-ladder has not registered it), and the "independent recheck" was a check by the runner's own Claude Code session, not an independent review. Its paths point into the runner's working tree: in this folder, `../../2026-10-21-farhan-run/probe-instruct-*` is `../step1-probe/probe-instruct-*`, `../../_gpu_run/diag/*.py` is `../code/diagnostic/*.py`, `../../_gpu_run/diag-20261021-235911/` is `../logs/diagnostic/` (shard folders not included), `../../_gpu_run/argmax256/` is `../logs/argmax256/` and `../code/argmax256/`, and its attempt folder is `../logs/argmax256/attempt/`. Three of its statements are corrected in ../README.md (Corrections to the runner's original note).

Run 2026-10-21 23:59 – 2026-10-22 00:04 MDT on gpu-node-1 (8× NVIDIA RTX PRO 4000 Blackwell 24 GB, driver 595.91.07,
torch 2.14.1+cu130, transformers 5.20.2, float32), the same card model and software as the registered probe in
`../../2026-10-21-farhan-run/probe-instruct-{256,768}/`.

**Why.** `baseline_probe.py` calls `model.generate(do_sample=False, ...)`, which inherits the pinned revision's
`generation_config.json`: `repetition_penalty: 1.1` (transformers applies it even without sampling) and
`eos_token_id: [151645, 151643]`. The registered (C) loop (`lag-ladder/tools/task_swap.py` `_greedy`) is plain argmax
that stops only on `tok.eos_token_id` (151645). So the registered probe's numbers were decoded under a penalty that
(C) does not apply.

**What changed.** Only decoding: `repetition_penalty` 1.0 and `eos_token_id` 151645, set on the loaded model's
generation_config by `../../_gpu_run/diag/norp_probe.py`, which then runs the package's `baseline_probe.main()`
unedited. Prompts (first 200, pinned file), system line, chat template, answer rule, float32, caps 256 and 768 all as
in the registered run. Each cap ran as 4 contiguous prompt shards (50 prompts each, one card each) and was merged
back with the package's own `summarize()` (`../../_gpu_run/diag/merge_diag.py`); `probe.json` carries a
`diagnostic` block saying so, and its `elapsed_s` is the slowest shard.

| cap | registered probe (penalty 1.1) | this diagnostic (plain argmax) |
|---|---|---|
| 256 | 44.5 [37.5, 51.5], band flips-verdict | 51.5 [44.5, 58.0], band accuracy-verdict |
| 768 | 64.5 [58.0, 71.0], band accuracy-verdict | 72.0 [65.5, 78.0], band accuracy-verdict |

**How firm (independent recheck, 2026-10-22).** Every number above recomputes exactly from the records, and 48 of
200 answers change at cap 256 (net +14) and 49 at 768 (net +15). The direction is consistent, but the gain is
statistically marginal on 200 prompts: exact McNemar p = 0.059 at 256 (paired 95% CI about [0, 13.5] points) and
p = 0.044 at 768 (about [1, 14]), and the two caps are not independent tests. The cap-256 band change rests on 3
prompts (103/200 vs the 100 threshold; exactly 50/100 on the brief's first 100). Cap 768 is accuracy-verdict either way.
The stop-set half of the override changed nothing (no completion in either run contains 151643), so the shift is the
penalty's. In the text, the penalty makes the model copy the question less, and its answers fail more often on
set-up, omitted quantities and arithmetic. Same node (gpu-node-1), Slurm job (JOB-A), venv, weights and prompts as the
registered run; cap256 shards 0-1 even used the registered run's two physical cards.

**What this is not.** It follows (C)'s decoding *rule* (raw argmax, stop on 151645) but is not a (C) run. (C)'s FRESH
arm decodes against a prompt cache rounded through float16 (on the base model that moved 0.15-0.53% of answers); this
diagnostic's unrounded float32 cache is (C)'s fp32 floor arm. It also uses step-0 weights rather than step-g, and 200 of
1,319 prompts. No Instruct cap is registered yet.

**Which decoding made which records** (`argmax_check.json`, `../../_gpu_run/diag/argmax_check.py`): each cap-768
completion teacher-forced through the pinned model in one forward pass. Registered: 198/200 records recoverable,
51,783 tokens, 100.00% equal the 1.1-penalized argmax, 94.62% the raw argmax. This diagnostic: 200/200, 50,712
tokens, 100.00% raw argmax, 94.72% penalized argmax.

Cap 256, the same script unedited (`argmax_check_256.json`, run 2026-10-23 19:47 MDT on an NVIDIA H200 NVL, gpu-node-2,
same venv, weights and float32; the card differs from the probe's). Registered: 199/200 records recoverable (index 3
does not re-encode to its stored hash), 42,930 tokens, 42,929 (99.998%) equal the 1.1-penalized argmax, 94.13% the
raw argmax; 198 of the 199 records match the penalized rule at every token. The one exception is a single token, the
near-tie disagreement a full-sequence forward pass is expected to produce now and then. This diagnostic: 200/200,
41,943 tokens, 100.00% raw argmax, 94.22% penalized argmax. So the registered cap-256 records, the ones the band
decision reads, were decoded under the penalty, as measured (not only inferred from the generation config).

Files: `probe-norp-256/`, `probe-norp-768/` (`probe.json`, `records.jsonl`), `sha256sums.txt`, `argmax_check.json`,
`argmax_check_256.json` (sha256 `8aada1fa05829fc0…`).
Logs: `../../_gpu_run/diag-20261021-235911/` (run.log, per-shard logs, shard folders, verify60.txt);
`../../_gpu_run/argmax256/` for the cap-256 check (manifest.md, watch.log, attempt-20261023-194656-gpu-node-2/).
