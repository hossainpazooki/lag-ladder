"""The own run (design §14): a short GRPO trajectory instrumented for statistic (B) at lags 1..32.

Run with the trainer's own interpreter (TRL + transformers + torch), never with the repository's; it imports
nothing from `lag_ladder` or `kvt`. Two commands:

  sample-rollouts  the step-0 policy samples one completion per held-out GSM8K prompt (the §14 "rollout" text);
                   writes the padded token file, its spans sidecar and the prompt indices the run must not train
                   on; prints the sha256 the registration pins. Run BEFORE registration (0017 pins the hashes).
  train            GRPO for N optimizer steps; after every saved step a callback dumps the held-out sequences
                   and the rollout sequences through the pinned instrument (`scripts.dump_kv --local-path`),
                   writes the whole-vector distance to step 0 (`tools/weight_distance.py`) under the labels
                   the config names, and deletes the weights unless the step is in the keep set.

Pure helpers (keep set, padding/spans, GSM8K answer extraction, rewards, the dump/distance commands) are
module-level and tested offline; everything that needs a GPU or TRL is behind the two commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

KVT_PAIR = "qwen2.5-1.5b-to-instruct"       # the instrument pair whose `source` is Qwen2.5-1.5B
DEFAULT_KEEP = (0, 1, 2, 4, 8, 16, 32)
ROW_LEN = 1024
MIN_ROLLOUT_TOKENS = 64


def decoding_spec(tok, *, do_sample: bool, temperature: float, max_new_tokens: int) -> dict:
    """Every decoding parameter `generate` would otherwise read from the checkpoint's generation_config.json,
    pinned here so a call decodes the same way on every checkpoint. Found 2026-10-24 on the Instruct probe:
    Qwen2.5-1.5B-Instruct ships `repetition_penalty 1.1` and a two-token stop set in that file, and transformers
    applies the penalty even under do_sample=False, so an unpinned call is not the decoder the statistic names. The
    three base checkpoints of 0017–0034 ship no penalty and their eos equals the tokenizer's, so this spec is what
    those rollouts ran under; the rule is per call, not per checkpoint. Recorded in the spans sidecar."""
    eos = int(tok.eos_token_id)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    return {"do_sample": bool(do_sample), "temperature": float(temperature) if do_sample else 1.0,
            "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "num_beams": 1,
            "eos_token_id": eos, "pad_token_id": int(pad), "max_new_tokens": int(max_new_tokens)}


# --- pure helpers -----------------------------------------------------------------------------------------------

def keep_set(spec: str | None) -> frozenset[int]:
    """`"0,1,2,4,8,16,32"` -> the steps whose weights are kept; default DEFAULT_KEEP."""
    if not spec:
        return frozenset(DEFAULT_KEEP)
    steps = sorted({int(x) for x in spec.split(",") if x.strip()})
    if any(s < 0 for s in steps):
        raise ValueError("keep set has a negative step")
    return frozenset(steps)


def pad_rows(rows: list[tuple[list[int], int]], row_len: int, pad_id: int) -> tuple[np.ndarray, dict]:
    """Rows of (tokens, prefix_len) -> one int64 array (n, row_len) padded with pad_id, and the spans sidecar
    {"prefix_len": [...], "length": [...]} the reader passes per sequence. Refuses a row longer than row_len or
    a prefix that leaves fewer than two continuation tokens."""
    arr = np.full((len(rows), row_len), pad_id, dtype=np.int64)
    spans = {"prefix_len": [], "length": []}
    for i, (toks, pl) in enumerate(rows):
        if len(toks) > row_len:
            raise ValueError(f"row {i}: {len(toks)} tokens exceed row_len {row_len}")
        if not (0 < pl and pl + 1 < len(toks)):
            raise ValueError(f"row {i}: prefix_len {pl} leaves no continuation in {len(toks)} tokens")
        arr[i, : len(toks)] = np.asarray(toks, dtype=np.int64)
        spans["prefix_len"].append(int(pl))
        spans["length"].append(int(len(toks)))
    return arr, spans


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K's `answer` ends with `#### <number>`; the number, commas removed."""
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", answer_field)
    if not m:
        raise ValueError("no '#### <number>' in the GSM8K answer field")
    return m.group(1).replace(",", "")


def final_number(text: str) -> str | None:
    """The model's answer: the number after the last `####` if present, else the last number in the text."""
    m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def _num_equal(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def reward_exact(completion_text: str, gold: str) -> float:
    """1.0 when the completion's final number equals the gold answer."""
    return 1.0 if _num_equal(final_number(completion_text), gold) else 0.0


def reward_format(completion_text: str) -> float:
    """0.2 for stating the answer as `#### <number>`; keeps the gradient alive when every answer is wrong."""
    return 0.2 if re.search(r"####\s*-?[\d,]*\.?\d+", completion_text) else 0.0


def select_prompt_indices(n_total: int, k: int, seed: int) -> list[int]:
    """k distinct GSM8K train indices for the rollout prompts, from a seeded generator; the run excludes them."""
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=k, replace=False))


def dump_command(kvt_python: Path, kvt_root: Path, ckpt: Path, tokens_npy: Path, out: Path,
                 kvt_pair: str = KVT_PAIR) -> list[str]:
    """`scripts.dump_kv` for one sequence file, as the (B) sittings ran it (float32 forward, stride 1). The
    instrument refuses a checkpoint that is not KV-matched to the pair's source (head dim, layers), so the pair
    must be the one whose source is the trained model."""
    return [str(kvt_python), "-m", "scripts.dump_kv", "--pair", kvt_pair, "--which", "source",
            "--local-path", str(ckpt), "--tokens", str(tokens_npy), "--stride", "1", "--dtype", "float32",
            "--out", str(out)]


def distance_command(python: Path, lag_ladder: Path, writer_ckpt: Path, reader_ckpt: Path, out: Path) -> list[str]:
    return [str(python), str(lag_ladder / "tools" / "weight_distance.py"), "--writer", str(writer_ckpt),
            "--reader", str(reader_ckpt), "--out", str(out)]


def labels_by_step(config_toml: Path) -> dict[int, list[str]]:
    """{step: [rung labels]} from the own-run config: every rung whose reader is `local:step_<g>` names a distance
    file results/distance/rungs/<label>/distance.json for step g. Read with tomllib; nothing from lag_ladder."""
    c = tomllib.loads(config_toml.read_text(encoding="utf-8"))["reader"]
    out: dict[int, list[str]] = {}
    for r in c["rungs"]:
        m = re.fullmatch(r"local:step_(\d+)", str(r["reader"]))
        if m:
            out.setdefault(int(m.group(1)), []).append(str(r["label"]))
    return out


def sha256_dir_safetensors(d: Path) -> str:
    """One sha256 over the checkpoint's safetensors files, in name order (the provenance `out_sha256`)."""
    h = hashlib.sha256()
    for f in sorted(d.glob("*.safetensors")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def write_per_seq_tokens(tokens_npy: Path, out_dir: Path, seqs: range) -> list[Path]:
    """`seq<i>.npy` (1, T) per held-out row, as the pod prepare scripts made them for dump_kv."""
    arr = np.load(tokens_npy)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in seqs:
        p = out_dir / f"seq{i}.npy"
        if not p.exists():
            np.save(p, arr[i : i + 1])
        paths.append(p)
    return paths


# --- the step hook ------------------------------------------------------------------------------------------------

class StepArtifacts:
    """What happens to one saved checkpoint: dump both token sets, measure its distance to step 0, keep or delete."""

    def __init__(self, *, lag_ladder: Path, kvt_root: Path, kvt_python: Path, tool_python: Path,
                 heldout_tokens: Path, heldout_seqs: range, heldout_pair: str,
                 rollout_tokens: Path, rollout_seqs: range, rollout_pair: str,
                 labels: dict[int, list[str]], keep: frozenset[int], log: Path, kvt_pair: str = KVT_PAIR):
        self.ll, self.kvt, self.kvt_py, self.tool_py = lag_ladder, kvt_root, kvt_python, tool_python
        self.kvt_pair = kvt_pair
        self.heldout = (heldout_tokens, heldout_seqs, heldout_pair)
        self.rollout = (rollout_tokens, rollout_seqs, rollout_pair)
        self.labels, self.keep, self.log = labels, keep, log
        self.ckpt_root = lag_ladder / "results" / "distance" / "checkpoints"
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def _log(self, msg: str) -> None:
        line = f"{msg}\n"
        sys.stdout.write(line); sys.stdout.flush()
        with self.log.open("a", encoding="utf-8") as f:
            f.write(line)

    def _run(self, cmd: list[str], cwd: Path | None = None) -> None:
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:4])}... rc={r.returncode}\n{r.stderr.decode(errors='replace')[-3000:]}")

    def process(self, step: int, ckpt: Path) -> None:
        name = f"step_{step}"
        kept = self.ckpt_root / name
        if kept.exists():
            shutil.rmtree(kept)
        shutil.move(str(ckpt), str(kept))
        prov = {"tool": "lag-ladder tools/own_run.py", "step": step, "out_sha256": sha256_dir_safetensors(kept),
                "files": sorted(p.name for p in kept.iterdir())}
        (kept / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
        for tokens, seqs, pair in (self.heldout, self.rollout):
            per_seq = write_per_seq_tokens(tokens, self.ll / "results" / "distance" / "pairs" / pair / "tokens", seqs)
            for i, p in zip(seqs, per_seq):
                out = self.ll / "results" / "distance" / "pairs" / pair / "dumps" / f"source@local-{name}" / f"seq{i}"
                if (out / "meta.json").exists():
                    continue
                self._run(dump_command(self.kvt_py, self.kvt, kept, p, out, self.kvt_pair), cwd=self.kvt)
        if step > 0:
            # every step's distance to step 0 (the trajectory on the axis), then a copy under each rung label
            # the config gives this step, where summarize_reader reads it (0015's layout)
            every = self.ll / "results" / "distance" / "ownrun" / f"step_{step}.json"
            every.parent.mkdir(parents=True, exist_ok=True)
            self._run(distance_command(self.tool_py, self.ll, self.ckpt_root / "step_0", kept, every))
            for label in self.labels.get(step, []):
                out = self.ll / "results" / "distance" / "rungs" / label / "distance.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(every, out)
            d = json.loads(every.read_text(encoding="utf-8"))
            self._log(f"STEP {step} distance rel_delta_all {d['rel_delta_all']:.4e} sha {prov['out_sha256'][:12]}"
                      f" labels {self.labels.get(step, [])}")
        else:
            self._log(f"STEP 0 sha {prov['out_sha256'][:12]}")
        if step not in self.keep:
            for f in kept.glob("*.safetensors"):
                f.unlink()
            self._log(f"STEP {step} weights deleted (not in keep set); dumps and provenance kept")


# --- commands ---------------------------------------------------------------------------------------------------------

SYSTEM_LINE = "Solve the problem. Reason step by step, then give the final answer on its own line as '#### <number>'."
PROMPT_FORMATS = ("chat", "plain")   # §14.5: `chat` renders the model's template (0017–0029, Qwen); `plain` is for
                                     # template-less bases (Llama-3.2-3B, SmolLM3-3B-Base): the same system line as text


def _messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_LINE}, {"role": "user", "content": question}]


def plain_prompt(question: str) -> str:
    """The registered plain rendering (design §14.5, R-F2): system line, blank line, `Question:` / `Answer:`."""
    return f"{SYSTEM_LINE}\n\nQuestion: {question}\nAnswer:"


def render_prompt(tokenizer, question: str, fmt: str, model: str = "") -> tuple[str, bool]:
    """The prompt text the step-0 policy samples from, and whether the tokenizer adds its special tokens to it.
    `chat`: the tokenizer's own template with the generation prompt, special tokens off (the template carries them);
    a tokenizer with no template fails loudly here, naming the model, instead of inside transformers. `plain`: the
    registered string, special tokens ON — TRL 1.15.0 tokenizes a string prompt with `processing_class(text=prompts)`
    and the default add_special_tokens (grpo_trainer.py `_tokenize_prompts`, line 1716), so the trainer's prompt
    carries the BOS the tokenizer adds and the rollout prefix must carry the same."""
    if fmt not in PROMPT_FORMATS:
        raise ValueError(f"prompt format {fmt!r} is not one of {PROMPT_FORMATS}")
    if fmt == "plain":
        return plain_prompt(question), True
    if getattr(tokenizer, "chat_template", None) is None:
        raise SystemExit(f"{model or 'the tokenizer'} has no chat template; a template-less base runs with "
                         "--prompt-format plain (design §14.5)")
    return tokenizer.apply_chat_template(_messages(question), tokenize=False, add_generation_prompt=True), False


def train_prompt(question: str, fmt: str):
    """The `prompt` field the trainer's dataset carries: role dicts under `chat` (TRL applies the template), the plain
    string under `plain` (TRL tokenizes it as text)."""
    if fmt not in PROMPT_FORMATS:
        raise ValueError(f"prompt format {fmt!r} is not one of {PROMPT_FORMATS}")
    return _messages(question) if fmt == "chat" else plain_prompt(question)


def count_stats(counts: list[int]) -> dict:
    """n / min / median / max of prompt token counts, for the record."""
    if not counts:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(counts), "min": int(min(counts)), "median": int(np.median(counts)), "max": int(max(counts))}


def cmd_sample_rollouts(a) -> int:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, _, rev = a.model.partition("@")
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev or None)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev or None, torch_dtype=torch.float32).to(a.device).eval()
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = select_prompt_indices(len(ds), a.n, a.seed)
    torch.manual_seed(a.seed)
    spec = decoding_spec(tok, do_sample=True, temperature=a.temperature, max_new_tokens=a.max_new_tokens)
    gc = model.generation_config
    print("DECODING " + json.dumps({"spec": spec, "checkpoint_generation_config": {
        k: getattr(gc, k, None) for k in ("repetition_penalty", "eos_token_id", "temperature", "top_p", "top_k")}}),
          flush=True)
    rows, meta = [], []
    for i in idx:
        prompt, special = render_prompt(tok, ds[i]["question"], a.prompt_format, a.model)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=special).to(a.device)
        pl = int(enc["input_ids"].shape[1])
        for attempt in range(5):
            out = model.generate(**enc, **spec)
            new = out[0, pl:].tolist()
            if tok.eos_token_id in new:
                new = new[: new.index(tok.eos_token_id) + 1]
            if len(new) >= MIN_ROLLOUT_TOKENS:
                break
        else:
            raise SystemExit(f"prompt {i}: five samples under {MIN_ROLLOUT_TOKENS} tokens; choose another seed")
        rows.append((out[0, :pl].tolist() + new, pl))
        meta.append({"gsm8k_train_index": i, "prefix_len": pl, "length": pl + len(new), "attempts": attempt + 1,
                     "gold": gsm8k_gold(ds[i]["answer"]), "sampled_final_number": final_number(tok.decode(new))})
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    arr, spans = pad_rows(rows, a.row_len, pad_id)
    tokens_out, spans_out = Path(a.tokens_out), Path(a.spans_out)
    tokens_out.parent.mkdir(parents=True, exist_ok=True); spans_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(tokens_out, arr)
    spans.update({"model": a.model, "seed": a.seed, "temperature": a.temperature, "pad_id": int(pad_id),
                  "decoding": spec, "system_line": SYSTEM_LINE, "prompt_format": a.prompt_format,
                  "prompt_token_counts": count_stats([m["prefix_len"] for m in meta]), "prompts": meta})
    spans_out.write_text(json.dumps(spans, indent=1), encoding="utf-8")
    Path(a.excluded_out).write_text(json.dumps(idx), encoding="utf-8")
    for f in (tokens_out, spans_out):
        print(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}")
    return 0


def cmd_train(a) -> int:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    ll = Path(a.lag_ladder).resolve()
    model_id, _, rev = a.model.partition("@")
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev or None)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev or None, torch_dtype=torch.float32)
    excluded = set(json.loads(Path(a.excluded_indices).read_text(encoding="utf-8")))
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.select([i for i in range(len(ds)) if i not in excluded]).shuffle(seed=a.seed)
    render_prompt(tok, "", a.prompt_format, a.model)   # the loud failure for a template-less tokenizer under chat
    ds = ds.map(lambda ex: {"prompt": train_prompt(ex["question"], a.prompt_format),
                            "gold": gsm8k_gold(ex["answer"])}, remove_columns=["question", "answer"])
    # the prompt token counts as TRL will tokenize them (template under chat, text with special tokens under plain)
    if a.prompt_format == "chat":
        prompt_tokens = [len(tok.apply_chat_template(p, tokenize=True, add_generation_prompt=True)) for p in ds["prompt"]]
    else:
        # a datasets Column is not a list[str] to the tokenizer (the 2026-10-24 smoke failed here); materialise it
        prompt_tokens = [len(ids) for ids in tok(list(ds["prompt"]))["input_ids"]]
    prompt_token_counts = count_stats(prompt_tokens)
    print(f"PROMPT_FORMAT {a.prompt_format} prompt_token_counts {json.dumps(prompt_token_counts)}", flush=True)

    def _texts(completions):
        return [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]

    def exact(completions, gold, **kw):
        return [reward_exact(t, g) for t, g in zip(_texts(completions), gold)]

    def fmt(completions, **kw):
        return [reward_format(t) for t in _texts(completions)]

    out_dir = Path(a.out).resolve()
    labels = labels_by_step(Path(a.config))
    if not labels:
        raise SystemExit(f"{a.config} names no local:step_<g> readers; nothing to label")
    # the instrument's interpreter is a venv symlink: resolving it would follow the link to the base interpreter
    # and lose the venv (its packages), so the path is made absolute without resolving symlinks
    hook = StepArtifacts(lag_ladder=ll, kvt_root=Path(a.kvt).resolve(), kvt_python=Path(a.kvt_python).expanduser().absolute(),
                         tool_python=Path(sys.executable), heldout_tokens=Path(a.heldout_tokens).resolve(),
                         heldout_seqs=range(a.heldout[0], a.heldout[1]), heldout_pair=a.heldout_pair,
                         rollout_tokens=Path(a.rollout_tokens).resolve(), rollout_seqs=range(0, a.n_rollouts),
                         rollout_pair=a.rollout_pair, labels=labels, keep=keep_set(a.keep), log=out_dir / "steps.log",
                         kvt_pair=a.kvt_pair)
    out_dir.mkdir(parents=True, exist_ok=True)
    step0 = out_dir / "checkpoint-0"
    model.save_pretrained(step0, safe_serialization=True)
    tok.save_pretrained(step0)
    hook.process(0, step0)

    class DumpOnSave(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt.exists():
                hook.process(int(state.global_step), ckpt)
            return control

    cfg = GRPOConfig(output_dir=str(out_dir), learning_rate=a.lr, beta=a.beta, num_generations=a.num_generations,
                     per_device_train_batch_size=a.num_generations,
                     gradient_accumulation_steps=a.prompts_per_step, max_steps=a.steps,
                     max_completion_length=a.max_completion_length, temperature=a.temperature,
                     save_strategy="steps", save_steps=1, save_only_model=True, logging_steps=1, seed=a.seed,
                     bf16=False, fp16=False, gradient_checkpointing=True, report_to=[], use_vllm=False,
                     log_completions=False)
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[exact, fmt], args=cfg,
                          train_dataset=ds, callbacks=[DumpOnSave()])

    def peak_bytes():
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    try:
        result = trainer.train()
    except torch.cuda.OutOfMemoryError:
        # the smoke's job (§14.5 R-F6): name the peak before the sitting decides on a bigger card
        print(f"TRAIN_OOM peak_cuda_mem_bytes {peak_bytes()}", flush=True)
        raise
    runtime = float(result.metrics.get("train_runtime", float("nan")))
    stats = {"prompt_format": a.prompt_format, "prompt_token_counts": prompt_token_counts,
             "peak_cuda_mem_bytes": peak_bytes(), "train_runtime_s": runtime,
             "seconds_per_step": runtime / a.steps if a.steps else None}
    (out_dir / "trainer_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps({**vars(a), **stats, "trl_config": cfg.to_dict()}, indent=1,
                                                        default=str), encoding="utf-8")
    print(f"TRAIN_STATS {json.dumps(stats)}", flush=True)
    print("TRAIN_DONE")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample-rollouts")
    s.add_argument("--model", required=True, help="model_id@revision (the step-0 policy)")
    s.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="chat",
                   help="chat: the tokenizer's template (0017); plain: the registered text for template-less bases (§14.5)")
    s.add_argument("--n", type=int, default=10); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--temperature", type=float, default=1.0); s.add_argument("--max-new-tokens", type=int, default=256)
    s.add_argument("--row-len", type=int, default=ROW_LEN); s.add_argument("--device", default="cuda")
    s.add_argument("--tokens-out", required=True, help="the token file at the path config/ownrun.toml names (under the instrument's data/tokens/)")
    s.add_argument("--spans-out", required=True, help="the spans sidecar at the path the config names")
    s.add_argument("--excluded-out", required=True, help="json list of the GSM8K train indices the run must not train on")
    s.set_defaults(fn=cmd_sample_rollouts)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True); t.add_argument("--steps", type=int, default=32)
    t.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="chat",
                   help="must equal the config's [trainer] prompt_format and the one the rollouts were sampled under")
    t.add_argument("--keep", default=",".join(str(k) for k in DEFAULT_KEEP))
    t.add_argument("--lr", type=float, default=1e-6); t.add_argument("--beta", type=float, default=0.0)
    t.add_argument("--num-generations", type=int, default=8); t.add_argument("--prompts-per-step", type=int, default=16)
    t.add_argument("--max-completion-length", type=int, default=256); t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lag-ladder", required=True); t.add_argument("--kvt", required=True); t.add_argument("--kvt-python", required=True)
    t.add_argument("--heldout-tokens", required=True); t.add_argument("--heldout", type=int, nargs=2, default=(40, 50))
    t.add_argument("--heldout-pair", default=KVT_PAIR, help="the lag-ladder pair name the held-out dumps are filed under")
    t.add_argument("--kvt-pair", default=KVT_PAIR, help="the instrument pair whose `source` is the trained model (KV-matched)")
    t.add_argument("--rollout-tokens", required=True); t.add_argument("--n-rollouts", type=int, default=10)
    t.add_argument("--rollout-pair", default="qwen2.5-1.5b-ownrun-rollout")
    t.add_argument("--excluded-indices", required=True, help="excluded_train_indices.json from sample-rollouts")
    t.add_argument("--config", required=True, help="config/ownrun.toml: its local:step_<g> rungs name the distance files")
    t.add_argument("--out", required=True); t.set_defaults(fn=cmd_train)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
