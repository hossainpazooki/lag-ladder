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


def dump_command(kvt_python: Path, kvt_root: Path, ckpt: Path, tokens_npy: Path, out: Path) -> list[str]:
    """`scripts.dump_kv` for one sequence file, as the (B) sittings ran it (float32 forward, stride 1)."""
    return [str(kvt_python), "-m", "scripts.dump_kv", "--pair", KVT_PAIR, "--which", "source",
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
                 labels: dict[int, list[str]], keep: frozenset[int], log: Path):
        self.ll, self.kvt, self.kvt_py, self.tool_py = lag_ladder, kvt_root, kvt_python, tool_python
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
                self._run(dump_command(self.kvt_py, self.kvt, kept, p, out), cwd=self.kvt)
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


def _chat(tokenizer, question: str):
    msgs = [{"role": "system", "content": SYSTEM_LINE}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


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
    rows, meta = [], []
    for i in idx:
        prompt = _chat(tok, ds[i]["question"])
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(a.device)
        pl = int(enc["input_ids"].shape[1])
        for attempt in range(5):
            out = model.generate(**enc, do_sample=True, temperature=a.temperature, top_p=1.0, top_k=0,
                                 max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
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
                  "system_line": SYSTEM_LINE, "prompts": meta})
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
    ds = ds.map(lambda ex: {"prompt": [{"role": "system", "content": SYSTEM_LINE},
                                       {"role": "user", "content": ex["question"]}],
                            "gold": gsm8k_gold(ex["answer"])}, remove_columns=["question", "answer"])

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
    hook = StepArtifacts(lag_ladder=ll, kvt_root=Path(a.kvt).resolve(), kvt_python=Path(a.kvt_python).resolve(),
                         tool_python=Path(sys.executable), heldout_tokens=Path(a.heldout_tokens).resolve(),
                         heldout_seqs=range(a.heldout[0], a.heldout[1]), heldout_pair=a.heldout_pair,
                         rollout_tokens=Path(a.rollout_tokens).resolve(), rollout_seqs=range(0, a.n_rollouts),
                         rollout_pair=a.rollout_pair, labels=labels, keep=keep_set(a.keep), log=out_dir / "steps.log")
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
    trainer.train()
    (out_dir / "trainer_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps({**vars(a), "trl_config": cfg.to_dict()}, indent=1,
                                                        default=str), encoding="utf-8")
    print("TRAIN_DONE")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample-rollouts")
    s.add_argument("--model", required=True, help="model_id@revision (the step-0 policy)")
    s.add_argument("--n", type=int, default=10); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--temperature", type=float, default=1.0); s.add_argument("--max-new-tokens", type=int, default=256)
    s.add_argument("--row-len", type=int, default=ROW_LEN); s.add_argument("--device", default="cuda")
    s.add_argument("--tokens-out", required=True, help="the token file at the path config/ownrun.toml names (under the instrument's data/tokens/)")
    s.add_argument("--spans-out", required=True, help="the spans sidecar at the path the config names")
    s.add_argument("--excluded-out", required=True, help="json list of the GSM8K train indices the run must not train on")
    s.set_defaults(fn=cmd_sample_rollouts)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True); t.add_argument("--steps", type=int, default=32)
    t.add_argument("--keep", default=",".join(str(k) for k in DEFAULT_KEEP))
    t.add_argument("--lr", type=float, default=1e-6); t.add_argument("--beta", type=float, default=0.0)
    t.add_argument("--num-generations", type=int, default=8); t.add_argument("--prompts-per-step", type=int, default=16)
    t.add_argument("--max-completion-length", type=int, default=256); t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lag-ladder", required=True); t.add_argument("--kvt", required=True); t.add_argument("--kvt-python", required=True)
    t.add_argument("--heldout-tokens", required=True); t.add_argument("--heldout", type=int, nargs=2, default=(40, 50))
    t.add_argument("--heldout-pair", default=KVT_PAIR)
    t.add_argument("--rollout-tokens", required=True); t.add_argument("--n-rollouts", type=int, default=10)
    t.add_argument("--rollout-pair", default="qwen2.5-1.5b-ownrun-rollout")
    t.add_argument("--excluded-indices", required=True, help="excluded_train_indices.json from sample-rollouts")
    t.add_argument("--config", required=True, help="config/ownrun.toml: its local:step_<g> rungs name the distance files")
    t.add_argument("--out", required=True); t.set_defaults(fn=cmd_train)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
