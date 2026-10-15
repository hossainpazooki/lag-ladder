"""Statistic (C), "does the task notice" (design §15): the reader at step g answers GSM8K test prompts greedily under an
injected prompt cache — its own (FRESH), the step-0 writer's (STALE), another prompt's (NEAR) or a web prefix of the
prompt's length (FAR) — and the final answers are compared against gold. Caches are built live by one float32 forward
of the arm's weights over the prompt, rounded through float16 (the dump path's precision) and injected exactly as
`score_continuation.py` injects a dump (its functions are imported from this directory; nothing from `kvt` or
`lag_ladder`). Run with the instrument's interpreter.

    python tools/task_swap.py equivalence --writer local:step_0 --dump-root DIR --tokens FILE.npy --spans FILE.json
                                          [--reader local:step_32 --expect-b1 8.914e-2] --out DIR
    python tools/task_swap.py cells --reader local:step_g --writer local:step_0 --prompts prompts.json
                                    --indices idx.json --far-tokens FILE.npy --far-seq 40 --out DIR [--max-new 256]

`equivalence` (once per sitting, before any cell): for every rollout sequence of 0017 the writer's live cache of the
prompt is compared layer by layer with the step-0 dump on disk (float16; equal to the last bit expected), and, with
--reader, 0018's rollout (B1) at that reader is reproduced by teacher-forcing the stored completions under the live
caches. `cells` writes one JSONL per arm under --out (fresh.jsonl, stale.jsonl, near.jsonl, far.jsonl) with the record
keys RECORD_KEYS; the STALE records also carry (B1) on the fresh path: the FRESH arm's completion teacher-forced under
both caches. For the first --control-n prompts it also writes fresh32.jsonl (the floor: FRESH from a cache NOT rounded
through float16) and fresh2.jsonl (the identity: FRESH generated a second time). Every pure helper is at module level and tested offline; torch is imported inside the commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from own_run import final_number, gsm8k_gold  # noqa: E402  (pure helpers; own_run imports nothing heavy at import)

SCHEMA = "task-record-v1"
ARMS = ("fresh", "stale", "near", "far")
CONTROL_ARMS = ("fresh32", "fresh2")   # the floor (FRESH built in float32, not rounded) and the identity (FRESH again)
RECORD_KEYS = frozenset({
    "schema", "index", "prompt_len", "n_tokens", "reached_cap", "answer", "gold", "correct",
    "completion_sha256", "first_divergence", "b1_fresh_path",
})


# --- pure helpers ------------------------------------------------------------------------------------------------

def near_pairing(lengths: list[int]) -> dict[int, int]:
    """NEAR's source per prompt: the next prompt in ascending token-length order (ties by index), so the source is at
    least as long and its prompt is cut to the target's length by the caller; the longest wraps to the shortest and is
    padded by the caller (the one pairing that is not a true cut)."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    return {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}


def through_float16(x: np.ndarray) -> np.ndarray:
    """The dump path's precision: float32 -> float16 -> float32."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def compare_layers(live: list, dump: list) -> dict:
    """Largest absolute gap between a live cache rounded through float16 and a dump, over every layer and side.
    live/dump: lists over layers of (K, V) arrays [rows, n_kv, d_h]; the dump may be float16."""
    if len(live) != len(dump):
        raise ValueError(f"layers differ: live {len(live)}, dump {len(dump)}")
    worst, max_diff, rows = None, 0.0, None
    for l, ((lk, lv), (dk, dv)) in enumerate(zip(live, dump)):
        for side, a, b in (("K", lk, dk), ("V", lv, dv)):
            a = through_float16(a)
            b = np.asarray(b, dtype=np.float32)
            if a.shape != b.shape:
                raise ValueError(f"layer {l} {side}: shape {a.shape} vs dump {b.shape}")
            rows = a.shape[0]
            d = float(np.abs(a - b).max()) if a.size else 0.0
            if d > max_diff:
                max_diff, worst = d, {"layer": l, "side": side}
    return {"layers": len(live), "rows": rows, "max_abs_diff": max_diff, "worst": worst}


def is_correct(answer: str | None, gold: str) -> bool:
    if answer is None:
        return False
    try:
        return abs(float(answer) - float(gold)) < 1e-6
    except ValueError:
        return answer == gold


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def tokens_sha256(tokens) -> str:
    return hashlib.sha256(np.asarray(list(tokens), dtype=np.int64).tobytes()).hexdigest()


def cell_record(*, index: int, prompt_len: int, completion: list[int], max_new: int, eos_id: int | None, text: str,
                gold: str, fresh_completion: list[int] | None, b1_fresh_path: float | None = None) -> dict:
    """One arm's record for one prompt. reached_cap: the generation hit max_new without an EOS."""
    answer = final_number(text)
    reached_cap = len(completion) >= max_new and (eos_id is None or completion[-1] != eos_id)
    return {"schema": SCHEMA, "index": int(index), "prompt_len": int(prompt_len), "n_tokens": len(completion),
            "reached_cap": bool(reached_cap), "answer": answer, "gold": gold, "correct": is_correct(answer, gold),
            "completion_sha256": tokens_sha256(completion),
            "first_divergence": None if fresh_completion is None else first_divergence(completion, fresh_completion),
            "b1_fresh_path": b1_fresh_path}


# --- GPU commands (torch imported here only) ----------------------------------------------------------------------

def _sc():
    import score_continuation as sc  # the (B) scorer's injection, loaded from this directory
    return sc


def _layers_through_float16(layers):
    return [(through_float16(k.numpy()) if hasattr(k, "numpy") else through_float16(k),
             through_float16(v.numpy()) if hasattr(v, "numpy") else through_float16(v)) for k, v in layers]


def _to_torch(layers):
    import torch
    return [(torch.from_numpy(np.ascontiguousarray(k)), torch.from_numpy(np.ascontiguousarray(v))) for k, v in layers]


def _greedy(model, layers, seed_token: int, past_len: int, max_new: int, eos_id: int | None, device):
    """Greedy tokens against the injected cache, stopping after EOS; same forward recipe as score_continuation."""
    import torch
    sc = _sc()
    cache = sc.build_cache(layers, device)
    tokens, tok = [], int(seed_token)
    for i in range(max_new):
        pos = torch.arange(past_len + i, past_len + i + 1, device=device)
        attn = torch.ones(1, past_len + i + 1, dtype=torch.long, device=device)
        out = model(input_ids=torch.tensor([[tok]], device=device), past_key_values=cache, attention_mask=attn,
                    position_ids=pos[None], cache_position=pos, use_cache=True)
        cache = out.past_key_values
        tok = int(out.logits[0, -1].float().argmax())
        tokens.append(tok)
        if eos_id is not None and tok == eos_id:
            break
    return tokens


def _mean_kl_fresh_path(model, fresh_layers, stale_layers, seed_token: int, past_len: int, cont: list[int], device) -> float | None:
    """(B1) on the fresh path: the FRESH completion teacher-forced under both caches; mean per-token KL(fresh||stale)."""
    import torch
    sc = _sc()
    if len(cont) < 1:
        return None
    inp = torch.tensor([[int(seed_token), *cont[:-1]]])
    lf = torch.log_softmax(sc.forward_logits(model, fresh_layers, inp, past_len, device), dim=-1)
    ls = torch.log_softmax(sc.forward_logits(model, stale_layers, inp, past_len, device), dim=-1)
    return float((lf.exp() * (lf - ls)).sum(dim=-1).mean())


def _tokenizer_for(spec: str):
    import os
    from transformers import AutoTokenizer
    if spec.startswith("local:"):
        return AutoTokenizer.from_pretrained(str(Path(os.environ["KVT_CHECKPOINT_ROOT"]) / spec[len("local:"):]))
    src, _, rev = spec.partition("@")
    return AutoTokenizer.from_pretrained(src, revision=rev or None)


def cmd_equivalence(a) -> int:
    import torch
    sc = _sc()
    device = sc.pick_device(a.device)
    writer = sc.load_reader(a.writer, torch.float32, device)
    tokens = np.load(a.tokens)
    spans = json.loads(Path(a.spans).read_text(encoding="utf-8"))
    out = {"schema": "task-equivalence-v1", "writer": a.writer, "dump_root": a.dump_root, "per_seq": [], "b1": None}
    worst = 0.0
    for i in range(tokens.shape[0]):
        pl = int(spans["prefix_len"][i])
        rows = pl - 1
        live = sc.live_prefix_layers(writer, torch.from_numpy(tokens[i, :pl].astype(np.int64)), rows, device)
        hashes: dict = {}
        dump, _meta = sc.load_dump(Path(a.dump_root) / f"seq{i}", rows, hashes)
        r = compare_layers([(k.numpy(), v.numpy()) for k, v in live], [(k.numpy(), v.numpy()) for k, v in dump])
        r["seq"] = i
        out["per_seq"].append(r)
        worst = max(worst, r["max_abs_diff"])
        print(f"EQUIV seq{i} rows {rows} max_abs_diff {r['max_abs_diff']:.3e}", flush=True)
    out["max_abs_diff"] = worst
    if a.reader:
        reader = sc.load_reader(a.reader, torch.float32, device)
        kls = []
        for i in range(tokens.shape[0]):
            pl, end = int(spans["prefix_len"][i]), int(spans["length"][i])
            rows = pl - 1
            prefix = torch.from_numpy(tokens[i, :pl].astype(np.int64))
            fresh = _to_torch(_layers_through_float16(sc.live_prefix_layers(reader, prefix, rows, device)))
            stale = _to_torch(_layers_through_float16(sc.live_prefix_layers(writer, prefix, rows, device)))
            cont = [int(t) for t in tokens[i, pl:end]]
            kls.append(_mean_kl_fresh_path(reader, fresh, stale, int(tokens[i, rows]), rows, cont, device))
        med = float(np.median(kls))
        out["b1"] = {"reader": a.reader, "per_seq_mean_kl": kls, "median": med, "expected": a.expect_b1,
                     "rel_gap": None if a.expect_b1 is None else abs(med - a.expect_b1) / a.expect_b1}
        print(f"EQUIV_B1 median {med:.4e} expected {a.expect_b1}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "equivalence.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"EQUIV_DONE max_abs_diff {worst:.3e}", flush=True)
    return 0


def cmd_cells(a) -> int:
    import torch
    sc = _sc()
    device = sc.pick_device(a.device)
    prompts = {int(p["index"]): p for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}
    indices = [int(i) for i in json.loads(Path(a.indices).read_text(encoding="utf-8"))]
    tok = _tokenizer_for(a.reader)
    eos = tok.eos_token_id
    reader = sc.load_reader(a.reader, torch.float32, device)
    writer = sc.load_reader(a.writer, torch.float32, device)
    far_tokens = np.load(a.far_tokens)
    from own_run import SYSTEM_LINE
    enc = {}
    for i in indices:
        msgs = [{"role": "system", "content": SYSTEM_LINE}, {"role": "user", "content": prompts[i]["question"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc[i] = tok(text, add_special_tokens=False)["input_ids"]
    lengths = [len(enc[i]) for i in indices]
    near_of = {indices[k]: indices[v] for k, v in near_pairing(lengths).items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_arms = ARMS + CONTROL_ARMS
    files = {arm: open(out / f"{arm}.jsonl", "a", encoding="utf-8") for arm in all_arms}
    done = {arm: set() for arm in all_arms}
    for arm in all_arms:
        p = out / f"{arm}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done[arm].add(json.loads(line)["index"])
    t0 = time.monotonic()
    for n, i in enumerate(indices):
        ids = enc[i]
        pl = len(ids)
        rows = pl - 1
        seed = ids[rows]
        with_controls = n < a.control_n
        if all(i in done[arm] for arm in ARMS) and (not with_controls or all(i in done[arm] for arm in CONTROL_ARMS)):
            continue
        prefix = torch.tensor(ids, dtype=torch.int64)
        fresh_live = sc.live_prefix_layers(reader, prefix, rows, device)
        fresh = _to_torch(_layers_through_float16(fresh_live))
        stale = _to_torch(_layers_through_float16(sc.live_prefix_layers(writer, prefix, rows, device)))
        near_ids = enc[near_of[i]]
        pad = eos if eos is not None else 0
        near_prefix = torch.tensor((near_ids + [pad] * pl)[:pl], dtype=torch.int64)   # cut to this prompt's length; the wrap pads
        near = _to_torch(_layers_through_float16(sc.live_prefix_layers(reader, near_prefix, rows, device)))
        far_row = far_tokens[a.far_seq + (n % (far_tokens.shape[0] - a.far_seq))]
        far_prefix = torch.from_numpy(far_row[:pl].astype(np.int64))
        far = _to_torch(_layers_through_float16(sc.live_prefix_layers(reader, far_prefix, rows, device)))
        layers = {"fresh": fresh, "stale": stale, "near": near, "far": far}
        if with_controls:
            layers["fresh32"] = [(k.float().cpu(), v.float().cpu()) for k, v in fresh_live]   # the unrounded cache
            layers["fresh2"] = fresh
        comps = {}
        for arm in layers:
            comps[arm] = _greedy(reader, layers[arm], seed, rows, a.max_new, eos, device)
        b1 = _mean_kl_fresh_path(reader, fresh, stale, seed, rows, comps["fresh"], device)
        for arm in layers:
            if i in done[arm]:
                continue
            rec = cell_record(index=i, prompt_len=pl, completion=comps[arm], max_new=a.max_new, eos_id=eos,
                              text=tok.decode(comps[arm], skip_special_tokens=True), gold=prompts[i]["gold"],
                              fresh_completion=None if arm == "fresh" else comps["fresh"],
                              b1_fresh_path=b1 if arm == "stale" else None)
            files[arm].write(json.dumps(rec) + "\n")
            files[arm].flush()
        if (n + 1) % 10 == 0:
            print(f"CELLS {n + 1}/{len(indices)} {time.monotonic() - t0:.0f}s", flush=True)
    for f in files.values():
        f.close()
    print(f"CELLS_DONE {len(indices)} prompts", flush=True)
    return 0


def cmd_prepare(a) -> int:
    """GSM8K test prompts to a JSON file [{index, question, gold}]; needs `datasets` (pod only)."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = [{"index": i, "question": ds[i]["question"], "gold": gsm8k_gold(ds[i]["answer"])} for i in range(len(ds))]
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}  {a.out}  n={len(rows)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("equivalence")
    e.add_argument("--writer", required=True); e.add_argument("--dump-root", required=True)
    e.add_argument("--tokens", required=True); e.add_argument("--spans", required=True)
    e.add_argument("--reader", default=None); e.add_argument("--expect-b1", type=float, default=None)
    e.add_argument("--out", required=True); e.add_argument("--device", default="auto")
    e.set_defaults(fn=cmd_equivalence)
    c = sub.add_parser("cells")
    c.add_argument("--reader", required=True); c.add_argument("--writer", required=True)
    c.add_argument("--prompts", required=True); c.add_argument("--indices", required=True)
    c.add_argument("--far-tokens", required=True); c.add_argument("--far-seq", type=int, default=40)
    c.add_argument("--max-new", type=int, default=256); c.add_argument("--out", required=True)
    c.add_argument("--control-n", type=int, default=0, help="prompts (from the first) that also get the floor and identity arms")
    c.add_argument("--device", default="auto")
    c.set_defaults(fn=cmd_cells)
    p = sub.add_parser("prepare"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_prepare)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
