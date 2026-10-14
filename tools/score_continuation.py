"""Score a 256-token continuation under an injected FRESH vs STALE KV cache (design section 13, statistic (B)).

Run with the instrument's interpreter (torch, transformers, numpy); imports nothing from the instrument
and nothing from lag_ladder. The injection recipe is re-implemented from the facts recorded at the pin
{kv-transfer-replication, kvt/cache.py + kvt/rope.py, 5d4aa4a}: dump layerNN.npz K/V are float16
[T, n_kv, d_h], K stored rope-applied, V position-free; the cache is built by DynamicCache.update per
layer in layer order; the forward runs with attention_mask all ones, position_ids = cache_position.

    python tools/score_continuation.py --reader ID@REV --fresh DIR --stale DIR \
        --tokens FILE.npy --seq 42 --prefix-len 768 --text corpus --out DIR

Indexing (the one place this file resolves the spec rather than restating it): a cache of all 768 prefix
rows leaves no input that yields the distribution over the FIRST continuation token and no seed token for
greedy generation. So the scoring forward injects rows 0..prefix_len-2 and feeds
[x[prefix_len-1], c_0..c_{C-2}] at positions prefix_len-1 .. prefix_len+C-2: logits[t] is then exactly the
distribution that produces c_t, every one of the 256 continuation tokens is scored, and generation seeds
from the last prefix token. The semantic prefix length reported is still prefix_len.

continuation_sha256 is sha256 over the scored token ids as little-endian int64 bytes
(np.asarray(ids, dtype=np.int64).tobytes()).

Refusals are documented and by SystemExit: --probe with an --out under results/, on-path without
--generate-under (or --generate-under on corpus), --fresh together with --live, a dump whose shapes or
layer count disagree with its meta.json.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

SCHEMA = "reader-report-v1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_reader(spec: str, dtype: torch.dtype, device: torch.device):
    """--reader is `model_id@revision` or `local:<name>` under $KVT_CHECKPOINT_ROOT."""
    if spec.startswith("local:"):
        root = os.environ.get("KVT_CHECKPOINT_ROOT")
        if not root:
            raise SystemExit("--reader local:<name> needs KVT_CHECKPOINT_ROOT set")
        path, rev = Path(root) / spec[len("local:"):], None
        if not path.is_dir():
            raise SystemExit(f"no local checkpoint at {path}")
        src = str(path)
    elif "@" in spec:
        src, rev = spec.rsplit("@", 1)
    else:
        raise SystemExit(f"--reader must be model_id@revision or local:<name>, got {spec!r}")
    model = AutoModelForCausalLM.from_pretrained(src, revision=rev, torch_dtype=dtype)
    return model.to(device).eval()


# --- dumps -------------------------------------------------------------------------------------

def load_dump(root: Path, rows: int, hashes: dict):
    """Rows 0..rows-1 of every layer of a stride-1 single-sequence dump, upcast float16 -> float32.
    Returns (layers, meta): layers is a list over layer index of (K, V) torch.float32 [rows, n_kv, d_h]."""
    root = Path(root)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"no meta.json under {root}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name in ("meta.json", "meta.npz", "checkpoint_manifest.json"):
        p = root / name
        if p.is_file():
            hashes[p.as_posix()] = sha256_file(p)
    n_layers, n_kv, d_h = int(meta["n_layers"]), int(meta["n_kv"]), int(meta["d_h"])
    files = sorted(root.glob("layer*.npz"), key=lambda p: int(p.stem[len("layer"):]))
    if len(files) != n_layers:
        raise SystemExit(f"{root}: {len(files)} layer files, meta.json says n_layers={n_layers}")
    layers = []
    for p in files:
        hashes[p.as_posix()] = sha256_file(p)
        with np.load(p) as z:
            k, v = z["K"], z["V"]
        for name, a in (("K", k), ("V", v)):
            if a.ndim != 3 or a.shape[1:] != (n_kv, d_h) or a.shape[0] < rows:
                raise SystemExit(f"{p}: {name} shape {a.shape} does not fit [T>={rows}, {n_kv}, {d_h}]")
        layers.append((torch.from_numpy(k[:rows].astype(np.float32)),
                       torch.from_numpy(v[:rows].astype(np.float32))))
    return layers, meta


# --- RoPE (contract formulas; spec read from meta.json["rope"], never re-derived) ----------------

def rope_cos_sin(positions: torch.Tensor, spec: dict):
    """Unscaled cos/sin plus the spec's attention factor m. The model writes K = m * R(pos) * k_content
    (transformers multiplies cos/sin by attention_scaling), so apply must MULTIPLY by m and strip must
    DIVIDE by m — the instrument's strip_rope_spec_tokens_first divides (kvt/rope.py at 5d4aa4a)."""
    inv_freq = torch.tensor(spec["inv_freq"], dtype=torch.float32)
    m = float(spec.get("attention_scaling", 1.0))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin(), m


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, m: float) -> torch.Tensor:
    """k: [T, n_kv, d_h]; cos/sin [T, d_h] unscaled, broadcast over heads; returns m * R * k."""
    return (k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]) * m


def strip_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, m: float) -> torch.Tensor:
    """Exact inverse of apply_rope: R^T * k / m."""
    return (k * cos[:, None, :] - rotate_half(k) * sin[:, None, :]) / m


def specs_differ(a: dict, b: dict) -> bool:
    return (list(map(float, a["inv_freq"])) != list(map(float, b["inv_freq"]))
            or float(a.get("attention_scaling", 1.0)) != float(b.get("attention_scaling", 1.0)))


def rerotate(layers, writer_spec: dict, reader_spec: dict, rows: int):
    """K on disk carries the WRITER's rotation; strip it with the writer's spec and apply the reader's.
    Only called when the specs differ (same-spec dumps are injected as is)."""
    pos = torch.arange(rows)
    cw, sw, mw = rope_cos_sin(pos, writer_spec)
    cr, sr, mr = rope_cos_sin(pos, reader_spec)
    return [(apply_rope(strip_rope(k, cw, sw, mw), cr, sr, mr), v) for k, v in layers]


# --- cache + forwards ----------------------------------------------------------------------------

def build_cache(layers, device: torch.device) -> DynamicCache:
    """layers: (K, V) float32 [rows, n_kv, d_h] -> DynamicCache of [1, n_kv, rows, d_h], layer order."""
    cache = DynamicCache()
    for l, (k, v) in enumerate(layers):
        cache.update(k.permute(1, 0, 2)[None].contiguous().to(device),
                     v.permute(1, 0, 2)[None].contiguous().to(device), l)
    return cache


@torch.no_grad()
def forward_logits(model, layers, input_ids: torch.Tensor, past_len: int, device: torch.device):
    """One forward of input_ids [1, C] at positions past_len..past_len+C-1 over a fresh cache
    built from `layers`. Logits float32 [C, vocab] on cpu."""
    cache = build_cache(layers, device)
    c = input_ids.shape[1]
    attn = torch.ones(1, past_len + c, dtype=torch.long, device=device)
    pos = torch.arange(past_len, past_len + c, device=device)
    out = model(input_ids=input_ids.to(device), past_key_values=cache, attention_mask=attn,
                position_ids=pos[None], cache_position=pos, use_cache=True)
    return out.logits[0].float().cpu()


@torch.no_grad()
def live_prefix_layers(model, prefix_ids: torch.Tensor, rows: int, device: torch.device):
    """The reader's own float32 cache of P: one forward over positions 0..prefix_len-1, keep rows 0..rows-1.
    Returned in the dump layout [rows, n_kv, d_h] so it flows through build_cache like a loaded dump."""
    ids = prefix_ids[None].to(device)
    attn = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=attn, use_cache=True)
    cache = out.past_key_values
    layers = []
    for l in range(model.config.num_hidden_layers):
        if hasattr(cache, "layers"):
            k, v = cache.layers[l].keys, cache.layers[l].values
        else:
            k, v = cache.key_cache[l], cache.value_cache[l]
        layers.append((k[0, :, :rows].permute(1, 0, 2).float().cpu(),
                       v[0, :, :rows].permute(1, 0, 2).float().cpu()))
    return layers


@torch.no_grad()
def greedy_generate(model, layers, seed_token: int, past_len: int, steps: int, device: torch.device):
    """`steps` argmax tokens against the injected cache, one token per forward, cache growing."""
    cache = build_cache(layers, device)
    tokens, tok = [], int(seed_token)
    t0 = time.monotonic()
    for i in range(steps):
        pos = torch.arange(past_len + i, past_len + i + 1, device=device)
        attn = torch.ones(1, past_len + i + 1, dtype=torch.long, device=device)
        out = model(input_ids=torch.tensor([[tok]], device=device), past_key_values=cache,
                    attention_mask=attn, position_ids=pos[None], cache_position=pos, use_cache=True)
        cache = out.past_key_values
        tok = int(out.logits[0, -1].float().argmax())
        tokens.append(tok)
    return tokens, time.monotonic() - t0


# --- main ----------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reader", required=True, help="model_id@revision | local:<name>")
    ap.add_argument("--fresh", default=None, help="reader's own dump dir of P (omit with --live)")
    ap.add_argument("--stale", required=True, help="writer's dump dir of P")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--prefix-len", type=int, default=768)
    ap.add_argument("--length", type=int, default=None,
                    help="end of the stored continuation (exclusive); default: the row's length. Rows padded past a "
                         "sampled completion (design §14 rollout text) are cut here")
    ap.add_argument("--text", choices=("corpus", "on-path", "rollout"), required=True,
                    help="corpus and rollout teacher-force the stored tokens after the prefix; on-path generates them")
    ap.add_argument("--generate-under", choices=("stale", "fresh"), default=None)
    ap.add_argument("--live", action="store_true",
                    help="floor arm: fresh = the reader's own live float32 cache of P")
    ap.add_argument("--probe", action="store_true", help="refuse any --out under results/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--dtype", default="float32", choices=("float32",))
    a = ap.parse_args(argv)

    out = Path(a.out)
    if a.probe and "results" in out.resolve().parts:
        raise SystemExit(f"probe mode refuses to write under results/: {out}")
    if a.text == "on-path" and a.generate_under is None:
        raise SystemExit("--text on-path requires --generate-under stale|fresh")
    if a.text in ("corpus", "rollout") and a.generate_under is not None:
        raise SystemExit(f"--generate-under is meaningless on {a.text} text")
    if a.live and a.fresh:
        raise SystemExit("--live replaces the fresh arm; do not also pass --fresh")
    if not a.live and not a.fresh:
        raise SystemExit("--fresh is required unless --live")

    device = pick_device(a.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    hashes: dict[str, str] = {}
    tokens_path = Path(a.tokens)
    hashes[tokens_path.as_posix()] = sha256_file(tokens_path)
    all_tokens = np.load(tokens_path)
    if all_tokens.ndim != 2 or not (0 <= a.seq < all_tokens.shape[0]):
        raise SystemExit(f"{tokens_path}: shape {all_tokens.shape} has no sequence {a.seq}")
    seq = all_tokens[a.seq].astype(np.int64)
    end = seq.shape[0] if a.length is None else a.length
    if not (0 < end <= seq.shape[0]):
        raise SystemExit(f"--length {a.length} is outside the row (T={seq.shape[0]})")
    if a.prefix_len + 1 >= end:
        raise SystemExit(f"prefix_len {a.prefix_len} leaves no continuation before {end}")
    rows = a.prefix_len - 1                      # injected cache rows 0..prefix_len-2 (see docstring)
    n_cont = end - a.prefix_len
    corpus = seq[a.prefix_len:end]

    model = load_reader(a.reader, torch.float32, device)

    stale_layers, stale_meta = load_dump(Path(a.stale), rows, hashes)
    if a.live:
        fresh_layers = live_prefix_layers(model, torch.from_numpy(seq[: a.prefix_len]), rows, device)
        fresh_meta = None
    else:
        fresh_layers, fresh_meta = load_dump(Path(a.fresh), rows, hashes)
        stale_spec, fresh_spec = stale_meta.get("rope"), fresh_meta.get("rope")
        if stale_spec and fresh_spec and specs_differ(stale_spec, fresh_spec):
            stale_layers = rerotate(stale_layers, stale_spec, fresh_spec, rows)

    gen_under, gen_seconds = None, None
    if a.text == "on-path":
        gen_under = a.generate_under
        gen_layers = stale_layers if gen_under == "stale" else fresh_layers
        cont, gen_seconds = greedy_generate(model, gen_layers, int(seq[rows]), rows, n_cont, device)
        cont = np.asarray(cont, dtype=np.int64)
    else:
        cont = corpus

    # Teacher-force both arms: input = [x[prefix_len-1], c_0..c_{C-2}], logits[t] produces c_t.
    inp = torch.from_numpy(np.concatenate([seq[rows: rows + 1], cont[:-1]]))[None]
    logits_f = forward_logits(model, fresh_layers, inp, rows, device)
    logits_s = forward_logits(model, stale_layers, inp, rows, device)

    lf = torch.log_softmax(logits_f, dim=-1)
    ls = torch.log_softmax(logits_s, dim=-1)
    kl = (lf.exp() * (lf - ls)).sum(dim=-1)      # KL(pi_fresh || pi_stale), full vocabulary, nats
    tgt = torch.from_numpy(cont)
    idx = torch.arange(n_cont)
    reader_meta = fresh_meta if fresh_meta is not None else stale_meta
    digest = (reader_meta.get("checkpoint") or {}).get("manifest_digest")

    report = {
        "schema": SCHEMA,
        "reader": a.reader,
        "fresh_dump": Path(a.fresh).as_posix() if a.fresh else None,
        "stale_dump": Path(a.stale).as_posix(),
        "tokens_file": tokens_path.as_posix(),
        "seq": a.seq,
        "prefix_len": a.prefix_len,
        "text": a.text,
        "generate_under": gen_under,
        "live": bool(a.live),
        "continuation_tokens": [int(t) for t in cont],
        "continuation_sha256": hashlib.sha256(np.asarray(cont, dtype=np.int64).tobytes()).hexdigest(),
        "logprob_fresh": lf[idx, tgt].tolist(),
        "logprob_stale": ls[idx, tgt].tolist(),
        "kl_per_token": kl.tolist(),
        "argmax_fresh": logits_f.argmax(dim=-1).tolist(),
        "argmax_stale": logits_s.argmax(dim=-1).tolist(),
        "max_abs_logit_diff": float((logits_f - logits_s).abs().max()),
        "inputs_sha256": hashes,
        "reader_checkpoint": {"manifest_digest": digest} if digest is not None else None,
        "device": device.type,
        "torch_version": torch.__version__,
        "peak_cuda_mem_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "gen_seconds": gen_seconds,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {(out / 'report.json').as_posix()}  text={a.text} live={a.live} "
          f"median_kl={float(kl.median()):.3e} max_logit_diff={report['max_abs_logit_diff']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
