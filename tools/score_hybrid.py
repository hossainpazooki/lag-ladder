"""Dump and inject a hybrid-state cache (MLA K/V + KDA recurrent and conv states) and score a continuation under
an arm of design §16: the (B) statistic on Kimi Linear 48B-A3B, by state kind.

Run with the pod's interpreter (torch, transformers with trust_remote_code, fla, numpy); imports nothing from the
instrument (`kvt`) and nothing from `lag_ladder`, as `score_continuation.py` does. torch, transformers and fla are
imported inside the functions that need them, so the layout, the shape check, the indexing and the report fields
import and test offline (tests/test_score_hybrid.py). The dump layout is `hybrid-dump-v1`, the one
`lag_ladder.hybrid_state` reads; it is re-implemented here, and the test suite holds the two implementations to the
same bytes.

    python tools/score_hybrid.py shapes --model DIR --tokens FILE.npy --seq 40 --length 1024 --out DIR
    python tools/score_hybrid.py dump   --model DIR --tokens FILE.npy --seq 40 --prefix-len 768 --out DIR
    python tools/score_hybrid.py score  --reader DIR --arm FULL-STALE --fresh DUMP --stale DUMP \
        --tokens FILE.npy --seq 40 --prefix-len 768 --out DIR [--live] [--probe] [--state-v-first 0|1]
    python tools/score_hybrid.py score  --reader DIR --arm ONE-PASS --tokens ... --out DIR          # the floor

Facts this file resolves from `modeling_kimi.py` (digest d79b365e37378881, read 2026-10-22) and FLA `main`:
  * `KimiDynamicCache` holds four lists, one slot per layer: `key_cache` / `value_cache` [B, H, T, k_dim|v_dim] on the
    full_attention layers; `recurrent_states` [B, H, K, V] float32 and `conv_states` = (q, k, v) each [B, D, W] on the
    linear_attention layers. The model reads its past length from the FIRST full_attention layer's key cache, so
    every arm carries an MLA cache of the cached rows.
  * `mla_use_nope` is asserted true: no rotary embedding touches Q or K, positions enter through the causal mask and
    the recurrence's order; a stale K/V is injected as is.
  * both the prefill and a 256-token continuation run the chunk kernel (fused only at q_len <= 64).

Indexing (as score_continuation.py): a cache of all P prefix rows leaves no input that yields the distribution over
the first continuation token, so the cache holds rows 0..P-2 (`cached_rows` = P-1: the forward over seq[:P-1]) and the
scoring forward feeds [seq[P-1], c_0..c_{C-2}]; logits[t] is the distribution that produces c_t. For a recurrent state
"rows 0..P-2" is not a slice, so the dump IS the forward over the first P-1 tokens; meta.extra records
`semantic_prefix_len` = P.

Activation-dtype states (K, V, conv windows) are stored as their bf16 bit patterns in uint16 (exact, half the bytes);
the recurrent state stays float32. ONE-PASS compares one forward over seq[:end-1] (logits at positions P-1..) against
the cached FRESH continuation on the same weights: the bf16 floor of R-K2. `--live` builds the fresh side from the
reader's own live cache instead of `--fresh`; FRESH with --live against the reader's own dump is the dump-equivalence
control; FULL-STALE with --fresh == --stale is the identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "hybrid-report-v1"
DUMP_SCHEMA = "hybrid-dump-v1"
MLA, KDA = "full_attention", "linear_attention"
ARMS = ("FRESH", "FULL-STALE", "MLA-STALE", "KDA-STALE", "ONE-PASS")
SOURCES = {
    "FRESH": {"mla": "fresh", "kda": "fresh"},
    "FULL-STALE": {"mla": "stale", "kda": "stale"},
    "MLA-STALE": {"mla": "stale", "kda": "fresh"},
    "KDA-STALE": {"mla": "fresh", "kda": "stale"},
    "ONE-PASS": {"mla": "none", "kda": "none"},
}
_CONV = ("conv_q", "conv_k", "conv_v")

# reader-report-v1's keys plus the hybrid axis; the summarizer reads these and nothing else
REPORT_KEYS = (
    "schema", "reader", "fresh_dump", "stale_dump", "tokens_file", "seq", "prefix_len",
    "text", "generate_under", "live",
    "continuation_tokens", "continuation_sha256",
    "logprob_fresh", "logprob_stale", "kl_per_token",
    "argmax_fresh", "argmax_stale", "max_abs_logit_diff",
    "inputs_sha256", "reader_checkpoint",
    "device", "torch_version", "peak_cuda_mem_bytes", "gen_seconds",
    "arm", "sources", "state_v_first", "dtype", "cached_rows", "seconds",
    "attn_implementation", "f32_precision",
)
DTYPES = ("bfloat16", "float32")


def dtype_name(name: str) -> str:
    """The two registered dtypes: bf16 as the Hub serves the weights (R-K2), float32 for the reference probe."""
    if name not in DTYPES:
        raise ValueError(f"dtype {name!r} is not one of {DTYPES}")
    return name


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- bf16 bits ------------------------------------------------------------------------------------------------

def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest, ties to even) and return the 16-bit patterns as uint16."""
    u = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = (u + np.uint32(0x7FFF) + lsb) >> 16
    nan = (u & 0x7F800000) == 0x7F800000
    rounded = np.where(nan & ((u & 0x007FFFFF) != 0), (u >> 16) | 0x40, rounded)
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return b.view(np.float32)


# --- shapes from config.json -----------------------------------------------------------------------------------

def expected_shapes(cfg: dict) -> dict:
    """What a KimiDynamicCache must hold per layer, from the config.json fields; refuses a layer list that
    does not partition the layers."""
    n = int(cfg["num_hidden_layers"])
    la = cfg["linear_attn_config"]
    full = {int(i) for i in la["full_attn_layers"]}
    kda = {int(i) for i in la["kda_layers"]}
    if full & kda or (full | kda) != set(range(1, n + 1)):
        raise ValueError(f"full_attn_layers {sorted(full)} and kda_layers {sorted(kda)} do not partition 1..{n}")
    layer_types = [MLA if (i + 1) in full else KDA for i in range(n)]
    heads = int(cfg["num_attention_heads"])
    return {
        "layer_types": layer_types,
        "mla": {"heads": heads, "k_dim": int(cfg["qk_nope_head_dim"]) + int(cfg["qk_rope_head_dim"]),
                "v_dim": int(cfg["v_head_dim"])},
        "kda": {"heads": int(la["num_heads"]), "k_dim": int(la["head_dim"]), "v_dim": int(la["head_dim"]),
                "conv_width": int(la["short_conv_kernel_size"]),
                "conv_channels": int(la["num_heads"]) * int(la["head_dim"])},
    }


def observed_shapes_from_lists(key_cache, value_cache, recurrent, conv) -> dict:
    """Shapes as plain tuples; `recurrent` entries are (dtype_name, shape) so the float32 rule is checkable."""
    return {"key_cache": list(key_cache), "value_cache": list(value_cache),
            "recurrent": list(recurrent), "conv": list(conv)}


def check_shapes(observed: dict, expected: dict, T: int) -> list[str]:
    problems = []
    m, k = expected["mla"], expected["kda"]
    for i, t in enumerate(expected["layer_types"]):
        if t == MLA:
            want_k, want_v = (1, m["heads"], T, m["k_dim"]), (1, m["heads"], T, m["v_dim"])
            if tuple(observed["key_cache"][i] or ()) != want_k:
                problems.append(f"layer {i}: key_cache {observed['key_cache'][i]} is not {want_k}")
            if tuple(observed["value_cache"][i] or ()) != want_v:
                problems.append(f"layer {i}: value_cache {observed['value_cache'][i]} is not {want_v}")
        else:
            r = observed["recurrent"][i]
            want_s = (1, k["heads"], k["k_dim"], k["v_dim"])
            if r is None or tuple(r[1]) != want_s or r[0] != "float32":
                problems.append(f"layer {i}: recurrent state {r} is not float32 {want_s}")
            c = observed["conv"][i]
            want_c = (1, k["conv_channels"], k["conv_width"])
            if c is None or len(c) != 3 or any(tuple(x) != want_c for x in c):
                problems.append(f"layer {i}: conv_states {c} are not three of {want_c}")
    return problems


def config_mismatches(a: dict, b: dict) -> list[str]:
    """Every cache-shaping field the writer and the reader must share (NoPE: rope fields are asserted equal
    although nothing rotates; a difference would mean a different model family)."""
    keys = ("num_hidden_layers", "num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
            "mla_use_nope", "rope_theta", "rope_scaling", "vocab_size", "kv_lora_rank")
    out = [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in keys if a.get(k) != b.get(k)]
    la, lb = a.get("linear_attn_config") or {}, b.get("linear_attn_config") or {}
    for k in sorted(set(la) | set(lb)):
        if la.get(k) != lb.get(k):
            out.append(f"linear_attn_config.{k}: {la.get(k)!r} vs {lb.get(k)!r}")
    return out


# --- indexing ---------------------------------------------------------------------------------------------------

def scored_input(seq: np.ndarray, prefix_len: int, end: int | None = None):
    """(cached_rows, input_ids, continuation): the cache holds rows 0..prefix_len-2, the forward feeds
    [seq[prefix_len-1], c_0..c_{C-2}], logits[t] produces c_t."""
    seq = np.asarray(seq, dtype=np.int64)
    end = seq.shape[0] if end is None else int(end)
    if not (0 < end <= seq.shape[0]):
        raise ValueError(f"end {end} is outside the row (T={seq.shape[0]})")
    if prefix_len < 2 or prefix_len + 1 > end:
        raise ValueError(f"prefix_len {prefix_len} leaves no scoreable continuation before {end}")
    cont = seq[prefix_len:end]
    inp = np.concatenate([seq[prefix_len - 1: prefix_len], cont[:-1]])
    return prefix_len - 1, inp, cont


# --- the dump layout ----------------------------------------------------------------------------------------------

@dataclass
class Dump:
    meta: dict
    mla: dict = field(default_factory=dict)     # layer -> (K [T,H,k], V [T,H,v]) as stored (uint16 bits or float32)
    kda: dict = field(default_factory=dict)     # layer -> (S [H,K,V] float32, (cq, ck, cv) [D,W] as stored)


def numpy_from_array(a, kind: str) -> np.ndarray:
    """The offline converter: numpy in, numpy out (the torch one is `torch_to_numpy`)."""
    return np.asarray(a)


def _pack_activation(arr: np.ndarray, activation_dtype: str) -> np.ndarray:
    if activation_dtype == "bfloat16":
        if arr.dtype == np.uint16:
            return arr
        if arr.dtype == np.float32:
            return f32_to_bf16_bits(arr)
        raise ValueError(f"cannot pack {arr.dtype} as bfloat16 bits")
    if activation_dtype == "float32":
        return arr.astype(np.float32)
    raise ValueError(f"unknown activation dtype {activation_dtype!r}")


def arrays_from_lists(key_cache, value_cache, recurrent, conv, layer_types, to_numpy, activation_dtype: str) -> dict:
    """The four cache lists (batch 1) -> the layout's arrays: K/V [T, H, d], S [H, K, V] float32, conv [D, W]."""
    layer_types = list(layer_types)
    out = {"mla": {}, "kda": {}, "activation_dtype": activation_dtype}
    for i, t in enumerate(layer_types):
        if t == MLA:
            K, V = to_numpy(key_cache[i], "activation"), to_numpy(value_cache[i], "activation")
            if K.ndim != 4 or K.shape[0] != 1 or V.shape[:3] != K.shape[:3]:
                raise ValueError(f"layer {i}: K {K.shape} / V {V.shape} are not [1, H, T, d]")
            out["mla"][i] = (_pack_activation(np.ascontiguousarray(K[0].transpose(1, 0, 2)), activation_dtype),
                             _pack_activation(np.ascontiguousarray(V[0].transpose(1, 0, 2)), activation_dtype))
        elif t == KDA:
            S = to_numpy(recurrent[i], "state")
            if S.ndim != 4 or S.shape[0] != 1:
                raise ValueError(f"layer {i}: recurrent state {S.shape} is not [1, H, K, V]")
            if S.dtype != np.float32:
                raise ValueError(f"layer {i}: recurrent state is {S.dtype}, the kernels keep it in float32")
            c = conv[i]
            if c is None or len(c) != 3:
                raise ValueError(f"layer {i}: conv_states must be the (q, k, v) triple")
            cs = []
            for x in c:
                x = to_numpy(x, "activation")
                if x.ndim != 3 or x.shape[0] != 1:
                    raise ValueError(f"layer {i}: conv window {x.shape} is not [1, D, W]")
                cs.append(_pack_activation(np.ascontiguousarray(x[0]), activation_dtype))
            out["kda"][i] = (np.ascontiguousarray(S[0]), tuple(cs))
        else:
            raise ValueError(f"unknown layer kind {t!r}")
    return out


def write_dump(out: Path, arrays: dict, layer_types, cached_rows: int, extra: dict) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layer_types = list(layer_types)
    mla_layers = [i for i, t in enumerate(layer_types) if t == MLA]
    kda_layers = [i for i, t in enumerate(layer_types) if t == KDA]
    if not mla_layers:
        raise ValueError("no full_attention layer: the model reads its past length from the first MLA key cache")
    adt = arrays["activation_dtype"]
    files, mla_meta, kda_meta = {}, None, None
    for i in mla_layers:
        K, V = arrays["mla"][i]
        if K.shape[0] != cached_rows or V.shape[0] != cached_rows:
            raise ValueError(f"layer {i}: K {K.shape} does not hold cached_rows={cached_rows}")
        m = {"heads": int(K.shape[1]), "k_dim": int(K.shape[2]), "v_dim": int(V.shape[2]), "dtype": adt}
        if mla_meta is None:
            mla_meta = m
        elif m != mla_meta:
            raise ValueError(f"layer {i}: MLA shape {m} differs from layer {mla_layers[0]}'s {mla_meta}")
        name = f"mla_layer{i:02d}.npz"
        np.savez(out / name, K=K, V=V)
        files[name] = sha256_file(out / name)
    for i in kda_layers:
        S, cs = arrays["kda"][i]
        m = {"heads": int(S.shape[0]), "k_dim": int(S.shape[1]), "v_dim": int(S.shape[2]),
             "state_dtype": "float32", "conv_width": int(cs[0].shape[1]), "conv_dtype": adt}
        if kda_meta is None:
            kda_meta = m
        elif m != kda_meta:
            raise ValueError(f"layer {i}: KDA shape {m} differs from layer {kda_layers[0]}'s {kda_meta}")
        name = f"kda_layer{i:02d}.npz"
        np.savez(out / name, recurrent=S, **dict(zip(_CONV, cs)))
        files[name] = sha256_file(out / name)
    meta = {"schema": DUMP_SCHEMA, "prefix_len": int(cached_rows), "layer_types": layer_types,
            "mla": {"layers": mla_layers, **mla_meta},
            "kda": ({"layers": kda_layers, **kda_meta} if kda_meta else {"layers": []}),
            "files": files, "extra": dict(extra or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


def read_dump(d: Path) -> Dump:
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != DUMP_SCHEMA:
        raise ValueError(f"{d}: schema {meta.get('schema')!r} is not {DUMP_SCHEMA!r}")
    for name, want in meta["files"].items():
        got = sha256_file(d / name)
        if got != want:
            raise ValueError(f"{d / name}: sha256 {got[:12]} is not the recorded {want[:12]}")
    dump = Dump(meta=meta)
    P, mm, km = int(meta["prefix_len"]), meta["mla"], meta["kda"]
    for i in mm["layers"]:
        z = np.load(d / f"mla_layer{i:02d}.npz")
        K, V = z["K"], z["V"]
        if K.shape != (P, mm["heads"], mm["k_dim"]) or V.shape != (P, mm["heads"], mm["v_dim"]):
            raise ValueError(f"{d}: layer {i} K {K.shape} / V {V.shape} disagree with meta.json")
        dump.mla[i] = (K, V)
    for i in km["layers"]:
        z = np.load(d / f"kda_layer{i:02d}.npz")
        S = z["recurrent"]
        if S.dtype != np.float32 or S.shape != (km["heads"], km["k_dim"], km["v_dim"]):
            raise ValueError(f"{d}: layer {i} recurrent state {S.dtype} {S.shape} disagrees with meta.json")
        cs = tuple(z[c] for c in _CONV)
        if any(c.shape != (km["heads"] * km["k_dim"], km["conv_width"]) for c in cs):
            raise ValueError(f"{d}: layer {i} conv windows disagree with meta.json")
        dump.kda[i] = (S, cs)
    return dump


def lists_from_arrays(dump: Dump, from_numpy):
    """The layout's arrays -> the four cache lists with the batch axis and the model's layouts restored:
    K/V [1, H, T, d], S [1, H, K, V], conv [1, D, W]. `from_numpy(array, kind)` converts (and moves) each."""
    n = len(dump.meta["layer_types"])
    key, val, rec, conv = [None] * n, [None] * n, [None] * n, [None] * n
    for i, (K, V) in dump.mla.items():
        key[i] = from_numpy(np.ascontiguousarray(K.transpose(1, 0, 2))[None], "activation")
        val[i] = from_numpy(np.ascontiguousarray(V.transpose(1, 0, 2))[None], "activation")
    for i, (S, cs) in dump.kda.items():
        rec[i] = from_numpy(S[None], "state")
        conv[i] = tuple(from_numpy(c[None], "activation") for c in cs)
    return key, val, rec, conv


def compose_lists(arm: str, fresh: Dump, stale: Dump, from_numpy):
    if arm not in SOURCES or arm == "ONE-PASS":
        raise ValueError(f"arm {arm!r} composes nothing; the composed arms are {ARMS[:4]}")
    if fresh.meta["prefix_len"] != stale.meta["prefix_len"] or fresh.meta["layer_types"] != stale.meta["layer_types"]:
        raise ValueError("the two dumps differ in prefix_len or layer_types")
    for key in ("mla", "kda"):
        a = {k: v for k, v in fresh.meta[key].items() if k != "layers"}
        b = {k: v for k, v in stale.meta[key].items() if k != "layers"}
        if a != b:
            raise ValueError(f"the two dumps' {key} shapes differ: {a} vs {b}")
    src = SOURCES[arm]
    m = fresh if src["mla"] == "fresh" else stale
    k = fresh if src["kda"] == "fresh" else stale
    key, val, _, _ = lists_from_arrays(Dump(meta=m.meta, mla=m.mla), from_numpy)
    _, _, rec, conv = lists_from_arrays(Dump(meta=k.meta, kda=k.kda), from_numpy)
    return key, val, rec, conv


# --- the report fields -----------------------------------------------------------------------------------------

def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def score_fields(logits_fresh, logits_stale, cont) -> dict:
    """KL(pi_fresh || pi_stale) per position over the full vocabulary, natural log, with the per-token fields
    reader-report-v1 carries."""
    lf, ls = _log_softmax(logits_fresh), _log_softmax(logits_stale)
    cont = np.asarray(cont, dtype=np.int64)
    idx = np.arange(cont.shape[0])
    return {
        "continuation_tokens": [int(t) for t in cont],
        "continuation_sha256": hashlib.sha256(cont.tobytes()).hexdigest(),
        "logprob_fresh": lf[idx, cont].tolist(),
        "logprob_stale": ls[idx, cont].tolist(),
        "kl_per_token": (np.exp(lf) * (lf - ls)).sum(axis=-1).tolist(),
        "argmax_fresh": np.asarray(logits_fresh).argmax(-1).tolist(),
        "argmax_stale": np.asarray(logits_stale).argmax(-1).tolist(),
        "max_abs_logit_diff": float(np.abs(np.asarray(logits_fresh, dtype=np.float64)
                                           - np.asarray(logits_stale, dtype=np.float64)).max()),
    }


def pairwise_gaps(reports: dict, lo: int, hi: int) -> dict:
    """The split-point readout of the float32 reference probe: several ONE-PASS reports of the SAME row with different
    cached-row counts, aligned by absolute position [lo, hi). Returns the median |delta logprob| of the realized token
    between every two cached paths and between each cached path and its one-pass, whether the one-pass vectors agree
    bit for bit at the shared positions, and the argmax agreement between cached paths. Refuses a report that does
    not cover the window."""
    cols = {}
    for name, r in reports.items():
        P, end = int(r["prefix_len"]), int(r["prefix_len"]) + len(r["continuation_tokens"])
        if P > lo or end < hi:
            raise ValueError(f"report {name!r} covers positions {P}..{end}, not {lo}..{hi}")
        a, b = lo - P, hi - P
        cols[name] = {k: np.asarray(r[k][a:b], dtype=np.float64) for k in ("logprob_fresh", "logprob_stale")}
        cols[name]["argmax_stale"] = np.asarray(r["argmax_stale"][a:b])
        cols[name]["tokens"] = np.asarray(r["continuation_tokens"][a:b])
    names = sorted(cols, key=lambda n: (0, int(n)) if str(n).isdigit() else (1, str(n)))
    toks = [cols[n]["tokens"] for n in names]
    if any(not np.array_equal(toks[0], t) for t in toks[1:]):
        raise ValueError("the reports do not score the same tokens at the shared positions")
    ones = [cols[n]["logprob_fresh"] for n in names]
    out = {"positions": [lo, hi], "n": int(hi - lo), "reports": names,
           "one_pass_identical": bool(all(np.array_equal(ones[0], o) for o in ones[1:])),
           "cached_vs_one_pass": {n: float(np.median(np.abs(cols[n]["logprob_stale"] - cols[n]["logprob_fresh"])))
                                  for n in names},
           "cached_vs_cached": {}, "argmax_agreement": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["cached_vs_cached"][f"{a}|{b}"] = float(
                np.median(np.abs(cols[a]["logprob_stale"] - cols[b]["logprob_stale"])))
            out["argmax_agreement"][f"{a}|{b}"] = float((cols[a]["argmax_stale"] == cols[b]["argmax_stale"]).mean())
    return out


# =================================================================================================================
# pod side: everything below imports torch and runs a model
# =================================================================================================================

def torch_to_numpy(t, kind: str) -> np.ndarray:
    import torch
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    if t.dtype == torch.float32:
        return t.numpy()
    raise ValueError(f"{kind}: unexpected dtype {t.dtype} (bf16 activations, float32 state)")


def torch_from_numpy(device, activation_dtype: str):
    import torch

    def conv(a: np.ndarray, kind: str):
        if a.dtype == np.uint16:
            t = torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)
        else:
            t = torch.from_numpy(a.copy())
        if kind == "state" and t.dtype != torch.float32:
            raise ValueError("recurrent state must be float32")
        return t.to(device)
    return conv


def f32_precision() -> dict:
    """What a float32 forward actually computes with: torch's matmul precision and Triton's default fp32 dot
    precision (`TRITON_F32_DEFAULT`; unset means TF32 on Ampere+ -- FLA 0.4.0's delta-rule kernels pass no
    `allow_tf32`, so an honest float32 reference sets it to `ieee`)."""
    import os
    import torch
    return {"torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch_allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT")}


def load_model(path: Path, device, dtype: str = "bfloat16", attn: str = "model", device_map: str | None = None):
    """`attn` 'model' keeps whatever the model's own __init__ forces (flash_attention_2); 'sdpa' or 'eager' is set
    on the config AFTER construction, since the model overrides anything passed in -- float32 needs it, flash-attn
    has no float32 kernels. `device_map` 'auto' shards across the visible GPUs (the float32 reference: 192 GB)."""
    import torch
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    td = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name(dtype)]
    if dtype == "float32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    kw = {"device_map": device_map} if device_map else {}
    model = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True, torch_dtype=td, **kw)
    if not device_map:
        model = model.to(device)
    model = model.eval()
    if attn != "model":
        model.config._attn_implementation = attn
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = attn
    cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    return model, cfg, time.time() - t0


def cache_class(model):
    mod = sys.modules[type(model).__module__]
    return getattr(mod, "KimiDynamicCache")


def checkpoint_digest(path: Path) -> dict:
    """sha256 of config.json and of the safetensors index (or the single weights file): what the dump was taken
    from, tied to bytes without hashing 96 GB."""
    path = Path(path)
    files = {}
    for name in ("config.json", "model.safetensors.index.json", "model.safetensors", "modeling_kimi.py"):
        if (path / name).exists():
            files[name] = sha256_file(path / name)
    return files


def _lists_of(cache):
    return cache.key_cache, cache.value_cache, cache.recurrent_states, cache.conv_states


def _fill(cache, lists):
    key, val, rec, conv = lists
    for i in range(len(key)):
        cache.key_cache[i], cache.value_cache[i] = key[i], val[i]
        cache.recurrent_states[i], cache.conv_states[i] = rec[i], conv[i]
    return cache


def prefix_cache(model, ids: np.ndarray, device):
    """Forward over `ids` with use_cache; returns the cache object and the seconds."""
    import torch
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(np.asarray(ids, dtype=np.int64))[None].to(device), use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out.past_key_values, time.time() - t0


def continue_logits(model, cache, inp: np.ndarray, device) -> tuple[np.ndarray, float]:
    """Teacher-force `inp` from `cache` (which must hold cached_rows positions); float32 logits [C, vocab]."""
    import torch
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(np.asarray(inp, dtype=np.int64))[None].to(device),
                    past_key_values=cache, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out.logits[0].float().cpu().numpy(), time.time() - t0


def one_pass_logits(model, ids: np.ndarray, start: int, device) -> tuple[np.ndarray, float]:
    import torch
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(np.asarray(ids, dtype=np.int64))[None].to(device), use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out.logits[0, start:].float().cpu().numpy(), time.time() - t0


def _device(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _tokens(path: Path, seq: int, hashes: dict) -> np.ndarray:
    hashes[path.as_posix()] = sha256_file(path)
    all_tokens = np.load(path)
    if all_tokens.ndim != 2 or not (0 <= seq < all_tokens.shape[0]):
        raise SystemExit(f"{path}: shape {all_tokens.shape} has no sequence {seq}")
    return all_tokens[seq].astype(np.int64)


def _write_json(out: Path, name: str, obj: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    return p


def cmd_shapes(a) -> int:
    import torch
    device = _device(a.device)
    hashes = {}
    seq = _tokens(Path(a.tokens), a.seq, hashes)
    if not (0 < a.length <= seq.shape[0]):
        raise SystemExit(f"--length {a.length} is outside the row (T={seq.shape[0]})")
    model, cfg, load_s = load_model(Path(a.model), device, a.dtype, a.attn, a.device_map)
    exp = expected_shapes(cfg)
    cache, fwd_s = prefix_cache(model, seq[: a.length], device)
    key, val, rec, conv = _lists_of(cache)
    obs = observed_shapes_from_lists(
        [None if k is None else tuple(k.shape) for k in key],
        [None if v is None else tuple(v.shape) for v in val],
        [None if r is None else (str(r.dtype).replace("torch.", ""), tuple(r.shape)) for r in rec],
        [None if c is None else [tuple(x.shape) for x in c] for c in conv])
    problems = check_shapes(obs, exp, a.length)
    act = sorted({str(x.dtype).replace("torch.", "") for x in key if x is not None}
                 | {str(x.dtype).replace("torch.", "") for c in conv if c is not None for x in c})
    if act != [a.dtype]:
        problems.append(f"activation-dtype states are {act}, the run asked for {a.dtype}: a kernel cast the state")
    rec_out = {"schema": "hybrid-shapes-v1", "model": Path(a.model).as_posix(), "checkpoint": checkpoint_digest(Path(a.model)),
               "length": a.length, "expected": exp, "observed": obs, "problems": problems, "activation_dtypes": act,
               "attn_implementation": getattr(model.config, "_attn_implementation", None),
               "dtype": a.dtype, "f32_precision": f32_precision(),
               "seconds": {"load": load_s, "prefill": fwd_s}, "torch_version": torch.__version__,
               "device": device.type, "inputs_sha256": hashes}
    _write_json(Path(a.out), "shapes.json", rec_out)
    print(f"STATE_SHAPES {'ok' if not problems else 'FAIL'} layers={len(exp['layer_types'])} "
          f"mla={exp['mla']} kda={exp['kda']} activations={act} prefill_s={fwd_s:.1f}")
    for p in problems:
        print("  " + p)
    return 0 if not problems else 3


def cmd_dump(a) -> int:
    device = _device(a.device)
    hashes = {}
    seq = _tokens(Path(a.tokens), a.seq, hashes)
    cached_rows, _, _ = scored_input(seq, a.prefix_len)
    model, cfg, load_s = load_model(Path(a.model), device, a.dtype, a.attn, a.device_map)
    exp = expected_shapes(cfg)
    cache, fwd_s = prefix_cache(model, seq[:cached_rows], device)
    t0 = time.time()
    arrays = arrays_from_lists(*_lists_of(cache), exp["layer_types"], torch_to_numpy, a.dtype)
    meta = write_dump(Path(a.out), arrays, exp["layer_types"], cached_rows,
                      extra={"semantic_prefix_len": a.prefix_len, "seq": a.seq, "tokens_file": Path(a.tokens).as_posix(),
                             "tokens_sha256": hashes[Path(a.tokens).as_posix()], "model": Path(a.model).as_posix(),
                             "checkpoint": checkpoint_digest(Path(a.model)),
                             "state_v_first": a.state_v_first, "dtype": a.dtype,
                             "attn_implementation": getattr(model.config, "_attn_implementation", None),
                             "seconds": {"load": load_s, "prefill": fwd_s}})
    write_s = time.time() - t0
    print(f"DUMP_OK {Path(a.out).as_posix()} cached_rows={cached_rows} files={len(meta['files'])} "
          f"prefill_s={fwd_s:.1f} write_s={write_s:.1f}")
    return 0


def cmd_score(a) -> int:
    import torch
    out = Path(a.out)
    if a.probe and "results" in out.resolve().parts:
        raise SystemExit(f"probe mode refuses to write under results/: {out}")
    if a.arm == "ONE-PASS" and (a.fresh or a.stale or a.live):
        raise SystemExit("ONE-PASS builds both sides live from the reader; no --fresh/--stale/--live")
    if a.arm != "ONE-PASS":
        if a.live and a.fresh:
            raise SystemExit("--live replaces the fresh dump; do not also pass --fresh")
        if not a.live and not a.fresh:
            raise SystemExit("--fresh is required unless --live")
        if not a.stale:
            raise SystemExit("--stale is required for a composed arm")
    device = _device(a.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hashes = {}
    seq = _tokens(Path(a.tokens), a.seq, hashes)
    cached_rows, inp, cont = scored_input(seq, a.prefix_len, a.length)
    model, cfg, load_s = load_model(Path(a.reader), device, a.dtype, a.attn, a.device_map)
    exp = expected_shapes(cfg)
    seconds = {"load": load_s}
    conv_in = torch_from_numpy(device, a.dtype)

    if a.arm == "ONE-PASS":
        end = seq.shape[0] if a.length is None else a.length
        logits_f, seconds["one_pass"] = one_pass_logits(model, seq[: end - 1], cached_rows, device)
        cache, seconds["prefill"] = prefix_cache(model, seq[:cached_rows], device)
        logits_s, seconds["continuation"] = continue_logits(model, cache, inp, device)
        fresh_dump = stale_dump = None
        live = True
    else:
        stale = read_dump(Path(a.stale))
        for name, h in stale.meta["files"].items():
            hashes[(Path(a.stale) / name).as_posix()] = h
        if a.live:
            cache, seconds["prefill"] = prefix_cache(model, seq[:cached_rows], device)
            arrays = arrays_from_lists(*_lists_of(cache), exp["layer_types"], torch_to_numpy, a.dtype)
            fresh = _dump_from_arrays(arrays, exp["layer_types"], cached_rows, stale.meta)
        else:
            fresh = read_dump(Path(a.fresh))
            for name, h in fresh.meta["files"].items():
                hashes[(Path(a.fresh) / name).as_posix()] = h
        if fresh.meta["prefix_len"] != cached_rows:
            raise SystemExit(f"dump holds {fresh.meta['prefix_len']} rows, --prefix-len {a.prefix_len} needs {cached_rows}")
        if fresh.meta["layer_types"] != exp["layer_types"]:
            raise SystemExit("the dump's layer_types are not the reader's")
        for d in (fresh, stale):
            ck = (d.meta.get("extra") or {}).get("checkpoint_config")
            if ck:
                mm = config_mismatches(ck, cfg)
                if mm:
                    raise SystemExit("writer/reader config mismatch: " + "; ".join(mm))
        lists_fresh = compose_lists("FRESH", fresh, stale, conv_in)
        lists_arm = compose_lists(a.arm, fresh, stale, conv_in)
        logits_f, seconds["continuation_fresh"] = continue_logits(model, _fill(cache_class(model)(model.config), lists_fresh), inp, device)
        logits_s, seconds["continuation_arm"] = continue_logits(model, _fill(cache_class(model)(model.config), lists_arm), inp, device)
        fresh_dump = None if a.live else Path(a.fresh).as_posix()
        stale_dump = Path(a.stale).as_posix()
        live = bool(a.live)

    fields = score_fields(logits_f, logits_s, cont)
    report = {
        "schema": SCHEMA, "reader": Path(a.reader).as_posix(), "fresh_dump": fresh_dump, "stale_dump": stale_dump,
        "tokens_file": Path(a.tokens).as_posix(), "seq": a.seq, "prefix_len": a.prefix_len,
        "text": a.text, "generate_under": None, "live": live, **fields,
        "inputs_sha256": hashes, "reader_checkpoint": checkpoint_digest(Path(a.reader)),
        "device": device.type, "torch_version": torch.__version__,
        "peak_cuda_mem_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "gen_seconds": None,
        "arm": a.arm, "sources": SOURCES[a.arm], "state_v_first": a.state_v_first, "dtype": a.dtype,
        "cached_rows": cached_rows, "seconds": seconds,
        "attn_implementation": getattr(model.config, "_attn_implementation", None), "f32_precision": f32_precision(),
    }
    assert set(report) == set(REPORT_KEYS)
    p = _write_json(out, "report.json", report)
    kl = np.asarray(fields["kl_per_token"])
    print(f"wrote {p.as_posix()} arm={a.arm} live={live} median_kl={float(np.median(kl)):.3e} "
          f"mean_kl={float(kl.mean()):.3e} max_logit_diff={fields['max_abs_logit_diff']:.3e}")
    return 0


def _dump_from_arrays(arrays: dict, layer_types, cached_rows: int, like_meta: dict) -> Dump:
    """An in-memory Dump of live arrays shaped like `like_meta` (for --live), never written."""
    meta = json.loads(json.dumps(like_meta))
    meta["prefix_len"] = cached_rows
    meta["files"] = {}
    meta["extra"] = {"live": True}
    d = Dump(meta=meta)
    for i, kv in arrays["mla"].items():
        d.mla[i] = kv
    for i, sc in arrays["kda"].items():
        d.kda[i] = sc
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model_flag):
        p.add_argument(model_flag, required=True, help="local checkpoint directory (config.json + safetensors)")
        p.add_argument("--tokens", required=True)
        p.add_argument("--seq", type=int, required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
        p.add_argument("--state-v-first", type=int, choices=(0, 1), default=None,
                       help="the FLA state layout flag as read from the installed fla source; recorded, not acted on")
        p.add_argument("--dtype", default="bfloat16", choices=DTYPES,
                       help="bfloat16 as served (R-K2); float32 for the reference probe (torch 'highest', no TF32; "
                            "set TRITON_F32_DEFAULT=ieee in the environment for the Triton kernels)")
        p.add_argument("--attn", default="model", choices=("model", "sdpa", "eager"),
                       help="'model' keeps the forced flash_attention_2; float32 needs sdpa or eager")
        p.add_argument("--device-map", default=None, choices=("auto",),
                       help="'auto' shards the weights across the visible GPUs (192 GB in float32)")

    p = sub.add_parser("shapes", help="forward --length tokens with a cache and assert the §16 shapes per layer")
    common(p, "--model")
    p.add_argument("--length", type=int, default=1024)
    p.set_defaults(fn=cmd_shapes)

    p = sub.add_parser("dump", help="write the hybrid-dump-v1 of the first prefix-len-1 tokens")
    common(p, "--model")
    p.add_argument("--prefix-len", type=int, default=768)
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("score", help="teacher-force the continuation under an arm; write hybrid-report-v1")
    common(p, "--reader")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--fresh", default=None, help="the reader's own dump (omit with --live or ONE-PASS)")
    p.add_argument("--stale", default=None, help="the writer's dump (composed arms)")
    p.add_argument("--prefix-len", type=int, default=768)
    p.add_argument("--length", type=int, default=None, help="end of the continuation (exclusive); default the row")
    p.add_argument("--text", default="corpus", choices=("corpus", "agentic"))
    p.add_argument("--live", action="store_true", help="fresh side = the reader's live cache (dump equivalence)")
    p.add_argument("--probe", action="store_true", help="refuse any --out under results/")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("gaps", help="split-point readout: pairwise gaps between ONE-PASS reports of one row (no torch)")
    p.add_argument("--report", action="append", required=True, metavar="NAME=PATH",
                   help="a ONE-PASS report.json named by its cached-row count, e.g. 767=/workspace/probe/floor/base/seq0/report.json")
    p.add_argument("--lo", type=int, required=True)
    p.add_argument("--hi", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_gaps)

    a = ap.parse_args(argv)
    return a.fn(a)


def cmd_gaps(a) -> int:
    reports, files = {}, {}
    for spec in a.report:
        name, path = spec.split("=", 1)
        reports[name] = json.loads(Path(path).read_text(encoding="utf-8"))
        files[Path(path).as_posix()] = sha256_file(Path(path))
    g = pairwise_gaps(reports, a.lo, a.hi)
    g["schema"] = "hybrid-gaps-v1"
    g["inputs_sha256"] = files
    g["dtypes"] = {n: r.get("dtype") for n, r in reports.items()}
    g["f32_precision"] = {n: r.get("f32_precision") for n, r in reports.items()}
    p = _write_json(Path(a.out), "gaps.json", g)
    c1 = {k: f"{v:.2e}" for k, v in g["cached_vs_one_pass"].items()}
    c2 = {k: f"{v:.2e}" for k, v in g["cached_vs_cached"].items()}
    print(f"wrote {p.as_posix()} n={g['n']} one_pass_identical={g['one_pass_identical']} "
          f"cached_vs_one_pass={c1} cached_vs_cached={c2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
