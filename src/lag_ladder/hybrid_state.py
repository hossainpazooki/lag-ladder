"""The inject path for a hybrid-state cache: a dump of three state kinds and the stale arms over them.

Built offline against `tests/fake_hybrid.py` for the Kimi Linear seed (a model whose cache is materialized MLA
K/V on some layers and a float32 delta-rule recurrent state plus three short-convolution windows on the others;
the shapes verified from `modeling_kimi.py` and FLA's kernels are recorded in the private seed brief). The pod
driver that fills a real `KimiDynamicCache` from a `Composed` is not here yet; this module owns what it will
depend on and what the summarizer will read back: the dump layout, its validation, and which source every state
kind takes under each arm. No torch: arrays are numpy, so the suite stays offline.

Dump layout (schema `hybrid-dump-v1`), one directory per (model, sequence, prefix):
  meta.json          schema, prefix_len, layer_types, mla {layers, heads, k_dim, v_dim, dtype},
                     kda {layers, heads, k_dim, v_dim, state_dtype, conv_width, conv_dtype}, files {name: sha256}, extra
  mla_layerNN.npz    K [T, H, k_dim], V [T, H, v_dim]            (the materialized per-head cache, batch dropped)
  kda_layerNN.npz    recurrent [H, k_dim, v_dim] float32, conv_q / conv_k / conv_v [H*k_dim, W]

Arms (design seed §4): FRESH takes everything from the reader's own dump; FULL-STALE everything from the writer's;
MLA-STALE the writer's K/V with the reader's recurrent and conv states; KDA-STALE the reverse. Every arm carries an
MLA cache of the full prefix, because the model reads its past length from the first MLA layer's key cache.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "hybrid-dump-v1"
ARMS = ("FRESH", "FULL-STALE", "MLA-STALE", "KDA-STALE")
MLA, KDA = "full_attention", "linear_attention"
_SOURCES = {
    "FRESH": {"mla": "fresh", "kda": "fresh"},
    "FULL-STALE": {"mla": "stale", "kda": "stale"},
    "MLA-STALE": {"mla": "stale", "kda": "fresh"},
    "KDA-STALE": {"mla": "fresh", "kda": "stale"},
}
_CONV = ("conv_q", "conv_k", "conv_v")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refuse(msg: str) -> ValueError:
    return ValueError(f"hybrid dump refused: {msg}")


@dataclass
class HybridDump:
    meta: dict
    mla: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)      # layer -> (K, V)
    kda: dict[int, tuple[np.ndarray, tuple[np.ndarray, ...]]] = field(default_factory=dict)  # layer -> (S, (cq, ck, cv))

    @property
    def prefix_len(self) -> int:
        return int(self.meta["prefix_len"])

    @property
    def layer_types(self) -> list[str]:
        return list(self.meta["layer_types"])


# --- writing -------------------------------------------------------------------------------------------------

def write_dump(out: Path, cache, layer_types, prefix_len: int, extra: dict | None = None) -> dict:
    """Write the four cache lists of a `KimiDynamicCache`-shaped object (`key_cache`, `value_cache`,
    `recurrent_states`, `conv_states`, one slot per layer; arrays or array-likes without a batch axis) under
    the dump layout. Returns the meta it wrote."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layer_types = list(layer_types)
    if set(layer_types) - {MLA, KDA}:
        raise _refuse(f"unknown layer kind in {layer_types}")
    mla_layers = [i for i, t in enumerate(layer_types) if t == MLA]
    kda_layers = [i for i, t in enumerate(layer_types) if t == KDA]
    if not mla_layers:
        raise _refuse("no full_attention layer: the model reads its past length from the first MLA key cache")
    files: dict[str, str] = {}
    mla_meta = kda_meta = None
    for i in mla_layers:
        K, V = np.asarray(cache.key_cache[i]), np.asarray(cache.value_cache[i])
        if K.ndim != 3 or V.ndim != 3 or K.shape[:2] != V.shape[:2] or K.shape[0] != prefix_len:
            raise _refuse(f"layer {i}: K {K.shape} / V {V.shape} are not [prefix_len, heads, dim]")
        m = {"heads": int(K.shape[1]), "k_dim": int(K.shape[2]), "v_dim": int(V.shape[2]), "dtype": str(K.dtype)}
        if mla_meta is None:
            mla_meta = m
        elif m != mla_meta:
            raise _refuse(f"layer {i}: MLA shape {m} differs from layer {mla_layers[0]}'s {mla_meta}")
        name = f"mla_layer{i:02d}.npz"
        np.savez(out / name, K=K, V=V)
        files[name] = sha256_file(out / name)
    for i in kda_layers:
        S = np.asarray(cache.recurrent_states[i])
        conv = cache.conv_states[i]
        if S.ndim != 3:
            raise _refuse(f"layer {i}: recurrent state {S.shape} is not [heads, k_dim, v_dim]")
        if S.dtype != np.float32:
            raise _refuse(f"layer {i}: recurrent state is {S.dtype}, the kernels keep it in float32")
        if conv is None or len(conv) != 3:
            raise _refuse(f"layer {i}: conv_states must be the (q, k, v) triple")
        conv = tuple(np.asarray(c) for c in conv)
        if any(c.ndim != 2 or c.shape[0] != S.shape[0] * S.shape[1] for c in conv):
            raise _refuse(f"layer {i}: conv windows {[c.shape for c in conv]} are not [heads*k_dim, W]")
        m = {"heads": int(S.shape[0]), "k_dim": int(S.shape[1]), "v_dim": int(S.shape[2]),
             "state_dtype": "float32", "conv_width": int(conv[0].shape[1]), "conv_dtype": str(conv[0].dtype)}
        if kda_meta is None:
            kda_meta = m
        elif m != kda_meta:
            raise _refuse(f"layer {i}: KDA shape {m} differs from layer {kda_layers[0]}'s {kda_meta}")
        name = f"kda_layer{i:02d}.npz"
        np.savez(out / name, recurrent=S, **dict(zip(_CONV, conv)))
        files[name] = sha256_file(out / name)
    meta = {"schema": SCHEMA, "prefix_len": int(prefix_len), "layer_types": layer_types,
            "mla": {"layers": mla_layers, **mla_meta},
            "kda": ({"layers": kda_layers, **kda_meta} if kda_meta else {"layers": []}),
            "files": files, "extra": dict(extra or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


# --- reading -------------------------------------------------------------------------------------------------

def load_dump(d: Path) -> HybridDump:
    """Read a dump back, refusing a file whose sha256 moved, a recurrent state that is not float32, or any
    shape that disagrees with `meta.json`."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise _refuse(f"{d}: schema {meta.get('schema')!r} is not {SCHEMA!r}")
    lt = list(meta["layer_types"])
    if set(lt) - {MLA, KDA} or lt.count(MLA) == 0:
        raise _refuse(f"{d}: layer_types {lt} need at least one full_attention layer and nothing else")
    if [i for i, t in enumerate(lt) if t == MLA] != list(meta["mla"]["layers"]):
        raise _refuse(f"{d}: meta.mla.layers does not match layer_types")
    if [i for i, t in enumerate(lt) if t == KDA] != list(meta["kda"]["layers"]):
        raise _refuse(f"{d}: meta.kda.layers does not match layer_types")
    for name, want in meta["files"].items():
        got = sha256_file(d / name)
        if got != want:
            raise _refuse(f"{d / name}: sha256 {got[:12]} is not the recorded {want[:12]}")
    dump = HybridDump(meta=meta)
    P, mm, km = int(meta["prefix_len"]), meta["mla"], meta["kda"]
    for i in mm["layers"]:
        z = np.load(d / f"mla_layer{i:02d}.npz")
        K, V = z["K"], z["V"]
        if K.shape != (P, mm["heads"], mm["k_dim"]):
            raise _refuse(f"{d}: layer {i} K {K.shape} is not (prefix_len, heads, k_dim) = "
                          f"({P}, {mm['heads']}, {mm['k_dim']})")
        if V.shape != (P, mm["heads"], mm["v_dim"]):
            raise _refuse(f"{d}: layer {i} V {V.shape} is not (prefix_len, heads, v_dim)")
        dump.mla[i] = (K, V)
    for i in km["layers"]:
        z = np.load(d / f"kda_layer{i:02d}.npz")
        S = z["recurrent"]
        if S.dtype != np.float32:
            raise _refuse(f"{d}: layer {i} recurrent state is {S.dtype}, not float32")
        if S.shape != (km["heads"], km["k_dim"], km["v_dim"]):
            raise _refuse(f"{d}: layer {i} recurrent state {S.shape} is not (heads, k_dim, v_dim)")
        conv = tuple(z[c] for c in _CONV)
        for c in conv:
            if c.shape != (km["heads"] * km["k_dim"], km["conv_width"]):
                raise _refuse(f"{d}: layer {i} conv window {c.shape} is not (heads*k_dim, conv_width)")
        dump.kda[i] = (S, conv)
    return dump


# --- the arms ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Composed:
    arm: str
    sources: dict
    fresh: HybridDump
    stale: HybridDump

    def _pick(self, kind: str) -> HybridDump:
        return self.fresh if self.sources[kind] == "fresh" else self.stale

    def fill(self, cache) -> None:
        """Fill a `KimiDynamicCache`-shaped object in place with copies of the chosen states, so a forward
        that grows the cache never writes into the dump's arrays."""
        m, k = self._pick("mla"), self._pick("kda")
        for i, (K, V) in m.mla.items():
            cache.key_cache[i], cache.value_cache[i] = K.copy(), V.copy()
        for i, (S, conv) in k.kda.items():
            cache.recurrent_states[i] = S.copy()
            cache.conv_states[i] = tuple(c.copy() for c in conv)


def compose(arm: str, fresh: HybridDump, stale: HybridDump) -> Composed:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; the arms are {ARMS}")
    if fresh.prefix_len != stale.prefix_len:
        raise ValueError(f"prefix_len differs: fresh {fresh.prefix_len}, stale {stale.prefix_len}")
    if fresh.layer_types != stale.layer_types:
        raise ValueError("the two dumps' layer_types differ; they are not dumps of the same architecture")
    for key in ("mla", "kda"):
        a = {k: v for k, v in fresh.meta[key].items() if k != "layers"}
        b = {k: v for k, v in stale.meta[key].items() if k != "layers"}
        if a != b:
            raise ValueError(f"the two dumps' {key} shapes differ: {a} vs {b}")
    return Composed(arm=arm, sources=dict(_SOURCES[arm]), fresh=fresh, stale=stale)


# --- the statistic -------------------------------------------------------------------------------------------

def kl_per_token(logits_fresh: np.ndarray, logits_stale: np.ndarray) -> np.ndarray:
    """KL(pi_fresh || pi_stale) per position over the full vocabulary, natural log — the (B) scorer's
    contract (design §13), as the hybrid driver will record it."""
    lf = np.asarray(logits_fresh, dtype=np.float64)
    ls = np.asarray(logits_stale, dtype=np.float64)
    lf = lf - lf.max(axis=-1, keepdims=True)
    ls = ls - ls.max(axis=-1, keepdims=True)
    lf = lf - np.log(np.exp(lf).sum(axis=-1, keepdims=True))
    ls = ls - np.log(np.exp(ls).sum(axis=-1, keepdims=True))
    return (np.exp(lf) * (lf - ls)).sum(axis=-1)
