"""A two-layer hybrid stand-in for the Kimi Linear cache shape, for tests only (numpy; no torch, no fla).

Layer 0 is MLA-like: softmax attention whose per-head K width (dk) differs from its V width (dv), materialized
K/V cached along the sequence. Layer 1 is KDA-like: a gated delta-rule recurrence over a short causal convolution,
whose cache is a float32 recurrent state [H, K, V] plus three conv windows [D, W]. The cache class mirrors
`KimiDynamicCache` field for field (`key_cache`, `value_cache`, `recurrent_states`, `conv_states`, one slot per
layer, None where the kind does not apply) so the inject path in `lag_ladder.hybrid_state` is exercised against
the real shape of the object it will fill on the pod. Nothing here is the model's math; the kernels are the
point on the pod, and this is the point offline: which state goes where under each arm.

The model is deterministic in its weights (seeded) and in its forward (float64), so "one pass over prefix +
continuation" and "prefix pass, cache, continuation from the cache" agree to rounding, which is the fake's floor.
"""
from __future__ import annotations

import numpy as np

VOCAB, EMB = 11, 8
MLA_HEADS, MLA_DK, MLA_DV = 2, 6, 4       # dk != dv on purpose (Kimi: 192 vs 128)
KDA_HEADS, KDA_D = 2, 3                    # K = V = KDA_D (Kimi: 128)
CONV_W = 4
LAYER_TYPES = ("full_attention", "linear_attention")


class FakeHybridCache:
    """Same four lists as KimiDynamicCache, one entry per layer."""

    def __init__(self, n_layers: int = 2):
        self.key_cache = [None] * n_layers
        self.value_cache = [None] * n_layers
        self.recurrent_states = [None] * n_layers
        self.conv_states = [None] * n_layers

    def get_seq_length(self) -> int:
        k = self.key_cache[0]
        return 0 if k is None else int(k.shape[0])


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class FakeHybrid:
    """Weights from `seed`; `mla_scale` / `kda_scale` let a test build a 'writer' that differs from the
    'reader' in one layer kind only, so the arm decomposition has a known answer."""

    def __init__(self, seed: int = 0, mla_scale: float = 1.0, kda_scale: float = 1.0):
        r = _rng(seed)
        self.emb = r.normal(size=(VOCAB, EMB)) * 0.5
        # MLA-like
        self.wq = r.normal(size=(EMB, MLA_HEADS * MLA_DK)) * 0.3 * mla_scale
        self.wk = r.normal(size=(EMB, MLA_HEADS * MLA_DK)) * 0.3 * mla_scale
        self.wv = r.normal(size=(EMB, MLA_HEADS * MLA_DV)) * 0.3 * mla_scale
        self.wo = r.normal(size=(MLA_HEADS * MLA_DV, EMB)) * 0.3
        # KDA-like
        D = KDA_HEADS * KDA_D
        self.pq = r.normal(size=(EMB, D)) * 0.3 * kda_scale
        self.pk = r.normal(size=(EMB, D)) * 0.3 * kda_scale
        self.pv = r.normal(size=(EMB, D)) * 0.3 * kda_scale
        self.conv = r.normal(size=(3, D, CONV_W)) * 0.3      # depthwise, one filter bank per q/k/v
        self.pg = r.normal(size=(EMB, KDA_HEADS)) * 0.3        # forget gate (per head)
        self.pb = r.normal(size=(EMB, KDA_HEADS)) * 0.3        # beta
        self.po = r.normal(size=(D, EMB)) * 0.3
        self.unemb = r.normal(size=(EMB, VOCAB)) * 0.5

    # --- layer 0: MLA-like ---------------------------------------------------------------------------
    def _mla(self, x: np.ndarray, cache: FakeHybridCache, layer: int = 0) -> np.ndarray:
        T = x.shape[0]
        q = (x @ self.wq).reshape(T, MLA_HEADS, MLA_DK)
        k = (x @ self.wk).reshape(T, MLA_HEADS, MLA_DK)
        v = (x @ self.wv).reshape(T, MLA_HEADS, MLA_DV)
        if cache.key_cache[layer] is not None:
            k = np.concatenate([cache.key_cache[layer], k], axis=0)
            v = np.concatenate([cache.value_cache[layer], v], axis=0)
        cache.key_cache[layer], cache.value_cache[layer] = k, v
        past = k.shape[0] - T
        out = np.zeros((T, MLA_HEADS, MLA_DV))
        for h in range(MLA_HEADS):
            s = q[:, h] @ k[:, h].T / np.sqrt(MLA_DK)                     # [T, past+T]
            mask = np.tril(np.ones((T, past + T), dtype=bool), k=past)
            s = np.where(mask, s, -np.inf)
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            out[:, h] = p @ v[:, h]
        return out.reshape(T, -1) @ self.wo

    # --- layer 1: KDA-like ---------------------------------------------------------------------------
    def _conv(self, u: np.ndarray, w: np.ndarray, state: np.ndarray | None):
        """Depthwise causal conv over a window of CONV_W inputs; state [D, W] holds the last W inputs
        (oldest first), the same window the FLA ShortConvolution cache carries."""
        T, D = u.shape
        win = np.zeros((D, CONV_W)) if state is None else state.copy()
        y = np.zeros_like(u)
        for t in range(T):
            win = np.concatenate([win[:, 1:], u[t][:, None]], axis=1)
            y[t] = (win * w).sum(axis=1)
        return np.tanh(y), win

    def _kda(self, x: np.ndarray, cache: FakeHybridCache, layer: int = 1) -> np.ndarray:
        T = x.shape[0]
        cs = cache.conv_states[layer] or (None, None, None)
        q, cq = self._conv(x @ self.pq, self.conv[0], cs[0])
        k, ck = self._conv(x @ self.pk, self.conv[1], cs[1])
        v, cv = self._conv(x @ self.pv, self.conv[2], cs[2])
        q = q.reshape(T, KDA_HEADS, KDA_D)
        k = k.reshape(T, KDA_HEADS, KDA_D)
        v = v.reshape(T, KDA_HEADS, KDA_D)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-9)
        k = k / np.linalg.norm(k, axis=-1, keepdims=True).clip(1e-9)
        g = -np.log1p(np.exp(x @ self.pg))            # log forget gate in (-inf, 0)
        beta = 1.0 / (1.0 + np.exp(-(x @ self.pb)))
        S = cache.recurrent_states[layer]
        S = np.zeros((KDA_HEADS, KDA_D, KDA_D), dtype=np.float32) if S is None else S
        if S.dtype != np.float32:
            raise TypeError("recurrent state must be float32 (the fused kernel asserts it)")
        S = S.astype(np.float64)
        out = np.zeros((T, KDA_HEADS, KDA_D))
        for t in range(T):
            for h in range(KDA_HEADS):
                S[h] *= np.exp(g[t, h])
                err = v[t, h] - S[h].T @ k[t, h]
                S[h] += beta[t, h] * np.outer(k[t, h], err)
                out[t, h] = S[h].T @ q[t, h]
        cache.recurrent_states[layer] = S.astype(np.float32)
        cache.conv_states[layer] = (cq, ck, cv)
        return out.reshape(T, -1) @ self.po

    # --- the model -----------------------------------------------------------------------------------
    def forward(self, tokens: np.ndarray, cache: FakeHybridCache) -> np.ndarray:
        """Logits [T, VOCAB] for `tokens` continuing whatever `cache` holds; `cache` is updated in place."""
        x = self.emb[np.asarray(tokens)]
        x = x + self._mla(x, cache)
        x = x + self._kda(x, cache)
        return x @ self.unemb

    def prefill(self, tokens: np.ndarray) -> FakeHybridCache:
        cache = FakeHybridCache()
        self.forward(tokens, cache)
        return cache
