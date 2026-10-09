"""Per-token deviation and the oracle selective-recompute fraction f*(tau), statistic (A).

Copied from linear-ceiling `src/linear_ceiling/e9_pertoken.py` at the chassis commit (UPSTREAM.md §3):
`centered_delta`, `token_mean`, `layer_mean`, `F_STAR_REL_TOL`, `f_star` and `band_outcome` are verbatim;
the seam, block, bootstrap and null-pairing helpers are dropped (they belong to a context handoff, which
this repository does not measure). The definitions are linear-ceiling ledger entry 0023, cited by entry
0001 here; nothing is recalibrated.

All of it is CPU arithmetic over the per-token record the pinned scorer writes (`--per-token`: squares
[n, L, H] float32 per read-out, plus the per-head SST in the score json). Nothing here reads a model.

Units (0023, rider 2): the centered per-token deviation
    delta_c(t, l, h) = ||x_hat(t,l,h) - x_R(t,l,h)||^2 / (SST(l,h) / n)
is the token's share of the layer-head's unexplained variance in R^2's own units: its mean over tokens
is exactly 1 - R^2(l, h). Here x_hat is the cache as the writer checkpoint wrote it and x_R the reader
checkpoint's own cache for the same tokens.
"""
import numpy as np


def centered_delta(sq: np.ndarray, sst: np.ndarray, n: int) -> np.ndarray:
    """[n, L, H] squares, [L, H] SST -> [n, L, H] centered deviation (float64)."""
    sq = np.asarray(sq, dtype=np.float64)
    sst = np.asarray(sst, dtype=np.float64)
    if sq.ndim != 3 or sst.shape != sq.shape[1:] or sq.shape[0] != n:
        raise ValueError(f"shape mismatch: squares {sq.shape}, sst {sst.shape}, n {n}")
    if (sst <= 0).any():
        raise ValueError("a layer-head SST is not positive; the reference set is degenerate")
    return sq / (sst / n)[None]


def token_mean(delta: np.ndarray) -> np.ndarray:
    """[n, L, H] -> [n]: mean over heads then layers (equal counts, so the plain mean)."""
    return np.asarray(delta, dtype=np.float64).reshape(len(delta), -1).mean(1)


def layer_mean(delta: np.ndarray) -> np.ndarray:
    """[n, L, H] -> [n, L]: mean over heads."""
    return np.asarray(delta, dtype=np.float64).mean(2)


F_STAR_REL_TOL = 1e-9            # "at or below tau" is judged to this relative tolerance (float32 record)


def f_star(delta_token: np.ndarray, tau: float) -> float:
    """Oracle selective-recompute fraction: the smallest fraction of tokens that, removed in
    descending order of deviation (each assumed restored exactly), leaves the MEAN deviation of
    the rest at or below tau (to F_STAR_REL_TOL relative, so a float32 per-token record whose
    mean lands 1e-11 above tau does not cost a token). 0 when the full-set mean is already
    <= tau; 1 when no proper subset qualifies. An oracle LOWER BOUND on real selective recompute
    (0023: restored-exactly assumption; no error propagation through the reused KV)."""
    d = np.asarray(delta_token, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("f* needs a one-dimensional array of token deviations")
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("token deviations must be finite and non-negative")
    d = np.sort(d)[::-1]
    n = len(d)
    if n == 0:
        raise ValueError("f* of an empty token set; refusing to invent a number")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError(f"tau must be a finite non-negative number, got {tau}")
    # mean of the suffix d[k:] for k = 0..n-1; suffix means are monotone non-increasing in k
    suffix = np.cumsum(d[::-1])[::-1]
    counts = np.arange(n, 0, -1)
    ok = (suffix / counts) <= tau * (1.0 + F_STAR_REL_TOL) + 1e-15
    if not ok.any():
        return 1.0
    return float(np.argmax(ok) / n)


def band_outcome(median_f: float, rule: dict) -> str:
    """0023 rule on median f*(tau_K) over included handoffs, K read-out."""
    if median_f <= rule["holds_max"]:
        return "HOLDS"
    if median_f >= rule["degrades_min"]:
        return "DEGRADES"
    return "UNRESOLVED"
