"""Read one scored job off disk and recompute its f*(tau) ladder; refuse on any disagreement.

A job is one `score_positions` call (the pinned instrument) over one held-out sequence: `report.json`
with the per-layer, per-head SSE/SST and R^2, and the per-token file it names by sha256. Every number
here is derived again from those two files: the per-token squares must hash to what the report
recorded, sum to the recorded SSE per layer-head, and their centered mean must reproduce
1 - R^2 (linear-ceiling 0023's exact bridge). Nothing here talks to a model or an engine.
"""
import json
from pathlib import Path

import numpy as np

from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.pertoken import centered_delta, f_star, token_mean

REPORT = "report.json"
SUM_RTOL = 1e-6          # float64 SSE vs the float32 per-token record
BRIDGE_ATOL = 1e-6       # mean centered deviation vs 1 - recorded R^2


def tau_key(tau: float) -> str:
    return f"{float(tau):g}"


def evaluate_job(job_dir: Path, taus: list[float], *, who: str = "summarize") -> dict:
    """{n, fstar: {tau_key: f*}, delta_mean, one_minus_r2, report_sha256, pertoken_sha256, pertoken_file}."""
    job_dir = Path(job_dir)
    report = job_dir / REPORT
    if not report.exists():
        raise ValueError(f"{who} REFUSED: {job_dir} has no {REPORT}")
    rec = json.loads(report.read_text(encoding="utf-8"))
    pt_meta = rec.get("per_token") or {}
    if not pt_meta.get("path") or not pt_meta.get("sha256"):
        raise ValueError(f"{who} REFUSED: {report} does not name a per-token file and its sha256")
    pt = job_dir / pt_meta["path"]
    if not pt.exists():
        raise ValueError(f"{who} REFUSED: per-token file {pt} is missing")
    pt_sha = sha256_file_bytes(pt)
    if pt_sha != pt_meta["sha256"]:
        raise ValueError(f"{who} REFUSED: {pt.name} does not match the sha256 {report.name} recorded at run time")
    n = int(rec["n_pairs"])
    layers = rec["same"]["K"]
    sse = np.asarray([l["sse"] for l in layers], dtype=np.float64)
    sst = np.asarray([l["sst"] for l in layers], dtype=np.float64)
    with np.load(pt) as z:
        if "same_K" not in z:
            raise ValueError(f"{who} REFUSED: {pt.name} carries no same_K array")
        sq = np.asarray(z["same_K"], dtype=np.float64)
    if sq.shape != (n,) + sse.shape:
        raise ValueError(f"{who} REFUSED: same_K has shape {sq.shape}, expected {(n,) + sse.shape}")
    if not np.isfinite(sq).all() or (sq < 0).any():
        raise ValueError(f"{who} REFUSED: same_K carries a non-finite or negative square")
    sums = sq.sum(0)
    if not np.allclose(sums, sse, rtol=SUM_RTOL, atol=SUM_RTOL * max(1.0, float(np.abs(sse).max()))):
        worst = float(np.max(np.abs(sums - sse) / np.maximum(np.abs(sse), 1e-300)))
        raise ValueError(f"{who} REFUSED: per-token squares do not sum to the recorded SSE (worst rel {worst:.2e})")
    delta = centered_delta(sq, sst, n)
    dt = token_mean(delta)
    one_minus_r2 = 1.0 - float(rec["same_K_r2_layer_mean"])
    if abs(float(dt.mean()) - one_minus_r2) > BRIDGE_ATOL:
        raise ValueError(f"{who} REFUSED: mean centered deviation {dt.mean():.8f} does not reproduce "
                         f"1 - R^2 = {one_minus_r2:.8f}")
    return {"n": n, "fstar": {tau_key(t): f_star(dt, float(t)) for t in taus},
            "delta_mean": float(dt.mean()), "one_minus_r2": one_minus_r2,
            "report_sha256": sha256_file_bytes(report), "pertoken_file": pt.name, "pertoken_sha256": pt_sha}
