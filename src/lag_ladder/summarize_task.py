"""Statistic (C) recomputed from the task records on disk (design §15): per lag the paired accuracy difference
STALE − FRESH in points with a seeded paired bootstrap over prompts carrying the interval, the (C) bands with the sign
fixed, the location clause the seal may use at g = 1, and the descriptives — answer-flip rate, no-answer rate, first
divergence, the rank correlation of per-prompt flips with (B1) on the fresh path. Pure functions first; the run-level
summarizer refuses on any mismatch and is built with the driver."""
from __future__ import annotations

import numpy as np

from lag_ladder.rng import make_rng


def paired_delta(stale_correct, fresh_correct, *, reps: int, seed: int) -> dict:
    """Δ = acc(STALE) − acc(FRESH) in points over paired prompts; a seeded paired bootstrap (resampling prompts
    with replacement) gives the 2.5/97.5 percentile interval."""
    s = np.asarray(stale_correct, dtype=bool)
    f = np.asarray(fresh_correct, dtype=bool)
    if s.shape != f.shape or s.ndim != 1 or s.size == 0:
        raise ValueError("paired arrays of equal non-zero length required")
    n = int(s.size)
    d = s.astype(np.float64) - f.astype(np.float64)
    point = float(100.0 * d.mean())
    rng = make_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    boots = 100.0 * d[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return {"n": n, "delta_points": point, "ci_low": lo, "ci_high": hi, "reps": int(reps), "seed": int(seed),
            "discordant": int((s != f).sum()), "acc_stale": float(100.0 * s.mean()), "acc_fresh": float(100.0 * f.mean())}


def band(delta: float, ci_low: float, ci_high: float, rule: dict) -> str:
    """HOLDS_C: |Δ| ≤ holds_abs_max with the interval inside ±holds_ci_abs_max; DEGRADES_C: Δ ≤ degrades_max with the
    interval's upper end ≤ degrades_ci_max; UNRESOLVED otherwise."""
    if abs(delta) <= float(rule["holds_abs_max"]) and max(abs(ci_low), abs(ci_high)) <= float(rule["holds_ci_abs_max"]):
        return "HOLDS_C"
    if delta <= float(rule["degrades_max"]) and ci_high <= float(rule["degrades_ci_max"]):
        return "DEGRADES_C"
    return "UNRESOLVED"


def location_ok(*, ci_low: float, ci_high: float, lower_bound: float) -> bool:
    """The seal's location clause: the interval contains 0 and its lower end lies above lower_bound."""
    return ci_low <= 0.0 <= ci_high and ci_low > lower_bound


def flip_rate(stale_answers, fresh_answers) -> float:
    a, b = list(stale_answers), list(fresh_answers)
    if len(a) != len(b) or not a:
        raise ValueError("paired answer lists of equal non-zero length required")
    return float(sum(x != y for x, y in zip(a, b)) / len(a))


def no_answer_rate(answers) -> float:
    a = list(answers)
    return float(sum(x is None for x in a) / len(a)) if a else 0.0


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def rank_correlation(a, b) -> float | None:
    """Spearman's rho with average ranks on ties; None when either side is constant."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size != y.size or x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


# --- the run-level summarizer ---------------------------------------------------------------------------------------

import json
import sys
from pathlib import Path

from lag_ladder import REPO_ROOT
from lag_ladder.config import TaskConfig, load_task_config
from lag_ladder.hashing import sha256_file_bytes

SUMMARY = "summary.json"
PREDICTIONS = Path("ledger") / "predictions"
RUN_RECORD = "run.json"
ARMS = ("stale", "near", "far")


def _refuse(msg: str) -> ValueError:
    return ValueError(f"summarize_task REFUSED: {msg}")


def _load_jsonl(path: Path, named_sha: str) -> dict[int, dict]:
    if not path.exists():
        raise _refuse(f"missing {path}")
    got = sha256_file_bytes(path)
    if got != named_sha:
        raise _refuse(f"{path} hashes to {got[:12]}, run.json names {named_sha[:12]} (sha256 mismatch)")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("schema") != "task-record-v2":
            raise _refuse(f"{path}: a record is not task-record-v2")
        if r["index"] in rows:
            raise _refuse(f"{path}: index {r['index']} twice")
        rows[int(r["index"])] = r
    return rows


def _arm_stats(fresh: dict[int, dict], arm: dict[int, dict], cfg: TaskConfig) -> dict:
    idx = sorted(fresh)
    if sorted(arm) != idx:
        raise _refuse("an arm's prompt indices differ from FRESH's")
    f_ok = [fresh[i]["correct"] for i in idx]
    a_ok = [arm[i]["correct"] for i in idx]
    d = paired_delta(a_ok, f_ok, reps=int(cfg.rule["bootstrap_reps"]), seed=cfg.seed)
    flips = [int(arm[i]["answer"] != fresh[i]["answer"]) for i in idx]
    b1 = [arm[i].get("b1_fresh_path") for i in idx]
    divs = [arm[i]["first_divergence"] for i in idx if arm[i]["first_divergence"] is not None]
    rho = rank_correlation(flips, b1) if all(x is not None for x in b1) else None
    return {"delta": d, "band": band(d["delta_points"], d["ci_low"], d["ci_high"], cfg.rule),
            "flip_rate": flip_rate([arm[i]["answer"] for i in idx], [fresh[i]["answer"] for i in idx]),
            "no_answer_rate": no_answer_rate([arm[i]["answer"] for i in idx]),
            "cap_rate": float(sum(bool(arm[i]["reached_cap"]) for i in idx) / len(idx)),
            "mean_first_divergence": (float(np.mean(divs)) if divs else None),
            "diverged_rate": float(len(divs) / len(idx)),
            "rank_correlation_b1": rho}


def summarize(cfg: TaskConfig, root: Path | None = None, out: Path | None = None) -> dict:
    """Recompute everything from the JSONL records; refuse on any mismatch before writing."""
    root = Path(root) if root is not None else cfg.results_dir
    rec_path = root / RUN_RECORD
    if not rec_path.exists():
        raise _refuse(f"no {RUN_RECORD} under {root}; nothing has run")
    run_rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if run_rec.get("registered_by") != cfg.registered_by or run_rec.get("upstream_sha") != cfg.upstream_sha:
        raise _refuse("the run record was written under another registration or pin than the config names")
    if run_rec.get("halted"):
        raise _refuse(f"the run halted: {run_rec['halted']}")
    summary = {"registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha, "rule": dict(cfg.rule),
               "controls_cfg": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in cfg.controls.items()},
               "equivalence": run_rec.get("equivalence"), "lags": {}, "seal_readout": None}
    if not summary["equivalence"] or not summary["equivalence"].get("passed"):
        raise _refuse("the dump-equivalence control did not pass; nothing below is comparable with (B)")
    for j in run_rec.get("jobs", []):
        if j["kind"] != "cells":
            continue
        if not j.get("done"):
            raise _refuse(f"job {j['dir']} is not done")
        d = root / j["dir"]
        shas = j["files_sha256"]
        fresh = _load_jsonl(d / "fresh.jsonl", shas["fresh.jsonl"])
        ctl = run_rec["controls"].get(j["reader"])
        if ctl is None:
            raise _refuse(f"{j['reader']}: no control judgement in run.json")
        f32 = _load_jsonl(d / "fresh32.jsonl", shas["fresh32.jsonl"])
        f2 = _load_jsonl(d / "fresh2.jsonl", shas["fresh2.jsonl"])
        floor_rate = float(sum(f32[i]["answer"] != fresh[i]["answer"] for i in f32) / len(f32)) if f32 else None
        ident = int(sum(f2[i]["completion_sha256"] != fresh[i]["completion_sha256"] for i in f2))
        if floor_rate is None or abs(floor_rate - ctl["floor_flip_rate"]) > 1e-12 or ident != ctl["identity_flips"]:
            raise _refuse(f"{j['reader']}: the controls recomputed from the records differ from run.json")
        lag = {"n": len(fresh), "acc_fresh": float(100.0 * np.mean([fresh[i]["correct"] for i in fresh])),
               "controls": {"n": len(f32), "floor_flip_rate": floor_rate,
                            "floor_passed": floor_rate <= float(cfg.controls["floor_max_flip_rate"]),
                            "identity_flips": ident, "identity_passed": ident == 0},
               "arms": {}}
        for arm in ARMS:
            lag["arms"][arm] = _arm_stats(fresh, _load_jsonl(d / f"{arm}.jsonl", shas[f"{arm}.jsonl"]), cfg)
        fire = all(lag["arms"][a]["band"] == "DEGRADES_C" for a in ("near", "far"))
        lag["positive_controls_fire"] = fire
        lag["evaluable"] = bool(fire and lag["controls"]["floor_passed"] and lag["controls"]["identity_passed"])
        if not lag["evaluable"]:
            lag["arms"]["stale"]["band"] = None          # UNEVALUABLE: the descriptives stay, the verdict is withheld
        summary["lags"][j["reader"]] = lag
    summary["seal_readout"] = seal_readout(summary, root.parent.parent, cfg.seal)
    target = Path(out) if out is not None else root / SUMMARY
    target.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def seal_readout(summary: dict, repo: Path, seal: str = "ownrun-task") -> dict | None:
    """The sealed clauses of section 15 read against the recomputed statistics: (i) NEAR and FAR DEGRADES_C at every
    named reader; (ii) at the first lag the STALE interval contains 0 with its lower end above a bound; (iii) the STALE
    answer-flip rate at the last lag at least a value. UNREAD when an input is withheld or missing."""
    f = Path(repo) / PREDICTIONS / f"{seal}.json"
    if not f.exists():
        return None
    P = json.loads(f.read_text(encoding="utf-8")).get("prediction", {}).get("predictions", {})
    out: dict = {"source": (PREDICTIONS / f"{seal}.json").as_posix(), "clauses": {}}
    lags = summary.get("lags", {})

    def verdict(checks):
        vals = [c for c in checks if c is not None]
        if len(vals) < len(checks) or not vals:
            return "UNREAD"
        return "HELD" if all(vals) else "FALSIFIED"

    i = P.get("i_positive_controls_fire")
    if i:
        fires = {r: (lags.get(r) or {}).get("positive_controls_fire") for r in i["readers"]}
        out["clauses"]["i"] = {"readers": list(i["readers"]), "fire": fires, "verdict": verdict(list(fires.values()))}
    ii = P.get("ii_location_at_first_lag")
    if ii:
        lag = lags.get(ii["reader"]) or {}
        d = ((lag.get("arms") or {}).get("stale") or {}).get("delta")
        ok = (None if not lag.get("evaluable") or d is None
              else location_ok(ci_low=d["ci_low"], ci_high=d["ci_high"], lower_bound=float(ii["lower_bound"])))
        out["clauses"]["ii"] = {"reader": ii["reader"], "lower_bound": float(ii["lower_bound"]),
                                "interval": None if d is None else [d["ci_low"], d["ci_high"]], "verdict": verdict([ok])}
    iii = P.get("iii_flip_rate_at_last_lag")
    if iii:
        lag = lags.get(iii["reader"]) or {}
        fr = ((lag.get("arms") or {}).get("stale") or {}).get("flip_rate")
        ok = None if not lag.get("evaluable") or fr is None else fr >= float(iii["at_least"])
        out["clauses"]["iii"] = {"reader": iii["reader"], "at_least": float(iii["at_least"]), "flip_rate": fr,
                                 "verdict": verdict([ok])}
    return out


def render(s: dict) -> str:
    lines = [f"statistic (C) (entry {s['registered_by']}, instrument {s['upstream_sha'][:12]}): greedy final-answer "
             "accuracy under an injected prompt cache; delta = STALE - FRESH in points, paired bootstrap interval"]
    eq = s["equivalence"]
    lines.append(f"dump equivalence: max_abs_diff {eq['max_abs_diff']:.3e} (limit {eq['limit']}), (B1) rel gap "
                 f"{eq['b1_rel_gap']} -> {'ok' if eq['passed'] else 'FAILED'}")
    for reader, lag in s["lags"].items():
        c = lag["controls"]
        lines.append(f"reader {reader}: n {lag['n']} acc_fresh {lag['acc_fresh']:.1f}; floor flip {c['floor_flip_rate']:.3f} "
                     f"{'ok' if c['floor_passed'] else 'FAILED'}; identity {'ok' if c['identity_passed'] else 'FAILED'}; "
                     f"positive controls {'fire' if lag['positive_controls_fire'] else 'DO NOT FIRE'}"
                     + ("" if lag["evaluable"] else "  -> UNEVALUABLE"))
        for arm, a in lag["arms"].items():
            d = a["delta"]
            lines.append(f"  {arm:6s} delta {d['delta_points']:+6.2f} [{d['ci_low']:+6.2f}, {d['ci_high']:+6.2f}]  "
                         f"flips {a['flip_rate']:.3f}  no-answer {a['no_answer_rate']:.3f}  cap {a['cap_rate']:.3f}  "
                         f"{a['band'] or 'UNEVALUABLE'}")
    ro = s.get("seal_readout")
    if ro:
        lines.append(f"sealed clauses ({ro['source']}):")
        for k, c in ro["clauses"].items():
            detail = {kk: vv for kk, vv in c.items() if kk != "verdict"}
            lines.append(f"  ({k}) {c['verdict']}  {json.dumps(detail, default=str)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="recompute statistic (C) from the task records on disk")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "ownrun_task.toml"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_task_config(Path(a.config), REPO_ROOT)
    root = Path(a.root) if a.root else cfg.results_dir
    try:
        out = Path(a.out) if a.out else root / SUMMARY
        s = summarize(cfg, root, out)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(render(s))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
