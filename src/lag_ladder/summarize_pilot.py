"""Recompute the pilot's ladder from what is on disk, and refuse on any disagreement.

Nothing here talks to an instrument. For every job the plan names, the score record and its per-token
file are read again (`fstar_record.evaluate_job`: hash, sums, the exact bridge to 1 - R^2) and f*(tau)
is recomputed; the run record the driver wrote must name the same files by the same hashes. Then:

  * identity: every held-out sequence must read f*(tau_K) at most `identity_max`;
  * scrambled: the median f*(tau_K) over held-out sequences must be at least `scrambled_min`;
  * if either fails for an anchor, no ladder table is produced for it -- the instrument cannot see
    the cache, and a number read off it would be a number about the instrument;
  * otherwise, per lag, the median over held-out sequences of f*(tau_K) and its band under the
    registered rule, with the tau ladder beside it.

Every f* is the oracle selective-recompute fraction of linear-ceiling 0023: an oracle LOWER BOUND on
real selective recompute, because it assumes a recomputed token is restored exactly and ignores that
real partial prefill recomputes the selected tokens against the reused cache of the others.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lag_ladder import REPO_ROOT
from lag_ladder.config import PilotConfig, load_pilot_config
from lag_ladder.fstar_record import evaluate_job, tau_key
from lag_ladder.pertoken import band_outcome
from lag_ladder.pilot import RUN_RECORD, Plan, pair_dir, plan, score_dir

SUMMARY = "summary.json"
ORACLE_SENTENCE = ("f* is an oracle lower bound on real selective recompute: it assumes a recomputed token is "
                   "restored exactly, and it ignores that real partial prefill recomputes the selected tokens "
                   "against the reused cache of the others.")


def _refuse(msg: str) -> ValueError:
    return ValueError(f"summarize REFUSED: {msg}")


def summarize(cfg: PilotConfig, p: Plan | None = None) -> dict:
    p = p or plan(cfg)
    rec_path = pair_dir(cfg) / RUN_RECORD
    if not rec_path.exists():
        raise _refuse(f"no {RUN_RECORD} under {pair_dir(cfg)}; nothing has run")
    run_rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if run_rec.get("registered_by") != cfg.registered_by or run_rec.get("upstream_sha") != cfg.upstream_sha:
        raise _refuse("the run record was written under another registration or pin than the config names")
    by_dir = {j["dir"]: j for j in run_rec.get("jobs", [])}
    taus = [float(t) for t in cfg.rule["tau_ladder"]]
    tk = tau_key(cfg.rule["tau_K"])
    jobs = {}
    for s in p.scores:
        d = score_dir(cfg, s)
        rel = d.relative_to(cfg.results_dir).as_posix()
        if rel not in by_dir:
            raise _refuse(f"{rel} is in the plan but not in {RUN_RECORD}; the run is incomplete (halted: "
                          f"{run_rec.get('halted')!r})")
        ev = evaluate_job(d, taus)
        if ev["report_sha256"] != by_dir[rel]["report_sha256"] or ev["pertoken_sha256"] != by_dir[rel]["pertoken_sha256"]:
            raise _refuse(f"{rel}: the files on disk are not the ones {RUN_RECORD} names")
        jobs[s] = ev
    out = {"pair": cfg.pair, "registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha,
           "tau_K": float(cfg.rule["tau_K"]), "tau_ladder": taus, "rule": dict(cfg.rule), "controls": dict(cfg.controls),
           "n_jobs": len(jobs), "oracle": ORACLE_SENTENCE, "anchors": {}}
    for a in cfg.anchors:
        ident = {s.seq: jobs[s]["fstar"][tk] for s in p.scores if s.kind == "identity" and s.anchor == a}
        scr = {s.seq: jobs[s]["fstar"][tk] for s in p.scores if s.kind == "scrambled" and s.anchor == a}
        id_ok = all(v <= float(cfg.controls["identity_max"]) for v in ident.values())
        scr_med = float(np.median(list(scr.values())))
        scr_ok = scr_med >= float(cfg.controls["scrambled_min"])
        entry = {"identity": {"per_seq": ident, "max": max(ident.values()), "passed": id_ok},
                 "scrambled": {"per_seq": scr, "median": scr_med, "passed": scr_ok},
                 "controls_passed": id_ok and scr_ok, "ladder": None}
        if id_ok and scr_ok:
            ladder = []
            for r in sorted({s.partner for s in p.scores if s.kind == "ladder" and s.anchor == a},
                            key=lambda x: int(x.split("_")[1])):
                cells = [jobs[s] for s in p.scores if s.kind == "ladder" and s.anchor == a and s.partner == r]
                lag = next(s.lag for s in p.scores if s.kind == "ladder" and s.anchor == a and s.partner == r)
                med = {tau_key(t): float(np.median([c["fstar"][tau_key(t)] for c in cells])) for t in taus}
                ladder.append({"partner": r, "lag": lag, "n_seqs": len(cells),
                               "per_seq_fstar_tau_K": [c["fstar"][tk] for c in cells],
                               "median_fstar": med, "band": band_outcome(med[tk], cfg.rule),
                               # descriptive, in R^2's own units (0023): where the whole-cache deviation sits
                               # when f* saturates at 0, i.e. the mean centered deviation is already under tau
                               "per_seq_one_minus_r2": [c["one_minus_r2"] for c in cells],
                               "median_one_minus_r2": float(np.median([c["one_minus_r2"] for c in cells]))})
            entry["ladder"] = ladder
        out["anchors"][a] = entry
    return out


def render(summary: dict) -> str:
    lines = [f"pilot {summary['pair']} (entry {summary['registered_by']}, instrument {summary['upstream_sha'][:12]}): "
             f"median over held-out sequences of f*(tau_K = {summary['tau_K']}), K read-out"]
    for a, e in summary["anchors"].items():
        lines.append(f"anchor {a}: identity max f* {e['identity']['max']:.6f} "
                     f"({'ok' if e['identity']['passed'] else 'FAILED'}); scrambled median f* "
                     f"{e['scrambled']['median']:.4f} ({'ok' if e['scrambled']['passed'] else 'FAILED'})")
        if not e["controls_passed"]:
            lines.append("  no ladder table: a control failed, so the instrument cannot see the cache here")
            continue
        lines.append("  lag     median f*(tau_K)  band        " + "  ".join(f"f*({tau_key(t)})" for t in summary["tau_ladder"][1:])
                     + "  median 1-R^2 (K)")
        for c in e["ladder"]:
            rest = "  ".join(f"{c['median_fstar'][tau_key(t)]:.4f}" for t in summary["tau_ladder"][1:])
            lines.append(f"  {c['lag']:<7d} {c['median_fstar'][tau_key(summary['tau_K'])]:.4f}            {c['band']:<10s}  {rest}"
                         f"  {c['median_one_minus_r2']:.3e}")
    lines.append(summary["oracle"])
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="recompute the pilot's f* ladder from disk")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "pilot.toml"))
    ap.add_argument("--out", default=None, help=f"where to write the summary (default: <pair dir>/{SUMMARY})")
    a = ap.parse_args(argv)
    cfg = load_pilot_config(Path(a.config), REPO_ROOT)
    try:
        s = summarize(cfg)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else pair_dir(cfg) / SUMMARY
    out.write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(render(s))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
