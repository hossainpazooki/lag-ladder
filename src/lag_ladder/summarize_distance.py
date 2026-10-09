"""Recompute the weight-distance ladder from what is on disk, and refuse on any disagreement.

Nothing here talks to an instrument. For every job the plan names, the score record and its per-token file are
read again (`fstar_record.evaluate_job`) and f*(tau) recomputed; the run record must name the same files by the
same hashes. Each rung's distance record is read again and its weight-file hashes checked against the checkpoint
manifests the two dumps carry, so the x-axis is the distance between the bytes that were dumped. Then, per
distinct writer ref, the identity and scrambled controls; a writer whose controls failed gets no rung table.
Rungs are reported sorted by relative distance with the median f*(tau_K), its band under the registered rule,
the tau ladder and the median 1 - R^2 beside it.

Every f* is linear-ceiling 0023's oracle selective-recompute fraction: an oracle LOWER BOUND on real selective
recompute (restored-exactly assumption; no error propagation through the reused cache).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lag_ladder import REPO_ROOT
from lag_ladder.config import DistanceConfig, load_distance_config
from lag_ladder.distance import DISTANCE_FILE, RUN_RECORD, DumpRef, Plan, dump_dir, job_dir, plan, rung_dir
from lag_ladder.fstar_record import evaluate_job, tau_key
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.pertoken import band_outcome

SUMMARY = "summary.json"
ORACLE_SENTENCE = ("f* is an oracle lower bound on real selective recompute: it assumes a recomputed token is "
                   "restored exactly, and it ignores that real partial prefill recomputes the selected tokens "
                   "against the reused cache of the others.")


def _refuse(msg: str) -> ValueError:
    return ValueError(f"summarize REFUSED: {msg}")


def _manifest_weights(cfg: DistanceConfig, ref: DumpRef, seq: int) -> dict:
    """{file name: sha256} of the weight files the dump's checkpoint manifest names."""
    m = dump_dir(cfg, ref, seq) / "checkpoint_manifest.json"
    if not m.exists():
        raise _refuse(f"{m} is missing; the dump carries no checkpoint manifest")
    rec = json.loads(m.read_text(encoding="utf-8"))
    return {f["path"]: f["sha256"] for f in rec.get("files", []) if str(f.get("path", "")).endswith(".safetensors")}


def _check_distance(cfg: DistanceConfig, rung, seq: int, drec: dict) -> None:
    for side, ref in (("writer", DumpRef.of(rung.pair, rung.writer)), ("reader", DumpRef.of(rung.pair, rung.reader))):
        read = drec.get(side, {}).get("files") or {}
        if not read:
            raise _refuse(f"rung {rung.label}: distance record names no {side} weight files")
        manifest = _manifest_weights(cfg, ref, seq)
        for name, sha in read.items():
            if manifest.get(name) != sha:
                raise _refuse(f"rung {rung.label}: the distance was computed over {side} file {name} "
                              f"{sha[:12]}, but the dump's manifest names {str(manifest.get(name))[:12]}")


def summarize(cfg: DistanceConfig, p: Plan | None = None) -> dict:
    p = p or plan(cfg)
    rec_path = cfg.results_dir / RUN_RECORD
    if not rec_path.exists():
        raise _refuse(f"no {RUN_RECORD} under {cfg.results_dir}; nothing has run")
    run_rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if run_rec.get("registered_by") != cfg.registered_by or run_rec.get("upstream_sha") != cfg.upstream_sha:
        raise _refuse("the run record was written under another registration or pin than the config names")
    by_dir = {j["dir"]: j for j in run_rec.get("jobs", [])}
    taus = [float(t) for t in cfg.rule["tau_ladder"]]
    tk = tau_key(cfg.rule["tau_K"])
    jobs = {}
    for j in p.jobs:
        d = job_dir(cfg, j)
        rel = d.relative_to(cfg.results_dir).as_posix()
        if rel not in by_dir:
            raise _refuse(f"{rel} is in the plan but not in {RUN_RECORD}; the run is incomplete (halted: "
                          f"{run_rec.get('halted')!r})")
        ev = evaluate_job(d, taus)
        if ev["report_sha256"] != by_dir[rel]["report_sha256"] or ev["pertoken_sha256"] != by_dir[rel]["pertoken_sha256"]:
            raise _refuse(f"{rel}: the files on disk are not the ones {RUN_RECORD} names")
        jobs[j] = ev
    lo = cfg.heldout[0]
    out = {"registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha, "tau_K": float(cfg.rule["tau_K"]),
           "tau_ladder": taus, "rule": dict(cfg.rule), "controls": dict(cfg.controls), "n_jobs": len(jobs),
           "oracle": ORACLE_SENTENCE, "writers": {}, "rungs": []}
    for w in p.writers:
        ident = {j.seq: jobs[j]["fstar"][tk] for j in p.jobs if j.kind == "identity" and j.src == w}
        scr = {j.seq: jobs[j]["fstar"][tk] for j in p.jobs if j.kind == "scrambled" and j.src == w}
        id_ok = all(v <= float(cfg.controls["identity_max"]) for v in ident.values())
        med = float(np.median(list(scr.values())))
        scr_ok = med >= float(cfg.controls["scrambled_min"])
        out["writers"][w.label()] = {"identity": {"per_seq": ident, "max": max(ident.values()), "passed": id_ok},
                                     "scrambled": {"per_seq": scr, "median": med, "passed": scr_ok},
                                     "controls_passed": id_ok and scr_ok}
    for r in cfg.rungs:
        w = DumpRef.of(r.pair, r.writer)
        dfile = rung_dir(cfg, r.label) / DISTANCE_FILE
        if not dfile.exists():
            raise _refuse(f"rung {r.label}: no {DISTANCE_FILE}")
        rr = run_rec["rungs"].get(r.label) or {}
        if rr.get("distance_sha256") != sha256_file_bytes(dfile):
            raise _refuse(f"rung {r.label}: {DISTANCE_FILE} is not the one {RUN_RECORD} names")
        drec = json.loads(dfile.read_text(encoding="utf-8"))
        _check_distance(cfg, r, lo, drec)
        cells = [jobs[j] for j in p.jobs if j.kind == "rung" and j.label == r.label]
        entry = {"label": r.label, "kind": r.kind, "pair": r.pair, "writer": w.label(),
                 "reader": DumpRef.of(r.pair, r.reader).label(), "n_seqs": len(cells),
                 "rel_delta_all": float(drec["rel_delta_all"]), "rel_delta_k_proj": drec.get("rel_delta_k_proj"),
                 "target_rel_norm": r.rel_norm, "realized_rel_norm": rr.get("realized_rel_norm"),
                 "controls_passed": out["writers"][w.label()]["controls_passed"],
                 "per_seq_fstar_tau_K": [c["fstar"][tk] for c in cells],
                 "per_seq_one_minus_r2": [c["one_minus_r2"] for c in cells],
                 "median_one_minus_r2": float(np.median([c["one_minus_r2"] for c in cells]))}
        if entry["controls_passed"]:
            med = {tau_key(t): float(np.median([c["fstar"][tau_key(t)] for c in cells])) for t in taus}
            entry["median_fstar"] = med
            entry["band"] = band_outcome(med[tk], cfg.rule)
        else:
            entry["median_fstar"] = None
            entry["band"] = None
        out["rungs"].append(entry)
    out["rungs"].sort(key=lambda e: e["rel_delta_all"])
    return out


def render(s: dict) -> str:
    lines = [f"weight-distance ladder (entry {s['registered_by']}, instrument {s['upstream_sha'][:12]}): median over "
             f"held-out sequences of f*(tau_K = {s['tau_K']}), K read-out, by relative weight distance"]
    for label, w in s["writers"].items():
        lines.append(f"writer {label}: identity max f* {w['identity']['max']:.6f} ({'ok' if w['identity']['passed'] else 'FAILED'}); "
                     f"scrambled median f* {w['scrambled']['median']:.4f} ({'ok' if w['scrambled']['passed'] else 'FAILED'})")
    lines.append("  rel distance  k_proj     kind   rung                          median f*(tau_K)  band        "
                 + "  ".join(f"f*({tau_key(t)})" for t in s["tau_ladder"][1:]) + "  median 1-R^2 (K)")
    for e in s["rungs"]:
        kp = f"{e['rel_delta_k_proj']:.2e}" if e["rel_delta_k_proj"] is not None else "   -    "
        if e["band"] is None:
            lines.append(f"  {e['rel_delta_all']:.3e}     {kp}  {e['kind']:5s}  {e['label']:28s}  (writer's controls failed; no reading)")
            continue
        rest = "  ".join(f"{e['median_fstar'][tau_key(t)]:.4f}" for t in s["tau_ladder"][1:])
        lines.append(f"  {e['rel_delta_all']:.3e}     {kp}  {e['kind']:5s}  {e['label']:28s}  "
                     f"{e['median_fstar'][tau_key(s['tau_K'])]:.4f}            {e['band']:<10s}  {rest}  {e['median_one_minus_r2']:.3e}")
    lines.append(s["oracle"])
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="recompute the weight-distance ladder from disk")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "distance.toml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_distance_config(Path(a.config), REPO_ROOT)
    try:
        s = summarize(cfg)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else cfg.results_dir / SUMMARY
    out.write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(render(s))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
