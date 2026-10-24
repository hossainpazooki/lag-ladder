"""Recompute statistic (B) "does the reader notice" from what is on disk, and refuse on any disagreement.

Nothing here talks to a model. Every number is recomputed from the `report.json` files the scorer wrote
(design doc section 13, ledger 0011): (B1) the mean per-token KL(pi_fresh || pi_stale) per sequence and its
median over sequences, per rung and text, with the per-token p90 and maximum beside it; (B2) the per-sequence
log importance weight sum(logprob_fresh - logprob_stale) and ESS/N over sequences; (B3) the same dispersion
at token level; (B4) the argmax agreement rate. (B2) and (B3) are importance-weight dispersion on corpus
text, not a trainer's number. Each report's continuation hash is verified against its stored tokens (sha256
over the token ids as little-endian int64 bytes), a corpus continuation is verified against the token file
itself, and the file on disk must be the one `run.json` names. Controls are judged per reader (float16
floor, identity, scrambled); a rung whose reader's controls failed, or whose floor limits are unset or
exceeded, is withheld (its per-sequence values are reported, its median and band are not). A rung whose
(B1) median sits within 2x the reader's floor KL of the same text is marked at the floor. Each rung is tied
to the checkpoint manifest digests its dumps carry. Bands per text: HOLDS_B <= holds_max, DEGRADES_B >=
degrades_min, UNRESOLVED between.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from lag_ladder import REPO_ROOT
from lag_ladder.config import ReaderConfig, load_reader_config
from lag_ladder.hashing import sha256_file_bytes

RUN_RECORD = "run.json"
DISTANCE_RESULTS = Path("results") / "distance"   # the distance ladder's record: rungs/<label>/distance.json, checkpoints/<name>/provenance.json
PREDICTIONS = Path("ledger") / "predictions"      # <cfg.seal>.json: the sealed clauses this config runs against
REPORT = "report.json"
SUMMARY = "summary.json"
SCHEMA = "reader-report-v1"
B2_LABEL = "importance-weight dispersion on corpus text"
REPORT_KEYS = frozenset((
    "schema", "reader", "fresh_dump", "stale_dump", "tokens_file", "seq", "prefix_len", "text",
    "generate_under", "live", "continuation_tokens", "continuation_sha256", "logprob_fresh", "logprob_stale",
    "kl_per_token", "argmax_fresh", "argmax_stale", "max_abs_logit_diff", "inputs_sha256",
    "reader_checkpoint", "device", "torch_version", "peak_cuda_mem_bytes", "gen_seconds"))


def _refuse(msg: str) -> ValueError:
    return ValueError(f"summarize_reader REFUSED: {msg}")


def continuation_sha256(tokens) -> str:
    """The scorer's continuation hash: sha256 over the scored token ids as little-endian int64 bytes."""
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def _resolve(raw: str, root: Path, upstream: Path) -> Path:
    """The file a report names. An absolute path from the run machine that does not exist here (the reports
    are pulled from a pod) is matched by its trailing components under the results root, the repository the
    results belong to and the instrument checkout the config pins; the caller's sha256 check decides whether
    the file found is the one the report hashed."""
    p = Path(raw)
    repo = root.parent.parent   # the checkout whose results these are (REPO_ROOT, or a test repo)
    bases = (root, repo, upstream)
    if p.is_absolute() or raw.startswith("/"):   # a POSIX path is not `absolute` to pathlib on Windows
        if p.exists():
            return p
        for n in (3, 2, 1):
            tail = Path(*p.parts[-n:])
            for base in bases:
                if (base / tail).exists():
                    return base / tail
        return p
    for base in bases[:2]:
        if (base / p).exists():
            return base / p
    return root / p


def _input_sha(rep: dict, raw: str) -> str | None:
    """The recorded sha256 for one input path, matched exactly or by trailing path components."""
    inputs = rep.get("inputs_sha256") or {}
    if raw in inputs:
        return inputs[raw]
    name = Path(raw).name
    hits = [v for k, v in inputs.items() if Path(k).name == name]
    return hits[0] if len(hits) == 1 else None


def _load_report(d: Path, named_sha: str, root: Path, upstream: Path) -> dict:
    f = d / REPORT
    if not f.exists():
        raise _refuse(f"{f} is missing; {RUN_RECORD} names a job with no report")
    if sha256_file_bytes(f) != named_sha:
        raise _refuse(f"{f} is not the file {RUN_RECORD} names")
    rep = json.loads(f.read_text(encoding="utf-8"))
    if rep.get("schema") != SCHEMA:
        raise _refuse(f"{f}: schema {rep.get('schema')!r} is not {SCHEMA!r}")
    missing, extra = REPORT_KEYS - rep.keys(), rep.keys() - REPORT_KEYS
    if missing or extra:
        raise _refuse(f"{f}: report keys differ from the contract (missing {sorted(missing)}, extra {sorted(extra)})")
    toks = rep["continuation_tokens"]
    n = len(toks)
    if n == 0:
        raise _refuse(f"{f}: empty continuation")
    for key in ("logprob_fresh", "logprob_stale", "kl_per_token", "argmax_fresh", "argmax_stale"):
        if len(rep[key]) != n:
            raise _refuse(f"{f}: {key} has {len(rep[key])} entries for a {n}-token continuation")
    if continuation_sha256(toks) != rep["continuation_sha256"]:
        raise _refuse(f"{f}: continuation_sha256 does not match the stored tokens")
    if rep["text"] in ("corpus", "rollout", "agentic"):   # a stored continuation: it must be the token file's own
        tf = _resolve(rep["tokens_file"], root, upstream)
        if not tf.exists():
            raise _refuse(f"{f}: token file {tf} is missing")
        named = _input_sha(rep, rep["tokens_file"])
        if named is not None and sha256_file_bytes(tf) != named:
            raise _refuse(f"{f}: {tf} is not the token file the report hashed")
        seq_tokens = np.load(tf)[int(rep["seq"])]
        pl = int(rep["prefix_len"])
        if not np.array_equal(np.asarray(toks, dtype=np.int64), seq_tokens[pl:pl + n].astype(np.int64)):
            raise _refuse(f"{f}: corpus continuation is not tokens[{rep['seq']}, {pl}:{pl + n}] of {tf.name}")
    return rep


def _rung_distance(label: str, reader: str, repo: Path) -> dict:
    """Where a rung sits on the axis (whole-vector relative L2 distance, 0007): the distance ladder's
    `distance.json` for the rung (0008's record, or a later rung's file written by the same tool and placed in
    the same layout); for a local noise reader without one, the regenerated checkpoint's provenance
    (`realized_rel_norm`). None when neither is on disk; the rung is then reported without a position."""
    kind = "noise" if ("noise-" in label or reader.startswith("local:noise-")) else "real"
    d = repo / DISTANCE_RESULTS / "rungs" / label / "distance.json"
    if d.exists():
        v = json.loads(d.read_text(encoding="utf-8")).get("rel_delta_all")
        return {"kind": kind, "rel_delta_all": v, "rel_norm": v if kind == "noise" else None,
                "distance_source": d.relative_to(repo).as_posix()}
    if reader.startswith("local:"):
        prov = repo / DISTANCE_RESULTS / "checkpoints" / reader[len("local:"):] / "provenance.json"
        if prov.exists():
            v = json.loads(prov.read_text(encoding="utf-8")).get("realized_rel_norm")
            return {"kind": kind, "rel_delta_all": v, "rel_norm": v, "distance_source": prov.relative_to(repo).as_posix()}
    return {"kind": kind, "rel_delta_all": None, "rel_norm": None, "distance_source": None}


def _sealed_backbone_rungs(repo: Path, seal: str) -> list[str] | None:
    """The backbone rungs the sealed prediction fits its slope over, when the seal names them."""
    f = repo / PREDICTIONS / f"{seal}.json"
    if not f.exists():
        return None
    try:
        return [str(r) for r in json.loads(f.read_text(encoding="utf-8"))["prediction"]["predictions"]["i_backbone_quadratic"]["rungs"]]
    except (KeyError, TypeError, ValueError):
        return None


def _digest(rep: dict, side: str, root: Path, upstream: Path) -> str | None:
    """The checkpoint manifest_digest a dump carries, read from its meta.json when the dump is on disk."""
    d = _resolve(rep[f"{side}_dump"], root, upstream) / "meta.json"
    if d.exists():
        meta = json.loads(d.read_text(encoding="utf-8"))
        return (meta.get("checkpoint") or {}).get("manifest_digest")
    return None


def _stats(reports: list[dict]) -> dict:
    """(B1)-(B4) recomputed from one rung-text cell's reports, one report per sequence."""
    # Continuations may differ in length across sequences (a stored rollout, design §14), so per-sequence
    # arrays are kept ragged: sequence-level statistics per row, token-level ones over the pooled tokens.
    kl = [np.asarray(r["kl_per_token"], dtype=np.float64) for r in reports]
    logw = [np.asarray(r["logprob_fresh"], dtype=np.float64) - np.asarray(r["logprob_stale"], dtype=np.float64)
            for r in reports]
    agree = np.array([np.mean(np.asarray(r["argmax_fresh"]) == np.asarray(r["argmax_stale"]))
                      for r in reports], dtype=np.float64)
    per_seq_mean_kl = np.array([k.mean() for k in kl])
    pooled_kl, pooled_logw = np.concatenate(kl), np.concatenate(logw)
    per_seq_logw = np.array([w.sum() for w in logw])
    return {"n_seqs": len(reports), "n_tokens": int(pooled_kl.size), "seqs": [int(r["seq"]) for r in reports],
            "b1": {"per_seq_mean_kl": per_seq_mean_kl.tolist(),
                   "median_kl": float(np.median(per_seq_mean_kl)),
                   "p90_token_kl": float(np.percentile(pooled_kl, 90)),
                   "max_token_kl": float(pooled_kl.max())},
            "b2": {"label": B2_LABEL, "per_seq_log_weight": per_seq_logw.tolist(),
                   "ess_over_n": _ess_over_n(per_seq_logw)},
            "b3": {"label": B2_LABEL, "ess_over_n": _ess_over_n(pooled_logw)},
            "b4": {"per_seq_agreement": agree.tolist(), "agreement_rate": float(agree.mean())}}


def _ess_over_n(logw: np.ndarray) -> float:
    w = np.exp(logw - logw.max())
    return float(w.sum() ** 2 / (len(w) * (w * w).sum()))


def _band(median_kl: float, rule: dict) -> str:
    if median_kl <= float(rule["holds_max"]):
        return "HOLDS_B"
    if median_kl >= float(rule["degrades_min"]):
        return "DEGRADES_B"
    return "UNRESOLVED"


def _control_key(reader: str, pair: str | None, cfg: ReaderConfig) -> str:
    """The unit a control judges: the reader (0012), or the reader on one pair (0020, per_pair)."""
    if cfg.controls.get("per_pair"):
        if not pair:
            raise _refuse(f"reader {reader}: a control job names no pair under per_pair")
        return f"{reader} @ {pair}"
    return reader


def _controls(reader: str, jobs: dict, cfg: ReaderConfig) -> dict:
    """One reader's controls judged from its floor/identity/scrambled reports. Floor limits unset => failed."""
    floor_lim_kl = cfg.controls.get("floor_max_median_kl") or None
    out = {"floor": None, "identity": None, "scrambled": None, "far": None, "far_over_scrambled": None}
    floor = [r for (kind, _t, _s), r in jobs.items() if kind == "floor"]
    if floor:
        if any(not r["live"] for r in floor):
            raise _refuse(f"reader {reader}: a floor control report was not scored --live")
        kl_by_text = {}
        for text in sorted({r["text"] for r in floor}):
            cell = [np.mean(r["kl_per_token"]) for r in floor if r["text"] == text]
            kl_by_text[text] = float(np.median(cell))
        max_logit = max(float(r["max_abs_logit_diff"]) for r in floor)
        unset = floor_lim_kl is None
        passed = not unset and all(v <= float(floor_lim_kl) for v in kl_by_text.values())   # 0013: KL only
        out["floor"] = {"median_kl_by_text": kl_by_text, "max_abs_logit_diff": max_logit,
                        "limits_set": not unset, "passed": passed}
    ident = [r for (kind, _t, _s), r in jobs.items() if kind == "identity"]
    if ident:
        if any(r["fresh_dump"] != r["stale_dump"] for r in ident):
            raise _refuse(f"reader {reader}: an identity control report names two different dumps")
        max_logit = max(float(r["max_abs_logit_diff"]) for r in ident)
        out["identity"] = {"max_abs_logit_diff": max_logit,
                           "passed": max_logit <= float(cfg.controls["identity_max_logit_diff"])}
    scr = [r for (kind, _t, _s), r in jobs.items() if kind == "scrambled"]
    if scr:
        if not cfg.controls.get("per_pair") and any(r["text"] != "corpus" for r in scr):
            raise _refuse(f"reader {reader}: a scrambled control report is not on corpus text")
        med = float(np.median([np.mean(r["kl_per_token"]) for r in scr]))
        out["scrambled"] = {"median_kl": med, "passed": med >= float(cfg.controls["scrambled_min"])}
    far = [r for (kind, _t, _s), r in jobs.items() if kind == "far"]
    if far:
        # 0022: the far prefix is another pair's dump under this pair's continuation; its limit is far_min.
        if cfg.controls.get("far_min") is None:
            raise _refuse(f"{reader}: far control reports exist but the config sets no far_min")
        if any(r["fresh_dump"] == r["stale_dump"] for r in far):
            raise _refuse(f"{reader}: a far control report names the fresh dump as its stale arm")
        med = float(np.median([np.mean(r["kl_per_token"]) for r in far]))
        out["far"] = {"median_kl": med, "passed": med >= float(cfg.controls["far_min"])}
        if out["scrambled"] and out["scrambled"]["median_kl"] > 0:
            out["far_over_scrambled"] = med / out["scrambled"]["median_kl"]
    # 0028: with evaluable_under_far, a scrambled miss is carried by a passing far control, and the unit says so.
    scr_ok = out["scrambled"] is not None and out["scrambled"]["passed"]
    far_ok = out["far"] is not None and out["far"]["passed"]
    rescued = (not scr_ok and out["scrambled"] is not None and far_ok and bool(cfg.controls.get("evaluable_under_far")))
    out["evaluable_under"] = "far" if rescued else None
    out["passed"] = (all(c is not None and c["passed"] for c in (out["floor"], out["identity"]))
                     and (scr_ok or rescued) and (out["far"] is None or far_ok))
    return out


def summarize(cfg: ReaderConfig, root: Path | None = None, out: Path | None = None) -> dict:
    """Recompute everything, then write summary.json (under root unless out names another file); a refusal
    raises before anything is written."""
    root = Path(root) if root is not None else cfg.results_dir
    rec_path = root / RUN_RECORD
    if not rec_path.exists():
        raise _refuse(f"no {RUN_RECORD} under {root}; nothing has run")
    run_rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if run_rec.get("registered_by") != cfg.registered_by or run_rec.get("upstream_sha") != cfg.upstream_sha:
        raise _refuse("the run record was written under another registration or pin than the config names")
    texts = [str(t) for t in cfg.rule["texts"]]
    texts_of = {r.label: [str(t) for t in (r.texts or texts)] for r in cfg.rungs}   # §14: per-rung texts
    # jobs: (reader, kind, label, text, seq) -> report; controls carry label None.
    by_reader: dict[str, dict] = {}
    rungs: dict[str, dict] = {}
    for j in run_rec.get("jobs", []):
        for k in ("dir", "kind", "reader", "done", "report_sha256"):
            if k not in j:
                raise _refuse(f"{RUN_RECORD} job {j.get('dir')!r} is missing {k}")
        if not j["done"]:
            raise _refuse(f"{RUN_RECORD} job {j['dir']} is not done; the run is incomplete "
                          f"(halted: {run_rec.get('halted')!r})")
        rep = _load_report(root / j["dir"], j["report_sha256"], root, cfg.upstream_path)
        key = (j["kind"], rep["text"], int(rep["seq"]))
        if j["kind"] in ("floor", "identity", "scrambled", "far"):
            cell = by_reader.setdefault(_control_key(j["reader"], j.get("pair"), cfg), {})
        elif j["kind"] == "rung":
            if "label" not in j:
                raise _refuse(f"{RUN_RECORD} rung job {j['dir']} names no rung label")
            cell = rungs.setdefault(j["label"], {"reader": j["reader"], "job": j, "reports": {}})
            if cell["reader"] != j["reader"]:
                raise _refuse(f"rung {j['label']}: jobs name two different readers")
            cell = cell["reports"]
        else:
            raise _refuse(f"{RUN_RECORD} job {j['dir']} has unknown kind {j['kind']!r}")
        if key in cell:
            raise _refuse(f"{RUN_RECORD} names {j['dir']} twice for {key}")
        cell[key] = rep
    controls = {r: _controls(r, jobs, cfg) for r, jobs in sorted(by_reader.items())}
    summary = {"registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha,
           "rule": dict(cfg.rule), "controls_cfg": dict(cfg.controls),
           "controls": controls, "rungs": [], "slope_fit": {}}
    repo = root.parent.parent
    pair_of = {r.label: r.pair for r in cfg.rungs}
    sealed = _sealed_backbone_rungs(repo, cfg.seal)
    for label, cell in sorted(rungs.items()):
        reader = cell["reader"]
        ckey = _control_key(reader, pair_of.get(label), cfg)
        ctl = controls.get(ckey)
        if ctl is None:
            raise _refuse(f"rung {label}: {ckey} has no control reports")
        meta = _rung_distance(label, reader, repo)
        reps = list(cell["reports"].values())
        digests = {side: {d for r in reps if (d := _digest(r, side, root, cfg.upstream_path)) is not None} for side in ("fresh", "stale")}
        for side, ds in digests.items():
            ds |= {rc["manifest_digest"] for r in reps if side == "fresh"
                   and isinstance(rc := r.get("reader_checkpoint"), dict) and "manifest_digest" in rc}
            if len(ds) > 1:
                raise _refuse(f"rung {label}: the {side} dumps carry conflicting manifest digests {sorted(ds)}")
        withheld = not ctl["passed"]
        entry = {"label": label, "reader": reader, "pair": pair_of.get(label), "kind": meta["kind"],
                 "rel_norm": meta["rel_norm"], "rel_delta_all": meta["rel_delta_all"], "distance_source": meta["distance_source"],
                 "manifest_digest": {s: (next(iter(d)) if d else None) for s, d in digests.items()},
                 "controls": ckey, "controls_passed": ctl["passed"], "withheld": withheld, "texts": {}}
        for text in texts_of.get(label, texts):
            trs = sorted((r for (_k, t, _s), r in cell["reports"].items() if t == text), key=lambda r: int(r["seq"]))
            if not trs:
                raise _refuse(f"rung {label}: no reports for text {text!r}")
            st = _stats(trs)
            if withheld:
                st["b1"]["median_kl"] = None
                st.update(band=None, at_floor=None)
            else:
                floor_kl = (ctl["floor"] or {}).get("median_kl_by_text", {}).get(text)
                st["band"] = _band(st["b1"]["median_kl"], cfg.rule)
                st["at_floor"] = (floor_kl is not None and st["b1"]["median_kl"] <= 2.0 * floor_kl)
                st["floor_kl"] = floor_kl
            # 0022, descriptive: the rung's median over its unit's scrambled and far medians (None without the control).
            for name in ("scrambled", "far"):
                c = ctl.get(name)
                st[f"over_{name}"] = (st["b1"]["median_kl"] / c["median_kl"]
                                      if not withheld and c and c["median_kl"] > 0 else None)
            entry["texts"][text] = st
        summary["rungs"].append(entry)
    # The slope of log median (B1) on log distance over the sealed backbone rungs (all evaluated noise rungs
    # when the seal names none), every backbone point beside it per pair, and the nearest backbone rung of
    # the same pair per real rung (by log distance).
    noise = [e for e in summary["rungs"] if e["kind"] == "noise" and not e["withheld"] and e["rel_norm"] is not None]
    # The fit runs over the rungs the seal names, whatever their kind (§14 seals a slope over real lag rungs), each
    # placed by its whole-vector distance; without a seal, over the backbone.
    if sealed is None:
        fit_rungs = noise
    else:
        fit_rungs = [e for e in summary["rungs"] if e["label"] in sealed and not e["withheld"]
                     and (e["rel_norm"] or e["rel_delta_all"]) is not None]

    def _x(e):
        return float(e["rel_norm"] if e["rel_norm"] is not None else e["rel_delta_all"])

    def _points(es, text):
        return sorted((_x(e), e["texts"][text]["b1"]["median_kl"]) for e in es
                      if text in e["texts"] and e["texts"][text]["b1"]["median_kl"] and e["texts"][text]["b1"]["median_kl"] > 0)

    # Fit every text the summary carries: the rule's texts and any text a rung carries on its own (0017's `rollout`,
    # 0030's `agentic`), so a seal whose slope clause names a per-rung text finds its fit instead of reading UNREAD.
    fit_texts = list(texts) + sorted({t for e in summary["rungs"] for t in e.get("texts", {})} - set(texts))
    for text in fit_texts:
        pts = _points(fit_rungs, text)
        fit = {"rungs": sorted(e["label"] for e in fit_rungs), "sealed_rungs": sealed is not None,
               "points": [list(p) for p in pts], "slope": None,
               "backbone_points": {str(pair): [list(p) for p in _points([e for e in noise if e["pair"] == pair], text)]
                                   for pair in sorted({e["pair"] for e in noise}, key=str)}}
        if len(pts) >= 2:
            x, y = np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts])
            fit["slope"] = float(np.polyfit(x, y, 1)[0])
        summary["slope_fit"][text] = fit
    for e in summary["rungs"]:
        e["matched_backbone"] = None
        if e["kind"] == "real" and e["rel_delta_all"]:
            cands = [(abs(np.log(float(e["rel_delta_all"])) - np.log(float(n["rel_norm"]))), n["label"])
                     for n in noise if n["pair"] == e["pair"]]
            if cands:
                e["matched_backbone"] = min(cands)[1]
    summary["rungs"].sort(key=lambda e: (e["kind"], float(e["rel_delta_all"] or e["rel_norm"] or 0.0), e["label"]))
    assert_distance_pins(summary, repo, cfg.seal)
    summary["seal_readout"] = seal_readout(summary, repo, cfg.seal)
    target = Path(out) if out is not None else root / SUMMARY
    target.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _b1(summary: dict, label: str, text: str):
    for e in summary["rungs"]:
        if e["label"] == label and not e["withheld"]:
            return e["texts"][text]["b1"]["median_kl"]
    return None


def assert_distance_pins(summary: dict, repo: Path, seal: str = "reader") -> None:
    """A seal may carry `distance_must_equal` ({rungs: {g: label}, reference: {g: rel_delta_all}, rel_tol}): rungs
    that re-read weights another entry already placed on the axis (0030: the same step-0 / step-g checkpoints as
    0018 under new labels) must sit where that entry put them. A rung without a position, or one whose re-derived
    `rel_delta_all` differs from the reference by more than rel_tol relative, REFUSES the summary: same weights and
    the same tool give the same number, so a different one is a wrong checkpoint, not a finding."""
    f = repo / PREDICTIONS / f"{seal}.json"
    if not f.exists():
        return
    pin = json.loads(f.read_text(encoding="utf-8")).get("prediction", {}).get("predictions", {}).get("distance_must_equal")
    if not pin:
        return
    tol = float(pin["rel_tol"])
    by_label = {e["label"]: e for e in summary["rungs"]}
    for g, label in pin["rungs"].items():
        ref = float(pin["reference"][g])
        e = by_label.get(label)
        if e is None:
            raise _refuse(f"distance pin: rung {label} is not in this summary")
        v = e.get("rel_delta_all")
        if v is None:
            raise _refuse(f"distance pin: rung {label} has no position (no results/distance/rungs/{label}/distance.json); "
                          f"the seal requires rel_delta_all = {ref:.6e} for it")
        if abs(float(v) - ref) > tol * abs(ref):
            raise _refuse(f"distance pin: rung {label} re-derived rel_delta_all {float(v):.9e} differs from the sealed "
                          f"reference {ref:.9e} by more than {tol:g} relative; the weights are not the ones that entry placed")


def seal_readout(summary: dict, repo: Path, seal: str = "reader") -> dict | None:
    """Each sealed clause of ledger/predictions/reader.json read against the recomputed statistics: the values
    it compares and HELD / FALSIFIED / UNREAD (an input withheld or missing). Nothing here is a number the
    summary does not already carry; this only applies the clause's comparison to them."""
    f = repo / PREDICTIONS / f"{seal}.json"
    if not f.exists():
        return None
    pred = json.loads(f.read_text(encoding="utf-8")).get("prediction", {})
    P = pred.get("predictions", {})
    rule = pred.get("rule", {})
    texts = [str(t) for t in pred.get("texts", summary["rule"]["texts"])]
    out: dict = {"source": (PREDICTIONS / f"{seal}.json").as_posix(), "clauses": {}}

    def verdict(checks):
        vals = [c for c in checks if c is not None]
        if len(vals) < len(checks) or not vals:
            return "UNREAD"
        return "HELD" if all(vals) else "FALSIFIED"

    i = P.get("i_backbone_quadratic")
    if i:
        lo, hi = (float(v) for v in i["slope_in"])
        # Only the texts the sealed rungs carry: a seal whose text list includes a rollout text (§14) still
        # fits its slope over corpus lag rungs, which have no rollout column, and that absence is not UNREAD.
        carried = {t for e in summary["rungs"] if e["label"] in set(i["rungs"]) for t in e.get("texts", {})}
        fit_texts = [t for t in texts if t in carried] or texts
        slopes = {t: summary["slope_fit"].get(t, {}).get("slope") for t in fit_texts}
        out["clauses"]["i"] = {"slope_in": [lo, hi], "slope": slopes, "rungs": list(i["rungs"]),
                               "verdict": verdict([None if v is None else lo <= v <= hi for v in slopes.values()])}
    ii = P.get("ii_backbone_holds_through")
    if ii:
        vals = {t: _b1(summary, "noise-0.01", t) for t in texts}
        hm = float(rule.get("holds_max", summary["rule"]["holds_max"]))
        out["clauses"]["ii"] = {"rung": "noise-0.01", "holds_max": hm, "median_kl": vals,
                                "verdict": verdict([None if v is None else v <= hm for v in vals.values()])}
    iii = P.get("iii_olmo_directions_above_backbone")
    if iii:
        per = {}
        checks = []
        for label in iii["rungs"]:
            e = next((e for e in summary["rungs"] if e["label"] == label), None)
            nb = e["matched_backbone"] if e else None
            row = {"matched_backbone": nb, "ratio": {}}
            for t in texts:
                a, b = _b1(summary, label, t), (_b1(summary, nb, t) if nb else None)
                r = (a / b) if (a is not None and b) else None
                row["ratio"][t] = r
                checks.append(None if r is None else r >= 10.0)
            per[label] = row
        out["clauses"]["iii"] = {"factor": 10.0, "rungs": per, "verdict": verdict(checks)}
    iv = P.get("iv_qwen_instruct_near_backbone")
    if iv:
        ratios = {}
        for t in texts:
            a, b = _b1(summary, iv["rung"], t), _b1(summary, "qwen-noise-0.01", t)
            ratios[t] = (a / b) if (a is not None and b) else None
        out["clauses"]["iv"] = {"rung": iv["rung"], "backbone": "qwen-noise-0.01", "within": 3.0, "ratio": ratios,
                                "verdict": verdict([None if r is None else (1 / 3.0 <= r <= 3.0) for r in ratios.values()])}
    v = P.get("v_on_path_at_or_below_corpus")
    if v and {"corpus", "on-path"} <= set(texts):
        rows, checks = {}, []
        for e in summary["rungs"]:
            if e["withheld"] or "corpus" not in e["texts"] or "on-path" not in e["texts"]:
                continue
            c, o = e["texts"]["corpus"], e["texts"]["on-path"]
            fl = c.get("floor_kl")
            if fl is None or c["b1"]["median_kl"] <= 2.0 * fl:
                continue   # at or under 2x the floor: not read by this clause
            ok = o["b1"]["median_kl"] <= c["b1"]["median_kl"]
            rows[e["label"]] = {"corpus": c["b1"]["median_kl"], "on-path": o["b1"]["median_kl"], "ok": ok}
            checks.append(ok)
        out["clauses"]["v"] = {"rungs_read": rows, "exceptions": sorted(k for k, r in rows.items() if not r["ok"]),
                               "verdict": verdict(checks)}
    # 0022: the far control. (i) far over scrambled on every named control unit; (ii) every named rung's median over
    # its unit's far median, on the named text. Both are numbers the summary already carries.
    fi = P.get("i_far_over_scrambled")
    if fi:
        vals = {u: (summary.get("controls", {}).get(u) or {}).get("far_over_scrambled") for u in fi["units"]}
        lo = float(fi["at_least"])
        out["clauses"]["i"] = {"units": list(fi["units"]), "at_least": lo, "far_over_scrambled": vals,
                               "verdict": verdict([None if v is None else v >= lo for v in vals.values()])}
    fii = P.get("ii_rollout_over_far")
    if fii:
        text, hi = str(fii["text"]), float(fii["at_most"])
        vals = {}
        for label in fii["rungs"]:
            e = next((e for e in summary["rungs"] if e["label"] == label), None)
            vals[label] = None if e is None or e.get("withheld") else e["texts"].get(text, {}).get("over_far")
        out["clauses"]["ii"] = {"rungs": list(fii["rungs"]), "text": text, "at_most": hi, "over_far": vals,
                                "verdict": verdict([None if v is None else v <= hi for v in vals.values()])}
    # 0028: the second seed's replication clauses, read against the comparators the seal itself carries.
    def _entry(label):
        e = next((e for e in summary["rungs"] if e["label"] == label), None)
        return None if e is None or e.get("withheld") else e

    ih = P.get("i_heldout_holds_at_every_lag")
    if ih:
        hm = float(rule.get("holds_max", summary["rule"]["holds_max"]))
        meds, checks = {}, []
        for label in ih["rungs"]:
            e = _entry(label)
            meds[label] = {}
            for t in ih["texts"]:
                st = (e or {}).get("texts", {}).get(t)
                m = st["b1"]["median_kl"] if st else None
                meds[label][t] = m
                checks.append(None if m is None or st.get("at_floor") is None else (m <= hm and not st["at_floor"]))
        out["clauses"]["i-holds"] = {"rungs": list(ih["rungs"]), "texts": list(ih["texts"]), "holds_max": hm,
                                     "median_kl": meds, "verdict": verdict(checks)}
    rr = P.get("ii_rollout_replicates_seed1")
    if rr:
        f, text = float(rr["within_factor"]), str(rr["text"])
        floor1 = rr.get("and_at_lag_1_at_least")
        meds, accept, checks = {}, {}, []
        for g, label in rr["rungs"].items():
            ref = float(rr["reference"][g])
            accept[g] = [ref / f, ref * f]
            e = _entry(label)
            m = e["texts"].get(text, {}).get("b1", {}).get("median_kl") if e else None
            meds[g] = m
            ok = None if m is None else (accept[g][0] <= m <= accept[g][1])
            if ok is not None and g == "1" and floor1 is not None:
                ok = ok and m >= float(floor1)
            checks.append(ok)
        out["clauses"]["ii"] = {"rungs": dict(rr["rungs"]), "text": text, "within_factor": f, "accept": accept,
                                "at_lag_1_at_least": floor1, "median_kl": meds, "verdict": verdict(checks)}
    ro = P.get("iii_rollout_over_heldout")
    if ro:
        k, text = float(ro["at_least_factor"]), str(ro["text"])
        ratios, checks = {}, []
        for g, (rl, hl) in ro["pairs_of_rungs"].items():
            re_, he = _entry(rl), _entry(hl)
            ratios[g] = {}
            for t in ro["heldout_texts"]:
                a = re_["texts"].get(text, {}).get("b1", {}).get("median_kl") if re_ else None
                b = he["texts"].get(t, {}).get("b1", {}).get("median_kl") if he else None
                r = (a / b) if (a is not None and b) else None
                ratios[g][t] = r
                checks.append(None if r is None else r >= k)
        out["clauses"]["iii"] = {"pairs_of_rungs": dict(ro["pairs_of_rungs"]), "text": text, "heldout_texts": list(ro["heldout_texts"]),
                                 "at_least_factor": k, "ratio": ratios, "verdict": verdict(checks)}
    lw = P.get("iii_lands_with")       # 0030: which of two earlier readings a new text's median lands with, per lag
    if lw:
        margin, text = float(lw["margin_log10"]), str(lw["text"])
        meds, dist, lands, checks = {}, {}, {}, []
        for g, label in lw["rungs"].items():
            e = _entry(label)
            st = (e or {}).get("texts", {}).get(text)
            m = st["b1"]["median_kl"] if st else None
            meds[g] = m
            c, r = float(lw["reference_corpus"][g]), float(lw["reference_rollout"][g])
            if m is None or m <= 0 or c <= 0 or r <= 0:
                dist[g], lands[g] = None, None
                checks.append(None)
                continue
            dc, dr = abs(math.log10(m) - math.log10(c)), abs(math.log10(m) - math.log10(r))
            dist[g] = {"to_corpus_log10": dc, "to_rollout_log10": dr}
            lands[g] = "corpus" if dc < dr else ("rollout" if dr < dc else "neither")
            checks.append(dr - dc >= margin)
        out["clauses"]["lands"] = {"rungs": dict(lw["rungs"]), "text": text, "margin_log10": margin, "median_kl": meds,
                                   "distance": dist, "lands_with": lands, "verdict": verdict(checks)}
    cu = P.get("iv_controls_as_0020")
    if cu:
        units, checks = {}, []
        for u in cu["units"]:
            c = summary.get("controls", {}).get(u)
            units[u] = None if c is None else {"passed": c.get("passed"), "evaluable_under": c.get("evaluable_under")}
            checks.append(None if c is None or c.get("passed") is None else bool(c["passed"]))
        out["clauses"]["iv"] = {"units": units, "verdict": verdict(checks)}
    # §14.5 (the families): 0017's sentence, HOLDS_B lost at the first optimizer step on the trainer's own text, as a
    # clause of its own — the rollout median at the lag-1 rung is at least `at_least` (0028 folded it into the
    # replication clause, which needs a seed-1 reference a new family does not have).
    lost = P.get("ii_rollout_lost_at_lag_1")
    if lost:
        e = _entry(lost["rung"])
        m = (e or {}).get("texts", {}).get(str(lost["text"]), {}).get("b1", {}).get("median_kl")
        lo = float(lost["at_least"])
        out["clauses"]["ii-lost"] = {"rung": lost["rung"], "text": str(lost["text"]), "at_least": lo, "median_kl": m,
                                     "verdict": verdict([None if m is None else m >= lo])}
    ex = P.get("iii_rl_excess_over_0015_backbone")
    if ex and "against_sha256" in ex:
        # 0017's seal carries this key as prose only ("read by hand in the outcome entry", 0018); the readout engages
        # on the machine-readable form §14.5's seals carry and leaves 0017's summary as it was written.
        out["clauses"]["iii-excess"] = backbone_excess_readout(summary, repo, ex, verdict)
    return out


def backbone_excess_readout(summary: dict, repo: Path, ex: dict, verdict) -> dict:
    """0017's clause (iii), read by hand in 0018 and by this function from §14.5 on: at every sealed rung whose
    whole-vector distance lies inside `distance_range`, median (B1) is at least `at_least_factor` times the backbone's
    median at the nearest backbone rung by log distance, on every text named. The backbone is another summary
    (`against`, 0015's `results/reader/summary.json`), pinned by `against_sha256` and REFUSED if the bytes moved;
    its backbone points are the non-withheld `noise` rungs of `backbone_pair` with a position. Sealed rungs outside
    the range are listed under `outside_range` and not read; with no rung inside, the clause is UNREAD (0018 read
    one rung, lag 32 at 1.049e-4: 11.1 on corpus, 7.88 on on-path — FALSIFIED on on-path)."""
    ref_path = repo / str(ex["against"])
    if not ref_path.exists():
        raise _refuse(f"backbone excess: the reference summary {ex['against']} is not on disk")
    got = sha256_file_bytes(ref_path)
    if got != str(ex["against_sha256"]):
        raise _refuse(f"backbone excess: {ex['against']} sha256 {got[:12]}... is not the sealed {str(ex['against_sha256'])[:12]}...; "
                      "the reference the clause names has moved")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    pair = str(ex["backbone_pair"])
    backbone = [(e["label"], float(e["rel_delta_all"])) for e in ref["rungs"]
                if e.get("kind") == "noise" and e.get("pair") == pair and not e.get("withheld")
                and e.get("rel_delta_all") is not None]
    if not backbone:
        raise _refuse(f"backbone excess: {ex['against']} carries no evaluated noise rung of pair {pair}")
    ref_b1 = {e["label"]: {t: s["b1"]["median_kl"] for t, s in e["texts"].items()} for e in ref["rungs"]}
    lo, hi = (float(v) for v in ex["distance_range"])
    k = float(ex["at_least_factor"])
    texts = [str(t) for t in ex["texts"]]
    rows, outside, checks = {}, {}, []
    for label in ex["rungs"]:
        e = next((e for e in summary["rungs"] if e["label"] == label), None)
        x = (e or {}).get("rel_delta_all")
        if x is None:
            rows[label] = {"rel_delta_all": None, "nearest_backbone": None, "ratio": {t: None for t in texts}}
            checks.append(None)       # a sealed rung with no position cannot be placed: UNREAD
            continue
        x = float(x)
        if not (lo <= x <= hi):
            outside[label] = x
            continue
        nb, _ = min(backbone, key=lambda b: abs(math.log10(x) - math.log10(b[1])))
        ratio = {}
        for t in texts:
            a = e["texts"].get(t, {}).get("b1", {}).get("median_kl") if not e.get("withheld") else None
            b = ref_b1[nb].get(t)
            r = (a / b) if (a is not None and b) else None
            ratio[t] = r
            checks.append(None if r is None else r >= k)
        rows[label] = {"rel_delta_all": x, "nearest_backbone": nb, "backbone_median_kl": {t: ref_b1[nb].get(t) for t in texts},
                       "ratio": ratio}
    return {"against": str(ex["against"]), "against_sha256": got, "backbone_pair": pair,
            "backbone_rungs": [b[0] for b in backbone], "distance_range": [lo, hi], "at_least_factor": k,
            "texts": texts, "rungs": rows, "outside_range": outside,
            "verdict": verdict(checks) if checks else "UNREAD"}


def render(s: dict) -> str:
    lines = [f"reader ladder, statistic (B) (entry {s['registered_by']}, instrument {s['upstream_sha'][:12]}): "
             "median over held-out sequences of mean per-token KL(fresh || stale), nats/token"]
    for reader, c in s["controls"].items():
        parts = []
        for name in ("floor", "identity", "scrambled"):
            v = c[name]
            parts.append(f"{name} {'ok' if v and v['passed'] else 'FAILED' if v else 'MISSING'}")
        if c.get("far") is not None:
            ratio = c.get("far_over_scrambled")
            parts.append(f"far {'ok' if c['far']['passed'] else 'FAILED'} ({c['far']['median_kl']:.3e}"
                         + (f", {ratio:.2f}x scrambled)" if ratio is not None else ")"))
        lines.append(f"reader {reader}: " + "; ".join(parts))
    for e in s["rungs"]:
        for text, st in e["texts"].items():
            if e["withheld"]:
                lines.append(f"  {e['label']:28s} {text:8s} (withheld: reader's controls failed or floor unset)")
                continue
            floor = " AT THE FLOOR" if st["at_floor"] else ""
            x = f"x {e['rel_delta_all']:.2e}" if e["rel_delta_all"] is not None else "x -"
            near = f"  ~{e['matched_backbone']}" if e["matched_backbone"] else ""
            lines.append(f"  {e['label']:28s} {text:8s} {x:11s} median KL {st['b1']['median_kl']:.3e}  "
                         f"p90 {st['b1']['p90_token_kl']:.3e}  max {st['b1']['max_token_kl']:.3e}  "
                         f"ESS/N {st['b2']['ess_over_n']:.3f}  agree {st['b4']['agreement_rate']:.3f}  "
                         f"{st['band']}{floor}{near}")
    for text, fit in s["slope_fit"].items():
        slope = f"{fit['slope']:.2f}" if fit["slope"] is not None else "-"
        which = "sealed rungs" if fit["sealed_rungs"] else "all backbone rungs"
        lines.append(f"slope of log median (B1) on log distance, {text}: {slope} over {which} {fit['rungs']}")
    ro = s.get("seal_readout")
    if ro:
        lines.append(f"sealed clauses ({ro['source']}):")
        for k, c in ro["clauses"].items():
            detail = {kk: vv for kk, vv in c.items() if kk not in ("verdict", "rungs_read")}
            lines.append(f"  ({k}) {c['verdict']}  {json.dumps(detail, default=str)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="recompute statistic (B) from the reader reports on disk")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "reader.toml"))
    ap.add_argument("--root", default=None, help="results root override (tests); default results/reader")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_reader_config(Path(a.config), REPO_ROOT)
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
