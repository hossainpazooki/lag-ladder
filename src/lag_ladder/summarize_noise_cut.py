"""The read-out of ledger 0042 (descriptive, nothing sealed): the noise readers and the prefix cut on the own run, from the five configurations'
summaries, each recomputed here from its reports by `summarize_reader.summarize`.

Per kept lag g in {1, 2, 4, 8, 16, 32}, every value a median (B1) of one rung and text as `summarize_reader` reports it:
  T_rollout   0017's rollout rung, read again in this sitting (config/ownrun_replicate.toml)
  T_heldout   0017's held-out rung, same sitting, texts corpus and on-path
  N_rollout   the noise reader at step g's distance, seed 0, rollout text (config/ownrun_noise.toml)
  N_heldout   the same reader on the held-out corpus
and the factors of T_rollout / T_heldout = (T_rollout / N_rollout) x (N_rollout / N_heldout) x (N_heldout / T_heldout).
Seeds 1 and 2 at lags 1 and 32 give the spread of N. The prefix cut: the held-out rung with the prefix cut (prompt length,
192, 384; configs ownrun_cut_*.toml) over the same rung at 768 in this sitting, and T_rollout over each cut reading.

A ratio carries its operands' floor flags (`at_floor`: a median within 2x of the unit's float16 floor, LL 0012): an
operand at the floor is an upper bound on the reading, so a ratio with its denominator at the floor is a lower bound
and one with its numerator there an upper bound. A withheld rung (its unit failed a control) gives no value. So does a
configured rung that the driver skipped because its unit was UNEVALUABLE: its cell is withheld and carries the unit's
scrambled median, and the rest is read from the rungs that are there. Which rungs were skipped is read from each
configuration's run record (`unevaluable_readers`), never inferred from the summary: a run that halted, did not finish,
or whose jobs are not the plan's jobs less those units' rungs (a run cut short with --limit, a crash partway) is
refused. An absent rung whose unit the record does not name, or whose unit passed its controls, is refused. The
replicate is also set beside 0018's medians when 0018's summary is on disk at its pinned sha256.

On on-path each reader is scored on its own greedy continuation, so values at the same g may be read on different
tokens; corpus and the rollout text are stored tokens every reader shares. Each held-out row and each text of the seed
spread says which (`shared_tokens`), and the printed on-path rows and seed spreads carry the mark.
"""
import argparse
import json
import sys
from pathlib import Path

from lag_ladder import REPO_ROOT, reader
from lag_ladder.config import ReaderConfig, load_reader_config
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.summarize_reader import RUN_RECORD, _control_key, summarize

LAGS = (1, 2, 4, 8, 16, 32)
HELDOUT_TEXTS = ("corpus", "on-path")
SHARED_TOKENS = {"corpus": True, "on-path": False}     # on-path: each reader's own greedy continuation
ON_PATH_MARK = "[on-path: each reader's own continuation, tokens may differ]"
SEEDS_AT = {1: (0, 1, 2), 32: (0, 1, 2)}
CONFIGS = {"replicate": "ownrun_replicate.toml", "noise": "ownrun_noise.toml", "cut_prompt": "ownrun_cut_prompt.toml",
           "cut_192": "ownrun_cut_192.toml", "cut_384": "ownrun_cut_384.toml"}
CUTS = ("cut_prompt", "cut_192", "cut_384")
REF_0018 = Path("results") / "ownrun" / "summary.json"
REF_0018_SHA256 = "f3b8f00ff6ed2f94e684845e87fa4da1e0b1e820ea2510d3ac0aa165fe1d11a6"
OUT = Path("results") / "ownrun_noise_cut" / "readout.json"


def _refuse(msg: str) -> ValueError:
    return ValueError(f"summarize_noise_cut REFUSED: {msg}")


def skipped_units(cfg: ReaderConfig, rec: dict, *, name: str) -> dict:
    """{rung label: control unit} for the rungs the driver skipped, read from the configuration's run record. Refuses
    a run that halted or did not finish, a record whose jobs are not the plan's jobs (in order) less the rungs of the
    units it names UNEVALUABLE, and a named unit that no configured rung has."""
    if rec.get("halted"):
        raise _refuse(f"{name}: the run halted: {rec['halted']}")
    if rec.get("finished") is None:
        raise _refuse(f"{name}: the run did not finish")
    unev = set(rec.get("unevaluable_readers") or {})
    want = [(j.kind, j.reader, j.label, j.pair, j.seq, j.text) for j in reader.plan(cfg).jobs
            if not (j.kind == "rung" and reader.control_key(j, cfg) in unev)]
    jobs = rec.get("jobs") or []
    got = [(j.get("kind"), j.get("reader"), j.get("label"), j.get("pair"), j.get("seq"), j.get("text")) for j in jobs]
    if got != want or not all(j.get("done") is True for j in jobs):
        raise _refuse(f"{name}: the run record's {len(got)} jobs are not the plan's {len(want)} jobs less the rungs of "
                      f"its UNEVALUABLE units {sorted(unev)}")
    units = {r.label: u for r in cfg.rungs if (u := _control_key(r.reader, r.pair, cfg)) in unev}
    if unev - set(units.values()):
        raise _refuse(f"{name}: the run record names UNEVALUABLE units no configured rung has: "
                      f"{sorted(unev - set(units.values()))}")
    return units


def cell(summary: dict, label: str, text: str, *, name: str, units: dict | None = None) -> dict:
    """One rung's median on one text: value None when withheld; refuses a rung or text the summary does not carry,
    except a rung the driver skipped (units: label -> its UNEVALUABLE unit, from `skipped_units`) whose unit failed its
    controls in the summary too."""
    for e in summary["rungs"]:
        if e["label"] == label:
            if text not in e["texts"]:
                raise _refuse(f"{name}: rung {label} carries no text {text!r}")
            t, w = e["texts"][text], bool(e.get("withheld"))
            v = None if w else t["b1"]["median_kl"]
            return {"value": v, "at_floor": None if w else bool(t.get("at_floor")), "withheld": w,
                    "distance": e["rel_delta_all"] if e.get("rel_delta_all") is not None else e.get("rel_norm"),
                    "unit": e.get("controls")}      # 0018's summary predates the per-rung unit key
    unit = (units or {}).get(label)
    if unit is None:
        raise _refuse(f"{name}: no rung {label} in the summary")
    c = summary["controls"].get(unit)
    if c is None:
        raise _refuse(f"{name}: no rung {label} in the summary, and its unit {unit} has no controls")
    if c["passed"]:
        raise _refuse(f"{name}: no rung {label} in the summary, though its unit {unit} passed its controls")
    return {"value": None, "at_floor": None, "withheld": True, "distance": None, "unit": unit, "skipped": True,
            "scrambled_median": (c.get("scrambled") or {}).get("median_kl")}


def ratio(a: dict, b: dict) -> dict:
    if a["value"] is None or b["value"] is None or not b["value"] > 0:
        return {"value": None, "bound": None, "why": "withheld" if (a["withheld"] or b["withheld"]) else "no value"}
    v = a["value"] / b["value"]
    bound = ("unread" if a["at_floor"] and b["at_floor"] else "upper" if a["at_floor"]
             else "lower" if b["at_floor"] else None)
    return {"value": v, "bound": bound, "why": None}


def _units(summary: dict) -> dict:
    return {u: {"passed": c["passed"], "evaluable_under": c.get("evaluable_under"),
                "scrambled_median": (c.get("scrambled") or {}).get("median_kl"),
                "far_median": (c.get("far") or {}).get("median_kl")} for u, c in summary["controls"].items()}


def readout(s: dict, ref: dict | None = None, units: dict | None = None) -> dict:
    """s: {config key: summary}. ref: 0018's summary or None. units: {config key: {rung label: control unit}} for the
    rungs each run record says the driver skipped (`skipped_units`), so such a rung reads as withheld; any other
    absent rung is refused."""
    missing = set(CONFIGS) - set(s)
    if missing:
        raise _refuse(f"summaries missing for {sorted(missing)}")

    def at(k: str, label: str, text: str) -> dict:
        return cell(s[k], label, text, name=k, units=(units or {}).get(k))

    out = {"lags": {}, "seeds": {}, "cuts": {}, "replicate_vs_0018": None,
           "controls": {k: _units(v) for k, v in s.items()}}
    for g in LAGS:
        T_r = at("replicate", f"ownrun-rollout-lag-{g}", "rollout")
        N_r = at("noise", f"ownrun-rollout-noise-lag-{g}-s0", "rollout")
        row = {"T_rollout": T_r, "N_rollout": N_r, "T_rollout_over_N_rollout": ratio(T_r, N_r), "heldout": {}}
        for text in HELDOUT_TEXTS:
            T_h = at("replicate", f"ownrun-lag-{g}", text)
            N_h = at("noise", f"ownrun-noise-lag-{g}-s0", text)
            f1, f2, f3 = ratio(T_r, N_r), ratio(N_r, N_h), ratio(N_h, T_h)
            whole = ratio(T_r, T_h)
            prod = (f1["value"] * f2["value"] * f3["value"]
                    if None not in (f1["value"], f2["value"], f3["value"]) else None)
            row["heldout"][text] = {"T_heldout": T_h, "N_heldout": N_h, "T_heldout_over_N_heldout": ratio(T_h, N_h),
                                    "N_rollout_over_N_heldout": f2, "T_rollout_over_T_heldout": whole,
                                    "factor_product": prod, "shared_tokens": SHARED_TOKENS[text]}
        dt, dn = T_r["distance"], N_r["distance"]
        row["distance"] = {"real": dt, "noise": dn, "noise_over_real": (dn / dt) if dt and dn else None}
        out["lags"][str(g)] = row
    for g, seeds in SEEDS_AT.items():
        per = {}
        for text in ("rollout",) + HELDOUT_TEXTS:
            vals = {}
            for sd in seeds:
                lab = f"ownrun-rollout-noise-lag-{g}-s{sd}" if text == "rollout" else f"ownrun-noise-lag-{g}-s{sd}"
                vals[str(sd)] = at("noise", lab, text)
            got = [v["value"] for v in vals.values() if v["value"] is not None and v["value"] > 0]
            per[text] = {"by_seed": vals, "max_over_min": (max(got) / min(got)) if len(got) == len(seeds) else None,
                         "any_at_floor": any(v["at_floor"] for v in vals.values()),
                         "shared_tokens": SHARED_TOKENS.get(text, True)}   # rollout: the stored completions
        out["seeds"][str(g)] = per
    for g in LAGS:
        T_r = at("replicate", f"ownrun-rollout-lag-{g}", "rollout")
        row = {}
        for text in HELDOUT_TEXTS:
            full = at("replicate", f"ownrun-lag-{g}", text)
            r = {"768": full}
            for c in CUTS:
                cc = at(c, f"ownrun-lag-{g}", text)
                r[c] = {"reading": cc, "over_768": ratio(cc, full), "T_rollout_over_cut": ratio(T_r, cc)}
            row[text] = r
        out["cuts"][str(g)] = row
    if ref is not None:
        rows = {}
        for e in s["replicate"]["rungs"]:
            for text in e["texts"]:
                a = at("replicate", e["label"], text)
                b = cell(ref, e["label"], text, name="0018")
                rows[f"{e['label']}/{text}"] = {"replicate": a["value"], "0018": b["value"], "ratio": ratio(a, b)["value"]}
        out["replicate_vs_0018"] = rows
    return out


def render(r: dict) -> str:
    def f(x):
        if x is None or x.get("value") is None:
            return "  --  "
        b = {"lower": ">", "upper": "<", "unread": "~"}.get(x.get("bound"), " ")
        return f"{b}{x['value']:.3g}"
    lines = ["noise readers, seed 0 (> lower bound: denominator at the floor; < upper bound; ~ both at the floor)"]
    for g, row in r["lags"].items():
        lines.append(f"g={g:>2} T_r {f(row['T_rollout'])} N_r {f(row['N_rollout'])} T_r/N_r {f(row['T_rollout_over_N_rollout'])}"
                     f"  distance noise/real {row['distance']['noise_over_real']}")
        for text, h in row["heldout"].items():
            lines.append(f"      {text:8} T_h {f(h['T_heldout'])} N_h {f(h['N_heldout'])} T_h/N_h {f(h['T_heldout_over_N_heldout'])}"
                         f" N_r/N_h {f(h['N_rollout_over_N_heldout'])} T_r/T_h {f(h['T_rollout_over_T_heldout'])}"
                         + ("" if h["shared_tokens"] else f"  {ON_PATH_MARK}"))
    for g, per in r["seeds"].items():
        lines.append(f"seeds at g={g}: " + "; ".join(f"{t} max/min {p['max_over_min']}"
                                                     + ("" if p["shared_tokens"] else f" {ON_PATH_MARK}")
                                                     for t, p in per.items()))
    lines.append("prefix cut: each cut over 768, and T_rollout over the cut reading")
    for g, row in r["cuts"].items():
        for text, c in row.items():
            lines.append(f"g={g:>2} {text:8} 768 {f(c['768'])} " + " ".join(
                f"{k[4:]} {f(c[k]['reading'])} (x{f(c[k]['over_768'])}, T_r/cut {f(c[k]['T_rollout_over_cut'])})" for k in CUTS))
    for k, units in r["controls"].items():
        bad = [u for u, c in units.items() if not c["passed"] or c["evaluable_under"]]
        if bad:
            lines.append(f"{k}: units withheld or evaluable under far: " + ", ".join(
                f"{u} (passed {units[u]['passed']}, scrambled {units[u]['scrambled_median']}, under {units[u]['evaluable_under']})" for u in bad))
    if r["replicate_vs_0018"] is None:
        lines.append("replicate vs 0018: UNREAD (0018's summary not on disk at its pinned sha256)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ledger 0042's read-out of the noise readers and the prefix cut, recomputed from the reports")
    ap.add_argument("--out", default=str(REPO_ROOT / OUT))
    a = ap.parse_args(argv)
    try:
        s, units = {}, {}
        for k, name in CONFIGS.items():
            cfg = load_reader_config(REPO_ROOT / "config" / name, REPO_ROOT)
            s[k] = summarize(cfg)                     # refuses first when the run record is missing
            rec = json.loads((cfg.results_dir / RUN_RECORD).read_text(encoding="utf-8"))
            units[k] = skipped_units(cfg, rec, name=k)
        ref_f, ref = REPO_ROOT / REF_0018, None
        if ref_f.exists():
            got = sha256_file_bytes(ref_f)
            if got != REF_0018_SHA256:
                raise _refuse(f"{REF_0018} hashes to {got[:12]}, not 0018's {REF_0018_SHA256[:12]}")
            ref = json.loads(ref_f.read_text(encoding="utf-8"))
        r = readout(s, ref, units)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(render(r))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
