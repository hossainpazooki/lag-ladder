"""The reduced-(C) summary (design 2026-11-09, reduced (C)): every number recomputed from the JSONL records
`tools/task_swap_tir.py` wrote through `lag_ladder.task_tir`, never from run.json's judgements (those are re-derived
and compared). Per reader: FRESH accuracy, MathAnswer rate, tool-call rate, turns and cap rate; per arm (STALE-prompt,
STALE-shared on every prompt; NEAR, FAR on the first controls.n_prompts): the final-answer flip rate against FRESH with a
seeded bootstrap interval over prompts (the statistic), the tool-call-sequence flip rate, the paired accuracy delta, the
no-answer and MathAnswer rates, turns, cap rate, the share of episodes whose stale span stayed inside the kept cache
prefix, the mean sandbox memo hits and, for STALE-prompt, the rank correlation of the flip with the fresh-path (B1).
The reader is UNEVALUABLE when NEAR or FAR flips fewer than control_min_flip_rate of its prompts or a control failed:
the descriptives stay and the sealed clauses that read it are UNREAD. The sealed clauses (PC1-PC3) are read against
`ledger/predictions/<seal>.json`. `summarize_task.py` (0025-0035's) is byte-unchanged; its pure functions are reused.
"""
import json
import sys
from pathlib import Path

import numpy as np

from lag_ladder import REPO_ROOT
from lag_ladder.hashing import sha256_file_bytes
from lag_ladder.summarize_task import (PREDICTIONS, RUN_RECORD, SUMMARY, flip_rate, flip_rate_interval, no_answer_rate,
                                       paired_delta, rank_correlation)
from lag_ladder.task_tir import ARMS, CONTROL_ARMS, FLOOR_ARM, RECORD_SCHEMA, TaskTirConfig, load_task_tir_config

STALE_ARMS = ("stale_prompt", "stale_shared")
POSITIVE_ARMS = ("near", "far")


def _refuse(msg: str) -> ValueError:
    return ValueError(f"summarize_task_tir REFUSED: {msg}")


def _load_jsonl(path: Path, named_sha: str) -> dict[int, dict]:
    if not path.exists():
        raise _refuse(f"missing {path}")
    got = sha256_file_bytes(path)
    if got != named_sha:
        raise _refuse(f"{path} hashes to {got[:12]}, run.json names {named_sha[:12]} (sha256 mismatch)")
    rows = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("schema") != RECORD_SCHEMA:
            raise _refuse(f"{path}: a record is not {RECORD_SCHEMA}")
        if r["index"] in rows:
            raise _refuse(f"{path}: index {r['index']} twice")
        rows[int(r["index"])] = r
    return rows


def _rate(rows: dict[int, dict], idx: list[int], pred) -> float:
    return float(sum(bool(pred(rows[i])) for i in idx) / len(idx))


def _arm_stats(fresh: dict[int, dict], arm: dict[int, dict], cfg: TaskTirConfig) -> dict:
    """The arm's statistics over ITS prompts (a control arm covers the first controls.n_prompts only), each paired with
    FRESH on the same prompt."""
    idx = sorted(arm)
    if any(i not in fresh for i in idx):
        raise _refuse("an arm holds a prompt FRESH does not")
    reps, seed = int(cfg.rule["bootstrap_reps"]), cfg.seed
    flips = [int(arm[i]["answer"] != fresh[i]["answer"]) for i in idx]
    fi = flip_rate_interval(flips, reps=reps, seed=seed)
    b1 = [arm[i].get("b1_fresh_path") for i in idx]
    rho = rank_correlation(flips, b1) if all(x is not None for x in b1) else None
    return {"n": len(idx),
            "flip_rate": flip_rate([arm[i]["answer"] for i in idx], [fresh[i]["answer"] for i in idx]),
            "flip_ci_low": fi["ci_low"], "flip_ci_high": fi["ci_high"],
            "tool_sequence_flip_rate": _rate(arm, idx, lambda r: r["tool_sequence_sha256"] != fresh[r["index"]]["tool_sequence_sha256"]),
            "delta": paired_delta([arm[i]["correct"] for i in idx], [fresh[i]["correct"] for i in idx], reps=reps, seed=seed),
            "acc": float(100.0 * np.mean([arm[i]["correct"] for i in idx])),
            "no_answer_rate": no_answer_rate([arm[i]["answer"] for i in idx]),
            "math_answer_rate": _rate(arm, idx, lambda r: r["answered_by"] == "MathAnswer"),
            "mean_turns": float(np.mean([arm[i]["n_turns"] for i in idx])),
            "cap_rate": _rate(arm, idx, lambda r: r["reached_cap"]),
            "stale_inside_lcp_rate": _rate(arm, idx, lambda r: r["stale_inside_lcp"]),
            "mean_sandbox_hits": float(np.mean([arm[i]["sandbox_hits"] for i in idx])),
            "rank_correlation_b1": rho}


def summarize(cfg: TaskTirConfig, root: Path | None = None, out: Path | None = None) -> dict:
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
    if not run_rec.get("finished"):
        # task_tir.run saves after every job and stamps `finished` only at the end: a killed sitting leaves it unset
        raise _refuse("the run did not finish (run.json has no finished time); resume it with run before summarizing")
    fid = run_rec.get("fidelity")
    if not fid or not fid.get("passed"):
        raise _refuse("the driver fidelity gate did not pass; the rendering is not upstream's")
    summary = {"registered_by": cfg.registered_by, "upstream_sha": cfg.upstream_sha, "rule": dict(cfg.rule),
               "controls_cfg": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in cfg.controls.items()},
               "fidelity": fid, "content_eos": run_rec.get("content_eos"),
               "readers_registered": list(run_rec.get("readers", [])),
               "readers_run": list(run_rec.get("readers_run", run_rec.get("readers", []))),
               "lags": {}, "seal_readout": None}
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
        f32 = _load_jsonl(d / f"{FLOOR_ARM}.jsonl", shas[f"{FLOOR_ARM}.jsonl"])
        f2 = _load_jsonl(d / "fresh2.jsonl", shas["fresh2.jsonl"])
        if not f32 or not f2:
            raise _refuse(f"{j['reader']}: a control arm is empty")
        floor_rate = float(sum(f32[i]["answer"] != fresh[i]["answer"] for i in f32) / len(f32))
        ident = int(sum(f2[i]["completion_sha256"] != fresh[i]["completion_sha256"] for i in f2))
        if (len(f32) != ctl["floor_n"] or len(f2) != ctl["control_n"] or abs(floor_rate - ctl["floor_flip_rate"]) > 1e-12
                or ident != ctl["identity_flips"]):
            raise _refuse(f"{j['reader']}: the controls recomputed from the records differ from run.json")
        idx = sorted(fresh)
        lag = {"n": len(idx), "acc_fresh": float(100.0 * np.mean([fresh[i]["correct"] for i in idx])),
               "math_answer_rate_fresh": _rate(fresh, idx, lambda r: r["answered_by"] == "MathAnswer"),
               "tool_call_rate_fresh": _rate(fresh, idx, lambda r: any(t["tool_calls"] for t in r["turns"])),
               "mean_turns_fresh": float(np.mean([fresh[i]["n_turns"] for i in idx])),
               "cap_rate_fresh": _rate(fresh, idx, lambda r: r["reached_cap"]),
               "controls": {"floor_n": len(f32), "floor_flip_rate": floor_rate,
                            "floor_passed": floor_rate <= float(cfg.controls["floor_max_flip_rate"]),
                            "control_n": len(f2), "identity_flips": ident, "identity_passed": ident == 0},
               "arms": {}}
        for arm in STALE_ARMS + POSITIVE_ARMS:
            lag["arms"][arm] = _arm_stats(fresh, _load_jsonl(d / f"{arm}.jsonl", shas[f"{arm}.jsonl"]), cfg)
        fire = all(lag["arms"][a]["flip_rate"] >= float(cfg.rule["control_min_flip_rate"]) for a in POSITIVE_ARMS)
        lag["positive_controls_fire"] = fire
        lag["evaluable"] = bool(fire and lag["controls"]["floor_passed"] and lag["controls"]["identity_passed"])
        summary["lags"][j["reader"]] = lag
    # the repo is the grandparent of the results tree; resolved, so a relative --root finds the same repo
    summary["seal_readout"] = seal_readout(summary, root.resolve().parent.parent, cfg.seal)
    target = Path(out) if out is not None else root / SUMMARY
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(summary, indent=1))
    return summary


def seal_readout(summary: dict, repo: Path, seal: str = "pipelinerl-tir-task") -> dict:
    """PC1: at a reader the named arm's flip rate is at least a value. PC2: at every named reader STALE-shared's flip
    rate is below STALE-prompt's (point estimates). PC3: the named arm's flip rate at the first reader is below its rate
    at the last (point estimates, the spec's wording; both bootstrap intervals are reported in the clause). UNREAD when
    a reader is unevaluable or missing. A missing seal file, or one
    that does not hold the three clauses, is refused (ValueError), never read as "no readout"."""
    f = Path(repo) / PREDICTIONS / f"{seal}.json"
    if not f.exists():
        # the registered readout must not go missing in silence: the summary of record is recomputed from a tree that
        # may sit outside <repo>/results/<name>, and summarize() takes the repo to be the root's grandparent
        raise ValueError(f"no sealed predictions at {f}: seal_readout reads <repo>/{PREDICTIONS.as_posix()}/{seal}.json and "
                         "summarize() takes <repo> to be the grandparent of --root; put the results tree under "
                         "<repo>/results/<name>")
    P = json.loads(f.read_text(encoding="utf-8")).get("prediction", {}).get("predictions", {})
    out: dict = {"source": (PREDICTIONS / f"{seal}.json").as_posix(), "clauses": {}}
    lags = summary.get("lags", {})

    def verdict(checks):
        vals = [c for c in checks if c is not None]
        if len(vals) < len(checks) or not vals:
            return "UNREAD"
        return "HELD" if all(vals) else "FALSIFIED"

    def arm(reader, name):
        lag = lags.get(reader) or {}
        a = ((lag.get("arms") or {}).get(name) or {})
        return a if lag.get("evaluable") and a.get("flip_rate") is not None else None

    c1 = P.get("pc1_flip_rate_at_first_lag")
    if c1:
        a = arm(c1["reader"], c1["arm"])
        ok = None if a is None else a["flip_rate"] >= float(c1["at_least"])
        out["clauses"]["pc1"] = {"reader": c1["reader"], "arm": c1["arm"], "at_least": float(c1["at_least"]),
                                 "flip_rate": None if a is None else a["flip_rate"], "verdict": verdict([ok])}
    c2 = P.get("pc2_shared_below_prompt")
    if c2:
        below = {}
        for r in c2["readers"]:
            p, s = arm(r, "stale_prompt"), arm(r, "stale_shared")
            below[r] = None if p is None or s is None else s["flip_rate"] < p["flip_rate"]
        out["clauses"]["pc2"] = {"readers": list(c2["readers"]), "below": below, "verdict": verdict(list(below.values()))}
    c3 = P.get("pc3_flip_growth")
    if c3:
        a, b = arm(c3["first"], c3["arm"]), arm(c3["last"], c3["arm"])
        # the spec's comparison is on the point estimates ("STALE-prompt's flip rate at g = 1 is below its rate at
        # g = 32"); the bootstrap intervals ride along for the reader's eye and are not the test
        ok = None if a is None or b is None else a["flip_rate"] < b["flip_rate"]
        out["clauses"]["pc3"] = {"first": c3["first"], "last": c3["last"], "arm": c3["arm"],
                                 "flip_rates": {r: (None if x is None else x["flip_rate"]) for r, x in ((c3["first"], a), (c3["last"], b))},
                                 "intervals": {r: (None if x is None else [x["flip_ci_low"], x["flip_ci_high"]])
                                               for r, x in ((c3["first"], a), (c3["last"], b))},
                                 "verdict": verdict([ok])}
    if set(out["clauses"]) != {"pc1", "pc2", "pc3"}:
        # a seal file that is there but does not hold the three sealed clauses is the same absence as a missing file
        raise ValueError(f"{f} does not hold the three sealed clauses (pc1_flip_rate_at_first_lag, pc2_shared_below_prompt, "
                         f"pc3_flip_growth); read {sorted(out['clauses'])}")
    return out


def render(s: dict) -> str:
    lines = [f"reduced statistic (C), PipelineRL TIR arm (entry {s['registered_by']}, instrument {s['upstream_sha'][:12]}): greedy "
             "multi-turn tool episodes under an injected prompt cache; verdict on the final-answer flip rate [bootstrap interval]"]
    fid = s["fidelity"]
    lines.append(f"driver fidelity: {fid['turns_matched']}/{fid['n_turns']} turns match upstream's prompt tokens "
                 f"(extend cleanly {fid['turns_extend_cleanly']}; content_eos {s.get('content_eos')}) -> {'ok' if fid['passed'] else 'FAILED'}")
    reg, ran = s.get("readers_registered") or [], s.get("readers_run") or []
    if reg and len(ran) < len(reg):
        lines.append(f"readers run: {len(ran)} of {len(reg)} registered ({', '.join(ran)}); not run: {', '.join(r for r in reg if r not in ran)}")
    for reader, lag in s["lags"].items():
        c = lag["controls"]
        lines.append(f"reader {reader}: n {lag['n']} acc_fresh {lag['acc_fresh']:.1f} MathAnswer {lag['math_answer_rate_fresh']:.3f} "
                     f"turns {lag['mean_turns_fresh']:.2f} cap {lag['cap_rate_fresh']:.3f}; floor flip {c['floor_flip_rate']:.4f} on {c['floor_n']} "
                     f"{'ok' if c['floor_passed'] else 'FAILED'}; identity {'ok' if c['identity_passed'] else 'FAILED'} on {c['control_n']}; "
                     f"positive controls {'fire' if lag['positive_controls_fire'] else 'DO NOT FIRE'}" + ("" if lag["evaluable"] else "  -> UNEVALUABLE"))
        for name, a in lag["arms"].items():
            d = a["delta"]
            lines.append(f"  {name:13s} n {a['n']:5d} flips {a['flip_rate']:.3f} [{a['flip_ci_low']:.3f}, {a['flip_ci_high']:.3f}]  "
                         f"tool-seq flips {a['tool_sequence_flip_rate']:.3f}  acc {a['acc']:.1f} (delta {d['delta_points']:+.2f})  "
                         f"no-answer {a['no_answer_rate']:.3f}  cap {a['cap_rate']:.3f}  span-in-lcp {a['stale_inside_lcp_rate']:.3f}"
                         + ("" if a["rank_correlation_b1"] is None else f"  rho(flip, B1) {a['rank_correlation_b1']:+.3f}"))
    ro = s.get("seal_readout")
    if ro:
        lines.append(f"sealed clauses ({ro['source']}):")
        for k, c in ro["clauses"].items():
            detail = {kk: vv for kk, vv in c.items() if kk != "verdict"}
            lines.append(f"  ({k}) {c['verdict']}  {json.dumps(detail, default=str)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="recompute the reduced statistic (C) from the TIR task records on disk")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "pipelinerl_tir_task.toml"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_task_tir_config(Path(a.config), REPO_ROOT)
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
