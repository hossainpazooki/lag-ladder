"""Recompute the figures README.md's Results tables claim, from this folder's own files; exit 1 on any mismatch.

    python verify_bundle.py                         # hashes, summaries, paired tests, argmax fractions (needs numpy)
    python verify_bundle.py --lag-ladder ../lag-ladder
        # also re-derives every record's answer and `correct` from completion_text with lag-ladder's registered rule
        # (tools/own_run.py final_number, tools/task_swap.py first_hash_number / is_correct), at commit e32d523
    python verify_bundle.py --package ../farhan-run
        # also re-runs the package's own baseline_probe.summarize() and compares it to probe.json field by field

The summary formulas below restate the package's `summarize()` (percentile bootstrap, 2,000 resamples,
numpy default_rng(0)); `--package` checks them against the original. Nothing here needs a GPU or the model.
Not checked here (README.md keys them to their files): which records cannot be rebuilt as tokens and the stop-token
statement (they need the tokenizer), `elapsed_s`, the environment strings, the chronology and the package hashes.
A standalone checker for an outside run: it does not import linear_ceiling and runs from this folder alone.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETS = {   # label -> folder; "reg" = the runbook's Step 1 probe run unedited (not registered anywhere),
    #                     "norp" = the no-repetition-penalty diagnostic
    "reg256": "step1-probe/probe-instruct-256",
    "reg768": "step1-probe/probe-instruct-768",
    "norp256": "diagnostic-no-rep-penalty/probe-norp-256",
    "norp768": "diagnostic-no-rep-penalty/probe-norp-768",
}
# Figures README.md states that are not fields of a probe.json (those are checked against the file itself).
CLAIMS = {
    "correct": {"reg256": 89, "reg768": 129, "norp256": 103, "norp768": 144},
    "correct_first100": {"reg256": 45, "reg768": 65, "norp256": 50, "norp768": 70},
    # paired, same prompt index: (probe right & diagnostic wrong, probe wrong & diagnostic right)
    "discordant": {"256": (17, 31), "768": (17, 32)},
    "mcnemar_exact_p": {"256": 0.0595, "768": 0.0444},
    # paired bootstrap of (diagnostic - probe) correctness, seed 0, 2,000 resamples, 95% percentile interval, points
    "paired_bootstrap": {"256": ("0.0", "13.5"), "768": ("1.0", "14.0")},
    # cap 256: (reached cap, of those correct, ended earlier, of those correct); cap 768: correct among the prompts
    # that reached cap 256; and how many cap-256 completions that ended earlier are identical at cap 768
    "cap_split": {"reg": (92, 15, 108, 74, 55, 108), "norp": (91, 15, 109, 88, 56, 109)},
    "answers_changed": {"256": 119, "768": 80},   # numerically different extracted answers ("29" == "29.00")
    "texts_changed": {"256": 200, "768": 200},
    # within one decoding: cap-768 texts that begin with the cap-256 text, and those identical to it
    "cap768_extends_cap256": {"reg": (200, 109), "norp": (200, 110)},
    "argmax": {   # set -> (records, ids_recovered, tokens, eq_raw_argmax, eq_penalized_argmax,
                  #         records_all_raw, records_all_penalized)
        "registered768": (200, 198, 51783, 48997, 51783, 0, 198), "norp768": (200, 200, 50712, 50712, 48032, 200, 0),
        "registered256": (200, 199, 42930, 40412, 42929, 0, 198), "norp256": (200, 200, 41943, 41943, 39519, 200, 0),
    },
}
FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("OK    " if ok else "FAIL  ") + what)
    if not ok:
        FAILS.append(what)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sums(listing: Path) -> None:
    for line in listing.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            h, name = line.split(None, 1)
            check(sha(listing.parent / name.strip()) == h, f"sha256 {listing.parent.relative_to(HERE)}/{name.strip()}")


def num_eq(a, b) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    correct = np.asarray([bool(r["correct"]) for r in recs], dtype=np.float64)
    rng = np.random.default_rng(0)
    means = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(2000)])
    acc = 100.0 * correct.sum() / n
    uncapped = [r for r in recs if not r["reached_cap"]]
    return {
        "n": n, "accuracy_points": acc,
        "accuracy_ci95": [float(100.0 * np.percentile(means, 2.5)), float(100.0 * np.percentile(means, 97.5))],
        "band": "accuracy-verdict" if acc >= 50.0 else "flips-verdict" if acc >= 20.0 else "stop",
        "cap_rate": sum(bool(r["reached_cap"]) for r in recs) / n,
        "no_answer_rate": sum(r["answer"] is None for r in recs) / n,
        "no_hash_marker_rate": sum(r["answer_first_hash"] is None for r in recs) / n,
        "first_vs_last_hash_disagree_rate":
            sum(r["answer_first_hash"] is not None and r["answer"] != r["answer_first_hash"] for r in recs) / n,
        "mean_new_tokens": float(np.mean([r["n_tokens"] for r in recs])),
        "correct_and_capped": sum(1 for r in recs if r["correct"] and r["reached_cap"]),
        "boxed_marker_rate": sum(r.get("answer_boxed") is not None for r in recs) / n,
        "accuracy_boxed_points": 100.0 * sum(num_eq(r.get("answer_boxed"), r["gold"]) for r in recs) / n,
        "accuracy_uncapped_only_points": 100.0 * sum(r["correct"] for r in uncapped) / max(1, len(uncapped)),
    }


def mcnemar_exact(b: int, c: int) -> float:
    k, m = min(b, c), b + c
    return min(1.0, 2.0 * sum(math.comb(m, i) for i in range(k + 1)) / 2.0 ** m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag-ladder", type=Path, help="lag-ladder checkout at e32d523 (re-derive answers from text)")
    ap.add_argument("--package", type=Path, help="the farhan-run package folder (re-run its summarize())")
    a = ap.parse_args()

    print("== hashes")
    sums(HERE / "SHA256SUMS")
    listed = {l.split(None, 1)[1].strip() for l in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if l.strip()}
    present = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*")
               if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    check(listed == present, f"SHA256SUMS lists every file in this folder except itself ({len(present)} files)"
          + (f"; unlisted {sorted(present - listed)}, missing {sorted(listed - present)}" if listed != present else ""))
    sums(HERE / "step1-probe/probe-instruct-sha256sums.txt")
    sums(HERE / "diagnostic-no-rep-penalty/sha256sums.txt")
    if a.lag_ladder:
        import subprocess
        head = subprocess.run(["git", "-C", str(a.lag_ladder), "rev-parse", "HEAD"], capture_output=True, text=True)
        print(f"INFO  lag-ladder checkout at {head.stdout.strip() or 'unknown'} (the records were scored at e32d523)")

    recs, probe = {}, {}
    for label, folder in SETS.items():
        print(f"== {label} ({folder})")
        path = HERE / folder / "records.jsonl"
        recs[label] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        probe[label] = json.loads((HERE / folder / "probe.json").read_text(encoding="utf-8"))
        r, p = recs[label], probe[label]
        check([x["index"] for x in r] == list(range(200)), f"{label}: 200 records, indices 0..199 in order, no duplicates")
        check(p["records_sha256"] == sha(path), f"{label}: probe.json records_sha256 equals records.jsonl")
        check(p["env"]["dtype"] == "float32" and p["env"]["model"].endswith("@989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
              f"{label}: float32, Qwen2.5-1.5B-Instruct @ 989aa79")
        check(p["prompts_sha256"] == "918c7e67f051d41f7b153911f0497bc62c82b76cb04ef84f6f5ee0e66ab22450",
              f"{label}: prompts file sha256 918c7e67...22450 (full)")
        s = summarize(r)
        for k, v in s.items():
            check(p[k] == v, f"{label}: probe.json {k} = {p[k]} (recomputed {v})")
        check(int(sum(x["correct"] for x in r)) == CLAIMS["correct"][label], f"{label}: {CLAIMS['correct'][label]}/200 correct")
        f100 = int(sum(x["correct"] for x in r[:100]))
        check(f100 == CLAIMS["correct_first100"][label], f"{label}: {f100}/100 correct on the first 100 prompts")
        if a.lag_ladder:
            sys.path.insert(0, str(a.lag_ladder / "tools"))
            from own_run import final_number          # noqa: E402
            from task_swap import first_hash_number, is_correct   # noqa: E402
            bad = [x["index"] for x in r if final_number(x["completion_text"]) != x["answer"]
                   or first_hash_number(x["completion_text"]) != x["answer_first_hash"]
                   or is_correct(x["answer"], x["gold"]) != x["correct"]]
            check(not bad, f"{label}: answer / first-#### / correct re-derived from text by lag-ladder's rule ({len(bad)} differ)")
        if a.package:
            sys.path.insert(0, str(a.package))
            import baseline_probe   # noqa: E402
            ps = baseline_probe.summarize(r)
            check(all(p[k] == v for k, v in ps.items()), f"{label}: package summarize() equals probe.json on every field")

    print("== paired, Step 1 probe vs diagnostic, same prompt index")
    for cap in ("256", "768"):
        reg, nrp = recs["reg" + cap], recs["norp" + cap]
        b = sum(x["correct"] and not y["correct"] for x, y in zip(reg, nrp))
        c = sum(y["correct"] and not x["correct"] for x, y in zip(reg, nrp))
        p = mcnemar_exact(b, c)
        changed = sum(not (num_eq(x["answer"], y["answer"]) or x["answer"] is y["answer"] is None) for x, y in zip(reg, nrp))
        texts = sum(x["completion_text"] != y["completion_text"] for x, y in zip(reg, nrp))
        check((b, c) == CLAIMS["discordant"][cap], f"cap {cap}: discordant {b} right->wrong, {c} wrong->right (net {c - b:+d})")
        check(round(p, 4) == CLAIMS["mcnemar_exact_p"][cap], f"cap {cap}: exact McNemar p = {p:.4f}")
        check(changed == CLAIMS["answers_changed"][cap], f"cap {cap}: extracted answer differs (numerically) on {changed}/200 prompts")
        check(texts == CLAIMS["texts_changed"][cap], f"cap {cap}: completion text differs on {texts}/200 prompts")
        d = np.asarray([float(y["correct"]) - float(x["correct"]) for x, y in zip(reg, nrp)])
        rng = np.random.default_rng(0)
        boots = np.array([d[rng.integers(0, 200, 200)].mean() for _ in range(2000)])
        lo, hi = f"{100 * np.percentile(boots, 2.5):.1f}", f"{100 * np.percentile(boots, 97.5):.1f}"
        check((lo, hi) == CLAIMS["paired_bootstrap"][cap], f"cap {cap}: paired bootstrap of the difference (diagnostic - "
              f"probe), seed 0, 2,000 resamples: {100 * d.mean():+.1f} points, 95% [{lo}, {hi}]")

    for dec in ("reg", "norp"):
        a256, a768 = recs[dec + "256"], recs[dec + "768"]
        pre = sum(y["completion_text"].startswith(x["completion_text"]) for x, y in zip(a256, a768))
        same = sum(x["completion_text"] == y["completion_text"] for x, y in zip(a256, a768))
        check((pre, same) == CLAIMS["cap768_extends_cap256"][dec],
              f"{dec}: {pre}/200 cap-768 texts begin with the cap-256 text, {same} identical")
        capped = [x for x in a256 if x["reached_cap"]]
        early = [x for x in a256 if not x["reached_cap"]]
        got = (len(capped), sum(x["correct"] for x in capped), len(early), sum(x["correct"] for x in early),
               sum(a768[x["index"]]["correct"] for x in capped),
               sum(x["completion_text"] == a768[x["index"]]["completion_text"] for x in early))
        check(got == CLAIMS["cap_split"][dec], f"{dec}: cap 256 reached by {got[0]} ({got[1]} correct), ended earlier "
              f"{got[2]} ({got[3]} correct); at cap 768 the {got[0]} cut prompts score {got[4]}; {got[5]}/{got[2]} "
              f"early-ended completions identical at cap 768")

    print("== argmax provenance checks")
    for fname in ("argmax_check.json", "argmax_check_256.json"):
        d = json.loads((HERE / "diagnostic-no-rep-penalty" / fname).read_text(encoding="utf-8"))
        check(d["penalty_tested"] == 1.1, f"{fname}: penalty tested 1.1 ({d['device']})")
        for k, s in d["sets"].items():
            got = (s["records"], s["ids_recovered"], s["tokens"], s["eq_raw_argmax"], s["eq_penalized_argmax"],
                   s["records_all_raw"], s["records_all_penalized"])
            check(got == CLAIMS["argmax"][k],
                  f"{fname} {k}: records/recovered/tokens/raw/penalized/all-raw/all-penalized = {got}")
            check(math.isclose(s["frac_raw"], s["eq_raw_argmax"] / s["tokens"])
                  and math.isclose(s["frac_penalized"], s["eq_penalized_argmax"] / s["tokens"]),
                  f"{fname} {k}: raw {s['frac_raw']:.4%}, penalized {s['frac_penalized']:.4%}")

    print(f"VERIFY_BUNDLE {'FAILED ' + str(len(FAILS)) if FAILS else 'OK'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
